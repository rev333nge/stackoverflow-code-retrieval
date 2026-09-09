"""HTML -> plain text for the filtered parquet, writing NEW *_clean.parquet.

body (raw HTML) becomes body_clean; other columns carried over. Keeps code
verbatim and link anchor text (not URLs), drops <img>, decodes entities,
block tags become newlines, case preserved. Parsed across all cores.

    python scripts/clean_text.py
"""

from __future__ import annotations

import html
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
from bs4 import BeautifulSoup

PROCESSED = Path("data/processed")

# block tags -> a newline after them (<br> handled separately)
_BLOCK_TAGS = [
    "p", "div", "li", "ul", "ol", "pre", "blockquote", "table", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr",
]
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_BLANKS = re.compile(r"\n{3,}")

# a "&" not starting a valid entity; escape it so html.parser doesn't choke
# on malformed refs like "&#46backends"
_STRAY_AMP = re.compile(
    r"&(?!(?:#\d+;|#x[0-9a-fA-F]+;|[a-zA-Z][a-zA-Z0-9]*;))"
)
_TAG = re.compile(r"<[^>]+>")
_HEX_ESC = re.compile(r"\\x([0-9A-Fa-f]{2})")


def unescape_bytes(s: str) -> str:
    # The dump stores newlines/quotes/backslashes/non-ASCII as literal "\xHH".
    # Map each back to its byte (via latin-1) and decode UTF-8 to get the real text.
    if "\\x" not in s:
        return s
    latin = _HEX_ESC.sub(lambda m: chr(int(m.group(1), 16)), s)
    return latin.encode("latin-1", "ignore").decode("utf-8", "replace")


def clean_html(raw: str | None) -> str:
    if not raw:
        return ""

    safe = _STRAY_AMP.sub("&amp;", unescape_bytes(raw))

    try:
        soup = BeautifulSoup(safe, "html.parser")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for tag in soup.find_all(_BLOCK_TAGS):
            tag.insert_after("\n")
        # get_text: anchor text without href, no <img>, entities decoded, <pre> kept
        text = soup.get_text()
    except Exception:
        # fallback for markup bs4 still can't parse: regex-strip tags
        text = html.unescape(_TAG.sub(" ", raw))

    text = _TRAILING_WS.sub("\n", text)
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip()


def parallel_clean(bodies: list[str]) -> list[str]:
    total = len(bodies)
    print(f"      cleaning {total:,} bodies across all cores (a few minutes) ...")
    cleaned: list[str] = [""] * total
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        # pool.map keeps input order
        for i, result in enumerate(pool.map(clean_html, bodies, chunksize=1000)):
            cleaned[i] = result
            if (i + 1) % 50_000 == 0:
                print(f"        {i + 1:,}/{total:,}")
    print(f"      done in {time.time() - t0:.0f}s")
    return cleaned


def clean_questions(con: duckdb.DuckDBPyConnection) -> None:
    src = (PROCESSED / "questions.parquet").as_posix()
    dst = (PROCESSED / "questions_clean.parquet").as_posix()
    print("  questions.parquet")

    rows = con.execute(f"SELECT id, title, body FROM read_parquet('{src}')").fetchall()
    ids = [r[0] for r in rows]
    titles = [html.unescape(unescape_bytes(r[1])) if r[1] else "" for r in rows]
    body_clean = parallel_clean([r[2] for r in rows])
    print(f"      sample: {body_clean[0][:160]!r}")

    # UNNEST the lists into a patch table, then join it back on id
    con.execute("DROP TABLE IF EXISTS patch")
    con.execute(
        "CREATE TABLE patch AS "
        "SELECT UNNEST(?) AS id, UNNEST(?) AS title, UNNEST(?) AS body_clean",
        [ids, titles, body_clean],
    )

    print(f"      writing {dst} ...")
    con.execute(f"""
        COPY (
            SELECT q.* EXCLUDE (title, body), p.title, p.body_clean
            FROM read_parquet('{src}') q
            JOIN patch p USING (id)
        ) TO '{dst}' (FORMAT parquet)
    """)
    print(f"      wrote {dst}")


def clean_answers(con: duckdb.DuckDBPyConnection) -> None:
    src = (PROCESSED / "answers.parquet").as_posix()
    dst = (PROCESSED / "answers_clean.parquet").as_posix()
    print("  answers.parquet")

    rows = con.execute(f"SELECT id, body FROM read_parquet('{src}')").fetchall()
    ids = [r[0] for r in rows]
    body_clean = parallel_clean([r[1] for r in rows])
    print(f"      sample: {body_clean[0][:160]!r}")

    con.execute("DROP TABLE IF EXISTS patch")
    con.execute(
        "CREATE TABLE patch AS "
        "SELECT UNNEST(?) AS id, UNNEST(?) AS body_clean",
        [ids, body_clean],
    )

    print(f"      writing {dst} ...")
    con.execute(f"""
        COPY (
            SELECT a.* EXCLUDE (body), p.body_clean
            FROM read_parquet('{src}') a
            JOIN patch p USING (id)
        ) TO '{dst}' (FORMAT parquet)
    """)
    print(f"      wrote {dst}")


def main() -> None:
    for name in ("questions.parquet", "answers.parquet"):
        if not (PROCESSED / name).exists():
            raise SystemExit(f"missing {PROCESSED / name} - run scripts/filter_posts.py first")

    con = duckdb.connect()
    t0 = time.time()

    clean_questions(con)
    clean_answers(con)

    for name in ("questions_clean.parquet", "answers_clean.parquet"):
        mb = (PROCESSED / name).stat().st_size / 1e6
        print(f"  {name}  ({mb:,.0f} MB)")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
