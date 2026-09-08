"""
Phase 1 - Step 4: turn the raw post HTML into clean plain text.

Reads the filtered parquet from step 3 and writes NEW files - the step-3
originals are never modified, so the cleaning rules can change and this step
can be re-run on its own.

    data/processed/questions.parquet  ->  questions_clean.parquet
    data/processed/answers.parquet    ->  answers_clean.parquet

`body` (raw HTML) is replaced by `body_clean` (plain text). Every other column
is carried over unchanged. Question titles are HTML-entity unescaped.

Cleaning rules (all deliberate - see the Phase 1 notes):
  - the ClickHouse dump byte-escaped the Body (newlines, quotes, backslashes
    and every non-ASCII char written as literal '\\xHH' text). We reverse that
    first so the text is real again (real newlines, "quotes", accents). The
    escaping is consistent - a real backslash was stored as '\\x5C' - so
    decoding every '\\xHH' is lossless. Angle brackets are NOT byte-escaped
    (they use &lt; entities), so decoding never invents fake tags.
  - code is KEPT: <pre> / <code> text passes through verbatim, indentation intact
  - links: keep the visible anchor text, drop the href URL
  - <img> dropped (no text content)
  - block tags (<p>, <li>, <pre>, headings, ...) become newlines so words do
    not run together across them
  - HTML entities decoded (&lt; -> <)
  - runs of 3+ blank lines collapsed to one; trailing spaces stripped
  - letter case preserved (the Phase 2 tokenizer needs camelCase)

The HTML parse is CPU-bound, so it runs across all cores.

Usage:
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

# Elements after which the output should have a line break.
# (<br> is handled separately, before this list is used.)
_BLOCK_TAGS = [
    "p", "div", "li", "ul", "ol", "pre", "blockquote", "table", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr",
]
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_BLANKS = re.compile(r"\n{3,}")

# A "&" that does NOT begin a well-formed entity. Python's html.parser crashes
# on malformed numeric char refs like "&#46backends" (missing ';', trailing
# letters), which real SO posts contain. We escape such stray "&" to "&amp;"
# so they survive as literal text instead of blowing up the parser.
_STRAY_AMP = re.compile(
    r"&(?!(?:#\d+;|#x[0-9a-fA-F]+;|[a-zA-Z][a-zA-Z0-9]*;))"
)
_TAG = re.compile(r"<[^>]+>")
_HEX_ESC = re.compile(r"\\x([0-9A-Fa-f]{2})")


def unescape_bytes(s: str) -> str:
    """Reverse the dump's byte-escaping: every '\\xHH' -> that byte, then read
    the whole thing back as UTF-8.

    The source text is pure ASCII with '\\xHH' escapes, so mapping each escape
    to codepoint U+00HH and then re-encoding as latin-1 rebuilds the exact
    original byte stream; decoding that as UTF-8 restores multi-byte characters
    (smart quotes, accents, dashes) as well as newlines/quotes/backslashes.
    """
    if "\\x" not in s:
        return s
    latin = _HEX_ESC.sub(lambda m: chr(int(m.group(1), 16)), s)
    return latin.encode("latin-1", "ignore").decode("utf-8", "replace")


def clean_html(raw: str | None) -> str:
    """Raw post HTML -> plain text, keeping code and link anchor text."""
    if not raw:
        return ""

    safe = _STRAY_AMP.sub("&amp;", unescape_bytes(raw))

    try:
        soup = BeautifulSoup(safe, "html.parser")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for tag in soup.find_all(_BLOCK_TAGS):
            tag.insert_after("\n")
        # get_text() already: emits anchor text but not href, ignores <img>,
        # decodes entities, and keeps whitespace inside <pre> verbatim.
        text = soup.get_text()
    except Exception:
        # Last-resort fallback for pathological markup: strip tags with a
        # regex and decode entities directly. Loses block-newline handling,
        # but never aborts the run over a single bad document.
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
        # pool.map yields results in input order, so enumerate() gives the
        # right index for each one.
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

    # Build the patch table in one columnar shot: bind the Python lists as
    # DuckDB LISTs and UNNEST them position-wise (row i = ids[i], titles[i], ...).
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
