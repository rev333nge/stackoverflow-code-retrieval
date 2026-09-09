"""Phase 2 tokenizer: code-aware, emits parts AND the whole compound.

  np.array   -> np, array, np.array
  read_csv   -> read, csv, read_csv
  DataFrame  -> data, frame, dataframe
  df.groupby -> df, groupby, df.groupby
Pure-numeric tokens (0, 42, 1.5) are dropped. Everything lowercased.

(Stemming was tried and dropped: it hurt every metric — code search relies on
exact technical terms, and blurring them lost more than form-matching gained.)

Keeping the signature tokenize(str) -> list[str] so it stays swappable.
The Phase 1 basic tokenizer was: re.findall(r"[a-z0-9]+", text.lower()).
"""

from __future__ import annotations

import re

# an identifier run, optionally joined by . or _ : pd.merge, read_csv, np.array
_CHUNK = re.compile(r"[A-Za-z0-9]+(?:[._][A-Za-z0-9]+)*")
_SEP = re.compile(r"[._]")
# camelCase / PascalCase / acronym boundaries: DataFrame -> Data, Frame ;
# HTMLParser -> HTML, Parser ; also isolates digit runs
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _is_numeric(s: str) -> bool:
    return _SEP.sub("", s).isdigit()


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for chunk in _CHUNK.findall(text):
        low = chunk.lower()
        has_sep = "." in chunk or "_" in chunk

        # parts: split on . and _, then split camelCase
        subs: list[str] = []
        for seg in _SEP.split(chunk):
            if seg:
                subs.extend(_CAMEL.findall(seg))
        for sp in subs:
            t = sp.lower()
            if not t.isdigit():
                out.append(t)

        # the whole compound, kept alongside its parts
        if has_sep:
            if not _is_numeric(chunk):
                out.append(low)          # np.array, read_csv
        elif len(subs) > 1 and not low.isdigit():
            out.append(low)              # dataframe (camelCase joined)

    return out
