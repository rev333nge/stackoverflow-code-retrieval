"""Phase 1 tokenizer: lowercase, split on non-alphanumeric.

Deliberately basic - it shatters code identifiers (pd.merge -> pd, merge;
snake_case -> snake, case). Phase 2 replaces this and measures the gain.
Keep the signature tokenize(str) -> list[str] so it stays swappable.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
