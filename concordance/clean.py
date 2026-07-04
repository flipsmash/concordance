"""Stage 2 — repair extracted text.

The single biggest source of fake candidate words is layout damage: words
split across line wraps (``inter-\nesting``) and ligature glyphs (``ﬁ`` ``ﬂ``).
Fixing them here means they never reach the validity gate as false rejects.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures that NFKC would fold anyway, kept explicit for clarity/coverage.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}

# Fancy punctuation -> ASCII so tokenizers and dictionary lookups behave.
_PUNCT = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}

# "inter-\n esting" -> "interesting"  (soft hyphen at a line break)
_HYPHEN_WRAP = re.compile(r"(\w+)[­-]\s*\n\s*(\w+)")
# collapse runs of whitespace, but keep paragraph breaks meaningful
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    for src, dst in _PUNCT.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    # Rejoin hyphenated line-wraps repeatedly (handles chains).
    prev = None
    while prev != text:
        prev = text
        text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()
