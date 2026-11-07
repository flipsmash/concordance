"""Stage 1 — ingest a book and return its chapters as clean-ish text.

Supports EPUB, text-based PDF, and plain .txt (handy for testing). Scanned,
image-only PDFs are detected and refused rather than half-processed with OCR
(a deliberate v1 scope decision).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chapter:
    title: str
    text: str


class ScannedPDFError(RuntimeError):
    """Raised when a PDF carries images but effectively no extractable text."""


class UnsupportedFormatError(RuntimeError):
    pass


def extract(path: str | Path) -> list[Chapter]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such book: {path}")
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _extract_epub(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in {".txt", ".text", ".md"}:
        from .archive_metadata import strip_gutenberg_boilerplate
        text = strip_producer_credits(strip_gutenberg_boilerplate(path.read_text(encoding="utf-8", errors="replace")))
        return [Chapter(title=path.stem, text=text)]
    raise UnsupportedFormatError(
        f"Concordance reads EPUB, text PDF, and .txt — not '{suffix}'."
    )


def _extract_epub(path: Path) -> list[Chapter]:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path))
    chapters: list[Chapter] = []
    for i, item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), start=1):
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")
        if not text.strip():
            continue
        # Prefer a real heading for the chapter label; fall back to index.
        heading = soup.find(["h1", "h2", "h3"])
        title = heading.get_text(" ", strip=True) if heading else f"Section {i}"
        chapters.append(Chapter(title=title[:80], text=text))
    chapters = strip_gutenberg_chapters(chapters)
    if chapters:
        chapters[0] = Chapter(title=chapters[0].title, text=strip_producer_credits(chapters[0].text))
    if not chapters:
        raise UnsupportedFormatError(f"No readable text found in {path.name}.")
    return chapters


# Volunteer credit paragraphs Gutenberg puts just inside the START marker --
# part of the "body" by the markers, but not the author's text: "Produced by
# Chuck Greif and the Online Distributed Proofreading Team" linked "greif"
# to 145 books. A paragraph is the line through the next blank line.
_CREDIT_RE = re.compile(
    r"^[ \t]*(?:\[?e-?book #\d+\]?|credits?:|produced by|e-?text (?:was )?(?:prepared|produced) by|"
    r"this e-?(?:text|book) (?:was|has been) (?:produced|prepared)|transcribed (?:from .{0,80}? )?by)"
    r"[^\n]*(?:\n(?![ \t]*\n)[^\n]*)*",
    re.IGNORECASE | re.MULTILINE)
_CREDIT_HEAD = 3000   # credits sit at the very start; never touch the rest


def strip_producer_credits(text: str) -> str:
    """Drop volunteer-credit paragraphs from the opening of a Gutenberg text
    (see _CREDIT_RE); the rest of the text is left exactly as it is."""
    head, rest = text[:_CREDIT_HEAD], text[_CREDIT_HEAD:]
    return _CREDIT_RE.sub("", head) + rest


def strip_gutenberg_chapters(chapters: list[Chapter]) -> list[Chapter]:
    """Drop Project Gutenberg's licence header and footer from an EPUB's
    documents: everything before the last START marker and from the last END
    marker on, the same boundaries archive_metadata uses for .txt files.
    Without this the licence's own vocabulary was counted as the book's --
    "nonproprietary" ended up linked to 6,571 books. A book with no markers
    is returned unchanged."""
    from .archive_metadata import _END_RE, _START_RE
    start = end = None                     # (chapter index, offset)
    for i, ch in enumerate(chapters):
        for m in _START_RE.finditer(ch.text):
            start = (i, m.end())
        for m in _END_RE.finditer(ch.text):
            end = (i, m.start())
    if start and end and end <= start:     # a stray END before the real START
        end = None
    out = []
    for i, ch in enumerate(chapters):
        if (start and i < start[0]) or (end and i > end[0]):
            continue
        lo = start[1] if start and i == start[0] else 0
        hi = end[1] if end and i == end[0] else len(ch.text)
        text = ch.text[lo:hi]
        if text.strip():
            out.append(Chapter(title=ch.title, text=text))
    return out


def _extract_pdf(path: Path) -> list[Chapter]:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    parts: list[str] = []
    char_count = 0
    image_pages = 0
    for page in doc:
        txt = page.get_text("text")
        char_count += len(txt.strip())
        if not txt.strip() and page.get_images():
            image_pages += 1
        parts.append(txt)
    doc.close()

    # Heuristic: lots of image-only pages and almost no text => scanned.
    if char_count < 200 and image_pages:
        raise ScannedPDFError(
            f"{path.name} looks scanned (image pages, no extractable text). "
            "OCR is out of scope for v1."
        )
    # PDFs rarely expose clean chapter boundaries; treat the book as one section.
    return [Chapter(title=path.stem, text="\n".join(parts))]
