"""Stage 1 — ingest a book and return its chapters as clean-ish text.

Supports EPUB, text-based PDF, and plain .txt (handy for testing). Scanned,
image-only PDFs are detected and refused rather than half-processed with OCR
(a deliberate v1 scope decision).
"""

from __future__ import annotations

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
        return [Chapter(title=path.stem, text=path.read_text(encoding="utf-8", errors="replace"))]
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
    if not chapters:
        raise UnsupportedFormatError(f"No readable text found in {path.name}.")
    return chapters


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
