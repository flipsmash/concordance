from concordance.extract import Chapter, extract, strip_gutenberg_chapters

START = "*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***"
LICENCE = "any binary, compressed, marked up, nonproprietary or proprietary form"


def test_epub_documents_lose_licence_header_and_footer():
    chs = [Chapter("Cover", "The Project Gutenberg eBook of Moby Dick\n" + START + "\nCall me Ishmael."),
           Chapter("Ch 1", "Some years ago."),
           Chapter("Ch 2", "The end of the voyage.\n" + END + "\n" + LICENCE),
           Chapter("Licence", "THE FULL PROJECT GUTENBERG LICENSE " + LICENCE)]
    out = strip_gutenberg_chapters(chs)
    body = "\n".join(c.text for c in out)
    assert [c.title for c in out] == ["Cover", "Ch 1", "Ch 2"]
    assert "Call me Ishmael" in body and "voyage" in body
    assert "nonproprietary" not in body and "Gutenberg" not in body


def test_epub_without_markers_is_unchanged():
    chs = [Chapter("A", "Plain text."), Chapter("B", "More.")]
    assert strip_gutenberg_chapters(chs) == chs


def test_single_document_with_both_markers():
    out = strip_gutenberg_chapters([Chapter("All", "header\n" + START + "\nbody text\n" + END + "\nlicence")])
    assert [c.text.strip() for c in out] == ["body text"]


def test_txt_extract_strips_licence(tmp_path):
    p = tmp_path / "book.txt"
    p.write_text("header\n" + START + "\nCall me Ishmael.\n" + END + "\n" + LICENCE, encoding="utf-8")
    (ch,) = extract(p)
    assert "Ishmael" in ch.text and "nonproprietary" not in ch.text
