"""OED headword detector: the pure text rules, the ink measure, and a recall
check against hand-labelled pages of the real scans (skipped without them)."""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest

from concordance.oed import extract, segment as s
from concordance.oed.config import OedConfig

RNG = ("slenderly", "sleuth")


def test_normalize_strips_dagger_stress_and_punctuation():
    assert s.normalize_headword("t'wombclout.") == ("wombclout", True)       # OCR'd dagger
    assert s.normalize_headword("f'slentwise,") == ("slentwise", True)
    assert s.normalize_headword("slent,") == ("slent", False)
    assert s.normalize_headword("anagra'mmatically,") == ("anagrammatically", False)
    assert s.normalize_headword("womward(e,") == ("womward", False)
    assert s.normalize_headword("incrysta.llizable") == ("incrystallizable", False)
    assert s.normalize_headword("slep(e)ry,") == ("slepery", False)


def test_bare_f_or_t_is_a_dagger_only_when_the_range_says_so():
    assert s.normalize_headword("fslep", RNG) == ("slep", True)             # fslep out of range, slep in
    assert s.normalize_headword("flag", ("fish", "flow")) == ("flag", False)


@pytest.mark.parametrize("token,rng,word,obsolete", [
    ("Hsolenium", ("solemn", "solid"), "solenium", False),      # the first letters are enough to place it
    ("ilken", ("kelt", "kent"), "ken", False),
    ("itorba,", ("tor", "torch"), "torba", False),
    ("fde'posure.", ("depose", "deposit"), "deposure", True),
    ("ilflos-ferri", ("flock", "flood"), "flos-ferri", False),
    ("hide", ("hick", "high"), "hide", False),                  # in range: nothing to strip
])
def test_a_marker_read_as_a_letter_is_stripped(token, rng, word, obsolete):
    assert s.normalize_headword(token, rng) == (word, obsolete)


def test_parallel_mark_is_stripped_only_when_the_range_says_so():
    assert s.normalize_headword("Ilsoleil,", ("sole", "soleil"))[0] == "soleil"
    assert s.normalize_headword("llcatabasion", ("cat", "catabolic"))[0] == "catabasion"
    assert s.normalize_headword("llama,", ("livid", "lobe"))[0] == "llama"
    assert s.normalize_headword("illness", ("ill", "image"))[0] == "illness"


def test_in_range_and_key():
    assert s.in_range("slent", RNG) and s.in_range("sleuth", RNG)
    assert not s.in_range("slow", RNG) and not s.in_range("having", RNG)
    assert s.headword_key("t'wombclout.") == "wombclout"


def test_body_starts_below_the_heads_not_at_a_fixed_fraction():
    heads = [{"text": "SYNTONISM", "bbox": [40, 35, 120, 45]}]
    first_row = {"text": "syntonism", "bbox": [40, 53, 90, 61]}     # above 6% of a 985pt page
    assert s.body_top(heads + [first_row], 985) < 53
    assert s.body_top([first_row], 800) < 53                        # nothing in the head zone: nothing to skip


def test_running_head_range_ignores_a_folio_misread_as_capitals():
    spans = [{"text": "SLENDERLY", "bbox": [40, 10, 120, 20]}, {"text": "691", "bbox": [350, 10, 370, 20]},
             {"text": "SLEUTH", "bbox": [600, 10, 660, 20]}, {"text": "body", "bbox": [40, 300, 80, 310]}]
    assert s.running_head_range(spans, 1000) == ("slenderly", "sleuth")
    assert s.running_head_range(spans[:1], 1000) is None
    misread = [spans[0], {"text": "III", "bbox": [350, 10, 370, 20]}, spans[3]]
    assert s.running_head_range(misread, 1000) is None


@pytest.mark.parametrize("rest,target", [
    (" obs. form of sluice sb.", "sluice"),
    (" variant of slaister Sc.", "slaister"),
    (" obs. ff. sleepery.", "sleepery"),
    (" Sc. var. of brain-wood a. Obs.", "brain-wood"),
    (": see under brand-new.", "brand-new"),
    (", sb. 2 : see branks 1 .", "branks"),
    (" corruption of brand-new.", "brand-new"),
    (" error for swyper, swipper.", "swyper"),
    (" Sc. pa. t. of bring v.", "bring"),
    (" obs. pa. t. of shove v.", "shove"),
    (": see next.", None),
    (", wombly, var. ff. wamble, wambly a.", "wamble"),
    (": see note to uptie sb.", "uptie"),
    (", to-tose, to-tray, etc.: see touch.", "touch"),
])
def test_pointer_names_its_target(rest, target):
    assert s.classify_pointer(rest) == (True, target)


@pytest.mark.parametrize("rest", [
    " ('snek), v. 1 Chiefly Sc. A form of latch.",
    ", v. Obs. exc. arch. [A cant form of slang]",
    ", a. [f. smile sb.] Having a form of a smile.",
    ", sb. A variant form of the dance; see quot.",
    ", used as combining form of L. totus, in certain cases",
    ", sb . 1 Obs. Forms: 1 slaewS, slaewj?, 2 slewSe, 4 sleawj?e",
    ". Obs. Forms: 4 wombecloute, 5 wamclowte, womclotte. [f. womb sb.",
    " (gæs), a. [ad. L.: see -ize] Of the stomach.",
])
def test_real_entries_are_not_pointers(rest):
    assert s.classify_pointer(rest) == (False, None)


def test_stroke_width_measures_bold_as_thicker():
    gray = np.full((40, 200), 255, dtype=np.uint8)
    gray[:, 10:12] = gray[:, 20:22] = gray[:, 30:32] = 0        # 2px strokes
    gray[:, 110:115] = gray[:, 120:125] = gray[:, 130:135] = 0  # 5px strokes
    thin = extract.stroke_width(gray, (5, 0, 40, 40), 1)
    thick = extract.stroke_width(gray, (105, 0, 140, 40), 1)
    assert thin == pytest.approx(2) and thick == pytest.approx(5)
    assert extract.stroke_width(gray, (60, 0, 90, 40), 1) is None      # no ink


def test_flattening_reads_grey_ink_on_a_shadowed_page_like_black_on_white():
    clean = np.full((200, 400), 240, dtype=np.uint8)
    clean[:, 40:42] = clean[:, 50:52] = clean[:, 60:62] = 20        # thin strokes
    clean[:, 240:244] = clean[:, 250:254] = clean[:, 260:264] = 20  # bold strokes
    # A lighter scan: dimmer paper, grey ink, and a spine shadow across the left half.
    faint = (clean * (150 / 240)).astype(np.uint8)
    faint[:, :200] = (faint[:, :200] * 0.6).astype(np.uint8)
    for gray in (clean, faint):
        flat = extract.flatten_background(gray, 48)
        assert extract.stroke_width(flat, (30, 0, 70, 200), 1) == pytest.approx(2)
        assert extract.stroke_width(flat, (230, 0, 270, 200), 1) == pytest.approx(4)
    assert (faint[:, :30] < extract.INK_DARK_LEVEL).all()       # paper the fixed level reads as ink
    assert not (extract.flatten_background(faint, 48)[:, :30] < extract.INK_DARK_LEVEL).any()


@pytest.mark.parametrize("row,expected", [
    ("bad (baed), a. (and sb.) Also 4-6 badd(e.", True),
    ("pink-eyed ('pnjkaid), a . 1 Obs. exc. dial.", True),
    ("breek (bri:k). Forms: 3-6 breke,", True),
    ("hooter ('hu:t9(r)). [f. hoot v.", True),
    ("mossful('mDsful), a. [f. moss sb.^", True),
    ("caper (see bean sb.); caper-bush,", False),
    ("everlasting (see the senses of the adj.). a. In", False),
    ("pace-board (cf. senses 9, 11); pace-bowling", False),
    ("pyrrhocorax (Pliny), a Gr. word", False),
    ("psychodrama (sense 2).", False),
    ("relieve (care). Obs,", True),     # shape alone passes; these are refused for not being bold
])
def test_pronunciation_opening(row, expected):
    assert bool(s._PRON_OPENING_RE.match(row)) is expected


def _span(text, x, y, w=30):
    return {"text": text, "bbox": [x, y, x + w, y + 8], "size": 8.0}


def _row(*spans):
    return {"spans": list(spans), "yc": spans[0]["bbox"][1] + 4, "n": len(spans)}


def _find(rows, bold_words, thin_prefix=0):
    """One column of rows at the margin x=20, with `bold_words` drawn heavy,
    except that many leading characters' worth of each (a ‖ mark)."""
    gray = np.full((1000, 400), 255, dtype=np.uint8)
    z = 1
    cfg = OedConfig(ink_zoom=z)
    for row in rows:
        for span in row["spans"]:
            x0, y0, x1, y1 = (int(v) for v in span["bbox"])
            heavy = span["text"].strip() in bold_words
            step, width = (6, 4) if heavy else (6, 2)
            thin_end = x0 + (x1 - x0) * thin_prefix // len(span["text"].strip()) if heavy else x0
            for x in range(x0, x1 - width, step):
                gray[y0:y1, x:x + (2 if x < thin_end else width)] = 0
    return s.find_headwords([rows], [20.0], 3, cfg, 1000, gray)


def _page(*extra):
    """Body rows at a 10pt pitch; `extra` rows are placed below a 16pt gap."""
    rows = [_row(_span("SLENDERLY", 20, 10, 60)), _row(_span("SLEUTH", 340, 10, 50))]
    y = 100
    for _ in range(4):
        rows.append(_row(_span("plain", 20, y), _span("body", 60, y), _span("text", 100, y)))
        y += 10
    for e in extra:
        y += 6
        rows.append(_row(*[_span(t, 20 + i * 40, y) for i, t in enumerate(e)]))
        y += 10
    return rows


def test_bold_flush_row_below_a_gap_is_a_headword():
    hits = _find(_page(["slent,", "sb.", "Obs."]), {"slent,"})
    assert [(h["headword"], h["obsolete"]) for h in hits] == [("slent", False)]


def test_a_pronunciation_opening_needs_no_extra_leading_but_plain_bold_does():
    tight = lambda *words: _page() + [_row(*[_span(t, 20 + i * 40, 140) for i, t in enumerate(words)])]
    hits = _find(tight("slent", "(slent),", "sb.", "Obs."), {"slent"})
    assert [h["headword"] for h in hits] == ["slent"]
    assert _find(tight("slent,", "sb.", "Obs."), {"slent,"}) == []


def test_thin_row_is_not_a_headword_however_it_reads():
    assert _find(_page(["slent,", "sb.", "Obs."]), set()) == []


def test_split_dagger_is_stepped_over_and_recorded():
    hits = _find(_page(["t", "'sleng,", "v.", "Obs."]), {"'sleng,"})
    assert [(h["headword"], h["obsolete"]) for h in hits] == [("sleng", True)]
    assert hits[0]["text"] == "t 'sleng,"


def test_stacked_markers_are_all_stepped_over():
    hits = _find(_page(["t", "'", "sleng,", "v.", "Obs."]), {"sleng,"})
    assert [(h["headword"], h["obsolete"]) for h in hits] == [("sleng", True)]


def test_a_first_letter_split_off_after_a_mark_stays_with_the_word():
    hits = _find(_page(["t", "s", "lent,", "v.", "Obs."]), {"s", "lent,"})
    assert [(h["headword"], h["obsolete"]) for h in hits] == [("slent", True)]


def test_a_combinations_list_item_does_not_start_an_entry():
    assert _find(_page(["resin,", "any", "of", "various"]), {"resin,"}) == []
    assert _find(_page(["slent,", "any", "of", "various"]), {"slent,"})      # in range: the comma will do
    assert _find(_page(["succeed,", "v.", "Obs."]), {"succeed,"})            # out of range, but an entry opening


def test_a_parallel_mark_does_not_dilute_the_bold_word_after_it():
    hits = _find(_page(["sleuth,", "sb."], ["Ilslent", "(slent),", "sb."]), {"sleuth,", "Ilslent"}, thin_prefix=2)
    assert [h["headword"] for h in hits] == ["sleuth", "slent"]


def test_short_headword_needs_to_read_like_an_entry():
    assert _find(_page(["sle,", "v.", "Obs."]), {"sle,"})
    assert not _find(_page(["sle", "plain", "text"]), {"sle"})


def test_out_of_range_bold_prose_is_dropped_but_a_real_entry_stays():
    assert not _find(_page(["succeeding", "plain", "text"]), {"succeeding"})
    assert not _find(_page(["with", "alternate", "text"]), {"with"})


def test_multi_word_headword_and_pointer_spellings():
    hits = _find(_page(["slepe", "man,", "sb."]), {"slepe", "man,"})
    assert hits[0]["headword"] == "slepe man"
    hits = _find(_page(["sleuce,", "sleugh,", "obs.", "ff.", "sluice."]), {"sleuce,", "sleugh,"})
    assert hits[0]["headword"] == "sleuce" and hits[0]["also"] == ["sleugh"]


# Hand-labelled against the page images. A pointer is any one-line cross-reference.
D = str.split
LABELLED = {
    ("02.pdf", 500): (D("branfulness brangle brangle brangle brangle branglement brangler brangling brangling brangling "
                        "brangling brank brank brank brank branks branks brankursine branky branle branle branlie "
                        "branling branling branner branny"),
                      D("branewod branfirenew brang brangill brank brankard branknew branling branne brannew branning")),
    ("15.pdf", 700): (D("slenderly slenderness slenderwise sleng slenk slenker slent slent slent slent slenting "
                        "slentwise slep slepe slerg slet sletch sleuth"),
                      D("slenter slepe slepery slepyr slester sletch sleuce sleugh sleughi")),
    ("19.pdf", 300): (D("upcovering upcreeper upcurl upcurling upcurrent upcurved upcut updart updatable update update "
                        "updip updo updoming updraught updraw updrawn updress updried updrinking"),
                      D("upcyne updelve upeat upen")),
    ("20.pdf", 500): (D("wombclout wombed wombful wombful womble wombling wombtie womby womenfolk womenish womenkin "
                        "womenkind womens womoonless won"),
                      D("wombill womble wome womell woment womerah womet wommal wommil wommle wompam womple womward "
                        "womyl")),
    ("06.pdf", 400): (D("gastrocnemial gastrocnemian gastrocnemius gastrodisc gastrodynia gastroid gastrolater "
                        "gastrolith gastrology gastromalacia gastromancy gastromantic gastronome gastronomer "
                        "gastronomic gastronomist gastronomous gastronomy gastropathy gastrophilanthropist gastrophile "
                        "gastrorrhaphy gastroscope gastroscopy gastrosoph gastrostomy gastrotomy gastrozooid"),
                      D("gastropod")),
    ("10.pdf", 900): (D("oration orational orationer oratist oratiuncle orator oratorial oratorially oratorian oratoric "
                        "oratorical oratorically oratorio"), []),
    ("13.pdf", 150): (D("raised raisedevil raisedly raisedness raiser raisevelvet raish raisin"), []),
    # Vols 17-18: a lighter scan, with the ink near the old fixed level and a spine shadow.
    ("17.pdf", 500): (D("syntonism syntonize syntono syntonous syntony syntractrix syntrierarch syntrochite syntrophy "
                        "syntropic syntype synulotic synusia synusiast syon syph sypher syphilide syphilis"),
                      D("syntoxoid synu synuwe synvy synyght synys synyster syon syour sypar sype syper syphareit "
                        "sypher")),
    ("18.pdf", 300): (D("totipresent totive toto toto toto toto totoll totonac totora totquot totread totsiens totsy "
                        "totted tottenham totter totter totter"),
                      D("totitive totle toto totog totorve totsane totter")),
    ("03.pdf", 800): (D("constitutioned constitutioner constitutioning constitutionist constitutionless constitutive "
                        "constitutively constitutiveness constitutor constrain"), []),
}


def test_recall_and_precision_on_labelled_pages():
    fitz = pytest.importorskip("fitz")
    from concordance.oed.pipeline import segment_page
    root = Path(__file__).resolve().parent.parent / "dictionaries"
    if not all((root / name).exists() for name, _ in LABELLED):
        pytest.skip("OED scans not present")
    cfg = OedConfig()
    found = total = extras = pointers = pointer_total = misclassified = 0
    for (name, page_no), (defs, ptrs) in LABELLED.items():
        _ordered, entries = segment_page(fitz.open(root / name)[page_no], cfg)
        keys = collections.Counter(s._key(e["headword"]) for e in entries)
        want_d, want_p = collections.Counter(defs), collections.Counter(ptrs)
        hit_d = want_d & keys
        hit_p = want_p & (keys - hit_d)
        found += sum(hit_d.values())
        total += len(defs)
        pointers += sum(hit_p.values())
        pointer_total += len(ptrs)
        extras += sum((keys - hit_d - hit_p).values())
        only_def = set(want_d) - set(want_p)
        misclassified += sum(1 for e in entries if e["pointer"] and s._key(e["headword"]) in only_def)
    assert found / total >= 0.93, f"definition recall {found}/{total}"
    assert pointers / pointer_total >= 0.85, f"pointer recall {pointers}/{pointer_total}"
    assert extras <= len(LABELLED), f"{extras} unexplained hits"     # mostly real entries with OCR'd spellings
    assert misclassified == 0


def test_running_head_range_ignores_a_citation_numeral_on_the_first_body_row():
    def span(text, x0, y0):
        return {"text": text, "bbox": [x0, y0, x0 + 50, y0 + 8]}
    spans = [span("DEPOSITURE", 40, 20), span("484", 280, 20), span("DEPRAVATIVE", 520, 20),
             span("he", 40, 38), span("left,", 300, 38), span("VIII", 530, 38)]
    assert s.running_head_range(spans, 800) == ("depositure", "depravative")


@pytest.mark.parametrize("token, rng, expected", [
    ("fde'post,", ("depositure", "depravative"), ("depost", True)),
    ("IjKalpa", ("kalmuck", "kamala"), ("kalpa", False)),
    ("Uplumula", ("plumpy", "plunder"), ("plumula", False)),
    ("J-fortu'nation.", ("fortune", "fossil"), ("fortunation", False)),
])
def test_marks_glued_to_the_word_are_stripped_by_the_page_range(token, rng, expected):
    assert s.normalize_headword(token, rng) == expected
