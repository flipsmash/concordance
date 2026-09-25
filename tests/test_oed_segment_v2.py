"""v2 OED headword detector rules (`oed-ingest --add-missing`) -- pure."""
from __future__ import annotations

from concordance.oed import segment as s

RNG = ("slenderly", "sleuth")


def test_normalize_strips_dagger_stress_and_punctuation():
    assert s.normalize_headword("t'wombclout.") == ("wombclout", True)       # OCR'd dagger
    assert s.normalize_headword("f'slentwise,") == ("slentwise", True)
    assert s.normalize_headword("slent,") == ("slent", False)
    assert s.normalize_headword("anagra'mmatically,") == ("anagrammatically", False)
    assert s.normalize_headword("womward(e,") == ("womward", False)
    assert s.normalize_headword("incrysta.llizable") == ("incrystallizable", False)


def test_bare_f_or_t_is_a_dagger_only_when_the_range_says_so():
    assert s.normalize_headword("fslep", RNG) == ("slep", True)             # fslep out of range, slep in
    assert s.normalize_headword("flag", ("fish", "flow")) == ("flag", False)


def test_in_range_and_key():
    assert s.in_range("slent", RNG) and s.in_range("sleuth", RNG)
    assert not s.in_range("slow", RNG) and not s.in_range("having", RNG)
    assert s.headword_key("t'wombclout.") == "wombclout"


def test_running_head_range():
    spans = [{"text": "SLENDERLY", "bbox": [40, 10, 120, 20]}, {"text": "691", "bbox": [350, 10, 370, 20]},
             {"text": "SLEUTH", "bbox": [600, 10, 660, 20]}, {"text": "body", "bbox": [40, 300, 80, 310]}]
    assert s.running_head_range(spans, 1000) == ("slenderly", "sleuth")
    assert s.running_head_range(spans[:1], 1000) is None


def test_pointer_and_continuation_rules():
    for rest in (": see snar v.", ", adv.: see snog.", ", variant of scary a.", " see snobberly adv.",
                 ", irreg. var. sleaved."):
        assert s._SEE_POINTER_RE.match(rest), rest
    for rest in (", obs. Sc. f. soar v.", ", obs. pa. t. of shove v.", ", var. of wayment"):
        assert s._XREF_RE.search(rest), rest
    for rest in (" sb. ] 1. trans.", " a.] One who", " (6 -lie); 5-6", ", 7 shaftmont"):
        assert s._CONTINUATION_RE.match(rest), rest
    for rest in (" ('snek), v. 1 Chiefly Sc.", ", v. Obs. exc. arch. [A cant", ", a. [f. smile sb.]"):
        assert not s._CONTINUATION_RE.match(rest) and not s._SEE_POINTER_RE.match(rest), rest
    assert s._PRON_START_RE.match(" ('snek), v.")
