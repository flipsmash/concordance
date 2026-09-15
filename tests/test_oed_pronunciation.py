"""OED pronunciation reconciliation: pure logic only (vision-LLM transcription
itself needs a live model, see concordance/oed/pronunciation.py's module
docstring). Cases below are drawn from a live sample of real pass1/pass2
disagreements in oed.entry, not invented -- see _reconcile_close's docstring
for why only these two narrow patterns are safe to auto-merge."""
from __future__ import annotations
from concordance.oed import pronunciation as p


# --- _count_nuclei -------------------------------------------------------

def test_count_nuclei_single_monophthong():
    assert p._count_nuclei("tenθ") == 1


def test_count_nuclei_treats_adjacent_diphthong_vowels_as_one_nucleus():
    # "pəʊp" (RP "pope"): əʊ is one diphthong, one syllable -- naively
    # counting vowel characters would wrongly say 2
    assert p._count_nuclei("pəʊp") == 1


def test_count_nuclei_counts_separated_vowel_runs_as_distinct_syllables():
    assert p._count_nuclei("kwɪntɪn") == 2


# --- _reconcile_close: safe merges ----------------------------------------

def test_reconcile_merges_ascii_apostrophe_stress_glyph():
    assert p._reconcile_close("ˈkwɪntɪn", "'kwɪntɪn") == "ˈkwɪntɪn"


def test_reconcile_merges_monosyllable_stress_presence_absence():
    assert p._reconcile_close("tenθ", "ˈtenθ") == "ˈtenθ"
    assert p._reconcile_close("ˈpəʊp", "pəʊp") == "ˈpəʊp"
    assert p._reconcile_close("skræn", "ˈskræn") == "ˈskræn"


def test_reconcile_prefers_the_stress_marked_variant():
    # both order combinations should land on the version WITH the mark
    assert p._reconcile_close("bel", "ˈbel") == "ˈbel"
    assert p._reconcile_close("ˈbel", "bel") == "ˈbel"


# --- _reconcile_close: must NOT merge (real sampled disagreements) -------

def test_reconcile_rejects_polysyllabic_stress_placement_difference():
    # real sample: genuinely different stress position, not noise
    assert p._reconcile_close("ˈskætərɒmɪtə(r", "skætəˈrɒmɪtə(r") is None


def test_reconcile_rejects_vowel_length_difference():
    assert p._reconcile_close("priˈtɛnsɪv", "priːˈtɛnsɪv") is None
    assert p._reconcile_close("ɪˈtɜːnəl", "iːˈtɜːnəl") is None


def test_reconcile_rejects_vowel_identity_difference():
    assert p._reconcile_close("kəˈrɛsənt", "kəˈresənt") is None


def test_reconcile_rejects_dropped_segment():
    assert p._reconcile_close("ˈtrʌkɪŋ", "ˈtrʌklɪŋ") is None
    assert p._reconcile_close("ˈbændəsnætʃ", "ˈændərsnætʃ") is None


def test_reconcile_rejects_unrelated_reads():
    assert p._reconcile_close("ˈnaɪnə(r", "ˈlɪnə(r") is None


def test_reconcile_apostrophe_normalization_can_still_expose_real_disagreement():
    # the apostrophe rule must not paper over a genuine stress-position
    # conflict just because it also involves the ascii-apostrophe glyph
    assert p._reconcile_close("ˈpræktɪk", "præk'tɪk") is None


# --- resolve_pronunciation: end-to-end wiring -----------------------------

def test_resolve_pronunciation_uses_reconciled_value_on_disagreement():
    ipa, needs_review = p.resolve_pronunciation("tenθ", "ˈtenθ")
    assert ipa == "ˈtenθ"
    assert needs_review is False


def test_resolve_pronunciation_still_needs_review_on_real_disagreement():
    ipa, needs_review = p.resolve_pronunciation("kəˈrɛsənt", "kəˈresənt")
    assert ipa is None
    assert needs_review is True


def test_resolve_pronunciation_leading_char_crosscheck_still_applies_to_reconciled_value():
    # the existing leading-schwa cross-check must still run on whatever
    # _reconcile_close produces, not just on a verbatim match -- "bɒnd"/
    # "ˈbɒnd" reconciles cleanly to "ˈbɒnd" (monosyllabic stress-mark-only
    # difference), but raw_ocr here shows a leading char before it that the
    # reconciled IPA doesn't have, which must still override back to review.
    ipa, needs_review = p.resolve_pronunciation("bɒnd", "ˈbɒnd", raw_ocr="(a'bond)")
    assert p._reconcile_close("bɒnd", "ˈbɒnd") == "ˈbɒnd"  # sanity: reconciles on its own
    assert ipa is None
    assert needs_review is True
