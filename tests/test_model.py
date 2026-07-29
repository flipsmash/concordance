"""normalize_pos: folds the accumulated mess of abbreviations/case variants
down to one consistent part-of-speech vocabulary."""

from concordance.model import RejectReason, junk_pos_reason, normalize_pos


def test_abbreviations_expand_to_full_words():
    assert normalize_pos("adj") == "adjective"
    assert normalize_pos("adv") == "adverb"
    assert normalize_pos("pron") == "pronoun"
    assert normalize_pos("num") == "numeral"


def test_spacy_universal_tags_get_readable_labels():
    # These came straight from spaCy's coarse tag as a fallback when
    # dictionary enrichment found no part-of-speech of its own.
    assert normalize_pos("adp") == "preposition"
    assert normalize_pos("sconj") == "conjunction"
    assert normalize_pos("propn") == "proper noun"
    assert normalize_pos("x") == "other"


def test_case_variants_and_whitespace_are_folded():
    assert normalize_pos("Noun") == "noun"
    assert normalize_pos("Adjective") == "adjective"
    assert normalize_pos("  VERB  ") == "verb"


def test_already_canonical_is_unchanged():
    assert normalize_pos("noun") == "noun"
    assert normalize_pos("symbol") == "symbol"


def test_blank_stays_blank_not_none():
    # This project's existing "no value" convention for text fields is ''.
    assert normalize_pos("") == ""
    assert normalize_pos(None) == ""


def test_unrecognized_value_passes_through_lowercased():
    assert normalize_pos("Gibberish") == "gibberish"


def test_mw_categories_are_junk_pos():
    # Found live via db.mw_backfill: MW's Collegiate API resolved real
    # leaked proper nouns ("bowell" -> a biographical name, "cochinchina" ->
    # a geographical name, "dilantin" -> a trademark) that this project's
    # existing symbol/proper-noun check didn't recognize, since it only knew
    # Free Dictionary/Wiktionary's simpler category vocabulary.
    assert junk_pos_reason("biographical name") is RejectReason.PROPER_NOUN
    assert junk_pos_reason("geographical name") is RejectReason.PROPER_NOUN
    assert junk_pos_reason("trademark") is RejectReason.PROPER_NOUN


def test_ordinary_pos_is_not_junk():
    assert junk_pos_reason("noun") is None
    assert junk_pos_reason("abbreviation") is None
