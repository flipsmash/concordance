"""derive.py — collapsing regular -ly adverbs and "un-" words to an
independently attested root. See the module docstring for why the "un-"
gate is stricter than validity_score._morph_root's: a false positive here
becomes the word's new identity, not just a scoring nudge."""

from __future__ import annotations

from concordance.derive import adverb_to_adjective, derived_root, un_root


def test_adverb_to_adjective_finds_attested_root():
    assert adverb_to_adjective("jauntily") == "jaunty"
    assert adverb_to_adjective("culpably") == "culpable"
    assert adverb_to_adjective("quickly") == "quick"


def test_adverb_to_adjective_none_when_no_attested_root():
    # no WordNet entry for "inobtrusive"/"deathful" — leave the adverb alone
    assert adverb_to_adjective("inobtrusively") is None
    assert adverb_to_adjective("deathfully") is None


def test_un_root_finds_attested_root():
    assert un_root("unhappy") == "happy"
    assert un_root("unsafe") == "safe"
    assert un_root("unwise") == "wise"


def test_un_root_rejects_non_decomposable_words():
    # "der"/"cle"/"ion"/"til" all pass a looser wordset-OR-wordnet-OR-zipf
    # gate (der is in WordNet on its own, cle has web-noise zipf, ion/til
    # are real short words unrelated to under/uncle/union/until) -- the
    # point of the stricter AND-gate is that none of them survive it.
    assert un_root("under") is None
    assert un_root("uncle") is None
    assert un_root("union") is None
    assert un_root("until") is None
    assert un_root("unique") is None
    assert un_root("unicorn") is None


def test_un_root_drops_the_one_known_casualty():
    # "fit" is a real 3-letter root, but the 4-letter floor that keeps
    # "cle"/"der"/"ion" out also excludes it -- documented trade-off.
    assert un_root("unfit") is None


def test_derived_root_adverb_substitution():
    assert derived_root("quickly", "ADV") == ("quick", "ADJ")


def test_derived_root_un_substitution_any_pos():
    assert derived_root("unhappy", "ADJ") == ("happy", "ADJ")


def test_derived_root_chains_un_then_ly():
    # un+(happy+ly): the -ily strip lands on "unhappy" first, which then
    # needs its own "un-" peeled to reach "happy".
    assert derived_root("unhappily", "ADV") == ("happy", "ADJ")


def test_derived_root_chains_ly_then_un():
    # (un+froward)+ly: "unfroward" isn't a WordNet adjective, so the -ly
    # strip on the whole word fails first; the "un-" peel must go first,
    # landing on "frowardly", which then reduces to "froward".
    assert derived_root("unfrowardly", "ADV") == ("froward", "ADJ")


def test_derived_root_none_for_non_decomposable_words():
    assert derived_root("under", "ADP") is None
    assert derived_root("uncle", "NOUN") is None
    assert derived_root("universe", "NOUN") is None


def test_derived_root_none_when_nothing_applies():
    assert derived_root("inobtrusively", "ADV") is None
    assert derived_root("book", "NOUN") is None
