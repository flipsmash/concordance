"""Putting back the linked words the old local Wiktionary table dropped
(concordance/wiktgaps.py)."""

from concordance.wiktgaps import repair


def links(*words):
    return frozenset(words)


def test_restores_dropped_links_inside_a_clause():
    assert repair("The or of a.", [("The author or speaker of a prologue.",
                                    links("author", "speaker", "prologue"))]) == "The author or speaker of a prologue."
    assert repair("Of or derived from or.", [("Of or derived from lye or wood ashes.",
                                             links("lye", "wood", "ashes"))]) == "Of or derived from lye or wood ashes."


def test_a_gap_at_the_end_counts_only_after_a_dangling_word():
    assert repair("A member of a.", [("A member of a slave patrol.", links("slave", "patrol"))]) \
        == "A member of a slave patrol."
    assert repair("The.", [("The sumpit.", links("sumpit"))]) == "The sumpit."


def test_rewordings_are_not_damage():
    # missing words that weren't links: Wiktionary reworded the gloss since
    assert repair("To bounce, rebound or ricochet.", [("To make bounce, rebound or ricochet.", links("bounce"))]) is None
    assert repair("A wing.", [("The joint of a bird's wing farthest from the body.", links("joint", "wing"))]) is None
    # a longer ending on a complete clause is a later edit too
    assert repair("A badger.", [("A badger or similar animal.", links("animal"))]) is None


def test_inline_link_asides_are_dropped_and_pointers_keep_their_head():
    assert repair("The manager or owner of a.",
                  [("The manager or owner of a wharf (“artificial landing place for ships”).", links("wharf"))]) \
        == "The manager or owner of a wharf."
    assert repair("Variant spelling of abnormality — The or quality of being abnormal.",
                  [("The state or quality of being abnormal.", links("state"))]) \
        == "Variant spelling of abnormality — The state or quality of being abnormal."


def test_only_the_damaged_clause_changes():
    assert repair("A firm.; A of the.", [("A firm handshake.", links("handshake")),
                                         ("A head of the barrel.", links("head", "barrel"))]) \
        == "A firm.; A head of the barrel."
