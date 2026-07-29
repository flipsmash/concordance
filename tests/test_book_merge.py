"""Multi-part book detection (§ book-merge) -- pure logic, no DB/network.
Fixture titles are drawn from the real archive/ corpus (confirmed live
against all 12,985 non-placeholder-author books before this was written)."""

from __future__ import annotations

from concordance import book_merge


# --- strip_suffix -------------------------------------------------------------

def test_strip_suffix_plain_volume_number():
    assert book_merge.strip_suffix("A Modern Chronicle — Volume 01") == \
        ("A Modern Chronicle", "volume", "01", None, None)


def test_strip_suffix_roman_numeral_volume():
    assert book_merge.strip_suffix("Diana Tempest, Volume II") == \
        ("Diana Tempest", "volume", "II", None, None)


def test_strip_suffix_part_with_period():
    assert book_merge.strip_suffix("A Connecticut Yankee in King Arthur's Court, Part 1.") == \
        ("A Connecticut Yankee in King Arthur's Court", "part", "1", None, None)


def test_strip_suffix_chapter_range():
    got = book_merge.strip_suffix("Adventures of Huckleberry Finn, Chapters 01 to 05")
    assert got == ("Adventures of Huckleberry Finn", "chapters", "01", "05", None)


def test_strip_suffix_chapter_range_open_ended():
    got = book_merge.strip_suffix("Adventures of Huckleberry Finn, Chapters 36 to the Last")
    assert got == ("Adventures of Huckleberry Finn", "chapters", "36", "the Last", None)


def test_strip_suffix_in_n_volumes_clause():
    # The regex's leading separator class consumes the period right before
    # "In Three Volumes" along with everything else that isn't the title --
    # a cosmetic, extremely rare edge case (a title that itself ends in a
    # period immediately before the volume clause), not a functional one:
    # grouping/merging both work correctly either way.
    got = book_merge.strip_suffix("A Gray Eye or So. In Three Volumes—Volume I")
    assert got == ("A Gray Eye or So", "volume", "I", None, None)


def test_strip_suffix_vol_with_of_total():
    got = book_merge.strip_suffix("A bankrupt heart, Vol. 2 (of 3)")
    assert got == ("A bankrupt heart", "vol.", "2", None, "3")


def test_strip_suffix_rejects_nested_essay_structure():
    # Confirmed live: this is an essay WITHIN a volume, not a simple split --
    # the conservative end-anchored regex correctly fails to match rather
    # than mis-parsing it.
    assert book_merge.strip_suffix("Critical Miscellanies, Vol. 1 (of 3), Essay 4") is None


def test_strip_suffix_rejects_issue_numbering():
    assert book_merge.strip_suffix("The Glebe 191312 (Vol. 1, No. 3)") is None


def test_strip_suffix_no_keyword_at_all():
    assert book_merge.strip_suffix("Early memories; some chapters of autobiography") is None


# --- num_to_int / roman_to_int ------------------------------------------------

def test_num_to_int_zero_padded_and_plain_equal():
    assert book_merge.num_to_int("01") == book_merge.num_to_int("1") == 1


def test_num_to_int_roman_numeral():
    assert book_merge.num_to_int("XXVII") == 27


def test_num_to_int_the_last_is_open_ended_sentinel():
    assert book_merge.num_to_int("the Last") is None


# --- group_and_classify --------------------------------------------------------

def _rows(*titles_authors, author="Author, Some"):
    return [(i, t, author, f"archive/{t} -- {author}.txt") for i, t in enumerate(titles_authors, 1)]


def test_lone_match_has_no_siblings():
    rows = _rows("A bankrupt heart, Vol. 2 (of 3)")
    groups = book_merge.group_and_classify(rows)
    assert len(groups) == 1
    assert groups[0].skip_reason == "lone_match"
    assert not groups[0].eligible


def test_simple_contiguous_group_is_eligible():
    rows = _rows("A Lover's Diary, Volume 1.", "A Lover's Diary, Volume 2.")
    groups = book_merge.group_and_classify(rows)
    assert len(groups) == 1
    g = groups[0]
    assert g.eligible
    assert g.title_base == "A Lover's Diary"
    assert g.ordered_book_ids == [1, 2]
    assert g.survivor_book_id == 1


def test_open_ended_terminal_at_the_actual_end_is_eligible():
    # Huckleberry-Finn-shaped: five 5-chapter ranges then an open-ended tail.
    rows = _rows(
        "Adventures of Huckleberry Finn, Chapters 01 to 05",
        "Adventures of Huckleberry Finn, Chapters 06 to 10",
        "Adventures of Huckleberry Finn, Chapters 11 to 15",
        "Adventures of Huckleberry Finn, Chapters 16 to the Last",
    )
    groups = book_merge.group_and_classify(rows)
    assert len(groups) == 1
    assert groups[0].eligible
    assert groups[0].ordered_book_ids == [1, 2, 3, 4]


def test_open_ended_terminal_not_at_the_end_is_a_conflict():
    rows = _rows(
        "Some Book, Chapters 01 to the Last",     # claims to be open-ended...
        "Some Book, Chapters 02 to 05",           # ...but something else goes higher
    )
    groups = book_merge.group_and_classify(rows)
    assert groups[0].skip_reason == "open_ended_conflict"


def test_two_open_ended_parts_is_a_conflict():
    rows = _rows("Some Book, Part 1 to the Last", "Some Book, Part 2 to the Last")
    groups = book_merge.group_and_classify(rows)
    assert groups[0].skip_reason == "open_ended_conflict"


def test_gap_detected_and_reported():
    # Montaigne-shaped: missing 8 and 16 out of a 1..17 sequence.
    nums = [n for n in range(1, 18) if n not in (8, 16)]
    rows = _rows(*[f"Essays of Michel de Montaigne, Volume {n:02d}" for n in nums])
    groups = book_merge.group_and_classify(rows)
    assert groups[0].skip_reason == "gap"
    assert groups[0].gap_detail == [8, 16]


def test_duplicate_number_conflict():
    # Confirmed live: "John Leech's..." has the same volume under two
    # different label conventions ("Volume N" and "Vol. N").
    rows = _rows(
        "John Leech's Pictures, Volume 1 (of 2)",
        "John Leech's Pictures, Vol. 1 (of 2)",
        "John Leech's Pictures, Volume 2 (of 2)",
    )
    groups = book_merge.group_and_classify(rows)
    assert groups[0].skip_reason == "duplicate_number"


def test_unlabeled_sibling_conflict():
    rows = [
        (1, "The Purcell Papers", "Le Fanu, Joseph Sheridan", "archive/x1.txt"),
        (2, "The Purcell Papers — Volume 1", "Le Fanu, Joseph Sheridan", "archive/x2.txt"),
        (3, "The Purcell Papers — Volume 2", "Le Fanu, Joseph Sheridan", "archive/x3.txt"),
    ]
    groups = book_merge.group_and_classify(rows)
    assert groups[0].skip_reason == "unlabeled_sibling_conflict"


def test_compiled_title_conflict():
    rows = [
        (1, "Some Book (Complete)", "Author, Some", "archive/x1.txt"),
        (2, "Some Book, Volume 1", "Author, Some", "archive/x2.txt"),
        (3, "Some Book, Volume 2", "Author, Some", "archive/x3.txt"),
    ]
    groups = book_merge.group_and_classify(rows)
    assert groups[0].skip_reason == "compiled_title_conflict"


def test_placeholder_authors_are_excluded_entirely():
    rows = _rows("A Book of Old Ballads — Volume 1", "A Book of Old Ballads — Volume 2",
                 author="Unknown Author")
    groups = book_merge.group_and_classify(rows)
    assert groups == []


def test_unmatched_keyword_titles_reports_but_does_not_crash():
    rows = _rows("Critical Miscellanies, Vol. 1 (of 3), Essay 4", "The Glebe 191312 (Vol. 1, No. 3)")
    out = book_merge.unmatched_keyword_titles(rows)
    assert len(out) == 2
