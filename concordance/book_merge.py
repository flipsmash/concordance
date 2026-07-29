"""Detect and compile multi-part books (§ book-merge).

Many books in ./archive/ arrived from Project Gutenberg split across
several files -- "Vol. 1"/"Volume I"/"Part 1."/"Chapters 01 to 05" etc. --
each currently its own separate `book` row, so the corpus double/triple/
N-counts vocabulary that really belongs to one work. `detect_merge_groups`
finds these (same exact title, once the part label is stripped, and same
exact author -- never PLACEHOLDER_AUTHORS); `compile_group` combines their
text into one new file; `concordance.db.merge_book_group` folds the
Postgres records together.

The detection regex was validated against the REAL corpus (12,985 books
with a non-placeholder author), not guessed:
  - 172 groups / 916 files are real multi-part candidates.
  - 30 "lone matches" -- a title matches the suffix pattern but has no
    sibling parts currently in the archive (e.g. "A bankrupt heart, Vol. 2
    (of 3)" -- only volume 2 of 3 exists here). Left untouched.
  - 56 titles have a vol/part/chapter-like keyword but don't match this
    (deliberately conservative, end-anchored) regex -- inspected samples
    confirm these are genuinely different shapes, not regex gaps:
    "Critical Miscellanies, Vol. 1 (of 3), Essay 4" (an essay WITHIN a
    volume, not a simple split) and "The Glebe 191312 (Vol. 1, No. 3)"
    (issue numbering). Better to under-merge than mis-merge these.
  - 28 of the 172 groups have gaps once ranges are correctly expanded
    (e.g. "Diary of Samuel Pepys" has only 13 of up to 72 volumes present).
    These must never be silently compiled into something labeled
    "Complete."
  - Only 2 of the 172 already have a pre-existing unlabeled sibling book
    with the exact stripped title -- rare but real; flagged for a human
    rather than guessed at.
  - Numbering in the wild: zero-padded ("01".."12"), roman numerals
    ("I".."XXVII"), plain ("1".."9"), and ranges with a non-numeric
    terminal ("Chapters 36 to the Last" -- Huckleberry Finn's 8 parts are
    CONTIGUOUS once ranges are expanded, not gappy: expanding only the
    range's start and ignoring its end is an easy mistake that produces
    false "gaps").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .archive_metadata import split_gutenberg_parts
from .db import PLACEHOLDER_AUTHORS

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int:
    s = s.upper()
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN_VALUES.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def num_to_int(s: str) -> int | None:
    """None sentinel = open-ended ('the last'); else int -- roman numerals
    and zero-padded arabic numbers ('01') compare equal to their plain form."""
    s = s.strip().lower()
    if s == "the last":
        return None
    if re.fullmatch(r"[ivxlcdm]+", s):
        return roman_to_int(s)
    return int(s)


# End-anchored on purpose: a trailing "vol/part/chapter NUMBER[-NUMBER][(of N)]"
# clause, with an optional leading "In Three Volumes" note. Deliberately
# conservative -- confirmed live that titles with more complex trailing
# structure ("...Vol. 1 (of 3), Essay 4", "...(Vol. 1, No. 3)") correctly fail
# to match rather than being mis-parsed, which is the safer failure mode here.
_SUFFIX_RE = re.compile(
    r"[\s,.;:—–-]+"
    r"(?:in\s+[\w-]+\s+(?:volumes?|parts?)[\s,.;:—–-]*)?"
    r"(?P<kind>vol(?:ume)?s?\.?|part\.?|chapters?|book)"
    r"\s*\.?\s*"
    r"(?P<num>[IVXLCDM]+|\d+|the\s+last)"
    r"(?:\s*(?:to|-|–)\s*(?P<num2>[IVXLCDM]+|\d+|the\s+last))?"
    r"(?:\s*\(?\s*of\s*(?P<total>\d+)\s*\)?)?"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)

_KEYWORD_RE = re.compile(r"\bvol(ume)?s?\.?\b|\bpart\b|\bchapters?\b", re.IGNORECASE)


def strip_suffix(title: str) -> tuple[str, str, str, str | None, str | None] | None:
    """(title_base, kind, num1_raw, num2_raw, total_raw), or None if the
    title doesn't end in a recognized volume/part/chapter marker."""
    m = _SUFFIX_RE.search(title)
    if not m:
        return None
    base = title[: m.start()].rstrip(" ,.;:—–-").strip()
    if not base:
        return None
    return base, m.group("kind").lower(), m.group("num"), m.group("num2"), m.group("total")


@dataclass
class PartInfo:
    book_id: int
    title: str
    archive_path: str
    num1: int
    num2: int | None   # None = open-ended ("...to the Last")


@dataclass
class CandidateGroup:
    title_base: str
    author: str
    parts: list[PartInfo]
    eligible: bool = False
    skip_reason: str | None = None       # None when eligible; else see module docstring's list
    gap_detail: list[int] = field(default_factory=list)
    ordered_book_ids: list[int] = field(default_factory=list)
    survivor_book_id: int | None = None


def _classify(title_base: str, author: str, parts: list[PartInfo],
              existing_titles: set[str]) -> CandidateGroup:
    group = CandidateGroup(title_base=title_base, author=author, parts=parts)

    if len(parts) == 1:
        group.skip_reason = "lone_match"
        return group

    # An open-ended terminal ("...to the Last") is valid only as the single
    # such part, and only if it's genuinely the group's last (highest) one
    # -- otherwise there's no principled way to know how far it actually
    # extends relative to its siblings.
    open_ended = [p for p in parts if p.num2 is None]
    if len(open_ended) > 1:
        group.skip_reason = "open_ended_conflict"
        return group
    closed_parts = [p for p in parts if p.num2 is not None]
    closed_max = max([p.num2 for p in closed_parts] + [p.num1 for p in closed_parts], default=0)
    if open_ended and open_ended[0].num1 <= closed_max:
        group.skip_reason = "open_ended_conflict"
        return group
    # The open-ended part's own num1 stands in as "the end of the sequence"
    # for range-expansion purposes below -- its true internal span (e.g. how
    # many chapters "36 to the Last" actually covers) doesn't matter for
    # detecting gaps in the OVERALL part sequence, only that it's the final
    # contiguous piece.
    group_max = open_ended[0].num1 if open_ended else closed_max

    # Duplicate-number check BEFORE gap-checking -- a duplicate ("Volume 1"
    # AND "Volume I" both claiming 1) would otherwise silently mask what's
    # really a gap somewhere else in the sequence.
    seen: dict[int, int] = {}
    for p in parts:
        end = p.num2 if p.num2 is not None else group_max
        for n in range(p.num1, end + 1):
            if n in seen and seen[n] != p.book_id:
                group.skip_reason = "duplicate_number"
                return group
            seen[n] = p.book_id

    present = set(seen.keys())
    lo, hi = min(present), max(present)
    missing = sorted(set(range(lo, hi + 1)) - present)
    if missing:
        group.skip_reason = "gap"
        group.gap_detail = missing
        return group

    compiled_title = f"{title_base} (Complete)"
    if title_base in existing_titles:
        group.skip_reason = "unlabeled_sibling_conflict"
        return group
    if compiled_title in existing_titles:
        group.skip_reason = "compiled_title_conflict"
        return group

    group.eligible = True
    group.ordered_book_ids = [p.book_id for p in sorted(parts, key=lambda p: p.num1)]
    group.survivor_book_id = group.ordered_book_ids[0]
    return group


def group_and_classify(rows: list[tuple[int, str, str, str]]) -> list[CandidateGroup]:
    """Pure logic: rows is (book_id, title, author, archive_path) for every
    book with a non-placeholder author. Returns one CandidateGroup per
    (title_base, author) that had at least one part matching the suffix
    regex -- callers filter on `.eligible` for what's actually mergeable and
    inspect `.skip_reason` for the rest."""
    from collections import defaultdict

    parts_by_key: dict[tuple[str, str], list[PartInfo]] = defaultdict(list)
    titles_by_author: dict[str, set[str]] = defaultdict(set)
    for book_id, title, author, archive_path in rows:
        titles_by_author[author].add(title)

    for book_id, title, author, archive_path in rows:
        if author in PLACEHOLDER_AUTHORS:
            continue
        parsed = strip_suffix(title)
        if not parsed:
            continue
        base, kind, num1_raw, num2_raw, total_raw = parsed
        num1 = num_to_int(num1_raw)
        if num1 is None:   # "the Last" as the ONLY number (no num2) can't anchor a sequence
            continue
        # num2 is None ONLY for a genuine open-ended terminal ("...to the
        # Last"); a simple "Volume 3" with no range at all is a closed
        # single-point part (num2 == num1), not open-ended -- conflating
        # the two made every plain, non-range volume look open-ended.
        num2 = num_to_int(num2_raw) if num2_raw else num1
        parts_by_key[(base, author)].append(
            PartInfo(book_id=book_id, title=title, archive_path=archive_path, num1=num1, num2=num2))

    groups = []
    for (base, author), parts in parts_by_key.items():
        existing = {t for t in titles_by_author.get(author, ()) if t not in {p.title for p in parts}}
        groups.append(_classify(base, author, parts, existing))
    return groups


def unmatched_keyword_titles(rows: list[tuple[int, str, str, str]]) -> list[tuple[int, str, str]]:
    """Report-only: titles carrying a vol/part/chapter-like keyword that
    never matched the suffix regex at all -- surfaced for human visibility,
    never auto-processed."""
    out = []
    for book_id, title, author, archive_path in rows:
        if author in PLACEHOLDER_AUTHORS:
            continue
        if strip_suffix(title) is None and _KEYWORD_RE.search(title):
            out.append((book_id, title, author))
    return out


def detect_merge_groups(conn, schema: str) -> list[CandidateGroup]:
    """Queries `book` fresh every call -- scanning `book`, not the
    filesystem, is what makes this idempotent for free: once a group is
    merged, its non-survivor rows are gone, so it structurally cannot
    re-form on a later scan."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, title, author, archive_path FROM {schema}.book "
                    f"WHERE author IS NOT NULL AND author <> ''")
        rows = cur.fetchall()
    return group_and_classify(rows)


def compile_group(group: CandidateGroup, archive_dir: Path) -> Path:
    """Read each part in numeric order, split each into (header, body,
    footer) around its own Gutenberg START/END markers, and assemble ONE
    compiled file: part 1's header (retitled), every part's body joined by
    a blank line, and the LAST part's footer -- not N redundant copies of
    the same license boilerplate."""
    ordered = sorted(group.parts, key=lambda p: p.num1)
    bodies = []
    header = footer = ""
    for i, part in enumerate(ordered):
        raw = (archive_dir / Path(part.archive_path).name).read_text(encoding="utf-8", errors="replace")
        h, b, f = split_gutenberg_parts(raw)
        bodies.append(b)
        if i == 0:
            header = re.sub(re.escape(part.title), group.title_base, h, flags=re.IGNORECASE)
        if i == len(ordered) - 1:
            footer = f

    compiled_text = header + "\n\n".join(bodies) + footer
    dest = archive_dir / f"{group.title_base} (Complete) -- {group.author}.txt"
    dest.write_text(compiled_text, encoding="utf-8")
    return dest
