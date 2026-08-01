"""Cross-entry ordering check: a real dictionary is alphabetical, so an
entry that's out of order relative to its neighbors is a strong signal it's
not a real entry at all -- the same class of noise
parse.looks_like_definition_entry catches (a body-text word coincidentally
followed by a bracket or POS token), just caught from a different angle.
Needs whole-volume context (every entry's position relative to every
other), so unlike looks_like_definition_entry this can't run per-page
during ingestion -- it's a post-processing pass over a volume's full,
already-ordered (by id, i.e. reading order) entry list.

Granularity is deliberately coarse: FIRST LETTER only, not the full
headword. A full-word check was tried first and rejected -- it found a real
failure mode. "abnormalism" (a genuine entry with a real resolved
pronunciation) sits between the real "abnormal" and a fake "abnormal"
fragment (leftover etymology text that happens to start with the same
root word). All three have nearly the same alphabetical key, and a
full-word longest-non-decreasing-subsequence has no way to prefer the real
entry over the fake one when both produce the same total sequence length --
it arbitrarily dropped the real "abnormalism" in testing. First-letter
granularity sidesteps this entirely: "abnormal" and "abnormalism" share a
first letter, so they never compete for the same slot. The tradeoff
(accepted -- Brian's call) is that same-letter noise like that fake
"abnormal" fragment slips through this check; it's still usually caught by
looks_like_definition_entry's bracket/POS check instead.

Uses the longest non-decreasing subsequence (LNDS), not a running-maximum
comparison against the previous accepted entry -- a running max is fragile
to a single early outlier (e.g. a stray "manner" false positive) which,
once accepted as the new floor, wrongly flags every real entry after it
for the rest of the volume. LNDS correctly identifies the single outlier
itself as the anomaly instead.
"""

from __future__ import annotations

import bisect
import re

_NON_ALPHA_RE = re.compile(r"[^a-z]")


def _first_letter(headword: str) -> str:
    letters = _NON_ALPHA_RE.sub("", headword.lower())
    return letters[:1]


def find_out_of_order_ids(entries: list[tuple[int, str]]) -> set[int]:
    """entries: [(entry_id, headword), ...] in reading order (ascending id
    within one volume). Returns the set of entry_ids NOT part of the
    longest non-decreasing first-letter subsequence -- i.e. the ones that
    broke alphabetical order and should be pruned."""
    if not entries:
        return set()
    keys = [_first_letter(hw) for _, hw in entries]

    tails_idx: list[int] = []
    tails_keys: list[str] = []
    prev = [-1] * len(keys)
    for i, k in enumerate(keys):
        pos = bisect.bisect_right(tails_keys, k)
        if pos == len(tails_keys):
            tails_keys.append(k)
            tails_idx.append(i)
        else:
            tails_keys[pos] = k
            tails_idx[pos] = i
        prev[i] = tails_idx[pos - 1] if pos > 0 else -1

    in_order = set()
    if tails_idx:
        c = tails_idx[-1]
        while c != -1:
            in_order.add(c)
            c = prev[c]

    return {entries[i][0] for i in range(len(entries)) if i not in in_order}
