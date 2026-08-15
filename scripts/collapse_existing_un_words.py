"""Retroactive follow-up to derive.py's "un-" root substitution (see
tokenize.py's wiring): apply the same policy Brian approved for future
ingestion -- an "un-" word with an independently attested root gets
ingested as the root, not as itself -- to the ~2,400 "un-" words already
active in the vocabulary before that policy existed.

For every active word whose lemma starts with "un" and whose stored lemma
resolves via derive.derived_root, group by resolved root (several
un-adjective/un-adverb pairs converge on the same root, e.g.
"unexpressible"/"unexpressibly" -> "expressible") and split into two cases:

  MERGE  the root already exists as its own word row (active or not) --
         every group member's word_book links move onto that row and the
         members are deactivated in place. The existing row's own
         active/definition state is never touched; it already went through
         its own validation when it was ingested.

  RENAME the root doesn't exist yet -- one member of the group (the one
         with the most word_book links) is re-run through the real
         floor -> proper-noun -> validity-gate -> LLM-judge pipeline, using
         every group member's own stored sentence as pooled occurrence
         evidence (a real, if resurrected, occurrence count -- NOT
         synthesized). If the root survives, that member is renamed to it,
         re-enriched, and every other group member's word_book links move
         onto it (all deactivated). If the root does NOT survive, nothing
         is renamed or reassigned -- every group member is simply
         deactivated in place, under its own original spelling, exactly as
         an ordinary floor/judge rejection would be.

Deliberately skips Merriam-Webster in the RENAME case (max_tier=FREE) to
avoid burning MW's daily quota on a batch this size; run `concordance
refill`/`deepen` afterward to backfill MW definitions for any survivors
still undefined.

Dry-run by default; pass --apply to write changes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from concordance import clean, db, derive, floor, judge, localdict, propernouns, resolve, tokenize, validity
from concordance.config import Config
from concordance.dictionary import make_session
from concordance.extract import Chapter
from concordance.model import Verdict, normalize_pos


def _pos_hint(lemma: str) -> str:
    return "ADV" if lemma.endswith("ly") else "ADJ"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    parser.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    args = parser.parse_args()

    conn = db.connect()
    s = db._safe_schema(args.schema)  # noqa: SLF001 -- same helper db.py's own commands use

    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma, part_of_speech, sentence, chapter
                        FROM {s}.word WHERE active AND lemma LIKE 'un%'""")
        rows = cur.fetchall()

    # --- Phase A: resolve + group -----------------------------------------
    # Computed straight off the stored lemma -- already correctly
    # spaCy-lemmatized from real ingestion, unlike the possessive cleanup's
    # garbage-in stored lemma, so no re-tokenize-to-rediscover-identity step
    # is needed here.
    groups: dict[str, list[tuple[int, str, str, str, str]]] = defaultdict(list)
    for word_id, lemma, pos, sentence, chapter in rows:
        sub = derive.derived_root(lemma, _pos_hint(lemma))
        if sub:
            root, _ = sub
            groups[root].append((word_id, lemma, pos, sentence or "", chapter or ""))

    if not groups:
        print("No active un-words resolve to a substitution root.")
        return

    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma_lc, id, active FROM {s}.word WHERE lemma_lc = ANY(%s)",
                    (list(groups.keys()),))
        existing = {lc: (wid, active) for lc, wid, active in cur.fetchall()}

    merge_groups = {root: members for root, members in groups.items() if root in existing}
    rename_groups = {root: members for root, members in groups.items() if root not in existing}

    print(f"{sum(len(m) for m in groups.values())} active un-word(s) resolve to "
          f"{len(groups)} distinct root(s): {len(merge_groups)} already exist "
          f"({sum(len(m) for m in merge_groups.values())} word(s) merging into them), "
          f"{len(rename_groups)} do not ({sum(len(m) for m in rename_groups.values())} "
          "word(s) up for re-validation).\n")

    # --- MERGE: report + apply ---------------------------------------------
    print("=== MERGE (root already exists) ===")
    for root, members in sorted(merge_groups.items()):
        target_id, target_active = existing[root]
        member_str = ", ".join(l for _, l, *_ in members)
        print(f"MERGE {member_str} -> {root!r} (existing id={target_id}, active={target_active})")
        if args.apply:
            with conn.cursor() as cur:
                for word_id, lemma, *_ in members:
                    cur.execute(
                        f"""INSERT INTO {s}.word_book (word_id, book_id)
                            SELECT %s, book_id FROM {s}.word_book WHERE word_id=%s
                            ON CONFLICT DO NOTHING""", (target_id, word_id))
                    cur.execute(f"DELETE FROM {s}.word_book WHERE word_id=%s", (word_id,))
                    cur.execute(f"UPDATE {s}.word SET active=false, updated_at=now() WHERE id=%s",
                                (word_id,))
            conn.commit()

    # --- RENAME: re-validate then report + apply ---------------------------
    print("\n=== RENAME (root not yet a word) ===")
    if rename_groups:
        cfg = Config()
        nlp = tokenize.load_nlp()
        gate = validity.ValidityGate(cfg)
        judge_obj = judge.get_judge(cfg)
        session = make_session()

        root_cands = {}
        canonical_of: dict[str, tuple[int, str, str]] = {}  # root -> (word_id, lemma, part_of_speech)
        with conn.cursor() as cur:
            for root, members in rename_groups.items():
                # Pick the member with the most word_book links as the
                # surviving row (tie-break: lowest id, i.e. oldest).
                counts = []
                for word_id, lemma, pos, *_ in members:
                    cur.execute(f"SELECT count(*) FROM {s}.word_book WHERE word_id=%s", (word_id,))
                    counts.append((cur.fetchone()[0], -word_id, word_id, lemma, pos))
                counts.sort(reverse=True)
                _, _, canonical_id, canonical_lemma, canonical_pos = counts[0]
                canonical_of[root] = (canonical_id, canonical_lemma, canonical_pos)

                chapters = [Chapter(title=chapter, text=clean.clean(sentence))
                            for _, _, _, sentence, chapter in members if sentence]
                if not chapters:
                    chapters = [Chapter(title="", text=root)]
                cands = tokenize.tokenize(chapters, nlp=nlp)
                cand = cands.get(root)
                if cand is None:
                    # Defensive fallback -- root didn't surface from re-tokenizing
                    # the pooled sentences (e.g. every sentence was too short/
                    # stripped); build a minimal one so it still gets judged.
                    from concordance.model import Candidate, Occurrence
                    cand = Candidate(lemma=root, pos=canonical_pos or "ADJ",
                                      occurrences=[Occurrence(sentence=s_, chapter=c_, surface=root)
                                                   for _, _, _, s_, c_ in members if s_])
                root_cands[root] = cand

        lexicon = localdict.build_lexicon(conn, set(root_cands.keys()))
        floor.apply_floor(root_cands, cfg)
        propernouns.strip_proper_nouns(root_cands, cfg)
        validity.apply_validity(root_cands, cfg, local_dict=lexicon, gate=gate)
        newly = [c for c in root_cands.values() if c.verdict in (Verdict.KEEP, Verdict.UNSURE)]
        if newly:
            judge_obj.judge(newly)

        for root, members in sorted(rename_groups.items()):
            cand = root_cands[root]
            canonical_id, canonical_lemma, canonical_pos = canonical_of[root]
            others = [(wid, l) for wid, l, *_ in members if wid != canonical_id]
            survives = cand.verdict in (Verdict.KEEP, Verdict.UNSURE)
            reason = cand.reject_reason.value if cand.reject_reason else cand.interesting_reason

            if survives:
                resolve.resolve_definition(cand, max_tier=resolve.Tier.FREE, lexicon=lexicon, session=session)
                print(f"KEEP  {canonical_lemma!r} -> {root!r}"
                      + (f" (+ merges {', '.join(l for _, l in others)})" if others else "")
                      + (f"  ({cand.definition[:60]})" if cand.definition else "  (still undefined)"))
            else:
                print(f"DROP  {', '.join(l for _, l, *_ in members)}  (stay as-is, deactivated; {reason or 'no reason recorded'})")

            if args.apply:
                with conn.cursor() as cur:
                    if survives:
                        cur.execute(f"SELECT id FROM {s}.word WHERE lemma_lc = lower(%s) AND id <> %s",
                                    (root, canonical_id))
                        if cur.fetchone():
                            print(f"      -- SKIPPED WRITE: {root!r} already exists as a separate word now.")
                            continue
                        cur.execute(
                            f"""UPDATE {s}.word SET lemma=%s, as_seen=%s, definition=%s,
                                    part_of_speech=%s, ipa=%s, synonyms=%s, etymology=%s,
                                    definition_source=%s, updated_at=now()
                                WHERE id=%s""",
                            (root, root, cand.definition,
                             normalize_pos(cand.part_of_speech) or normalize_pos(canonical_pos),
                             cand.ipa, list(cand.synonyms), cand.etymology,
                             cand.definition_source or ", ".join(cand.validity_sources), canonical_id),
                        )
                        for word_id, _ in others:
                            cur.execute(
                                f"""INSERT INTO {s}.word_book (word_id, book_id)
                                    SELECT %s, book_id FROM {s}.word_book WHERE word_id=%s
                                    ON CONFLICT DO NOTHING""", (canonical_id, word_id))
                            cur.execute(f"DELETE FROM {s}.word_book WHERE word_id=%s", (word_id,))
                            cur.execute(f"UPDATE {s}.word SET active=false, updated_at=now() WHERE id=%s",
                                        (word_id,))
                    else:
                        for word_id, _, *_ in members:
                            cur.execute(f"UPDATE {s}.word SET active=false, updated_at=now() WHERE id=%s",
                                        (word_id,))
                conn.commit()
    else:
        print("(none)")

    if not args.apply:
        print("\nDry-run only -- pass --apply to write these changes.")

    conn.close()


if __name__ == "__main__":
    main()
