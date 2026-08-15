"""One-time retroactive cleanup for the possessive-apostrophe ingestion bug.

concordance/clean.py didn't normalize U+02BC (MODIFIER LETTER APOSTROPHE), so
spaCy tokenized "publisherʼs" as one alphabetic token instead of splitting off
the possessive "'s" -- see clean.py's _PUNCT comment, which stops new
occurrences going forward. This script fixes the words that already got in
under the bug.

For each currently-active word whose lemma ends in the contaminated "ʼs":
strip the suffix, then re-run the bare form through the SAME pipeline stages
a fresh candidate goes through (floor -> proper-noun strip -> validity gate
-> LLM judge -> definition resolution), using the word's own stored
`sentence`/`chapter` as its one occurrence so proper-noun detection has real
(if thin) signal to work with. If the bare form survives, the row is renamed
to it in place (preserving its word_book link, i.e. the book's vocabulary
count). If not -- and empirically most of these don't: the contaminated
apostrophe made wordfreq see a novel zero-frequency token, so a perfectly
ordinary possessive noun like "publisher's" looked like a rare word purely
as an OCR artifact -- the row is renamed for hygiene and deactivated.

Dry-run by default; pass --apply to write changes.
"""

from __future__ import annotations

import argparse

from concordance import clean, db, floor, judge, localdict, mw, propernouns, resolve, tokenize, validity
from concordance.config import Config
from concordance.dictionary import make_session
from concordance.extract import Chapter
from concordance.model import Verdict, normalize_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    parser.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    args = parser.parse_args()

    conn = db.connect()
    s = db._safe_schema(args.schema)  # noqa: SLF001 -- same helper db.py's own commands use

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, sentence, chapter FROM {s}.word
                WHERE active AND lemma LIKE '%ʼs' ORDER BY lemma"""
        )
        rows = cur.fetchall()

    if not rows:
        print("No contaminated active words found.")
        return

    print(f"{len(rows)} active possessive-contaminated word(s) found.\n")

    cfg = Config()
    nlp = tokenize.load_nlp()
    gate = validity.ValidityGate(cfg)
    judge_obj = judge.get_judge(cfg)
    session = make_session()
    mw_key = mw.mw_api_key()
    if mw_key and mw.quota_exhausted():
        mw_key = ""

    kept, dropped, skipped = 0, 0, 0

    for word_id, lemma, sentence, chapter in rows:
        surface = lemma[:-2]  # strip the trailing "ʼs" -- the surface form as seen, not necessarily the final lemma

        # Re-derive real occurrence signal (capitalization, tagger POS) from
        # the word's own stored example sentence, cleaned the same way a
        # fresh ingest would clean it -- so proper-noun detection (e.g.
        # "merlinʼs" -> "merlin", the wizard) has something to work with
        # instead of a bare lemma with no context.
        text = clean.clean(sentence) if sentence else surface
        cands = tokenize.tokenize([Chapter(title=chapter or "", text=text)], nlp=nlp)
        # Match by SURFACE form, not by assuming surface == its own lemma --
        # spaCy may reduce it further (workmen's -> workman, singular) the
        # same way it would for any other candidate.
        cand = next((c for c in cands.values()
                     if any(o.surface.lower() == surface for o in c.occurrences)), None)
        if cand is None:
            print(f"SKIP  {lemma!r}: surface form {surface!r} didn't survive re-tokenizing "
                  "its stored sentence -- needs manual review.")
            skipped += 1
            continue

        resolved = cand.lemma  # the real target identity (may differ from `surface`, e.g. workman)

        floor.apply_floor({resolved: cand}, cfg)
        propernouns.strip_proper_nouns({resolved: cand}, cfg)
        lexicon = localdict.build_lexicon(conn, {resolved})
        validity.apply_validity({resolved: cand}, cfg, local_dict=lexicon, gate=gate)
        if cand.verdict in (Verdict.KEEP, Verdict.UNSURE):
            judge_obj.judge([cand])

        survives = cand.verdict in (Verdict.KEEP, Verdict.UNSURE)
        reason = cand.reject_reason.value if cand.reject_reason else cand.interesting_reason

        if survives:
            resolve.resolve_definition(cand, max_tier=resolve.Tier.MW, lexicon=lexicon,
                                        session=session, mw_api_key=mw_key)
            print(f"KEEP  {lemma!r} -> {resolved!r}"
                  + (f"  ({cand.definition[:60]})" if cand.definition else "  (still undefined)"))
            kept += 1
        else:
            print(f"DROP  {lemma!r} -> {resolved!r}  ({reason or 'floored as too common once corrected'})")
            dropped += 1

        if args.apply:
            with conn.cursor() as cur:
                # Defensive: someone else's ingest could have created a
                # `resolved` word row since the SELECT above ran.
                cur.execute(f"SELECT id FROM {s}.word WHERE lemma_lc = lower(%s) AND id <> %s",
                            (resolved, word_id))
                if cur.fetchone():
                    print(f"      -- SKIPPED WRITE: {resolved!r} already exists as a separate word now.")
                    continue
                if survives:
                    cur.execute(
                        f"""UPDATE {s}.word SET lemma=%s, as_seen=%s, definition=%s,
                                part_of_speech=%s, ipa=%s, synonyms=%s, etymology=%s,
                                definition_source=%s, updated_at=now()
                            WHERE id=%s""",
                        (resolved, resolved, cand.definition, normalize_pos(cand.part_of_speech or cand.pos),
                         cand.ipa, list(cand.synonyms), cand.etymology,
                         cand.definition_source or ", ".join(cand.validity_sources), word_id),
                    )
                else:
                    cur.execute(
                        f"""UPDATE {s}.word SET lemma=%s, as_seen=%s, active=false, updated_at=now()
                            WHERE id=%s""",
                        (resolved, resolved, word_id),
                    )
            conn.commit()

    print(f"\n{kept} kept (renamed), {dropped} dropped (renamed + deactivated), {skipped} skipped.")
    if not args.apply:
        print("Dry-run only -- pass --apply to write these changes.")

    conn.close()


if __name__ == "__main__":
    main()
