"""Definition passes: fill/refill/deepen, Merriam-Webster backfill, imports, plural/synonym
consolidation, and in-definition links."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import RejectReason, normalize_pos
from .core import DEFAULT_SCHEMA, _safe_schema
from .sync import _invalidate_definition_dependents


_POS_TO_TAGGER = {"noun": "NOUN", "verb": "VERB", "adjective": "ADJ", "adverb": "ADV"}


def fill_definitions(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                     use_web: bool = False, model_path: str | None = None,
                     recheck_after_days: int = 14, oed_schema: str = "oed",
                     validity_labels: set[str] | None = None) -> dict:
    """The single definition-acquisition pass for words whose definition is
    still blank: one candidate SELECT, one lexicon build, one per-row trip
    through resolve.resolve_definition at whatever depth `use_web` allows
    (YOURDICT without it, WEB with it). History (it replaced two passes; MW
    and the dropped WEB pre-gate): docs/decisions/0003-fill-definitions-single-pass.md

    Tier.OED (oed_schema, default "oed") is included automatically too --
    local/free like Tier LOCAL, so there's no reason to gate it behind
    use_web the way YOURDICT/WEB are. Degrades to a no-op if that schema/
    table doesn't exist yet.

    Tier.MW (mw_api_key auto-discovered from MW_DICTIONARY_API_KEY, same as
    Wordnik) is included automatically, so a word only MW can define doesn't
    wait for a manual `mw-backfill`. A foreign-language loanword MW
    catches (its own "<Language> noun/verb/..." fl convention) is cast out
    below exactly like a symbol/proper-noun-only resolution -- see the
    cand.reject_reason check a few lines down.

    A word that resolves to a symbol/proper-noun-only sense (see
    model.junk_pos_reason -- the same gate ingest's pipeline.process()
    applies) is cast out (active=false) instead of being filled in: these
    words were ACCEPTED with no definition at all, so this is the first
    real evidence of what they actually are. Never clears flagged_undefined
    -- that flag is a permanent "this one needed a second look" marker, not
    a live status (see apply_schema).

    Whatever's still undefined after the full cascade gets a
    validity_score.estimate() written to word.validity_* -- the DB-native
    version of deepen.py's <book>.undefined.csv report, so a word that's
    both flagged_undefined AND scored likely-artifact is an obvious prune
    candidate, not silent noise in the accepted list. WEB (when use_web) is
    tried for EVERY word nothing else defined, regardless of that estimate:
    "matches no dictionary" is exactly the rare vocabulary this project prizes.

    `recheck_after_days`: a word already scored by validity_score recently
    is skipped entirely rather than re-run through the full cascade (Wordnik
    pacing included) again -- without this, every `maintain` run would
    re-grind the entire permanently-undefined tail through Wordnik/web-search
    forever, not just the first time it's ever seen.

    `validity_labels`: restrict to words already scored (by an earlier pass
    through this same function) with one of these validity_score.estimate()
    labels -- e.g. {'likely-valid', 'uncertain'} to skip the 'likely-artifact'
    tail (probably OCR noise) on an expensive WEB/LLM-backed run. A word
    never scored yet (validity_label IS NULL) is excluded when this filter
    is given, matching the intent of "only the words already triaged as
    worth the deeper search" -- run once without the filter first if the
    backlog hasn't been scored yet at all.

    Only considers currently-active words: an inactive word with a blank
    definition (already cast out/pruned some other way) isn't shown to
    anyone, so re-running the full network+LLM cascade against it is pure
    waste -- confirmed 521 such rows live in the table at once."""
    from .. import deepdef, localdict, resolve, validity_score
    from ..config import Config
    from ..dictionary import make_session
    from ..model import Candidate, Occurrence, junk_pos_reason

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, part_of_speech, sentence, chapter, as_seen, validity_label
                FROM {s}.word
                WHERE active AND coalesce(definition,'') = ''
                  AND (validity_checked_at IS NULL
                       OR validity_checked_at < now() - (%s * interval '1 day'))
                  {"AND validity_label = ANY(%s)" if validity_labels else ""}
                ORDER BY flagged_undefined_at NULLS LAST, lemma""" +
            (f" LIMIT {int(limit)}" if limit else ""),
            (recheck_after_days, list(validity_labels)) if validity_labels else (recheck_after_days,))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "defined": 0, "still_undefined": 0, "cast_out": 0}
    if not rows:
        return stats

    lexicon = localdict.build_lexicon(conn, {lemma.lower() for _, lemma, *_ in rows})
    localdict.expand_lexicon_for_stubs(conn, lexicon)
    from ..oed import definitions as oed_definitions
    oed_lexicon = oed_definitions.definition_lexicon(
        conn, {lemma.lower() for _, lemma, *_ in rows}, schema=oed_schema)
    session = make_session()
    key = deepdef.wordnik_key()
    max_tier = resolve.Tier.WEB if use_web else resolve.Tier.YOURDICT

    # Same quota pre-check pipeline.py's ingest-time enrichment already does
    # (mw.mw_api_key() + mw.quota_exhausted()) -- checked ONCE for the whole
    # batch, not per word: without this, resolve_definition's Tier.MW would
    # auto-discover the key regardless, silently burn the shared 1000/day cap
    # on the first however-many words of a real backlog, then every
    # remaining word's Tier.MW call returns [] from mw.lookup_api for free
    # but still means a wasted cache re-read/re-parse per word (see mw.py's
    # lookup_api) -- and a word that reaches WORDNIK/YOURDICT/WEB and still
    # misses gets validity_checked_at stamped, so recheck_after_days pushes
    # its next MW attempt out two weeks instead of to tomorrow's run. This
    # also protects mw_backfill/lookup_mw.py, which share the same on-disk
    # quota counter, from starving for the rest of the day.
    from .. import mw as mw_module
    mw_key = mw_module.mw_api_key()
    if mw_key and mw_module.quota_exhausted():
        mw_key = ""

    llm = None
    if use_web:
        cfg = Config()
        mp = model_path or cfg.model_path
        if mp and Path(mp).exists():
            from llama_cpp import Llama
            llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    with conn.cursor() as cur:
        for i, (wid, lemma, pos, sentence, chapter, as_seen, prior_label) in enumerate(rows, 1):
            cand = Candidate(lemma=lemma, pos=_POS_TO_TAGGER.get((pos or "").lower(), ""))
            if sentence:
                cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "",
                                                    surface=as_seen or lemma))
            # llm=None here even when a model is loaded: max_tier already
            # includes WEB when use_web is set, but resolve_definition would
            # try it before validity_score ever runs -- deliberately not
            # skipped here (the likely-artifact pre-gate was removed; WEB is
            # now the true last resort, tried for anything nothing else
            # defined), just sequenced so validity_score's estimate() always
            # gets computed and is available to write if WEB also misses.
            est = None
            found = resolve.resolve_definition(
                cand, max_tier=max_tier, lexicon=lexicon, oed_lexicon=oed_lexicon, session=session,
                wordnik_key=key, mw_api_key=mw_key, llm=None,
                # Wordnik's paced 12.5 s call almost never lands for a word an
                # earlier pass already scored as probable OCR/scan noise.
                skip_wordnik=prior_label == "likely-artifact") is not None
            if not found:
                est = validity_score.estimate(lemma, session=session, sentence=sentence or "")
                if llm is not None:
                    from .. import websearch
                    found = websearch.define_via_web(cand, llm)
                    if found:
                        resolve.apply_pos_repair(cand, lexicon)

            # validity_score.variant_reject_reason (foreign-word / archaic-
            # spelling-variant detection) is a human-review flag here too,
            # same as pipeline.py: NOT a hard cast-out (a real-scale dry-run
            # sweep against the live word table found it flags ~21% of
            # already-accepted vocabulary, mostly genuine rare words --
            # haft, glaive, thurible, discomfit, kickshaw -- rather than the
            # junk it was built to catch) but still worth recording so a
            # human can review + prune via the webapp.
            # cand.reject_reason: set directly by resolve.Tier.MW when the hit
            # revealed a foreign-language loanword (MW's own raw POS
            # convention, e.g. "Swahili noun" -- see resolve._from_mw) --
            # caught here alongside junk_pos_reason's symbol/proper-noun
            # check since MW's signal doesn't survive normalize_pos.
            reason = (junk_pos_reason(cand.part_of_speech) or cand.reject_reason) if found else None
            variant = validity_score.variant_reject_reason(lemma) if (found and not reason) else None
            # variant (a foreign-word/archaic-spelling suspicion) is a
            # stronger, more specific signal than cand.variant_flag_reason
            # (which resolve.py's Tier.OED may have already set, marking
            # this definition as OED-sourced and unreviewed) -- prefer
            # variant when both are present, otherwise fall back to
            # whatever _from_oed already put on the candidate. Without this
            # merge, the UPDATE below only ever threads `variant` through
            # COALESCE against the word row's PRIOR value, silently
            # dropping cand.variant_flag_reason entirely.
            flag_reason = variant[0].value if variant else (cand.variant_flag_reason or None)
            flag_note = variant[1] if variant else (cand.variant_flag_note or None)
            if reason:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=%s, active=false, updated_at=now()
                        WHERE id=%s AND NOT admin_suggested""",
                    (cand.definition, cand.definition_source,
                     normalize_pos(cand.part_of_speech), wid))
                stats["cast_out"] += 1
            elif found:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=COALESCE(NULLIF(%s,''), part_of_speech),
                            ipa=COALESCE(NULLIF(%s,''), ipa),
                            etymology=COALESCE(NULLIF(%s,''), etymology),
                            synonyms=CASE WHEN %s THEN %s ELSE synonyms END,
                            variant_flag_reason=COALESCE(%s, variant_flag_reason),
                            variant_flag_note=COALESCE(%s, variant_flag_note),
                            variant_flagged_at=CASE WHEN %s::text IS NOT NULL THEN now() ELSE variant_flagged_at END,
                            updated_at=now()
                        WHERE id=%s""",
                    (cand.definition, cand.definition_source, normalize_pos(cand.part_of_speech),
                     cand.ipa, cand.etymology, bool(cand.synonyms), list(cand.synonyms),
                     flag_reason, flag_note, flag_reason, wid))
                stats["defined"] += 1
            else:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            validity_label=%s, validity_score=%s, validity_notes=%s,
                            suggested_correction=%s, validity_checked_at=now()
                        WHERE id=%s""",
                    (est.label, est.score, est.notes, est.suggestion or None, wid))
                stats["still_undefined"] += 1
            # Committed every word, not batched every 200: each iteration's
            # slow network call (Wordnik/yourdictionary, rate-limited) can
            # itself take longer than the whole old batch interval, so a
            # 200-row batch left one transaction open for tens of minutes at
            # a time -- long enough to block a webapp restart's schema-check
            # ALTER TABLE, which needs an ACCESS EXCLUSIVE lock on this same
            # table and would otherwise queue behind it. Per-word commits cap
            # any held lock at one row's write.
            conn.commit()
            if i % 25 == 0:
                print(f"  ...{i}/{len(rows)} words attempted "
                      f"({stats['defined']} defined, {stats['still_undefined']} still undefined)")
    # Explicit, deterministic close rather than leaving it to whenever
    # Python happens to collect `llm` after this function returns -- found
    # live: `maintain` chains straight into classify's own fresh ~9GB model
    # load immediately after this step, and relying on implicit cleanup
    # timing let a `Failed to load model from file` crash slip through when
    # the previous instance's GPU memory hadn't actually been released yet.
    if llm is not None:
        llm.close()
    return stats


def refill_definitions(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Standalone `concordance refill`: the cheap/free tiers only (LOCAL,
    FREE), never Wordnik/yourdictionary/web -- a thin wrapper around
    fill_definitions for the independent, human-scheduled command. Doesn't
    write validity_score (that's specifically deepen/fill_definitions'
    deep-pass signal; a word cheap tiers missed hasn't earned an artifact
    verdict yet, it just hasn't been tried deeply). Returns refill's
    historical stat vocabulary (filled/still_missing) rather than
    fill_definitions' (defined/still_undefined) for backward compatibility
    with existing callers/scripts."""
    from .. import localdict, resolve, validity_score
    from ..dictionary import make_session
    from ..model import Candidate, Occurrence, junk_pos_reason

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, part_of_speech, sentence, chapter, as_seen
                FROM {s}.word WHERE coalesce(definition,'') = ''
                ORDER BY flagged_undefined_at NULLS LAST, lemma""" +
            (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "filled": 0, "still_missing": 0, "cast_out": 0}
    if not rows:
        return stats

    lexicon = localdict.build_lexicon(conn, {lemma.lower() for _, lemma, *_ in rows})
    localdict.expand_lexicon_for_stubs(conn, lexicon)
    session = make_session()

    with conn.cursor() as cur:
        for i, (wid, lemma, pos, sentence, chapter, as_seen) in enumerate(rows, 1):
            cand = Candidate(lemma=lemma, pos=_POS_TO_TAGGER.get((pos or "").lower(), ""))
            if sentence:
                cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "",
                                                    surface=as_seen or lemma))
            resolve.resolve_definition(cand, max_tier=resolve.Tier.FREE, lexicon=lexicon, session=session)
            reason = junk_pos_reason(cand.part_of_speech)
            variant = validity_score.variant_reject_reason(lemma) if (cand.definition and not reason) else None
            if reason:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=%s, active=false, updated_at=now()
                        WHERE id=%s AND NOT admin_suggested""",
                    (cand.definition, cand.definition_source,
                     normalize_pos(cand.part_of_speech), wid))
                stats["cast_out"] += 1
            elif cand.definition:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=COALESCE(NULLIF(%s,''), part_of_speech),
                            ipa=COALESCE(NULLIF(%s,''), ipa),
                            etymology=COALESCE(NULLIF(%s,''), etymology),
                            synonyms=CASE WHEN %s THEN %s ELSE synonyms END,
                            variant_flag_reason=COALESCE(%s, variant_flag_reason),
                            variant_flag_note=COALESCE(%s, variant_flag_note),
                            variant_flagged_at=CASE WHEN %s::text IS NOT NULL THEN now() ELSE variant_flagged_at END,
                            updated_at=now()
                        WHERE id=%s""",
                    (cand.definition, cand.definition_source, normalize_pos(cand.part_of_speech),
                     cand.ipa, cand.etymology, bool(cand.synonyms), list(cand.synonyms),
                     variant[0].value if variant else None, variant[1] if variant else None,
                     variant[0].value if variant else None, wid))
                stats["filled"] += 1
            else:
                stats["still_missing"] += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return stats


def deepen_definitions(conn, schema: str = DEFAULT_SCHEMA, use_web: bool = False,
                       model_path: str | None = None, limit: int = 0,
                       oed_schema: str = "oed") -> dict:
    """Standalone `concordance deepen`: a thin wrapper around fill_definitions
    with no cooldown (recheck_after_days=0) -- an explicit, human-invoked
    deepen run should always retry the undefined tail regardless of when it
    was last checked; the cooldown exists to stop `maintain`'s *automatic*
    re-grinding, not to gate a deliberate one-off command."""
    return fill_definitions(conn, schema, limit=limit, use_web=use_web,
                            model_path=model_path, recheck_after_days=0, oed_schema=oed_schema)


def mw_backfill(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                use_scrape: bool = True, headless: bool = False,
                scrape_timeout_ms: int = 10000) -> dict:
    """Standalone `concordance mw-backfill`: check Merriam-Webster (the
    scripts/lookup_mw.py cascade -- API first, then a Playwright site-scrape
    fallback for words the API misses) for every accepted word that's still
    undefined AND not already written off as likely-artifact -- exactly the
    words fill_definitions'/deepen's own cascade (Free Dictionary/Wiktionary/
    Wordnik/yourdictionary/web-search) couldn't resolve, where MW's own
    Collegiate coverage sometimes succeeds anyway.

    `scrape_timeout_ms` (default 10s, half lookup_mw.py's own 20s default):
    a genuine miss on the live site still costs the full page-load wait --
    confirmed empirically (MW's "isn't in the dictionary" suggestions page
    never satisfies the entry-container selector, so it always times out
    rather than returning fast) -- and most candidates reaching this scrape
    tier already failed Wordnik/yourdictionary/web-search too, so misses here
    are the common case, not the exception. A warmed-up profile with a
    valid cleared cookie loads in well under a second; 20s was sized for
    lookup_mw.py's interactive one-or-few-word use, where patience costs
    nothing, not a batch scan that may hit this tier hundreds of times.

    `mw_checked_at` is a STICKY marker (never cleared) set the moment a word
    is attempted here, hit or miss -- re-querying MW for the same word
    tomorrow is very unlikely to produce a different answer, so this is a
    permanent "already tried" flag, same convention as flagged_undefined,
    not a recheck-after-N-days cooldown. A repeated daily run just keeps
    working through whatever's left.

    Only definition/part_of_speech/etymology/definition_source/
    first_known_use are ever written -- NOT ipa. MW's pronunciation field is
    its own proprietary respelling, not true IPA (no ahd.py-style converter
    exists for it yet), and word.ipa is trusted elsewhere (audio.py's Azure
    TTS synthesis) to actually contain IPA; writing MW's respelling there
    would silently corrupt that pipeline. Never overwrites an existing
    non-blank value in any of the columns it does write (COALESCE(NULLIF(...))
    guard), same as fill_definitions.

    The API's free tier caps out at 1000 queries/day (tracked in
    concordance/mw.py's own on-disk cache, shared with lookup_mw.py -- a word
    either tool already looked up today never costs a second query). Once
    that cap is hit, THE WHOLE RUN STOPS (not just the API tier) -- remaining
    candidates are left untouched (mw_checked_at not set) for tomorrow's run,
    rather than falling through to an unbounded scrape-only tail that would
    hammer the live site far harder than the polite, quota-capped API path.

    Commits every word (not batched), same lock-safety rationale as
    fill_definitions: a long-running batch holding one open transaction can
    block a webapp restart's schema-check ALTER TABLE, which needs an ACCESS
    EXCLUSIVE lock on this same table."""
    from contextlib import ExitStack

    from .. import mw as mw_module
    from ..dictionary import make_session
    from ..model import junk_pos_reason

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, part_of_speech
                FROM {s}.word
                WHERE active
                  AND coalesce(definition,'') = ''
                  AND (validity_label IS NULL OR validity_label IN ('uncertain','likely-valid'))
                  AND mw_checked_at IS NULL
                ORDER BY lemma""" +
            (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"attempted": 0, "defined": 0, "cast_out": 0, "no_entry": 0,
             "quota_stopped": False, "remaining": 0}
    if not rows:
        return stats

    api_key = mw_module.mw_api_key()
    session = make_session()
    scraper = None

    with conn.cursor() as cur, ExitStack() as stack:
        for i, (wid, lemma, pos) in enumerate(rows, 1):
            if api_key and mw_module.quota_exhausted():
                stats["quota_stopped"] = True
                stats["remaining"] = len(rows) - i + 1
                break
            stats["attempted"] += 1

            # exact_matches: MW's search is fuzzy and will return a same-
            # ballpark idiom for a query that isn't a real headword at all
            # (confirmed on live data -- see exact_matches' own docstring),
            # so a returned entry only counts here if its own headword
            # literally is this word. If the API's fuzzy hit doesn't survive
            # that filter, still give the scrape tier its own chance (the
            # site's ranking isn't guaranteed identical) before calling it a
            # genuine miss.
            entries = mw_module.exact_matches(
                mw_module.lookup_api(lemma, api_key, session) if api_key else [], lemma)
            if not entries and use_scrape:
                if scraper is None:
                    from .. import mw_scrape
                    scraper = stack.enter_context(mw_scrape.MWScraper(headless=headless))
                entries = mw_module.exact_matches(
                    scraper.lookup(lemma, timeout_ms=scrape_timeout_ms), lemma)

            if not entries:
                cur.execute(f"UPDATE {s}.word SET mw_checked_at=now() WHERE id=%s", (wid,))
                stats["no_entry"] += 1
                conn.commit()
                if i % 25 == 0:
                    print(f"  ...{i}/{len(rows)} words attempted ({stats['defined']} defined, "
                          f"{stats['no_entry']} no MW entry)")
                continue

            tagger_pos = _POS_TO_TAGGER.get((pos or "").lower(), "")
            entry = mw_module.pick_entry(entries, tagger_pos)
            definition = "; ".join(entry.definitions)
            resolved_pos = normalize_pos(entry.part_of_speech)
            # is_foreign_pos checks the RAW (pre-normalize_pos) string --
            # MW's "<Language> noun" foreign-loanword tag is a capitalized
            # demonym, a signal normalize_pos's lowercasing would destroy.
            reason = junk_pos_reason(resolved_pos) or (
                RejectReason.FOREIGN_LANGUAGE if mw_module.is_foreign_pos(entry.part_of_speech) else None)

            if reason:
                # Same safety net as fill_definitions: an ACCEPTED word whose
                # only resolvable sense turns out to be a symbol/proper-noun/
                # foreign-language entry gets cast out now that there's real
                # evidence of what it is, rather than sitting active with a
                # junk definition.
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s, definition_source=%s, part_of_speech=%s,
                            active=false, mw_checked_at=now(), updated_at=now()
                        WHERE id=%s AND NOT admin_suggested""",
                    (definition, entry.source, resolved_pos, wid))
                stats["cast_out"] += 1
            else:
                # COALESCE(NULLIF(%s,''), column) -- new value preferred, existing
                # kept only if the new one is blank -- same direction as
                # fill_definitions' own UPDATE. Not the reverse: definition_source
                # (and, in principle, the others) can carry a stale non-blank
                # value from history even while definition itself is blank (the
                # candidate filter only guarantees the latter), so getting this
                # backwards silently keeps old metadata under a brand-new
                # definition instead of recording MW as its real source --
                # caught empirically on a live word ("aglance") during testing.
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=COALESCE(NULLIF(%s,''), definition),
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=COALESCE(NULLIF(%s,''), part_of_speech),
                            etymology=COALESCE(NULLIF(%s,''), etymology),
                            first_known_use=COALESCE(NULLIF(%s,''), first_known_use),
                            mw_checked_at=now(), updated_at=now()
                        WHERE id=%s""",
                    (definition, entry.source, resolved_pos, entry.etymology,
                     entry.first_known_use, wid))
                stats["defined"] += 1
            conn.commit()
            if i % 25 == 0:
                print(f"  ...{i}/{len(rows)} words attempted ({stats['defined']} defined, "
                      f"{stats['no_entry']} no MW entry, {stats['cast_out']} cast out)")
    return stats


def import_defined_words(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                         commit_every: int = 500) -> dict:
    """One-time/occasional bootstrap: pull genuinely-new terms from the
    legacy `vocab.defined` table (a predecessor project's term/POS/definition
    list, collected outside any book) into `word` as book-less words -- no
    `word_book` row, since there's no book occurrence to attach. They pick up
    all the normal `maintain` processing (classify, difficulty, quizdef,
    etc.) the next time it runs, same as any book-sourced word; only the
    three purely book/author-relatedness computations
    (compute_book_similarity/compute_author_similarity/compute_author_clustering)
    read exclusively FROM word_book and so simply won't see these words --
    a correct no-op, not something this import needs to handle.

    Excludes: phrases (the `phrase` flag column, confirmed to match 100% of
    space-containing terms -- skipped outright per instruction, not just
    deprioritized), rows flagged `bad=1`, terms already in `word`, and terms
    ever rejected in ANY book for ANY reason (not just "hard" rejection
    reasons -- confirmed with the user). Multiple `vocab.defined` rows per
    term (different senses/POS) are collapsed to the single richest row
    before insert, since `word.lemma_lc` is UNIQUE.

    `vocab.defined` has no ipa/etymology/synonyms columns. fill_definitions'
    gate (`WHERE coalesce(definition,'') = ''`) will never revisit these
    words to backfill ipa/etymology since they arrive with a non-blank
    definition, so this reuses localdict.build_lexicon (the same
    vocab.wiktionary lookup the ingestion pipeline already does) once for
    the whole batch to best-effort fill those two; synonyms stays blank (no
    source has it).

    Commits every `commit_every` words, not once at the end -- same
    crash-safety rationale as classify_and_store (a run over ~11k candidates
    that dies partway through should keep whatever it already inserted).

    Every inserted word is stamped vocab1_import=true (vocab1_import_at=now())
    so it can be found later regardless of what definition_source says --
    definition_source is deliberately left as vocab.defined's own per-row
    label (falls back to "vocab.defined import" only when that's blank),
    not overwritten with a blanket tag, since those original source labels
    are more informative. Words imported before this flag existed were
    backfilled once via scripts/backfill_vocab1_import_flag.py."""
    from .. import localdict

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('vocab.defined')")
        if cur.fetchone()[0] is None:
            return {"available": False, "candidates": 0, "imported": 0, "skipped_conflict": 0}

        cur.execute(
            f"""SELECT DISTINCT ON (lower(d.term))
                    d.term, d.part_of_speech,
                    COALESCE(NULLIF(d.corrected_definition,''), d.definition),
                    d.definition_source
                FROM vocab.defined d
                WHERE d.phrase IS DISTINCT FROM 1
                  AND position(' ' in d.term) = 0
                  AND COALESCE(d.bad,0) != 1
                  AND NOT EXISTS (SELECT 1 FROM {s}.word w WHERE w.lemma_lc = lower(d.term))
                  AND NOT EXISTS (SELECT 1 FROM {s}.rejected_word r WHERE r.lemma_lc = lower(d.term))
                ORDER BY lower(d.term),
                    (d.part_of_speech IS NOT NULL AND upper(d.part_of_speech) NOT IN ('', 'TBD')) DESC,
                    length(COALESCE(NULLIF(d.corrected_definition,''), d.definition)) DESC,
                    d.id"""
            + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"available": True, "candidates": len(rows), "imported": 0, "skipped_conflict": 0}
    if not rows:
        return stats

    lexicon = localdict.build_lexicon(conn, {term.lower() for term, *_ in rows})

    with conn.cursor() as cur:
        for i, (term, pos, definition, def_source) in enumerate(rows, 1):
            raw_pos = "" if (pos or "").strip().upper() == "TBD" else (pos or "")
            norm_pos = normalize_pos(raw_pos)

            ipa = etymology = ""
            entries = lexicon.get(term.lower())
            if entries:
                match = next((e for e in entries if normalize_pos(e[0]) == norm_pos), entries[0])
                ipa, etymology = match[2], match[3]

            cur.execute(
                f"""INSERT INTO {s}.word
                        (lemma, as_seen, definition, part_of_speech, ipa, etymology,
                         definition_source, first_added, vocab1_import, vocab1_import_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE, true, now())
                    ON CONFLICT (lemma_lc) DO NOTHING""",
                (term, term, definition, norm_pos, ipa, etymology,
                 def_source or "vocab.defined import"))
            if cur.rowcount:
                stats["imported"] += 1
            else:
                stats["skipped_conflict"] += 1
            if i % commit_every == 0:
                conn.commit()
    conn.commit()
    return stats


def dedupe_plural_definitions(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                              use_web: bool = True, model_path: str | None = None) -> dict:
    """`concordance dedupe-plurals`: a definition that just says "plural of X"
    isn't real vocabulary content -- the word IS real (a dictionary vouched
    for it as its own headword), but it's redundant scaffolding once X exists
    as its own properly-defined entry. quizdef.quizzable() already excludes
    these from quizzes (crossref.mentions_pointer matches "plural of"), so this isn't a
    correctness fix -- it's consolidation: for every such word, resolve its
    singular X and soft-delete the plural (active=false, same reversible
    pattern as every other removal in this codebase -- never a hard delete).

    Only considers currently-active words -- an already-pruned plural isn't
    cluttering anything and doesn't need reprocessing. Idempotent: a plural
    already deactivated by an earlier run won't be selected again.

    Three outcomes per plural, tracked separately:
      - `linked`    the singular already exists and is active -- just needed
                     the plural deactivated.
      - `left_inactive` the singular exists but is currently inactive.
                     DELIBERATELY left untouched, whatever the reason it's
                     inactive -- checked against real data before building
                     this: every one of the handful of cases found already
                     has a real definition (not a blank/unresolved one),
                     meaning "inactive" here is near-certainly a deliberate
                     decision (a human prune via the review webapp, or a
                     justified automated cast-out) that a plural merely
                     existing is not good evidence to override.
      - `created`    the singular didn't exist at all -- a new word row,
                     resolved through the full cascade (same as any newly
                     ingested word), inheriting the plural's own
                     sentence/chapter context since there's no literal book
                     occurrence of the singular form to draw from. Cast out
                     (active=false) rather than accepted if the resolution
                     itself reveals a symbol/proper-noun sense
                     (junk_pos_reason) -- same gate every other
                     definition-acceptance path applies. Otherwise still
                     gets flagged_undefined if the cascade can't define it,
                     same as any other word -- refill/deepen will keep
                     trying on later runs."""
    from .. import crossref, deepdef, localdict, resolve
    from ..config import Config
    from ..dictionary import make_session
    from ..model import Candidate, Occurrence, junk_pos_reason
    from ..validity_score import effective_zipf

    s = _safe_schema(schema)
    # Broad SQL prefilter (crossref.PLURAL_PREFILTER_SQL, a superset) +
    # precise Python-side match below -- NOT a direct `~*` on
    # crossref._PLURAL_OF_RE.pattern: Postgres's POSIX ERE has no non-greedy
    # `+?`, so the same pattern string matches a different row set there.
    # crossref.plural_target() (needed anyway, to parse the singular out)
    # does the real matching.
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, definition, part_of_speech, sentence, chapter, as_seen
                FROM {s}.word
                WHERE active AND NOT admin_suggested AND definition ~* %s
                ORDER BY id""" + (f" LIMIT {int(limit)}" if limit else ""),
            (crossref.PLURAL_PREFILTER_SQL,))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "linked": 0, "left_inactive": 0, "created": 0,
             "cast_out": 0, "still_undefined": 0, "unparsed": 0, "common_singular": 0}
    if not rows:
        return stats

    parsed = []
    for wid, lemma, defn, pos, sentence, chapter, as_seen in rows:
        singular = crossref.plural_target(defn)
        if not singular:
            stats["unparsed"] += 1
            continue
        parsed.append((wid, lemma, pos, sentence, chapter, as_seen, singular))
    if not parsed:
        return stats

    lexicon = localdict.build_lexicon(conn, {sing for *_, sing in parsed})
    localdict.expand_lexicon_for_stubs(conn, lexicon)
    session = make_session()
    key = deepdef.wordnik_key()
    max_tier = resolve.Tier.WEB if use_web else resolve.Tier.YOURDICT

    llm = None
    if use_web:
        cfg = Config()
        mp = model_path or cfg.model_path
        if mp and Path(mp).exists():
            from llama_cpp import Llama
            llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    with conn.cursor() as cur:
        for plural_id, plural_lemma, plural_pos, sentence, chapter, as_seen, singular in parsed:
            cur.execute(f"SELECT id, active FROM {s}.word WHERE lemma_lc = %s", (singular,))
            existing = cur.fetchone()

            if existing and existing[1]:
                stats["linked"] += 1

            elif existing:
                stats["left_inactive"] += 1

            elif effective_zipf(singular) >= Config().min_zipf:
                # The ingest frequency floor would reject the singular as too
                # common to be vocabulary (tooken -> "took"): don't create it.
                # The plural still goes -- it isn't vocabulary either.
                stats["common_singular"] += 1

            else:
                cand = Candidate(lemma=singular, pos=_POS_TO_TAGGER.get((plural_pos or "").lower(), ""))
                if sentence:
                    cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "",
                                                        surface=singular))
                found = resolve.resolve_definition(
                    cand, max_tier=max_tier, lexicon=lexicon, session=session,
                    wordnik_key=key, llm=llm) is not None
                reason = junk_pos_reason(cand.part_of_speech) if found else None
                is_blank = not found
                cur.execute(
                    f"""INSERT INTO {s}.word
                            (lemma, as_seen, definition, part_of_speech, sentence, chapter,
                             definition_source, first_added, active, flagged_undefined, flagged_undefined_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE, %s, %s, CASE WHEN %s THEN now() ELSE NULL END)
                        ON CONFLICT (lemma_lc) DO UPDATE SET active=EXCLUDED.active, updated_at=now()""",
                    (singular, singular, cand.definition, normalize_pos(cand.part_of_speech),
                     sentence or "", chapter or "", cand.definition_source,
                     not reason, is_blank, is_blank))
                if reason:
                    stats["cast_out"] += 1
                else:
                    stats["created"] += 1
                    if is_blank:
                        stats["still_undefined"] += 1

            cur.execute(f"UPDATE {s}.word SET active=false, updated_at=now() WHERE id=%s", (plural_id,))
            conn.commit()
    return stats


# "Synonym of X" from a source that embedded a real gloss right there --
# either quoted ("...") or bare -- e.g. 'Synonym of nithing ("a coward...").'
_SYNONYM_OF_RE = re.compile(
    r"^synonym of ([^(\n]+?)\s*(?:\(\s*[“\"]?(.+?)[”\"]?\s*\))?\.?\s*$", re.IGNORECASE)
# Wiktionary REST occasionally leaves a raw CSS rule trailing a gloss --
# ".mw-parser-output .defdate{font-size:smaller}" -- a copy-through of the
# page's own stylesheet class, not content.
_CSS_JUNK_RE = re.compile(r"\.mw-parser-output[^{]*\{[^}]*\}")


def expand_synonym_definitions(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                               use_web: bool = True, model_path: str | None = None) -> dict:
    """`concordance expand-synonyms`: a definition that just says "synonym of
    X" is a real data-quality problem, not merely a quizzability one (unlike
    "plural of X", crossref.mentions_pointer doesn't even exclude these from
    quizzing today -- "synonym" was never in its word list). But the fix is
    the OPPOSITE of dedupe-plurals': a synonym is a genuinely distinct
    headword worth keeping on its own, not redundant scaffolding for another
    surface form of the same word -- so this never deletes/deactivates the
    word carrying the "synonym of X" definition. It replaces that
    definition with real content, and separately assesses X (the synonym
    target) for inclusion in the corpus, mirroring dedupe-plurals' handling
    of a plural's singular.

    Three ways a word's definition gets upgraded:
      - the source already embedded a real gloss right in the cross-
        reference -- 'Synonym of nithing ("a coward...").' -- extracted
        directly, no lookup needed.
      - the source put the real definition on a later line after the
        "Synonym of X." sentence (seen in a couple of live rows) -- used
        as-is.
      - bare 'Synonym of X.' with nothing else -- X's OWN definition is
        reused (or freshly resolved through the same cascade every other
        definition-acceptance path uses, creating X as a new word if it
        doesn't exist yet). Never done if X exists but is currently
        inactive -- checked against real data before building this: the one
        live case is inactive WITH a real definition already, meaning
        "inactive" here is (as with dedupe-plurals) near-certain evidence of
        a deliberate earlier decision that a bare synonym pointer is not
        good reason to override, and definitely not good reason to import
        that same word's content into a DIFFERENT word's definition.
        Likewise never done if a fresh resolution of X reveals a
        symbol/proper-noun sense (junk_pos_reason) -- X still gets created,
        cast out (same as dedupe-plurals), but W's definition is left
        untouched rather than "upgraded" with content that isn't real
        vocabulary.

    Whenever a word's own definition text actually changes, its stale
    downstream artifacts (quiz_definition, USAS categories, definition
    embedding) are invalidated via _invalidate_definition_dependents --
    the same "changeful"-bug fix sync_book_results already applies, needed
    here for the identical reason (this is a direct word.definition write,
    not going through that upsert path)."""
    from .. import deepdef, localdict, resolve
    from ..config import Config
    from ..dictionary import make_session
    from ..model import Candidate, Occurrence, junk_pos_reason

    s = _safe_schema(schema)
    # Same POSIX-ERE-vs-Python-re caveat as dedupe_plural_definitions: a
    # plain literal substring for the SQL prefilter, precise parsing here.
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, definition, part_of_speech, sentence, chapter, as_seen
                FROM {s}.word
                WHERE active AND definition ~* 'synonym of'
                ORDER BY id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "extracted": 0, "reused_existing": 0, "target_created": 0,
             "target_cast_out": 0, "target_inactive": 0, "target_still_undefined": 0, "unparsed": 0}
    if not rows:
        return stats

    def _parse(raw: str) -> tuple[str, str | None] | None:
        """(target, gloss_or_None) if `raw` cleanly parses, else None."""
        d = _CSS_JUNK_RE.sub("", raw or "").strip()
        if "\n" in d:
            first, rest = d.split("\n", 1)
            if rest.strip():
                return "", rest.strip()  # real content on a later line -- no target needed
            d = first.strip()
        m = _SYNONYM_OF_RE.match(d)
        if not m:
            return None
        target = m.group(1).strip().rstrip(".")
        gloss = m.group(2)
        return target, (gloss.strip() if gloss and len(gloss.strip()) >= 4 else None)

    parsed = []
    for wid, lemma, defn, pos, sentence, chapter, as_seen in rows:
        result = _parse(defn)
        if result is None:
            stats["unparsed"] += 1
            continue
        target, gloss = result
        parsed.append((wid, lemma, pos, sentence, chapter, target.lower() if target else "", gloss))

    bare_targets = {t for *_, t, gloss in parsed if t and gloss is None}
    lexicon = localdict.build_lexicon(conn, bare_targets)
    localdict.expand_lexicon_for_stubs(conn, lexicon)
    session = make_session()
    key = deepdef.wordnik_key()
    max_tier = resolve.Tier.WEB if use_web else resolve.Tier.YOURDICT

    llm = None
    if use_web:
        cfg = Config()
        mp = model_path or cfg.model_path
        if mp and Path(mp).exists():
            from llama_cpp import Llama
            llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    with conn.cursor() as cur:
        for wid, lemma, pos, sentence, chapter, target, gloss in parsed:
            if gloss is not None:
                # Already-embedded content -- straight extraction, no lookup.
                cur.execute(f"UPDATE {s}.word SET definition=%s, updated_at=now() WHERE id=%s",
                            (gloss, wid))
                _invalidate_definition_dependents(cur, s, wid)
                stats["extracted"] += 1
                conn.commit()
                continue

            cur.execute(f"SELECT active, coalesce(definition,''), definition_source "
                        f"FROM {s}.word WHERE lemma_lc = %s", (target,))
            existing = cur.fetchone()

            if existing and not existing[0]:
                stats["target_inactive"] += 1
                conn.commit()
                continue

            if existing and existing[1]:
                _, target_def, target_src = existing
                cur.execute(f"UPDATE {s}.word SET definition=%s, "
                            f"definition_source=%s, updated_at=now() WHERE id=%s",
                            (target_def, f"{target_src} (synonym of '{target}')", wid))
                _invalidate_definition_dependents(cur, s, wid)
                stats["reused_existing"] += 1
                conn.commit()
                continue

            # Target doesn't exist, or exists active with a blank definition
            # (existing is (True, '', ...) at this point -- the inactive and
            # active-with-content cases were both already handled above) --
            # resolve it fresh, same cascade as any other definition-
            # acceptance path. ON CONFLICT DO UPDATE actually fills in the
            # definition/POS this time (not just active/updated_at) -- a
            # plain no-op update here would silently drop a genuine
            # resolution for an existing-but-blank target.
            cand = Candidate(lemma=target, pos=_POS_TO_TAGGER.get((pos or "").lower(), ""))
            if sentence:
                cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "", surface=target))
            found = resolve.resolve_definition(
                cand, max_tier=max_tier, lexicon=lexicon, session=session,
                wordnik_key=key, llm=llm) is not None
            reason = junk_pos_reason(cand.part_of_speech) if found else None
            is_blank = not found

            cur.execute(
                f"""INSERT INTO {s}.word
                        (lemma, as_seen, definition, part_of_speech, sentence, chapter,
                         definition_source, first_added, active, flagged_undefined, flagged_undefined_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE, %s, %s, CASE WHEN %s THEN now() ELSE NULL END)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        definition=COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition),
                        part_of_speech=COALESCE(NULLIF(EXCLUDED.part_of_speech,''), {s}.word.part_of_speech),
                        definition_source=COALESCE(NULLIF(EXCLUDED.definition_source,''), {s}.word.definition_source),
                        active=EXCLUDED.active, updated_at=now()""",
                (target, target, cand.definition, normalize_pos(cand.part_of_speech),
                 sentence or "", chapter or "", cand.definition_source,
                 not reason, is_blank, is_blank))

            if reason:
                stats["target_cast_out"] += 1
            elif is_blank:
                stats["target_still_undefined"] += 1
            else:
                cur.execute(f"UPDATE {s}.word SET definition=%s, "
                            f"definition_source=%s, updated_at=now() WHERE id=%s",
                            (cand.definition, f"{cand.definition_source} (synonym of '{target}')", wid))
                _invalidate_definition_dependents(cur, s, wid)
                stats["target_created"] += 1
            conn.commit()
    return stats


def compute_definition_links(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                              chunk_size: int = 500) -> dict:
    """`concordance link-definitions`: find every case where an active word's
    OWN definition text uses another word that's ALSO active in this app's
    vocabulary, so the webapp can render a real clickable link
    (word_definition_link) instead of inert prose. Matching is lemma-aware
    (spaCy, tokenize._resolve_lemma -- the same mis-lemmatization guards the
    ingestion pipeline itself relies on, not a fresh/divergent heuristic) so
    an inflected mention ("proscribed") still matches the target's headword
    ("proscribe"). Target pool is this app's own `word` table only, never
    the broader local Wiktionary dump -- `word` already holds only curated
    rare/interesting vocabulary, so there's no meaningful over-linking risk
    the way there would be linking against ordinary-word coverage.

    Full recompute every run, no watermark column: a word's definition can
    be rewritten later by refill/deepen/mw-backfill/expand-synonyms, and
    with ~67k short definitions a full spaCy pass is cheap, so "just rerun
    this command" is the staleness fix, same manual-backfill convention
    already used everywhere else in this codebase (mw-backfill, dedupe-
    plurals, expand-synonyms) rather than a new automatic-invalidation hook."""
    from .. import tokenize

    s = _safe_schema(schema)
    nlp = tokenize.load_nlp()

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma_lc, definition FROM {s}.word
                WHERE active AND definition IS NOT NULL AND definition != ''
                ORDER BY id""" + (f" LIMIT {int(limit)}" if limit else "")
        )
        rows = cur.fetchall()

    stats = {"words_examined": len(rows), "words_with_links": 0, "links_created": 0}
    if not rows:
        return stats

    lemma_to_id = {lemma_lc: wid for wid, lemma_lc, _ in rows}
    docs = nlp.pipe((definition for _, _, definition in rows), batch_size=200)

    with conn.cursor() as cur:
        for i, ((word_id, lemma_lc, _definition), doc) in enumerate(zip(rows, docs), 1):
            # target lemma -> first surface form seen for it in this definition
            matches: dict[str, str] = {}
            for tok in doc:
                if not tok.is_alpha or tok.is_stop or len(tok) < 3:
                    continue
                target_lemma = tokenize._resolve_lemma(tok)
                if target_lemma == lemma_lc or target_lemma not in lemma_to_id:
                    continue
                matches.setdefault(target_lemma, tok.text)

            cur.execute(f"DELETE FROM {s}.word_definition_link WHERE source_word_id = %s", (word_id,))
            if matches:
                stats["words_with_links"] += 1
                stats["links_created"] += len(matches)
                cur.executemany(
                    f"""INSERT INTO {s}.word_definition_link (source_word_id, target_word_id, surface)
                        VALUES (%s, %s, %s)""",
                    [(word_id, lemma_to_id[target_lemma], surface) for target_lemma, surface in matches.items()],
                )

            if i % chunk_size == 0:
                conn.commit()
        conn.commit()

    return stats


# The legacy vocab.defined bootstrap (import_defined_words) kept that
# project's own per-row definitions. Its "datamuse"/"dm" rows came from a
# spelling-similarity lookup, which often returned a DIFFERENT, similar-
# looking word's entry: serpenticide got serpentinize's "To convert (another
# magnesium silicate mineral) into serpentine", pavidly got avidly's.
# Refills never overwrite an existing definition, so these stuck.
_IMPORTED_FUZZY_SOURCES = ("datamuse", "dm")


def _def_norm(text: str | None) -> str:
    text = re.sub(r"^\s*(?:\([^)]*\)\s*)+", "", (text or "").lower())
    return re.sub(r"[^a-z ]", "", text).strip()


def _spelling_kin(word: str, other: str) -> bool:
    """`other` is `word`'s singular or the same word under another spelling
    convention -- borrowing its definition is right, not a mix-up."""
    w, o = word.lower(), other.lower()
    if w in (o + "s", o + "es") or (w.endswith("ies") and w[:-3] + "y" == o):
        return True
    swaps = (("ise", "ize"), ("isation", "ization"), ("our", "or"), ("tre", "ter"), ("ae", "e"), ("oe", "e"))
    return any(w.replace(a, b) == o.replace(a, b) for a, b in swaps)


def redefine_imported(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False,
                      oed_schema: str = "oed", samples: int = 12) -> dict:
    """`concordance redefine-imported`: fix definitions the legacy import took
    from a fuzzy lookup (see _IMPORTED_FUZZY_SOURCES) when the imported text
    is verbatim ANOTHER headword's definition (not the word's own, its
    singular, or a spelling variant's):
      - replaced: the local Wiktionary / 0 Dict entry for THIS word, when it
        is a real gloss;
      - cleared: otherwise, so fill-definitions retries the full cascade on
        the next maintain.
    Everything else is left alone.
    The replaced text goes to previous_definition. Dry run unless `apply`."""
    from collections import defaultdict

    from .. import crossref, localdict, resolve
    from ..model import Candidate
    from ..oed import definitions as oed_definitions

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma, part_of_speech, definition FROM {s}.word
                        WHERE active AND definition_source = ANY(%s) ORDER BY id""",
                    (list(_IMPORTED_FUZZY_SOURCES),))
        rows = cur.fetchall()
        cur.execute("SELECT lower(term), definition FROM vocab.wiktionary")
        owners: dict[str, set[str]] = defaultdict(set)
        for term, d in cur:
            for part in (d or "").split(";"):
                n = _def_norm(part)
                if len(n) > 15:
                    owners[n].add(term)
    lemmas = {lemma.lower() for _, lemma, _, _ in rows}
    lexicon = localdict.build_lexicon(conn, lemmas)
    oed_lexicon = oed_definitions.definition_lexicon(conn, lemmas, schema=oed_schema)

    stats = {"imported": len(rows), "replaced": 0, "cleared": 0, "kept": 0}
    shown: dict[str, list] = {"replaced": [], "cleared": []}
    with conn.cursor() as cur:
        for wid, lemma, pos, old in rows:
            # The mix-up signal: the imported text is verbatim some OTHER
            # headword's definition (and not this word's own, its singular,
            # or a spelling variant's). Text that merely words this word's
            # meaning differently from the local copy is left alone.
            parts = [_def_norm(p) for p in old.split(";") if len(_def_norm(p)) > 15]
            others = set().union(*(owners.get(p, set()) for p in parts)) if parts else set()
            if (not others or lemma.lower() in others or any(_spelling_kin(lemma, o) for o in others)
                    # "Alternative form of dulocracy [...]": the source itself
                    # vouched for the link to the other word
                    or (crossref.mentions_pointer(old)
                        and any(re.search(rf"\b{re.escape(o)}\b", old, re.I) for o in others))):
                stats["kept"] += 1
                continue
            cand = Candidate(lemma=lemma, pos=_POS_TO_TAGGER.get((pos or "").lower(), ""))
            if localdict.enrich(cand, lexicon) or resolve._from_oed(cand, oed_lexicon):
                resolve.apply_pos_repair(cand, lexicon)
            # only a real gloss replaces it -- not a bare "abbreviation of X"
            # stub or a 0 Dict "= X" cross-reference
            if cand.definition and crossref.classification_gloss(cand.definition) \
                    and not crossref.is_abbreviation_stub(cand.definition) \
                    and not re.match(r"^[\w. ]{0,12}=\s", cand.definition):
                stats["replaced"] += 1
                if len(shown["replaced"]) < samples:
                    shown["replaced"].append((lemma, sorted(others)[:2], old[:60], cand.definition[:70]))
                if apply:
                    cur.execute(
                        f"""UPDATE {s}.word SET previous_definition=definition, definition=%s,
                                definition_source=%s,
                                part_of_speech=COALESCE(NULLIF(%s,''), part_of_speech), updated_at=now()
                            WHERE id=%s""",
                        (cand.definition, cand.definition_source, normalize_pos(cand.part_of_speech), wid))
                    _invalidate_definition_dependents(cur, s, wid)
                continue
            stats["cleared"] += 1
            if len(shown["cleared"]) < samples:
                shown["cleared"].append((lemma, sorted(others)[:2], old[:70]))
            if apply:
                cur.execute(
                    f"""UPDATE {s}.word SET previous_definition=definition, definition='',
                            definition_source='', updated_at=now()
                        WHERE id=%s""", (wid,))
                _invalidate_definition_dependents(cur, s, wid)
    if apply:
        conn.commit()
    else:
        conn.rollback()
    stats["samples"] = shown
    return stats
