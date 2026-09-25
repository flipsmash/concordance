"""Pronunciations: Wordnik/Commons lookups, IPA, and audio synthesis."""

from __future__ import annotations

import requests

from .core import DEFAULT_SCHEMA, _safe_schema
from .definitions import _POS_TO_TAGGER


def fetch_wordnik_pronunciations(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                                  limit: int = 0, delay: float = 0.1) -> dict:
    """Fetch RAW pronunciation strings from Wordnik (ahd-5 diacritic respelling,
    arpabet, or gcide-diacritical — whichever it has) and store them as-is, with
    no IPA conversion here. Rate-limited (~1 word per several seconds observed on
    the free tier) but that cost is paid once: wordnik_checked_at gates re-fetch,
    so converting to IPA later is a separate, fast, freely-iterable pass that never
    re-triggers this fetch. only_missing also skips inactive words and anything
    that already has a valid ipa — those wouldn't gain anything from a Wordnik
    round trip, and at several seconds/word skipping them saves real hours."""
    import time
    from collections import Counter
    from .. import deepdef
    s = _safe_schema(schema)
    key = deepdef.wordnik_key()
    if not key:
        return {"error": "no WORDNIK_API_KEY in .env"}

    where = (f" WHERE wordnik_checked_at IS NULL AND active"
             f" AND (ipa IS NULL OR ipa = '')") if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word{where}" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    import requests
    from ..dictionary import _get
    session = requests.Session()
    dist: Counter = Counter()
    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(rows, start=1):
            r = _get(session, f"https://api.wordnik.com/v4/word.json/{lemma}/pronunciations",
                     {"api_key": key, "limit": 5})
            raw, rtype = None, None
            if r is not None and r.status_code == 200:
                data = r.json()
                if data:
                    raw, rtype = data[0].get("raw"), data[0].get("rawType")
                    dist[rtype or "unknown"] += 1
            if raw is None:
                dist["none"] += 1
            cur.execute(f"UPDATE {s}.word SET wordnik_pron_raw=%s, wordnik_pron_type=%s, "
                        "wordnik_checked_at=now() WHERE id=%s", (raw, rtype, wid))
            if i % 25 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} checked")
            time.sleep(delay)
    conn.commit()
    return {"words": len(rows), **dist}


def search_commons_direct(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                           only_missing: bool = True, limit: int = 0, delay: float = 2.5) -> dict:
    """Second-pass Commons search for words kaikki's dump reported no audio for
    (confirmed empirically to under-count: kaikki missed real, exact-match English
    recordings for words like "unpeople"/"enkindle"). Stores only the search
    result (title + constructed URL) — actually downloading is a separate,
    fast, freely-retriable step. Deliberately slow (Commons rate-limits hard);
    meant to run for hours unattended."""
    import time
    from collections import Counter
    from .. import commons_search, wiktextract
    s = _safe_schema(schema)

    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_commons_search c WHERE c.word_id=w.id)"
             if only_missing else "")
    with conn.cursor() as cur:
        cur.execute(f"SELECT w.id, w.lemma FROM {s}.word w{where}" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    if not rows:
        return {"candidates": 0}

    # skip words kaikki already solved — only worth the slow search for real gaps
    lemmas = {lemma.strip().lower() for _, lemma in rows}
    lexicon = wiktextract.sound_lexicon(
        conn, lemmas, dump_path, progress_cb=lambda n: print(f"  ...{n} lines scanned"))
    candidates = [(wid, lemma) for wid, lemma in rows
                  if not lexicon.get(lemma.strip().lower(), {}).get("audio")]

    dist: Counter = Counter(total=len(rows), skipped_kaikki_has_audio=len(rows) - len(candidates))
    session = requests.Session()
    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(candidates, start=1):
            titles = commons_search.search_word(lemma, session)
            match = commons_search.best_english_exact_match(titles, lemma)
            url = commons_search.download_url(match) if match else None
            cur.execute(
                f"""INSERT INTO {s}.word_commons_search (word_id, found_title, download_url, checked_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET found_title=EXCLUDED.found_title,
                        download_url=EXCLUDED.download_url, checked_at=now()""",
                (wid, match, url))
            dist["found"] += 1 if match else 0
            dist["not_found"] += 0 if match else 1
            if i % 20 == 0:
                conn.commit()
                print(f"  ...{i}/{len(candidates)} searched")
            time.sleep(delay)
        # words skipped because kaikki already has audio still need a checked_at
        # row so a re-run doesn't re-parse the dump for them pointlessly
        for wid, lemma in rows:
            if (wid, lemma) not in candidates:
                cur.execute(
                    f"""INSERT INTO {s}.word_commons_search (word_id, found_title, download_url, checked_at)
                        VALUES (%s, NULL, NULL, now()) ON CONFLICT (word_id) DO NOTHING""", (wid,))
    conn.commit()
    return dict(dist)


def compute_ipa(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                 only_missing: bool = True, limit: int = 0, oed_schema: str = "oed") -> dict:
    """Backfill + clean word.ipa via resolve_pronunciation's unified cascade
    (kaikki -> Wordnik-converted -> local Wiktionary -> oed, see that
    module's docstring for the full per-tier rationale) -- one candidate
    SELECT, four lexicons built once for the whole batch, one per-word trip
    through resolve_pronunciation.resolve_ipa. Folds in what used to be the
    separate standalone `oed-ipa` command's own candidate-selection/priority
    logic (backfill_ipa_from_oed still exists for a targeted OED-only run,
    but every regular `ipa`/`maintain` pass now covers OED too, so it's
    rarely needed on its own anymore).

    Also NULLs out any existing transcription that fails the English-language
    sanity check (the pre-existing ad hoc scrape occasionally grabbed a
    cross-referenced foreign cognate's IPA instead of the word's own — e.g.
    "murmurer" had the French verb's transcription). Idempotent: with
    only_missing=True (default), only words with an empty or invalid ipa are
    candidates, so a re-run after everything's resolved does no dump parsing
    at all and is a no-op. ipa_source is kept in sync with every write here
    (set to whichever tier won, cleared alongside a NULLed ipa) -- see
    audio.ipa_dialect_for_source for why this matters downstream."""
    from collections import Counter
    from .. import audio, localdict, resolve_pronunciation, wiktextract
    from ..oed import db as oed_db
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, ipa, wordnik_pron_raw, wordnik_pron_type, ipa_checked_at "
                    f"FROM {s}.word ORDER BY id")
        all_rows = cur.fetchall()

    def is_valid(ipa):
        return bool(ipa) and audio.looks_like_english_ipa(ipa)

    # only_missing also skips anything already confirmed-checked (not just
    # already-valid): without that, a word the cascade already walked and
    # found nothing for gets the full dump-scan-plus-cascade cost paid again
    # on every future run, forever -- the exact "very slow, barely moves"
    # complaint this was part of fixing. A deliberate --refetch (only_missing
    # =False) still re-walks everything, which is the right way to pick up
    # newly-unlocked upstream data (e.g. after an OED reconciliation pass
    # resolves entries that were needs_review before).
    candidates = all_rows if not only_missing else [
        r for r in all_rows if not is_valid(r[2]) and r[5] is None]
    already_valid = sum(1 for r in all_rows if is_valid(r[2]))
    already_checked_empty = (len(all_rows) - already_valid - len(candidates)) if only_missing else 0
    dist: Counter = Counter(total=len(all_rows), already_valid=already_valid,
                             already_checked_empty=already_checked_empty)
    # `limit` slices the already-filtered candidate set, not the raw fetch --
    # applying it beforehand (the original bug) could silently hand back
    # fewer than `limit` words, or zero, depending on where the first N rows
    # in scan order happened to already be valid. The already_* counts above
    # are computed from the full filtered set, before this slice, so they
    # still reflect the whole table regardless of `limit`.
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        return dict(dist)

    lemmas = {lemma.strip().lower() for _, lemma, _, _, _, _ in candidates}
    kaikki_lexicon = wiktextract.sound_lexicon(
        conn, lemmas, dump_path, progress_cb=lambda n: print(f"  ...{n} lines scanned"))
    local_lexicon = localdict.build_lexicon(conn, lemmas)
    oed_lexicon = oed_db.pronunciation_lexicon(conn, lemmas, schema=oed_schema)

    with conn.cursor() as cur:
        for i, (wid, lemma, existing_ipa, wn_raw, wn_type, _checked_at) in enumerate(candidates, 1):
            if i % 5000 == 0:
                conn.commit()
                print(f"  ...{i}/{len(candidates)} words checked")
            had_valid_existing = is_valid(existing_ipa)
            lemma_lc = lemma.strip().lower()
            hit = resolve_pronunciation.resolve_ipa(
                kaikki_entry=kaikki_lexicon.get(lemma_lc),
                wordnik_raw=wn_raw, wordnik_type=wn_type,
                local_entries=local_lexicon.get(lemma_lc),
                oed_matches=oed_lexicon.get(lemma_lc),
            )
            replacement, source = hit if hit else (None, None)
            source = source.name.lower() if source else None

            if had_valid_existing and not replacement:
                dist["already_valid"] += 1  # nothing to do, no change
                continue
            if not (existing_ipa or "").strip() and replacement:
                cur.execute(f"UPDATE {s}.word SET ipa=%s, ipa_source=%s WHERE id=%s", (replacement, source, wid))
                dist[f"backfilled_{source}"] += 1
            elif (existing_ipa or "").strip() and not had_valid_existing and replacement:
                cur.execute(f"UPDATE {s}.word SET ipa=%s, ipa_source=%s WHERE id=%s", (replacement, source, wid))
                dist[f"corrected_{source}"] += 1
            elif (existing_ipa or "").strip() and not had_valid_existing:
                cur.execute(f"UPDATE {s}.word SET ipa=NULL, ipa_source=NULL WHERE id=%s", (wid,))
                dist["cleared_no_replacement"] += 1
            else:
                dist["unresolved"] += 1
            # Every candidate reaches here regardless of outcome -- the whole
            # cascade was walked for this word even when nothing was found,
            # and that's the distinction ipa_checked_at exists to preserve
            # (see apply_schema's comment on this column): a still-empty ipa
            # after this means "confirmed nothing anywhere", not "never
            # looked". Without it, only_missing's own is_valid() filter would
            # keep re-walking the full cascade for the same permanently-empty
            # words on every future run forever.
            cur.execute(f"UPDATE {s}.word SET ipa_checked_at=now() WHERE id=%s", (wid,))
    conn.commit()
    return dict(dist)


def backfill_ipa_from_oed(conn, schema: str = DEFAULT_SCHEMA, oed_schema: str = "oed",
                           only_missing: bool = True, limit: int = 0) -> dict:
    """Backfill word.ipa from the oed schema's double-pass-verified
    pronunciation_ipa (see oed/pronunciation.py's module docstring for how
    that's gated -- confirmed live: pronunciation_ipa IS NOT NULL implies
    pronunciation_needs_review=false for all 5007 resolved entries in this
    corpus, 0 counterexamples).

    Superseded for routine use: compute_ipa (the `ipa`/`maintain` step) now
    tries OED as its own tier too (see resolve_pronunciation.py), so a
    regular `ipa` run already picks up whatever OED coverage exists as of
    that run. Kept as a standalone command for a targeted OED-only pass
    (e.g. right after a fresh `oed-ingest` run, without re-touching the
    kaikki/Wordnik/local-Wiktionary tiers) -- no local LLM or embedding
    model, so safe to run alongside GPU-bound steps.

    OED coverage is partial and grows with each future `oed-ingest` run.

    Same only_missing candidate gate as compute_ipa (empty or invalid
    existing ipa). A headword can map to several oed.entry rows (homographs
    -- "bay"/"fleet"/"back" each have up to 10 in this corpus); this only
    writes when every entry with a resolved pronunciation for that headword
    agrees on the same IPA (confirmed live: true for 4408/4424 headwords
    with any resolved pronunciation at all -- OED's part_of_speech field is
    unparsed OCR abbreviation soup, not usable to pick the "right" homograph,
    so agreement is the disambiguation signal instead of POS matching). A
    genuine conflict is skipped, not guessed at. Only fills currently-empty
    slots -- never overrides an existing valid IPA from another source, even
    though OED is higher-confidence; upgrading a lower-confidence existing
    IPA to OED's is a deliberate non-goal for now."""
    from collections import Counter
    from .. import audio
    from ..oed import db as oed_db
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, ipa FROM {s}.word ORDER BY id")
        all_rows = cur.fetchall()

    def is_valid(ipa):
        return bool(ipa) and audio.looks_like_english_ipa(ipa)

    candidates = all_rows if not only_missing else [r for r in all_rows if not is_valid(r[2])]
    dist: Counter = Counter(total=len(all_rows), already_valid=len(all_rows) - len(candidates))
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        return dict(dist)

    lemmas = {lemma.strip().lower() for _, lemma, _ in candidates}
    lexicon = oed_db.pronunciation_lexicon(conn, lemmas, schema=oed_schema)

    with conn.cursor() as cur:
        for i, (wid, lemma, _existing_ipa) in enumerate(candidates, 1):
            if i % 5000 == 0:
                conn.commit()
                print(f"  ...{i}/{len(candidates)} words checked")
            matches = lexicon.get(lemma.strip().lower())
            if not matches:
                dist["no_match"] += 1
                continue
            if len(matches) > 1:
                dist["ambiguous_homograph"] += 1
                continue
            candidate_ipa = matches[0]
            if not audio.looks_like_english_ipa(candidate_ipa):
                dist["failed_sanity_check"] += 1
                continue
            cur.execute(f"UPDATE {s}.word SET ipa=%s, ipa_source='oed' WHERE id=%s", (candidate_ipa, wid))
            dist["backfilled"] += 1
    conn.commit()
    return dict(dist)


def download_commons_direct_finds(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0,
                                   delay: float = 4.0) -> dict:
    """Download the real recordings `commons-search` confirmed exist, upgrading
    any word currently on 'azure' or 'none' to the real recording. Split out
    from `compute_audio` because interleaving Commons downloads with fast Azure
    calls exhausted Commons' upload-CDN rate limit mid-run (429s that the
    per-request backoff wasn't patient enough for — this earlier in the session
    took over a minute to clear even at near-zero request volume). Paced like
    `commons-search` itself: slow, meant to run unattended."""
    import time
    from collections import Counter
    from .. import audio
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT w.id, w.lemma, cs.download_url, a.source
            FROM {s}.word w
            JOIN {s}.word_commons_search cs ON cs.word_id = w.id
            LEFT JOIN {s}.word_audio a ON a.word_id = w.id
            WHERE cs.found_title IS NOT NULL AND (a.source IS NULL OR a.source <> 'commons')
        """ + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter(candidates=len(rows))
    if not rows:
        return dict(dist)
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma, url, prior_source) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
            if audio.fetch_commons_audio(url, dest, tries=6):
                cur.execute(
                    f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice, license_note, generated_at)
                        VALUES (%s,'commons',%s,NULL,%s,%s, now())
                        ON CONFLICT (word_id) DO UPDATE SET source='commons', file_path=EXCLUDED.file_path,
                            ipa_used=NULL, voice=EXCLUDED.voice, license_note=EXCLUDED.license_note, generated_at=now()""",
                    (wid, str(dest), url,
                     "Wikimedia Commons recording (direct search — kaikki's dump missed it); "
                     "verify per-file license before public reuse"))
                dist["downloaded"] += 1
                dist[f"upgraded_from_{prior_source}"] += 1 if prior_source else 0
            else:
                dist["failed"] += 1
            if i % 20 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} downloaded")
            time.sleep(delay)
    conn.commit()
    return dict(dist)


def compute_audio(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                   only_missing: bool = True, limit: int = 0, delay: float = 0.3,
                   upgrade_guesses: bool = False) -> dict:
    """Fill in word_audio: real Commons recordings where kaikki/Wiktextract has
    one, else a real recording the direct Commons search found that kaikki
    missed, else a real Merriam-Webster recording (mw.py's Collegiate API --
    same cached, quota-respecting mw.lookup_api every other MW caller uses,
    so a word already looked up during fill_definitions'/maintain's Tier.MW
    costs nothing extra here, a genuinely fresh one still costs one of the
    shared 1000/day API calls), else Azure IPA-guided synthesis where a
    transcription is known (ours, kaikki's, or Wordnik's — backfilling
    word.ipa along the way), else local Piper grapheme-only synthesis (no
    IPA needed -- see audio.synthesize_piper's docstring for why this stays
    off Azure), else a 'none' placeholder so re-runs don't keep re-parsing
    the dump for words with nothing to find (in practice this should now be
    rare: Piper almost never fails). Azure synthesis picks its voice/lang
    from the word's ipa_source (audio.ipa_dialect_for_source/voice_for_dialect)
    so an OED-sourced (British RP) transcription gets the UK voice and the
    RP linking-r convention, not the US voice's rhotic default -- see
    audio.normalize_ipa's docstring for why that distinction matters.

    MW audio is real human speech (same trust tier as Commons, ahead of any
    synthesis), NOT gated on word.ipa the way Azure is -- MW's own
    pronunciation field is a proprietary respelling, never written to
    word.ipa (see resolve.py's Tier.MW), but the recording itself doesn't
    need that respelling to be usable."""
    import time
    from collections import Counter
    from .. import audio, mw as mw_module, wiktextract
    from ..dictionary import make_session
    s = _safe_schema(schema)

    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_audio a WHERE a.word_id=w.id)"
             if only_missing else "")
    if upgrade_guesses:
        # Words whose audio is only a spelling guess (Piper / legacy
        # azure_guess) but that have gained a transcription since -- e.g. an
        # 0 Dict IPA backfilled by `oed-ipa`, or a batch that fell back to
        # Piper while Azure's quota was spent. Only-missing never revisits
        # these (they HAVE audio), so they'd keep the guess forever.
        where = (f" WHERE w.active AND coalesce(w.ipa, '') <> '' AND EXISTS (SELECT 1 FROM {s}.word_audio a "
                 f"WHERE a.word_id = w.id AND a.source IN ('piper', 'azure_guess'))")
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma, w.ipa, w.ipa_source, w.part_of_speech, cs.download_url
                        FROM {s}.word w
                        LEFT JOIN {s}.word_commons_search cs ON cs.word_id = w.id{where}""" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    if upgrade_guesses:
        rows = [r for r in rows if audio.looks_like_english_ipa(r[2] or "")]

    dist: Counter = Counter()
    if not rows:
        return {"candidates": 0, **dist}

    lemmas = {lemma.strip().lower() for _, lemma, _, _, _, _ in rows}
    lexicon = wiktextract.sound_lexicon(
        conn, lemmas, dump_path, progress_cb=lambda n: print(f"  ...{n} lines scanned"))

    key, region = audio.azure_credentials()
    if not (key and region):
        print("  (no AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in .env — skipping synthesis, Commons-only pass)")

    # NOT gated on mw_module.quota_exhausted() here, unlike
    # fill_definitions/pipeline.py's batch-level pre-check: mw.lookup_api
    # already checks its on-disk cache BEFORE it ever looks at the quota
    # (a cache hit costs nothing, exhausted or not), so a blanket pre-check
    # at this level was silently throwing away already-cached, already-paid
    # -for MW recordings on any day the shared 1000/day cap had already been
    # spent by another caller earlier that day -- confirmed live: hundreds
    # of words with real cached MW audio (e.g. "callipygian") still ended up
    # 'azure' or 'none' because this batch happened to run after the quota
    # was gone. lookup_api's own internal check still protects the quota
    # for words that truly aren't cached yet.
    mw_key = mw_module.mw_api_key()
    mw_session = make_session() if mw_key else None

    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma, existing_ipa, ipa_source, pos, direct_search_url) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            entry = lexicon.get(lemma_lc, {})

            existing_ipa = existing_ipa if audio.looks_like_english_ipa(existing_ipa or "") else None

            kaikki_ipa = wiktextract.best_ipa(entry.get("ipa", []))
            if kaikki_ipa and not audio.looks_like_english_ipa(kaikki_ipa):
                kaikki_ipa = None
            if kaikki_ipa and not (existing_ipa or "").strip():
                cur.execute(f"UPDATE {s}.word SET ipa=%s, ipa_source='kaikki' WHERE id=%s", (kaikki_ipa, wid))
                existing_ipa = kaikki_ipa
                ipa_source = "kaikki"

            # tries=2 (not fetch_commons_audio's default 4-6): this loop needs to
            # move fast through many candidates and has Azure as a good fallback.
            # A sustained Commons rate-limit block turned a handful of slow
            # downloads into an hours-long stall here — `commons-download` is the
            # dedicated, patient (tries=6) pass for real recovery, run separately.
            best_recording = wiktextract.best_audio(entry.get("audio", []))
            row = None
            if best_recording:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                if audio.fetch_commons_audio(best_recording["url"], dest, tries=1):
                    row = ("commons", str(dest), None, best_recording["url"],
                           "Wikimedia Commons recording; verify per-file license before public reuse")
                    dist["commons"] += 1
            if row is None and direct_search_url:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                if audio.fetch_commons_audio(direct_search_url, dest, tries=1):
                    row = ("commons", str(dest), None, direct_search_url,
                           "Wikimedia Commons recording (direct search — kaikki's dump missed it); "
                           "verify per-file license before public reuse")
                    dist["commons_direct_search"] += 1
            if row is None and mw_key:
                tagger_pos = _POS_TO_TAGGER.get((pos or "").lower(), "")
                mw_entries = mw_module.exact_matches(
                    mw_module.lookup_api(lemma, mw_key, mw_session), lemma)
                mw_audio_url = None
                if mw_entries:
                    mw_entry = mw_module.pick_entry(mw_entries, tagger_pos)
                    mw_audio_url = next(
                        (p.audio_url for p in mw_entry.pronunciations if p.audio_url), None)
                if mw_audio_url:
                    dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                    if audio.fetch_commons_audio(mw_audio_url, dest, tries=1):
                        row = ("mw", str(dest), None, mw_audio_url,
                               "Merriam-Webster recording; verify terms before public reuse")
                        dist["mw"] += 1
            if row is None and (existing_ipa or "").strip() and key and region:
                dialect = audio.ipa_dialect_for_source(ipa_source)
                voice, lang = audio.voice_for_dialect(dialect)
                keep_optional = dialect == "us"
                clip = audio.synthesize_azure(lemma, existing_ipa, key, region,
                                               voice=voice, lang=lang, keep_optional=keep_optional)
                if clip:
                    dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                    dest.write_bytes(clip)
                    ipa_used = audio.normalize_ipa(existing_ipa, keep_optional=keep_optional)
                    row = ("azure", str(dest), ipa_used, voice, None)
                    dist["azure"] += 1
            if row is None and upgrade_guesses:
                dist["kept_guess"] += 1          # nothing better than the Piper clip it already has
                continue
            if row is None:
                # Piper: no curated IPA needed, so this is the tier that
                # actually closes the gap the others structurally can't --
                # see audio.py's module docstring for why it's local/
                # grapheme-only rather than an Azure-IPA substitute. Almost
                # never fails (any spelling can be phonemized), so this is
                # what 'none' below is now reserved for: Piper itself not
                # installed/configured, not an ordinary per-word miss.
                clip = audio.synthesize_piper(lemma)
                if clip:
                    dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                    dest.write_bytes(clip)
                    row = ("piper", str(dest), None, audio.PIPER_VOICE_NAME,
                           "Synthesized from spelling by local Piper TTS -- no verified pronunciation available")
                    dist["piper"] += 1
            if row is None:
                row = ("none", None, None, None, None)
                dist["none"] += 1

            cur.execute(
                f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice, license_note, generated_at)
                    VALUES (%s,%s,%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET source=EXCLUDED.source, file_path=EXCLUDED.file_path,
                        ipa_used=EXCLUDED.ipa_used, voice=EXCLUDED.voice, license_note=EXCLUDED.license_note,
                        generated_at=now()""",
                (wid, *row))
            if i % 50 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} words processed")
            time.sleep(delay)
    conn.commit()
    return {"candidates": len(rows), **dist}


def synthesize_unverified_guesses(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0,
                                   delay: float = 0.0) -> dict:
    """Sweeps up any pre-existing source='none' backlog (words compute_audio
    processed before Piper was wired in as its own final tier, or a run where
    Piper itself wasn't installed/configured) using the same local Piper
    grapheme-only synthesis compute_audio now does inline for new words --
    see audio.synthesize_piper's docstring for why this stays off Azure.
    Recorded with source='piper' — deliberately distinct from 'azure'
    (IPA-guided) so the quiz app can flag these as unverified rather than
    presenting a guess with the same confidence as a verified pronunciation.
    No external service, so no real rate limit; delay defaults to 0 and only
    exists for callers that want to throttle disk I/O."""
    import time
    from collections import Counter
    from .. import audio
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma FROM {s}.word w
                        JOIN {s}.word_audio a ON a.word_id = w.id
                        WHERE a.source = 'none'""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter(candidates=len(rows))
    if not rows:
        return dict(dist)
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            clip = audio.synthesize_piper(lemma)
            if clip:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                dest.write_bytes(clip)
                cur.execute(
                    f"""UPDATE {s}.word_audio SET source='piper', file_path=%s, ipa_used=NULL,
                        voice=%s, license_note='Synthesized from spelling by local Piper TTS -- no verified pronunciation available',
                        generated_at=now() WHERE word_id=%s""",
                    (str(dest), audio.PIPER_VOICE_NAME, wid))
                dist["synthesized"] += 1
            else:
                dist["failed"] += 1
            if i % 50 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} words processed")
            time.sleep(delay)
    conn.commit()
    return dict(dist)
