"""Writing words in: the master-CSV sync, per-book ingest results, known verdicts, POS cleanup."""

from __future__ import annotations

import csv
from pathlib import Path

import psycopg

from ..model import RejectReason, normalize_pos
from .core import DEFAULT_SCHEMA, _safe_schema


def _synonyms(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def _books(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def _read_master_rows(path: Path) -> list[dict]:
    """master_vocab.csv is tool-written with a full MASTER_COLUMNS header (it is not
    hand-edited in Excel like the per-book files), so a plain DictReader keeps every
    column — crucially date_added and source_book, which the vocab-only reader drops."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if (r.get("word") or "").strip()]


def sync_master(csv_path: Path, conn: psycopg.Connection,
                schema: str = DEFAULT_SCHEMA) -> dict:
    """Upsert every row of master_vocab.csv into the DB. Idempotent."""
    s = _safe_schema(schema)
    rows = _read_master_rows(Path(csv_path))
    stats = {"words": 0, "books": 0, "links": 0, "rows": len(rows)}
    seen_books: dict[str, int] = {}

    with conn.cursor() as cur:
        for r in rows:
            word = (r.get("word") or "").strip()
            if not word:
                continue
            definition = r.get("definition") or ""
            is_blank = not definition.strip()

            cur.execute(f"SELECT definition FROM {s}.word WHERE lemma_lc = lower(%s)", (word,))
            prior = cur.fetchone()
            old_definition = (prior[0] or "").strip() if prior else None

            cur.execute(
                f"""INSERT INTO {s}.word
                    (lemma, as_seen, definition, part_of_speech, ipa, sentence,
                     chapter, synonyms, etymology, definition_source, first_added,
                     flagged_undefined, flagged_undefined_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NULLIF(%s,'')::date,
                            %s, CASE WHEN %s THEN now() ELSE NULL END)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        as_seen=EXCLUDED.as_seen,
                        definition=COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition),
                        part_of_speech=EXCLUDED.part_of_speech,
                        ipa=COALESCE(NULLIF(EXCLUDED.ipa,''), {s}.word.ipa),
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter,
                        synonyms=CASE WHEN cardinality(EXCLUDED.synonyms) > 0
                                      THEN EXCLUDED.synonyms ELSE {s}.word.synonyms END,
                        etymology=COALESCE(NULLIF(EXCLUDED.etymology,''), {s}.word.etymology),
                        definition_source=COALESCE(NULLIF(EXCLUDED.definition_source,''),
                                                    {s}.word.definition_source),
                        first_added=LEAST(
                            {s}.word.first_added,
                            COALESCE(EXCLUDED.first_added, {s}.word.first_added)),
                        flagged_undefined={s}.word.flagged_undefined OR
                            (COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''),
                        flagged_undefined_at=CASE
                            WHEN {s}.word.flagged_undefined THEN {s}.word.flagged_undefined_at
                            WHEN COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''
                                THEN now()
                            ELSE {s}.word.flagged_undefined_at
                        END,
                        updated_at=now()
                    RETURNING id, definition""",
                (word, r.get("as_seen"), definition, normalize_pos(r.get("part_of_speech")),
                 r.get("ipa"), r.get("sentence"), r.get("chapter"), _synonyms(r.get("synonyms", "")),
                 r.get("etymology"), r.get("source"), (r.get("date_added") or ""),
                 is_blank, is_blank),
            )
            word_id, new_definition = cur.fetchone()
            stats["words"] += 1

            if old_definition and (new_definition or "").strip() != old_definition:
                _invalidate_definition_dependents(cur, s, word_id)

            for title in _books(r.get("source_book", "")):
                if title not in seen_books:
                    cur.execute(
                        f"""INSERT INTO {s}.book (title) VALUES (%s)
                            ON CONFLICT (title) DO UPDATE SET title=EXCLUDED.title
                            RETURNING id""", (title,))
                    seen_books[title] = cur.fetchone()[0]
                    stats["books"] += 1
                cur.execute(
                    f"""INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s,%s)
                        ON CONFLICT DO NOTHING""", (word_id, seen_books[title]))
                if cur.rowcount:
                    stats["links"] += 1
    conn.commit()
    return stats


def _invalidate_definition_dependents(cur, s: str, word_id: int) -> None:
    """Clear the downstream artifacts computed FROM word.definition text whose
    recompute is only-missing/NOT-EXISTS gated -- i.e. the ones that would
    otherwise silently go stale and never get revisited once this word's
    definition changes (e.g. the same lemma resolving to a different
    dictionary sense when a later book re-ingests it -- see the "changeful"
    bug this was written for: its quiz_definition was a redaction of an
    earlier, longer definition no longer stored anywhere).

    Deliberately NOT touched here: archaic, difficulty, and quizzable. All
    three fully recompute every row unconditionally whenever their command
    runs (no only-missing filter), so they self-correct on the next
    maintenance pass with no help -- invalidating them would just be a
    no-op that adds noise."""
    cur.execute(f"UPDATE {s}.word SET quiz_definition=NULL, quiz_def_source=NULL WHERE id=%s", (word_id,))
    cur.execute(f"DELETE FROM {s}.word_category WHERE word_id=%s", (word_id,))
    cur.execute(
        f"""UPDATE {s}.word_embedding SET definition_vector=NULL, definition_model=NULL, definition_source=NULL
            WHERE word_id=%s""",
        (word_id,))


def sync_book_results(conn, book_title: str, kept: list, rejected: list,
                       schema: str = DEFAULT_SCHEMA, author: str | None = None) -> dict:
    """Upsert one book's ingestion results straight into Postgres — no CSV, no
    hand-edit, no `finalize`. KEEP/UNSURE candidates go into word/word_book
    exactly like sync_master; DROPped ones go into rejected_word, one row per
    (book, lemma). Review/pruning happens afterward in the review webapp
    (word.active) rather than before promotion. Idempotent: re-running the
    same book updates both tables in place. `author` is COALESCEd on conflict
    so re-ingesting a book without a parsed author never blanks a known one."""
    s = _safe_schema(schema)
    stats = {"kept": 0, "rejected": 0, "cast_out": 0}

    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {s}.book (title, author) VALUES (%s, %s)
                ON CONFLICT (title) DO UPDATE SET title=EXCLUDED.title,
                    author=COALESCE(EXCLUDED.author, {s}.book.author)
                RETURNING id""", (book_title, author))
        book_id = cur.fetchone()[0]

        for c in kept:
            rep = c.representative
            definition = c.definition or ""
            is_blank = not definition.strip()

            # Fetched before the upsert so it reflects the pre-upsert value --
            # needed to tell "this lemma's definition just changed" apart from
            # "first time seeing this lemma" / "same value again", the only
            # case _invalidate_definition_dependents needs to fire for.
            cur.execute(f"SELECT definition FROM {s}.word WHERE lemma_lc = lower(%s)", (c.lemma,))
            prior = cur.fetchone()
            old_definition = (prior[0] or "").strip() if prior else None

            cur.execute(
                f"""INSERT INTO {s}.word
                    (lemma, as_seen, definition, part_of_speech, ipa, sentence,
                     chapter, synonyms, etymology, definition_source, first_added,
                     flagged_undefined, flagged_undefined_at,
                     variant_flag_reason, variant_flag_note, variant_flagged_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE,
                            %s, CASE WHEN %s THEN now() ELSE NULL END,
                            NULLIF(%s,''), NULLIF(%s,''), CASE WHEN %s <> '' THEN now() ELSE NULL END)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        as_seen=EXCLUDED.as_seen,
                        definition=COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition),
                        part_of_speech=EXCLUDED.part_of_speech,
                        ipa=COALESCE(NULLIF(EXCLUDED.ipa,''), {s}.word.ipa),
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter,
                        synonyms=CASE WHEN cardinality(EXCLUDED.synonyms) > 0
                                      THEN EXCLUDED.synonyms ELSE {s}.word.synonyms END,
                        etymology=COALESCE(NULLIF(EXCLUDED.etymology,''), {s}.word.etymology),
                        definition_source=COALESCE(NULLIF(EXCLUDED.definition_source,''),
                                                    {s}.word.definition_source),
                        flagged_undefined={s}.word.flagged_undefined OR
                            (COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''),
                        flagged_undefined_at=CASE
                            WHEN {s}.word.flagged_undefined THEN {s}.word.flagged_undefined_at
                            WHEN COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''
                                THEN now()
                            ELSE {s}.word.flagged_undefined_at
                        END,
                        variant_flag_reason=COALESCE(EXCLUDED.variant_flag_reason, {s}.word.variant_flag_reason),
                        variant_flag_note=COALESCE(EXCLUDED.variant_flag_note, {s}.word.variant_flag_note),
                        variant_flagged_at=COALESCE(EXCLUDED.variant_flagged_at, {s}.word.variant_flagged_at),
                        updated_at=now()
                    RETURNING id, definition""",
                (c.lemma, rep.surface if rep else "", definition,
                 normalize_pos(c.part_of_speech or c.pos), c.ipa,
                 rep.sentence if rep else "", rep.chapter if rep else "",
                 list(c.synonyms), c.etymology,
                 c.definition_source or ", ".join(c.validity_sources),
                 is_blank, is_blank,
                 c.variant_flag_reason, c.variant_flag_note, c.variant_flag_reason))
            word_id, new_definition = cur.fetchone()
            stats["kept"] += 1

            if old_definition and (new_definition or "").strip() != old_definition:
                _invalidate_definition_dependents(cur, s, word_id)

            cur.execute(
                f"""INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s,%s)
                    ON CONFLICT DO NOTHING""", (word_id, book_id))

        for c in rejected:
            rep = c.representative
            stats["rejected"] += 1
            # FREQUENCY_FLOOR is the one reject reason that's NOT book-specific
            # -- a lemma's zipf frequency doesn't vary by book, so this verdict
            # is the same everywhere and recomputable in <1ms (floor.py:29),
            # unlike every other reason here (judge/human/junk-POS calls that
            # can genuinely differ book to book, which is why rejected_word is
            # deliberately one row per (book, lemma) rather than deduped --
            # see its own table comment). Persisting it anyway was pure
            # duplication with no reader: confirmed live at 76.3M rows behind
            # just 58,413 distinct lemmas (a 1,307x duplication factor) before
            # this was skipped, ~21GB of the table's ~40GB, and nothing
            # queries rejected_word by this reason except the admin reason-
            # filter dropdown (now sourced from the RejectReason enum
            # directly instead, see /api/rejected/reasons). Still counted in
            # stats["rejected"] above so the per-book console summary stays
            # accurate -- only the DB write is skipped.
            if c.reject_reason is RejectReason.FREQUENCY_FLOOR:
                continue
            cur.execute(
                f"""INSERT INTO {s}.rejected_word
                    (book_id, lemma, reason, detail, count, zipf, pos, as_seen, sentence, chapter)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (book_id, lemma_lc) DO UPDATE SET
                        reason=EXCLUDED.reason, detail=EXCLUDED.detail,
                        count=EXCLUDED.count, zipf=EXCLUDED.zipf,
                        pos=EXCLUDED.pos, as_seen=EXCLUDED.as_seen,
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter""",
                (book_id, c.lemma, c.reject_reason.value if c.reject_reason else None,
                 c.interesting_reason or None, c.count, c.zipf,
                 c.pos, rep.surface if rep else None,
                 rep.sentence if rep else None, rep.chapter if rep else None))

            # A symbol/proper-noun rejection can happen for a lemma that's
            # already an active word from an earlier book (pipeline.py's
            # post-enrichment junk-POS check now applies on every
            # re-encounter, not just the first) -- cast it out here too, same
            # as refill/deepen already do for their own junk-POS
            # resolutions. A no-op UPDATE (0 rows) for a lemma with no
            # existing word row, so this is safe to run unconditionally
            # rather than needing to first check whether one exists.
            if c.reject_reason in (RejectReason.PROPER_NOUN, RejectReason.NUMERIC_OR_SYMBOL):
                cur.execute(
                    f"""UPDATE {s}.word SET active=false, updated_at=now()
                        WHERE lemma_lc = lower(%s) AND active AND NOT admin_suggested""",
                    (c.lemma,))
                stats["cast_out"] += cur.rowcount

    conn.commit()
    return stats


def fetch_known_verdicts(conn, schema: str = DEFAULT_SCHEMA) -> dict[str, str]:
    """Map lemma_lc -> a cached verdict from EARLIER books, so the (expensive)
    LLM judge is only ever run on lemmas whose verdict isn't already known.

    The judge's input for a word is purely (lemma, its wordfreq band) — no
    book/sentence/POS context — and it runs at temp 0, so a given lemma's
    verdict is the same in every book. Re-judging "refectory" from scratch in
    every book of a shared-vocabulary corpus is pure waste; this is the cache
    that eliminates it.

      'keep'    -> in `word`, active    (judge kept it; human hasn't pruned)
      'pruned'  -> in `word`, inactive  (human manually pruned via the webapp)
      <reason>  -> in `rejected_word`, one of 'not_interesting', 'numeric_or_symbol',
                                        or 'proper_noun' -- the specific reason, not a
                                        generic 'reject', so pipeline.py's _VERDICT_MAP
                                        can restore the true original reason on a cached
                                        hit (judge, or the post-enrichment junk-POS gate,
                                        rejected it before — both are purely lemma-derived,
                                        like the judge verdict, so caching them is exactly
                                        as safe: see pipeline.py's junk_pos_reason gate)

    `word` wins over `rejected_word` for a lemma present in both: a promoted
    row is authoritative and its `active` flag reflects the human's latest
    call. Re-fetched per book (cheap, indexed) so book N sees the new keeps
    that books 1..N-1 added earlier in the same batch."""
    s = _safe_schema(schema)
    verdicts: dict[str, str] = {}
    with conn.cursor() as cur:
        # Both session-scoped (this connection is opened and closed once per
        # book, see pipeline.process) -- neither touches the server default.
        # work_mem: without this the Sort/HashAggregate over however many
        # rows currently match spills to disk under the 4MB default (measured
        # live: 47.7s with the default vs 9.9s at 128MB, same plan otherwise).
        # random_page_cost: measured live that the planner otherwise picks a
        # Bitmap Heap Scan over rejected_word_reason_lemma_key_idx (still
        # ~9-10s) instead of the actually-fastest Parallel Index Only Scan
        # (~4s) that same index supports -- its default (4.0, tuned for
        # spinning disks) overweights random I/O relative to this machine's
        # actual (SSD) storage. 1.1 is the standard SSD-tuning value and gets
        # the planner to pick the fast plan on its own.
        cur.execute("SET work_mem = '128MB'")
        cur.execute("SET random_page_cost = 1.1")
        # The specific reason (not a generic "reject") so pipeline.py's
        # _VERDICT_MAP can restore the true original reason on a cached hit
        # instead of relabeling every cached reject as not_interesting. DISTINCT
        # because rejected_word is deliberately one row per (book, lemma) --
        # a common lemma rejected the same way in thousands of books would
        # otherwise ship one duplicate tuple per book instead of one per
        # lemma, which is what blew this call up to tens of GB of client-side
        # buffering (28M rows fetched here at ~106M total rejected_word rows)
        # and OOM-killed a live ingest run on 2026-08-16.
        cur.execute(f"""SELECT DISTINCT lemma_lc, reason FROM {s}.rejected_word
                        WHERE reason IN ('not_interesting', 'numeric_or_symbol', 'proper_noun')""")
        for lemma, reason in cur.fetchall():
            verdicts[lemma] = reason
        cur.execute(f"SELECT lemma_lc, active FROM {s}.word")
        for lemma, active in cur.fetchall():
            verdicts[lemma] = "keep" if active else "pruned"   # word overrides rejected_word
    return verdicts


def normalize_word_pos(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Clean up word.part_of_speech in place: folds abbreviations/case variants
    (adj, adv, pron, adp, sconj, num, Noun, Adjective, ...) accumulated from
    older write paths down to the canonical vocabulary via normalize_pos().
    Idempotent — safe to re-run any time a new inconsistency creeps in.
    Always recomputes every word in scope (no only_missing gate): the source
    column is mutable and there's no separate signal to gate a re-check on,
    so freezing a word's normalized POS after the one time this ran would
    silently stop it from self-correcting if part_of_speech changes later."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, part_of_speech FROM {s}.word ORDER BY id" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        changed = 0
        for wid, pos in rows:
            new_pos = normalize_pos(pos)
            if new_pos != (pos or ""):
                cur.execute(f"UPDATE {s}.word SET part_of_speech = %s WHERE id = %s", (new_pos, wid))
                changed += 1
    conn.commit()
    return {"words": len(rows), "changed": changed}
