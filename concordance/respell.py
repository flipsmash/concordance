"""An admin's respelling of a word, and the upkeep it sets off.

zirbájeh was ingested from one book under a spelling no dictionary has; the
usual spelling is zirbaja. `respell` renames it (or, when zirbaja already
exists, merges it into that word), keeps the old spelling as an alias, looks
the definition up again under the new spelling, clears everything that was
derived from the old spelling, and queues a refresh job. `refresh_words`
(run in the background as `concordance refresh-words`) then redoes that
upkeep for just those words: categories, quiz definition, frequency,
pronunciation, origin, difficulty, embeddings and definition links.

An alias is honoured at ingest (pipeline.process maps it onto its word), so a
later book that uses the old spelling attaches to the respelled word.

A respelled word is marked admin_suggested, which the automated cleanup
sweeps respect -- otherwise a sweep could cast out the word the admin fixed
(a Persian dish is exactly what clean-foreign-words looks for).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Config
from .db import _safe_schema
from .model import Candidate

# The tagger's coarse tag the resolvers pick senses by (as pointer_targets).
_COARSE = {"noun": "NOUN", "verb": "VERB", "adjective": "ADJ", "adverb": "ADV"}

# Per-word rows derived from the spelling; the refresh job rebuilds them.
_DERIVED_TABLES = ("word_category", "word_ngram", "word_audio", "word_commons_search",
                   "word_embedding", "word_origin", "wn_relation_term")


class RespellError(ValueError):
    """The respelling can't be done as asked (bad spelling, unknown word)."""


@dataclass
class Collision:
    """The new spelling already belongs to another word: merging needs the
    admin's say-so (and, for an inactive word, bringing it back)."""
    word_id: int
    lemma: str
    active: bool
    inactive_reason: str | None
    books: int


class RespellCollision(Exception):
    def __init__(self, collision: Collision):
        super().__init__(f"{collision.lemma!r} already exists")
        self.collision = collision


def clean_spelling(raw: str) -> str:
    lemma = re.sub(r"\s+", " ", (raw or "").strip())
    if not lemma or len(lemma) > 80 or not re.fullmatch(r"[^\W\d_]+(?:[ '\-][^\W\d_]+)*", lemma):
        raise RespellError("a spelling is letters, optionally joined by spaces, hyphens or apostrophes")
    return lemma


def find_collision(cur, s: str, new_lc: str, word_id: int) -> Collision | None:
    """The word the new spelling already names -- as its lemma, or as one of
    its aliases -- other than the word being respelled."""
    cur.execute(f"""SELECT w.id, w.lemma, w.active, w.variant_flag_reason,
                           (SELECT count(*) FROM {s}.word_book wb WHERE wb.word_id = w.id)
                    FROM {s}.word w
                    WHERE w.id <> %s AND (w.lemma_lc = %s OR w.id IN
                          (SELECT a.word_id FROM {s}.word_alias a WHERE a.alias_lc = %s))
                    ORDER BY (w.lemma_lc = %s) DESC LIMIT 1""", (word_id, new_lc, new_lc, new_lc))
    row = cur.fetchone()
    if not row:
        return None
    wid, lemma, active, flag, books = row
    return Collision(word_id=wid, lemma=lemma, active=active, books=books,
                     inactive_reason=None if active else (flag or "pruned by hand"))


def respell(conn, schema: str, word_id: int, new_spelling: str, *, by: str,
            merge: bool = False, cfg: Config | None = None) -> dict:
    """Respell word `word_id` as `new_spelling`. Raises RespellCollision when
    the spelling belongs to another word and `merge` is False; with `merge`,
    folds this word into that one (reactivating it if it was inactive).
    Returns {"word_id": the surviving word, "lemma", "merged": bool,
    "definition_refreshed": bool}. Commits."""
    cfg = cfg or Config()
    s = _safe_schema(schema)
    new = clean_spelling(new_spelling)
    new_lc = new.lower()

    with conn.cursor() as cur:
        cur.execute(f"""SELECT lemma, definition, part_of_speech, flagged_undefined, definition_edited_by
                        FROM {s}.word WHERE id = %s AND active""", (word_id,))
        row = cur.fetchone()
        if row is None:
            raise RespellError("no such active word")
        old, definition, pos, flagged_undefined, hand_edited = row
        if new == old:
            raise RespellError("that's already its spelling")
        collision = find_collision(cur, s, new_lc, word_id)
    if collision and not merge:
        raise RespellCollision(collision)

    # Look the definition up again under the new spelling -- outside the
    # transaction (it can go to the network) -- when the old spelling left
    # it undefined. A real or hand-edited definition is kept.
    looked_up = None
    if not collision and not hand_edited and (not definition or flagged_undefined):
        from .pipeline import lookup_definitions
        looked_up = Candidate(lemma=new, pos=_COARSE.get((pos or "").lower(), ""))
        lookup_definitions(conn, [looked_up], cfg)
        if not looked_up.definition:
            looked_up = None

    with conn.cursor() as cur:
        if collision:
            survivor = collision.word_id
            _merge(cur, s, word_id, survivor)
            if not collision.active:
                cur.execute(f"UPDATE {s}.word SET active = true, variant_flag_reason = NULL, "
                            f"variant_flag_note = NULL, variant_flagged_at = NULL WHERE id = %s", (survivor,))
            lemma = collision.lemma
        else:
            survivor = word_id
            cur.execute(f"UPDATE {s}.word SET lemma = %s WHERE id = %s", (new, word_id))
            _clear_derived(cur, s, word_id)
            if looked_up:
                cur.execute(
                    f"""UPDATE {s}.word SET definition = %s, part_of_speech = %s, ipa = %s, synonyms = %s,
                            etymology = %s, definition_source = %s WHERE id = %s""",
                    (looked_up.definition, looked_up.part_of_speech or pos, looked_up.ipa or None,
                     list(looked_up.synonyms), looked_up.etymology, looked_up.definition_source, word_id))
            lemma = new

        # the old spelling now names the survivor; the new one is no alias
        cur.execute(f"""INSERT INTO {s}.word_alias (alias_lc, word_id, created_by) VALUES (%s, %s, %s)
                        ON CONFLICT (alias_lc) DO UPDATE SET word_id = EXCLUDED.word_id""",
                    (old.lower(), survivor, by))
        cur.execute(f"DELETE FROM {s}.word_alias WHERE alias_lc = %s", (lemma.lower(),))
        cur.execute(f"""UPDATE {s}.word SET admin_suggested = true,
                            admin_suggested_by = coalesce(admin_suggested_by, %s),
                            admin_suggested_at = coalesce(admin_suggested_at, now()), updated_at = now()
                        WHERE id = %s""", (by, survivor))
        cur.execute(f"""INSERT INTO {s}.word_refresh_job (word_id, old_lemma, new_lemma, requested_by)
                        VALUES (%s, %s, %s, %s)""", (survivor, old, lemma, by))
    conn.commit()
    return {"word_id": survivor, "lemma": lemma, "merged": bool(collision),
            "definition_refreshed": looked_up is not None}


def _clear_derived(cur, s: str, word_id: int) -> None:
    """Forget everything that was worked out from the old spelling. The
    book's sentence and as_seen stay: they're the book's real text."""
    cur.execute(f"""UPDATE {s}.word SET
                        ipa = NULL, ipa_source = NULL, ipa_checked_at = NULL,
                        wordnik_pron_raw = NULL, wordnik_pron_type = NULL, wordnik_checked_at = NULL,
                        mw_checked_at = NULL, first_known_use = NULL,
                        quiz_definition = NULL, quiz_def_source = NULL,
                        flagged_undefined = false, flagged_undefined_at = NULL,
                        validity_label = NULL, validity_score = NULL, validity_notes = NULL,
                        suggested_correction = NULL, validity_checked_at = NULL,
                        variant_flag_reason = NULL, variant_flag_note = NULL, variant_flagged_at = NULL
                    WHERE id = %s""", (word_id,))
    for table in _DERIVED_TABLES:
        cur.execute(f"DELETE FROM {s}.{table} WHERE word_id = %s", (word_id,))
    cur.execute(f"DELETE FROM {s}.word_definition_link WHERE source_word_id = %s OR target_word_id = %s",
                (word_id, word_id))


def _merge(cur, s: str, src: int, dst: int) -> None:
    """Move everything people and books attach to word `src` onto `dst`, then
    delete `src`. Where both have a row (same book, set, or user), dst's is
    kept -- except a review schedule, where the one with more answers wins."""
    cur.execute(f"""INSERT INTO {s}.word_book (word_id, book_id) SELECT %s, book_id FROM {s}.word_book
                    WHERE word_id = %s ON CONFLICT DO NOTHING""", (dst, src))
    cur.execute(f"""INSERT INTO {s}.word_set_item (set_id, word_id, mastered, mastered_at, added_at)
                    SELECT set_id, %s, mastered, mastered_at, added_at FROM {s}.word_set_item
                    WHERE word_id = %s ON CONFLICT DO NOTHING""", (dst, src))
    cur.execute(f"""INSERT INTO {s}.word_review_schedule AS r
                        (user_id, word_id, streak, last_seen_at, next_eligible_at, correct_count, incorrect_count)
                    SELECT user_id, %s, streak, last_seen_at, next_eligible_at, correct_count, incorrect_count
                    FROM {s}.word_review_schedule WHERE word_id = %s
                    ON CONFLICT (user_id, word_id) DO UPDATE SET
                        streak = EXCLUDED.streak, last_seen_at = EXCLUDED.last_seen_at,
                        next_eligible_at = EXCLUDED.next_eligible_at,
                        correct_count = EXCLUDED.correct_count, incorrect_count = EXCLUDED.incorrect_count
                    WHERE EXCLUDED.correct_count + EXCLUDED.incorrect_count
                          > r.correct_count + r.incorrect_count""", (dst, src))
    cur.execute(f"""INSERT INTO {s}.word_personal_difficulty
                        (user_id, word_id, item_rating, personal_difficulty, based_on_correct, calibrated_at)
                    SELECT user_id, %s, item_rating, personal_difficulty, based_on_correct, calibrated_at
                    FROM {s}.word_personal_difficulty WHERE word_id = %s ON CONFLICT DO NOTHING""", (dst, src))
    cur.execute(f"UPDATE {s}.quiz_answer SET word_id = %s WHERE word_id = %s", (dst, src))
    cur.execute(f"""UPDATE {s}.quiz_question SET target_word_ids = array_replace(target_word_ids, %s, %s)
                    WHERE %s = ANY(target_word_ids)""", (src, dst, src))
    cur.execute(f"UPDATE {s}.word_alias SET word_id = %s WHERE word_id = %s", (dst, src))
    cur.execute(f"UPDATE {s}.word_refresh_job SET word_id = %s WHERE word_id = %s", (dst, src))
    cur.execute(f"DELETE FROM {s}.word WHERE id = %s", (src,))


# --- the background upkeep ---------------------------------------------------

def open_jobs(conn, schema: str) -> list[tuple[int, int, str, str]]:
    """Claim every pending job: [(job id, word id, old lemma, new lemma)]."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""UPDATE {s}.word_refresh_job SET status = 'running', started_at = now()
                        WHERE status = 'pending' RETURNING id, word_id, old_lemma, new_lemma""")
        jobs = cur.fetchall()
    conn.commit()
    return jobs


def refresh_words(conn, schema: str, cfg: Config | None = None, *, database_url: str | None = None,
                  log=print) -> dict:
    """Run the upkeep for every pending job, as `maintain` would for these
    words alone. The GPU steps wait for the GPU lock (gpu_lock), so a
    running ingest or maintain finishes first. Returns {"jobs", "words"}."""
    from . import compound, origins
    from . import db
    from .classify import classify_and_store
    from .gpu_lock import gpu_lock

    cfg = cfg or Config()
    s = _safe_schema(schema)
    jobs = open_jobs(conn, schema)
    if not jobs:
        return {"jobs": 0, "words": 0}
    ids = sorted({wid for _, wid, _, _ in jobs})
    spellings = sorted({x.lower() for _, _, old, new in jobs for x in (old, new)})
    try:
        log(f"refreshing {len(ids)} word(s): frequency, origin, pronunciation…")
        db.fetch_ngrams(conn, schema, only_missing=True)
        origins.compute(conn, schema, progress=lambda *_: None)
        db.compute_ipa(conn, schema, only_missing=True)
        db.fetch_wordnik_pronunciations(conn, schema, only_missing=True)

        with gpu_lock(database_url, on_wait=lambda: log("waiting for the GPU (another job is using it)…")):
            log("categories, compound check and quiz definition…")
            classify_and_store(conn, schema, cfg, word_ids=ids)
            compound.judge_and_store(conn, schema, cfg)
            db.compute_quiz_definitions(conn, schema, cfg, word_ids=ids)

        log("archaic, difficulty, quizzable, embeddings, links…")
        db.compute_archaic(conn, schema)
        db.compute_difficulty(conn, schema)
        db.compute_quizzable(conn, schema)
        db.compute_definition_embeddings(conn, schema, only_missing=True, word_ids=ids)
        from .cli import _FASTTEXT_MODEL_PATH
        if _FASTTEXT_MODEL_PATH.exists():
            db.compute_fasttext_embeddings(conn, schema, model_path=str(_FASTTEXT_MODEL_PATH), only_missing=True)
        relink(conn, schema, ids, spellings)
    except Exception as exc:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f"""UPDATE {s}.word_refresh_job SET status = 'failed', finished_at = now(), error = %s
                            WHERE id = ANY(%s)""", (f"{type(exc).__name__}: {exc}"[:500], [j[0] for j in jobs]))
        conn.commit()
        raise
    with conn.cursor() as cur:
        cur.execute(f"""UPDATE {s}.word_refresh_job SET status = 'done', finished_at = now()
                        WHERE id = ANY(%s)""", ([j[0] for j in jobs],))
    conn.commit()
    return {"jobs": len(jobs), "words": len(ids)}


def relink(conn, schema: str, word_ids: list[int], spellings: list[str]) -> None:
    """Definition links for the respelled words and for every definition
    that mentions either spelling -- not a full 70k-definition recompute."""
    from . import db
    s = _safe_schema(schema)
    patterns = [f"%{sp}%" for sp in spellings]
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id FROM {s}.word WHERE active AND (id = ANY(%s) OR definition ILIKE ANY(%s))""",
                    (word_ids, patterns))
        sources = [r[0] for r in cur.fetchall()]
    db.compute_definition_links(conn, schema, source_ids=sources)


# --- ingest: old spellings attach to their word -------------------------------

def alias_map(conn, schema: str) -> dict[str, str]:
    """alias -> the lemma (lowercased) of the word it now belongs to."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT a.alias_lc, w.lemma_lc FROM {s}.word_alias a
                        JOIN {s}.word w ON w.id = a.word_id""")
        return dict(cur.fetchall())


def apply_aliases(candidates: dict[str, Candidate], aliases: dict[str, str]) -> int:
    """Rename each candidate spelled as an alias to its word's lemma, folding
    its occurrences into that lemma's candidate when the book has both.
    Mutates `candidates` (keyed by lemma); returns how many were mapped.
    Pure, so ingest's behaviour is unit-testable."""
    mapped = 0
    for alias in [k for k in candidates if k.lower() in aliases]:
        target = aliases[alias.lower()]
        cand = candidates.pop(alias)
        if target in candidates:
            candidates[target].occurrences.extend(cand.occurrences)
        else:
            cand.lemma = target
            candidates[target] = cand
        mapped += 1
    return mapped
