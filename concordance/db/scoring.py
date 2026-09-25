"""Per-word scores: archaic signal, difficulty, quiz definitions/quizzability, personal difficulty."""

from __future__ import annotations

from .core import DEFAULT_SCHEMA, _safe_schema


def compute_archaic(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Set the archaic-currency ordinal on word_difficulty for every word. Uses the
    definition register-label + (if present) vocab.wiktionary is_archaic/is_obsolete
    (no Google Books signal -- see archaic.py's docstring for why).
    Always recomputes every word in scope (no only_missing gate) -- definition
    text can change after the first run, and there's no
    signal to gate a re-check on other than just running it again."""
    from collections import Counter
    from .. import archaic as _archaic
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute("select to_regclass('vocab.wiktionary')")
        have_wik = cur.fetchone()[0] is not None
    join = ("LEFT JOIN (select lower(term) t, bool_or(is_archaic) arc, bool_or(is_obsolete) obs "
            "from vocab.wiktionary group by lower(term)) k on k.t = lower(w.lemma)") if have_wik else ""
    cols = "coalesce(k.arc,false), coalesce(k.obs,false)" if have_wik else "false, false"
    dist: Counter = Counter()
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.definition, {cols}
                        FROM {s}.word w {join}
                        ORDER BY w.id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        for wid, defn, arc, obs in rows:
            flag, evid, conf = _archaic.classify(defn, arc, obs)
            dist[flag] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, archaic, archaic_evidence, archaic_confidence, updated_at)
                    VALUES (%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        archaic=EXCLUDED.archaic, archaic_evidence=EXCLUDED.archaic_evidence,
                        archaic_confidence=EXCLUDED.archaic_confidence, updated_at=now()""",
                (wid, flag, evid, conf))
    conn.commit()
    return dict(dist)


def compute_difficulty(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Compute the ex-ante difficulty scalar (+ factor breakdown) for every word.
    Always recomputes every word in scope (no only_missing gate) -- ngram,
    archaic, and domain data are all mutable upstream inputs with no signal
    to gate a re-check on."""
    import statistics
    from psycopg.types.json import Json
    from .. import difficulty as _diff
    from wordfreq import zipf_frequency
    from ..validity_score import _morph_root
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT w.id, w.lemma, g.peak, g.recent, d.archaic, d.archaic_confidence, coalesce(dom.fields,'')
            FROM {s}.word w
            LEFT JOIN {s}.word_ngram g ON g.word_id = w.id
            LEFT JOIN {s}.word_difficulty d ON d.word_id = w.id
            LEFT JOIN (SELECT wc.word_id, string_agg(DISTINCT left(c.code,1), '') fields
                       FROM {s}.word_category wc JOIN {s}.category c ON c.id = wc.category_id
                       GROUP BY wc.word_id) dom ON dom.word_id = w.id
            ORDER BY w.id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        scores = []
        for wid, lemma, peak, recent, archaic, aconf, fields in rows:
            key = lemma.strip().lower()
            root = _morph_root(key)
            has_domain = any(f in _diff.DOMAIN_FIELDS for f in fields)
            sc, factors = _diff.score(
                zipf_frequency(key, "en"), recent, peak, archaic or "current", aconf,
                has_domain, morph_transparent=root is not None,
                root_zipf=zipf_frequency(root, "en") if root else None)
            scores.append(sc)
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, difficulty, difficulty_factors, updated_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        difficulty=EXCLUDED.difficulty, difficulty_factors=EXCLUDED.difficulty_factors,
                        updated_at=now()""",
                (wid, sc, Json(factors)))
    conn.commit()
    return {"words": len(scores),
            "mean": round(statistics.mean(scores), 1) if scores else 0,
            "median": statistics.median(scores) if scores else 0}


def compute_quiz_definitions(conn, schema: str = DEFAULT_SCHEMA, cfg=None,
                             only_missing: bool = True, limit: int = 0) -> dict:
    """Set quiz_definition/quiz_def_source. Clean defs pass through free; leakers are
    LLM-rewritten (validated) or redacted. Resumable via only_missing (scale-ready)."""
    from collections import Counter
    from .. import quizdef
    s = _safe_schema(schema)
    where = "quiz_definition IS NULL AND " if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, definition FROM {s}.word "
                    f"WHERE {where}coalesce(definition,'') <> ''" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    clean = [(i, l, d) for i, l, d in rows if not quizdef.has_leak(l, d)]
    leakers = [(i, l, d) for i, l, d in rows if quizdef.has_leak(l, d)]
    stats = Counter()

    with conn.cursor() as cur:
        for wid, lemma, defn in clean:                       # free — no model
            cur.execute(f"UPDATE {s}.word SET quiz_definition=%s, quiz_def_source='clean' WHERE id=%s",
                        (defn, wid))
            stats["clean"] += 1
        conn.commit()

    if leakers:
        rw = quizdef.Rewriter(cfg)
        res = rw.rewrite([{"word": l, "definition": d} for _, l, d in leakers])
        with conn.cursor() as cur:
            for wid, lemma, defn in leakers:
                qd, src = res.get(lemma.lower(), (quizdef.redact(lemma, defn), "redacted"))
                cur.execute(f"UPDATE {s}.word SET quiz_definition=%s, quiz_def_source=%s WHERE id=%s",
                            (qd, src, wid))
                stats[src] += 1
        conn.commit()
        # Deterministic release, not left to implicit GC timing -- see
        # fill_definitions' matching comment for the live crash this
        # pattern is fixing (a fresh model load right after this step
        # returns, racing the previous instance's GPU memory release).
        rw.close()
    return {"words": len(rows), "clean": stats["clean"],
            "rewritten": stats["rewritten"], "redacted": stats["redacted"]}


def compute_quizzable(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Set the quizzable flag (+ reason) on word_difficulty for every word.
    Always recomputes every word in scope (no only_missing gate) -- definition
    and quiz_definition are both mutable upstream inputs with no signal to
    gate a re-check on."""
    from collections import Counter
    from wordfreq import zipf_frequency
    from .. import quizdef
    from ..validity_score import _morph_root
    s = _safe_schema(schema)
    dist: Counter = Counter()
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, definition, quiz_definition, quiz_def_source "
                    f"FROM {s}.word WHERE coalesce(definition,'') <> '' ORDER BY id" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        for wid, lemma, defn, quiz_defn, quiz_def_source in rows:
            root = _morph_root(lemma)
            rz = zipf_frequency(root, "en") if root else None
            ok, reason = quizdef.quizzable(defn, root, rz, quiz_defn, quiz_def_source, word=lemma)
            dist["quizzable" if ok else "excluded"] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, quizzable, quizzable_reason, updated_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        quizzable=EXCLUDED.quizzable, quizzable_reason=EXCLUDED.quizzable_reason, updated_at=now()""",
                (wid, ok, reason or None))
    conn.commit()
    return dict(dist)


def compute_personal_difficulty(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0) -> dict:
    """`concordance calibrate-difficulty` / `maintain`'s calibration step: a
    per-(user, word) personalized adjustment to the ex-ante difficulty
    score, from that user's own FIRST exposure to the word in a quiz --
    see concordance/calibration.py's module docstring for the model and
    why this is deliberately NOT written into the shared, all-users-facing
    word_difficulty.difficulty column (one dominant rater's response data
    never identifies population-level item difficulty, no matter how much
    of it accumulates -- see calibration.py).

    Only a word's FIRST quiz exposure per user counts (a window-function
    row_number() = 1 filter, below) -- a later re-exposure of the same
    word is evidence the person is LEARNING it (word_review_schedule's own
    reason for existing), not independent evidence about a fixed item
    difficulty; folding repeat exposures in would read "he learned it" as
    "it got easier," a confound that gets worse, not better, as the same
    user answers the same words repeatedly over a long time.

    KNOWN GAP: "first" here means first quiz_answer row WITH a guessing_floor
    (the WHERE below), not first ever. guessing_floor didn't exist before
    this feature shipped, so a (user, word) pair quizzed pre-migration and
    then answered again post-migration gets that later answer treated as
    rn=1 -- a real repeat exposure miscounted as a first one, the exact
    confound the paragraph above is trying to avoid. Accepted rather than
    fixed: pre-migration rows have no guessing_floor to build a response-
    probability model from, so they're unusable as an anchor regardless: the
    alternative (skip any pair with prior history, migration-era or not)
    trades this confound for discarding real data. Revisit if it turns out
    to matter in practice.

    Always recomputes every first-exposure row in scope on every run (no
    only-missing gate) -- cheap, pure-local arithmetic, and it must re-run
    whenever the underlying ex-ante difficulty changes upstream anyway,
    same "recompute is fine, it's cheap" reasoning as archaic/difficulty/
    quizzable. Truncates the whole table before repopulating on an
    unqualified (limit=0) run rather than a targeted delete -- see
    compute_book_similarity's own comment on this: a targeted delete only
    reaches rows still in scope THIS run, so a (user, word) pair that drops
    out of scope (its quiz_answer/quiz_question/quiz_session deleted, say)
    would otherwise keep a stale row forever."""
    from .. import calibration as calib

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT value FROM {s}.app_settings WHERE key = 'calibration_eta'")
        row = cur.fetchone()
        eta = (row[0] or {}).get("value", calib.DEFAULT_ETA) if row else calib.DEFAULT_ETA
        cur.execute(f"SELECT value FROM {s}.app_settings WHERE key = 'calibration_scale'")
        row = cur.fetchone()
        scale = (row[0] or {}).get("value", calib.DEFAULT_SCALE) if row else calib.DEFAULT_SCALE

        cur.execute(f"""
            WITH first_exposure AS (
                SELECT qa.word_id, qs.user_id, qa.is_correct, qa.guessing_floor,
                       row_number() OVER (PARTITION BY qs.user_id, qa.word_id
                                           ORDER BY qa.answered_at) AS rn
                FROM {s}.quiz_answer qa
                JOIN {s}.quiz_question qq ON qq.id = qa.question_id
                JOIN {s}.quiz_session  qs ON qs.id = qq.session_id
                WHERE qa.guessing_floor IS NOT NULL
            )
            SELECT word_id, user_id, is_correct, guessing_floor
            FROM first_exposure WHERE rn = 1""" + (f" LIMIT {int(limit)}" if limit else ""))
        exposures = cur.fetchall()

        if not exposures:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"words": 0}

        word_ids = list({r[0] for r in exposures})
        cur.execute(f"""SELECT word_id, difficulty FROM {s}.word_difficulty
                        WHERE word_id = ANY(%s) AND difficulty IS NOT NULL""", (word_ids,))
        difficulty_by_word = dict(cur.fetchall())

    stored = 0
    skipped_no_baseline = 0
    with conn.cursor() as cur:
        if limit:
            # A composite (user_id, word_id) = ANY(%s) isn't a portable psycopg
            # parameter binding (would need an actual Postgres row-type array,
            # not a plain Python list of tuples) -- exposures is small whenever
            # limit is set anyway (that's the point of limit), so a per-pair
            # delete is simpler and just as correct.
            for word_id, user_id, *_ in exposures:
                cur.execute(f"""DELETE FROM {s}.word_personal_difficulty
                                WHERE user_id = %s AND word_id = %s""", (user_id, word_id))
        else:
            cur.execute(f"TRUNCATE {s}.word_personal_difficulty")

        for i, (word_id, user_id, is_correct, c_q) in enumerate(exposures, 1):
            base_difficulty = difficulty_by_word.get(word_id)
            if base_difficulty is None:   # no ex-ante score yet -- nothing to anchor a nudge to
                skipped_no_baseline += 1
                continue
            b0 = calib.difficulty_to_logit(base_difficulty, scale)
            b_new = calib.update_rating(b0, is_correct, c_q, eta)
            personal_difficulty = calib.logit_to_difficulty(b_new, scale)
            cur.execute(
                f"""INSERT INTO {s}.word_personal_difficulty
                        (user_id, word_id, item_rating, personal_difficulty, based_on_correct, calibrated_at)
                    VALUES (%s,%s,%s,%s,%s, now())
                    ON CONFLICT (user_id, word_id) DO UPDATE SET
                        item_rating=EXCLUDED.item_rating, personal_difficulty=EXCLUDED.personal_difficulty,
                        based_on_correct=EXCLUDED.based_on_correct, calibrated_at=now()""",
                (user_id, word_id, b_new, personal_difficulty, is_correct))
            stored += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return {"words": stored, "skipped_no_baseline": skipped_no_baseline}
