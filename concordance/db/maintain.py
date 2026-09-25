"""Progress reporting for `concordance maintain` (§ maintain-status)."""

from __future__ import annotations

from .core import DEFAULT_SCHEMA, _safe_schema


# --- maintain progress (§ maintain-status) -------------------------------------
#
# Read-only: safe to run at any time, including against a schema another
# `maintain`/`backfill-analogies` process is actively writing to (it never
# takes a lock beyond a plain SELECT). Exists because `maintain` chains ~16
# steps and, for a run kicked off hours ago in a terminal nobody's watching
# anymore, its own step-by-step console output is long gone -- this
# reconstructs an equivalent picture from DB state instead.
#
# Each step's "done" fraction is a proxy, not a real percentage the step
# itself reports -- most steps loop over "words not yet touched" and this
# just counts touched-vs-total the same way. A few are legitimately capped
# below 100% forever (e.g. fill-definitions: some words are permanently
# undefined and cast out; quizdef/quizzable are similar) -- that's not a
# stalled step, just a step whose target was never "every active word."
MAINTAIN_STEPS = [
    "fill-definitions", "classify", "normalize-pos", "ngram", "archaic",
    "difficulty", "quizdef", "quizzable", "calibrate-difficulty",
    "wordnik-pron", "ipa", "embed", "backfill-analogies",
]


def maintain_status(conn, schema: str = DEFAULT_SCHEMA) -> list[dict]:
    """One dict per MAINTAIN_STEPS entry, in maintain's own run order:
    {name, done, total, note}. `done`/`total` are None for a step with no
    meaningful fraction (normalize-pos, quizzable, calibrate-difficulty --
    each recomputes fully in one fast pass rather than incrementally
    skipping already-touched rows, so there's nothing to be partway through
    between runs). book-similarity/book-clustering/author-similarity/
    author-clustering are NOT listed here -- no longer part of `maintain`'s
    own chain (see maintain's own docstring for why), track their own
    freshness by re-running those commands directly."""
    s = _safe_schema(schema)
    steps: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active")
        active_total = cur.fetchone()[0]

        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND coalesce(definition,'') <> ''")
        steps.append({"name": "fill-definitions", "done": cur.fetchone()[0], "total": active_total,
                      "note": "some words are permanently undefined (cast out) -- won't reach 100%"})

        cur.execute(f"""SELECT count(DISTINCT wc.word_id) FROM {s}.word_category wc
                        JOIN {s}.word w ON w.id = wc.word_id WHERE w.active""")
        steps.append({"name": "classify", "done": cur.fetchone()[0], "total": active_total,
                      "note": "a word can legitimately classify to zero categories -- "
                              "won't reach 100% either way"})

        steps.append({"name": "normalize-pos", "done": None, "total": None,
                      "note": "recomputes fully each run -- fast, no meaningful fraction"})

        cur.execute(f"SELECT count(*) FROM {s}.word_ngram")
        steps.append({"name": "ngram", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"SELECT count(*) FROM {s}.word_difficulty WHERE archaic_confidence IS NOT NULL")
        steps.append({"name": "archaic", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"SELECT count(*) FROM {s}.word_difficulty WHERE difficulty IS NOT NULL")
        steps.append({"name": "difficulty", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND quiz_definition IS NOT NULL")
        steps.append({"name": "quizdef", "done": cur.fetchone()[0], "total": active_total,
                      "note": "only words with a definition to build one from are in scope"})

        steps.append({"name": "quizzable", "done": None, "total": None,
                      "note": "recomputes fully each run -- fast, no meaningful fraction"})

        steps.append({"name": "calibrate-difficulty", "done": None, "total": None,
                      "note": "driven by real quiz answers, not a fixed word-count target"})

        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND wordnik_checked_at IS NOT NULL")
        steps.append({"name": "wordnik-pron", "done": cur.fetchone()[0], "total": active_total,
                      "note": "rate-limited (~seconds/word on the free tier) -- typically the slowest step"})

        # ipa IS NOT NULL alone overcounts: a prior compute_ipa pass clears an
        # unresolvable word to '' (empty string, not NULL) rather than leaving
        # a stale bad value -- see compute_ipa's cleared_no_replacement branch.
        # Matching the same "still needs work" test the pipeline itself uses
        # (fetch_wordnik_pronunciations' and compute_ipa's own WHERE clauses).
        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND ipa IS NOT NULL AND ipa <> ''")
        steps.append({"name": "ipa", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"""SELECT count(*) FROM {s}.word_embedding e
                        JOIN {s}.word w ON w.id = e.word_id
                        WHERE w.active AND e.definition_vector IS NOT NULL""")
        steps.append({"name": "embed", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"""SELECT count(*) FROM {s}.word w JOIN {s}.word_difficulty wd ON wd.word_id = w.id
                        WHERE w.active AND wd.quizzable = true AND w.part_of_speech IS NOT NULL
                          AND w.definition IS NOT NULL""")
        analogy_total = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM {s}.wn_relation_scan")
        steps.append({"name": "backfill-analogies", "done": cur.fetchone()[0], "total": analogy_total,
                      "note": "eligible = active, quizzable, WordNet-mappable POS"})

    return steps
