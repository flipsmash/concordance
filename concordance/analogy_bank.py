"""Analogy item bank (`concordance build-analogy-bank`, inside `maintain`).

Live assembly fails for roughly a third of words (no usable edge/anchor/
options), and nothing checks that exactly one option fits. The bank builds
items ahead of time, while the GPU is free, and has the local model check
each one twice:

  solve  -- given A:B, C (with its definition) and the options, pick the
            answer and name any other option that would also fit. Kept only
            if it picks D and nothing else fits.
  blind  -- the same options and C but NO example pair (and no definition,
            which often contains the answer word itself): which
            option is the hidden answer? Asked over up to three shuffles of
            the option order (a single pass is right by chance a quarter of
            the time); D picked in a majority = 'associative', answerable
            without the relation, served only when a word has nothing better.

The quiz serves banked items first (relation families balanced, recent
anchors avoided) and falls back to live assembly for words not banked yet.
An item is only served while its tested and anchor edges still pass the
same checks live assembly applies (verified; anchor not rejected or scored
unclear), so a later review quietly retires it; a word is rescanned when one
of its edges is verified after its last scan.
"""

from __future__ import annotations

import json
import random

from . import analogy_select as asel
from . import distractors as dx

_BLIND_SHUFFLES = 3
_CHECK_BATCH = 8


def assemble(conn, schema: str, word_id: int, exclude_ids: set[int], option_count: int = 4,
             **select_kwargs) -> dict | None:
    """One analogy item for `word_id` (tested edge, anchor, options with
    the answer included and each option's source), or None."""
    a = asel.select_analogy_edge(conn, schema, word_id, exclude_ids, option_count=option_count, **select_kwargs)
    if a is None:
        return None
    result = dx.select_analogy_distractors(
        conn, schema, a.style, a.d_pos, a.a_lemma, a.b_lemma, a.target_word_id, a.d_term_id,
        a.exclusion_lemmas, a.trap_lemmas, option_count - 1, exclude_word_ids=exclude_ids,
        d_word_id=a.d_word_id, anchor_synonyms=a.anchor_synonyms, d_lemma=a.d_lemma,
        prompt_lemmas=[a.a_lemma, a.b_lemma, a.c_lemma], contrast_lemmas=a.contrast_lemmas,
    )
    if not result.candidates:
        return None
    options = [{"label": c["lemma"], "word_id": c.get("id"), "strategy": c.get("strategy")}
               for c in result.candidates[:option_count - 1]]
    options.append({"label": a.d_lemma, "word_id": a.d_word_id, "strategy": "answer"})
    return {"target_word_id": a.target_word_id, "tested_edge_id": a.tested_edge_id,
            "anchor_edge_id": a.anchor_edge_id, "relation_type": a.relation_type,
            "relation_family": a.relation_family, "style": a.style,
            "a_lemma": a.a_lemma, "b_lemma": a.b_lemma, "c_lemma": a.c_lemma, "d_lemma": a.d_lemma,
            "d_word_id": a.d_word_id, "options": options, "degraded": result.degraded}


def payload_fields(item: dict) -> dict:
    """The analogy quiz payload fields common to banked and live items:
    shuffled keyed options, the answer key, and the analysis fields."""
    opts = list(item["options"])
    random.shuffle(opts)
    options = [{"key": f"opt{i}", "label": o["label"], "word_id": o.get("word_id"), "strategy": o.get("strategy")}
               for i, o in enumerate(opts)]
    correct = next(o["key"] for o in options if o["strategy"] == "answer")
    return {
        "prompt": f"{item['a_lemma'].upper()} is to {item['b_lemma'].upper()} as {item['c_lemma'].upper()} is to ___",
        "options": options,
        "correct_option_key": correct,
        "style": item["style"],
        "relation_type": item["relation_type"],
        "relation_family": item["relation_family"],
        "tested_edge_id": item["tested_edge_id"],
        "anchor_edge_id": item["anchor_edge_id"],
        "a_lemma": item["a_lemma"], "b_lemma": item["b_lemma"], "c_lemma": item["c_lemma"],
        "d_lemma": item["d_lemma"],
        "explanation": asel.explain(item["relation_type"], item["a_lemma"], item["b_lemma"],
                                    item["c_lemma"], item["d_lemma"]),
    }


# --- model checks -------------------------------------------------------------

_SOLVE_SYSTEM = (
    "You are checking vocabulary analogy questions of the form 'A is to B as C is to ___'. "
    "For each item: work out exactly how a relates to b, then choose the ONE option that relates to c "
    "the same way (c's definition is given). Then list every OTHER option that would also complete the "
    "analogy acceptably -- be strict: an option fits only if c relates to it the same way a relates to b. "
    'Output ONLY a JSON array, one object per input, in order: [{"i": <int>, "answer": <option number>, '
    '"also": [<option numbers>]}].'
)

_BLIND_SYSTEM = (
    "Each item gives a word and numbered options. One option is the hidden answer to a "
    "word-relationship question about this word; you are not told which relationship. Pick the option "
    'most likely to be the answer. Output ONLY a JSON array, in order: [{"i": <int>, "answer": <option number>}].'
)


def _ask(llm, system: str, items: list[dict]) -> dict[int, dict]:
    from .analogies import _parse_verify_response

    for _ in range(3):
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                temperature=0.0, max_tokens=len(items) * 30 + 64)
            verdicts = _parse_verify_response(out["choices"][0]["message"]["content"]) or []
            return {v["i"]: v for v in verdicts if isinstance(v, dict) and isinstance(v.get("i"), int)}
        except Exception:  # noqa: BLE001 -- a flaky call shouldn't end the run
            continue
    return {}


def _as_int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def check_items(llm, items: list[dict], c_defs: dict[int, str]) -> None:
    """Set status / check_note / blind_hits on each item in place."""
    for i in range(0, len(items), _CHECK_BATCH):
        batch = items[i:i + _CHECK_BATCH]
        orders = []
        for it in batch:
            order = list(range(len(it["options"])))
            random.shuffle(order)
            orders.append(order)
        solve = _ask(llm, _SOLVE_SYSTEM, [
            {"i": j, "a": it["a_lemma"], "b": it["b_lemma"], "c": it["c_lemma"],
             "c_definition": (c_defs.get(it["target_word_id"]) or "")[:300],
             "options": {str(n + 1): it["options"][k]["label"] for n, k in enumerate(order)}}
            for j, (it, order) in enumerate(zip(batch, orders))])
        for j, (it, order) in enumerate(zip(batch, orders)):
            v = solve.get(j)
            answer_n = order.index(len(it["options"]) - 1) + 1  # answer is stored last
            if v is None:
                it["status"], it["check_note"] = "failed", "no verdict"
                continue
            also = {_as_int(x) for x in (v.get("also") or [])} - {answer_n, None}
            if _as_int(v.get("answer")) != answer_n:
                pick = _as_int(v.get("answer"))
                label = it["options"][order[pick - 1]]["label"] if pick and 1 <= pick <= len(order) else "?"
                it["status"], it["check_note"] = "failed", f"model chose {label}"
            elif also:
                labels = [it["options"][order[n - 1]]["label"] for n in also if 1 <= n <= len(order)]
                it["status"], it["check_note"] = "failed", "also fits: " + ", ".join(labels)
            else:
                it["status"], it["check_note"] = "ok", None
        live = [it for it in batch if it["status"] == "ok"]
        for it in live:
            it["blind_hits"] = 0
        for shuffle_n in range(_BLIND_SHUFFLES):
            # Stop asking once the majority is settled either way.
            need = [it for it in live if it["blind_hits"] < 2 and it["blind_hits"] + (_BLIND_SHUFFLES - shuffle_n) >= 2]
            if not need:
                break
            orders = []
            for it in need:
                order = list(range(len(it["options"])))
                random.shuffle(order)
                orders.append(order)
            blind = _ask(llm, _BLIND_SYSTEM, [
                {"i": j, "word": it["c_lemma"],
                 "options": {str(n + 1): it["options"][k]["label"] for n, k in enumerate(order)}}
                for j, (it, order) in enumerate(zip(need, orders))])
            for j, (it, order) in enumerate(zip(need, orders)):
                pick = _as_int((blind.get(j) or {}).get("answer"))
                if pick and 1 <= pick <= len(order) and order[pick - 1] == len(it["options"]) - 1:
                    it["blind_hits"] += 1
        for it in live:
            if it["blind_hits"] >= 2:
                it["status"], it["check_note"] = "associative", "answerable without the example pair"


# --- build --------------------------------------------------------------------

def _words_to_scan(cur, schema: str, rebuild: bool, limit: int) -> list[tuple[int, str]]:
    cur.execute(
        f"""SELECT w.id, w.quiz_definition FROM {schema}.word w
            JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
            JOIN {schema}.wn_relation_term t ON t.word_id = w.id
            LEFT JOIN {schema}.analogy_bank_scan s ON s.word_id = w.id
            WHERE w.active AND wd.quizzable AND w.quiz_definition IS NOT NULL
              AND EXISTS (SELECT 1 FROM {schema}.word_relation_edge e
                          WHERE e.term_a_id = t.id AND e.verification_status = 'verified'
                            AND (%s OR s.word_id IS NULL OR e.verified_at > s.scanned_at))
            GROUP BY w.id, w.quiz_definition
            ORDER BY random()""" + (" LIMIT %s" if limit else ""),
        (rebuild, limit) if limit else (rebuild,))
    return cur.fetchall()


_ROUNDS = 3  # first build + up to two repairs


def _assemble_for_word(conn, schema: str, wid: int, want: int, m: dict) -> list[dict]:
    """Up to `want` new items for one word: an untried tested edge first,
    then a fresh anchor for one already used; never a tested edge whose
    answer the model rejected; never an option the
    model flagged (m['excl'])."""
    got: list[dict] = []
    for attempt in range(want * 2 + 1):
        fresh_edges = attempt < want
        skip = (m["tested"] if fresh_edges else set()) | m["bad_d"]
        it = assemble(conn, schema, wid, {wid}, skip_tested_edge_ids=skip, avoid_anchor_ids=m["anchors"],
                      extra_exclusions=m["excl"])
        if it is None:
            if not fresh_edges:
                break
            continue
        m["tested"].add(it["tested_edge_id"])
        m["anchors"].add(it["anchor_edge_id"])
        got.append(it)
        if len(got) >= want:
            break
    return got


def build_bank(conn, schema: str, cfg, per_word: int = 2, limit: int = 0, rebuild: bool = False,
               llm=None, log=print) -> dict:
    """Assemble up to `per_word` items per word (different tested edges
    first, then different anchors), check them with the local model, store
    them. Commits per batch of words; resumable."""
    with conn.cursor() as cur:
        words = _words_to_scan(cur, schema, rebuild, limit)
    log(f"[build-analogy-bank] {len(words)} words to scan")
    stats = {"words": 0, "built": 0, "ok": 0, "associative": 0, "failed": 0, "unbuildable": 0, "words_served": 0}
    if not words:
        return stats
    own_llm = llm is None
    if own_llm:
        from .analogies import _load_llm

        llm = _load_llm(cfg)
    try:
        step = 40
        for i in range(0, len(words), step):
            chunk = words[i:i + step]
            c_defs = {wid: d for wid, d in chunk}
            items_by_word: dict[int, list[dict]] = {wid: [] for wid, _ in chunk}
            memo = {wid: {"tested": set(), "anchors": set(), "bad_d": set(), "excl": set()} for wid, _ in chunk}
            todo = [wid for wid, _ in chunk]
            for round_n in range(_ROUNDS):
                fresh: list[dict] = []
                for wid in todo:
                    m = memo[wid]
                    want = per_word if round_n == 0 else 1
                    got = _assemble_for_word(conn, schema, wid, want, m)
                    items_by_word[wid] += got
                    fresh += got
                conn.rollback()  # assembly only reads
                if not fresh:
                    break
                check_items(llm, fresh, c_defs)
                for it in fresh:
                    m = memo[it["target_word_id"]]
                    note = it.get("check_note") or ""
                    if note.startswith("also fits: "):
                        m["excl"] |= set(note[len("also fits: "):].split(", "))
                    elif note.startswith("model chose "):
                        # Either D is a poor answer or that option is a rival one: avoid both.
                        m["excl"].add(note[len("model chose "):])
                        m["bad_d"].add(it["tested_edge_id"])
                # Repair rounds only for words with nothing servable yet.
                todo = [wid for wid, got in items_by_word.items()
                        if got and not any(it["status"] in ("ok", "associative") for it in got)]
            with conn.cursor() as cur:
                for wid, got in items_by_word.items():
                    cur.execute(f"DELETE FROM {schema}.analogy_item WHERE target_word_id = %s", (wid,))
                    for it in got:
                        cur.execute(
                            f"""INSERT INTO {schema}.analogy_item
                                    (target_word_id, tested_edge_id, anchor_edge_id, relation_type, relation_family,
                                     style, a_lemma, b_lemma, c_lemma, d_lemma, d_word_id, options, status,
                                     check_note, blind_hits)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                ON CONFLICT (tested_edge_id, anchor_edge_id) DO NOTHING""",
                            (wid, it["tested_edge_id"], it["anchor_edge_id"], it["relation_type"],
                             it["relation_family"], it["style"], it["a_lemma"], it["b_lemma"], it["c_lemma"],
                             it["d_lemma"], it["d_word_id"], json.dumps(it["options"]), it["status"],
                             it.get("check_note"), it.get("blind_hits")))
                        stats[it["status"]] += 1
                    n_ok = sum(1 for it in got if it["status"] == "ok")
                    cur.execute(
                        f"""INSERT INTO {schema}.analogy_bank_scan (word_id, scanned_at, built, ok)
                            VALUES (%s, now(), %s, %s)
                            ON CONFLICT (word_id) DO UPDATE SET scanned_at = now(), built = EXCLUDED.built,
                                                                ok = EXCLUDED.ok""",
                        (wid, len(got), n_ok))
                    stats["built"] += len(got)
                    stats["unbuildable"] += not got
                    stats["words_served"] += any(it["status"] in ("ok", "associative") for it in got)
            conn.commit()
            stats["words"] += len(chunk)
            log(f"[build-analogy-bank]   ...{stats['words']}/{len(words)} words: {stats['ok']} ok, "
                f"{stats['associative']} associative, {stats['failed']} failed, {stats['unbuildable']} unbuildable; "
                f"{stats['words_served']} words servable")
    finally:
        if own_llm:
            llm.close()
    return stats
