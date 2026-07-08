"""USAS multi-label classifier (§ taxonomy).

Assigns each word 1-3 USAS category codes from word + POS + definition + sentence.
The WordNet-Domains prior (wndomains.usas_prior) is injected as a *candidate hint*
the model prunes/confirms against the sentence — not a hard seed — which grounds
the model and kills multi-sense over-generation. The expressive/abstract words
(no prior) are carried by the model alone, so this reuses the judge's hard-won
reliability scaffolding: compact output, every-word-returned, retry-on-omission,
temperature 0, and — critically — every returned code is validated against the
assignable USAS set so hallucinated codes (G3.1, K5.3) are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import db, usas, wndomains
from .config import Config

# assignable code -> label, and a compact reference block for the prompt
_CATS = [c for c in usas.categories() if c["assignable"]]
_LABEL = {c["code"]: c["name"] for c in _CATS}
_ASSIGNABLE = set(_LABEL)
_REFERENCE = "\n".join(f"{c['code']} {c['name']}" for c in _CATS)

_SYSTEM = (
    "You tag English words with USAS semantic category codes. Use ONLY codes from "
    "this list:\n\n" + _REFERENCE + "\n\n"
    "For each word choose the FEWEST codes that capture its core meaning IN THE GIVEN "
    "SENSE — usually ONE, at most three, most specific first. Add a 2nd or 3rd code ONLY when "
    "it captures a genuinely distinct, well-supported aspect (e.g. a word that is both a "
    "physical object AND a food). Do NOT pad with loosely-related codes; if unsure about an "
    "extra code, leave it out. A word may legitimately be both a domain and an expressive term. "
    "A 'hint' lists domain codes suggested by a lexicon — keep the ones that fit the "
    "sentence, drop the rest, and ADD codes for meaning the hint misses (emotion, quality, "
    "manner, thought, social, etc.). "
    'Output ONLY a JSON array, one object per input word IN ORDER, no prose: '
    '[{"w":"<word>","c":["CODE",...]}]. Include EVERY input word exactly once.'
)


def _prompt_items(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        hint = sorted(wndomains.usas_prior(it["word"]))
        out.append({
            "word": it["word"],
            "pos": it.get("pos", ""),
            "def": (it.get("definition") or "")[:200],
            "sentence": (it.get("sentence") or "")[:200],
            "hint": hint,
        })
    return out


class Classifier:
    _MAX_PASSES = 3

    def __init__(self, cfg: Config | None = None, model_path: str | None = None):
        from llama_cpp import Llama
        cfg = cfg or Config()
        mp = model_path or cfg.model_path
        if not mp or not Path(mp).exists():
            raise RuntimeError(f"classifier model not found: {mp!r}")
        self.llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)
        self.batch = 15

    def classify(self, items: list[dict]) -> dict[str, list[str]]:
        """word(lower) -> list of validated USAS codes."""
        result: dict[str, list[str]] = {}
        for i in range(0, len(items), self.batch):
            self._classify_batch(items[i : i + self.batch], result)
        return result

    def _classify_batch(self, batch: list[dict], result: dict) -> None:
        pending = list(batch)
        for _ in range(self._MAX_PASSES):
            got = self._query(pending)
            for word, codes in got.items():
                result[word] = codes
            pending = [it for it in batch if it["word"].lower() not in result]
            if not pending:
                break
        for it in pending:                       # unresolved after retries: leave empty
            result.setdefault(it["word"].lower(), [])

    def _query(self, items: list[dict]) -> dict[str, list[str]]:
        payload = json.dumps(_prompt_items(items), ensure_ascii=False)
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": payload}],
            temperature=0.0, max_tokens=len(items) * 40 + 128)
        parsed = _parse(out["choices"][0]["message"]["content"])
        got = {}
        for obj in parsed:
            w = str(obj.get("w", "")).strip().lower()
            if not w:
                continue
            codes = [c for c in _validate(obj.get("c", []))]
            got[w] = codes
        return got


def _validate(codes) -> list[str]:
    """Keep only real assignable codes; repair a bad subcode to its nearest valid
    ancestor (G3.1 -> G3), drop anything unrecognisable."""
    out: list[str] = []
    for raw in codes if isinstance(codes, list) else []:
        c = str(raw).strip().upper().rstrip("+-")   # USAS uses +/- polarity; ignore for now
        # normalise case of the letter, keep dotted digits
        if not c:
            continue
        c = c[0].upper() + c[1:]
        if c in _ASSIGNABLE:
            out.append(c)
            continue
        while "." in c:                              # repair G3.1 -> G3 -> G
            c = c.rsplit(".", 1)[0]
            if c in _ASSIGNABLE:
                out.append(c)
                break
        else:
            if c[:1] in _ASSIGNABLE:
                out.append(c[:1])
    # dedupe, cap at 3, preserve order
    seen, capped = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); capped.append(c)
    return capped[:3]


def _parse(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    start = text.find("[")
    if start == -1:
        return []
    snippet = text[start:]
    for end in range(len(snippet), 0, -1):
        try:
            data = json.loads(snippet[:end])
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            continue
    return []


def classify_and_store(conn, schema: str, cfg: Config | None = None, limit: int = 0) -> dict:
    """Classify every word in {schema}.word and write tags to word_category.
    Idempotent for the LLM-sourced rows (cleared and rewritten each run)."""
    ssch = db._safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, part_of_speech, definition, sentence FROM {ssch}.word"
                    + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        cur.execute(f"SELECT code, id FROM {ssch}.category WHERE taxonomy='usas'")
        code_id = dict(cur.fetchall())

    items = [{"word": r[1], "pos": r[2] or "", "definition": r[3] or "",
              "sentence": r[4] or "", "_id": r[0]} for r in rows]
    tags = Classifier(cfg).classify(items)

    stats = {"words": len(items), "classified": 0, "assignments": 0}
    with conn.cursor() as cur:
        if limit:
            cur.execute(f"DELETE FROM {ssch}.word_category WHERE source IN ('llm','wnd+llm') "
                        "AND word_id = ANY(%s)", ([r[0] for r in rows],))
        else:
            cur.execute(f"DELETE FROM {ssch}.word_category WHERE source IN ('llm','wnd+llm')")
        for it in items:
            codes = tags.get(it["word"].lower(), [])
            if not codes:
                continue
            stats["classified"] += 1
            prior_fields = {c[0] for c in wndomains.usas_prior(it["word"])}
            for rank, code in enumerate(codes):
                cid = code_id.get(code)
                if cid is None:
                    continue
                src = "wnd+llm" if code[0] in prior_fields else "llm"
                conf = round(max(0.4, 1.0 - 0.25 * rank), 2)
                cur.execute(
                    f"""INSERT INTO {ssch}.word_category (word_id, category_id, confidence, source, is_primary)
                        VALUES (%s,%s,%s,%s,%s)
                        ON CONFLICT (word_id, category_id) DO UPDATE SET
                            confidence=EXCLUDED.confidence, source=EXCLUDED.source, is_primary=EXCLUDED.is_primary""",
                    (it["_id"], cid, conf, src, rank == 0))
                stats["assignments"] += 1
    conn.commit()
    return stats
