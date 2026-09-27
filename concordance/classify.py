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
import re
from pathlib import Path

from . import db, usas, wndomains
from .crossref import bare_pointer_target, classification_gloss
from .config import Config

# assignable code -> label, and a compact reference block for the prompt
_CATS = [c for c in usas.categories() if c["assignable"]]
_LABEL = {c["code"]: c["name"] for c in _CATS}
_ASSIGNABLE = set(_LABEL)
_REFERENCE = "\n".join(f"{c['code']} {c['name']}" for c in _CATS)

# Two steps per word (docs/decisions/0006): the broad field first, from the
# 21 top-level fields each listed with what it covers; then the specific
# codes, choosing only among that field's own. Picking straight from all
# ~230 codes, the model understood the word but grabbed the wrong line
# (a snake -> "Objects generally"): on a hand-labelled test set this took
# exact-code accuracy from 63 to 78 of 143, and is fully repeatable.
_FIELDS = [c for c in usas.categories() if c["level"] == 0]
_FIELD_CODES = {c["code"] for c in _FIELDS}
_FIELD_SYSTEM = (
    "You sort English words into broad semantic fields. Fields:\n"
    + "\n".join(f"{f['code']} {f['name']} -- covers: "
                + "; ".join(n for code, n in _LABEL.items() if code[0] == f["code"] and code != f["code"])
                for f in _FIELDS)
    + "\n\nClassify ONLY the sense given by the definition (or, if there is no definition, the "
      "example sentence). Answer with the field letter(s) that fit, best first, at most two, comma-separated, nothing else."
)


def _code_system(letters: list[str]) -> str:
    sub = "\n".join(f"{c} {n}" for c, n in _LABEL.items() if c[0] in letters)
    return ("Choose USAS codes for the word from THIS list only:\n" + sub + "\n\n"
            "Pick the FEWEST codes that capture the given sense, most specific and best first, "
            "at most three. Answer with the codes, comma-separated, nothing else.")


def _word_block(p: dict) -> str:
    """One _prompt_items entry as the plain-text block both steps see."""
    lines = [f"word: {p['word']}", f"part of speech: {p.get('pos') or 'unknown'}",
             f"definition: {p.get('def') or '(none)'}"]
    if p.get("sentence"):
        lines.append(f"example sentence from a book: {p['sentence']}")
    if p.get("hint"):
        lines.append(f"lexicon hint (may be wrong): {', '.join(p['hint'])}")
    return "\n".join(lines)


def _prompt_items(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        # Never the cross-referenced spelling: "Variant spelling of faggot —
        # A bundle of sticks" must reach the model as just its gloss, and the
        # hint must come from THAT sense (see validity_score.
        # classification_gloss / wndomains.usas_prior_for_sense).
        # A bare pointer ("Archaic form of vampirism.") has no gloss of its
        # own: use the target's FIRST dictionary sense (never the target word,
        # never its other senses) -- see bare_pointer_senses.
        gloss = classification_gloss(it.get("definition")) or it.get("pointer_sense", "")
        hint = sorted(wndomains.usas_prior_for_sense(it["word"], gloss))
        out.append({
            "word": it["word"],
            "pos": it.get("pos", ""),
            "def": gloss[:300],
            # The book sentence only when there is no definition to go on:
            # measured against a hand-labelled set it added nothing beside a
            # definition, and it is often misleading (glossary/index lines).
            "sentence": "" if gloss else (it.get("sentence") or "")[:200],
            "hint": hint,
        })
    return out


class Classifier:
    def __init__(self, cfg: Config | None = None, model_path: str | None = None):
        from llama_cpp import Llama
        cfg = cfg or Config()
        mp = model_path or cfg.model_path
        if not mp or not Path(mp).exists():
            raise RuntimeError(f"classifier model not found: {mp!r}")
        self.llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    def close(self) -> None:
        """Deterministically frees the model's GPU memory -- see
        classify_and_store's call site for why this can't just be left to
        whenever Python garbage-collects the object."""
        self.llm.close()

    def classify(self, items: list[dict]) -> dict[str, list[str]]:
        """word(lower) -> list of validated USAS codes ([] if nothing usable)."""
        result: dict[str, list[str]] = {}
        for it in items:
            result[it["word"].lower()] = self._classify_one(it)
        return result

    def _ask(self, system: str, user: str, max_tokens: int) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=max_tokens)
        return out["choices"][0]["message"]["content"] or ""

    def _classify_one(self, it: dict) -> list[str]:
        block = _word_block(_prompt_items([it])[0])
        raw = self._ask(_FIELD_SYSTEM, block, 12)
        # "L, B" or "L LIFE & LIVING THINGS": the first letter of each
        # comma-separated answer, never of every word in it
        letters = [x.strip().upper()[:1] for x in re.split(r"[,;]", raw) if x.strip()]
        letters = list(dict.fromkeys(l for l in letters if l in _FIELD_CODES))[:2]
        if not letters:
            return []
        raw = self._ask(_code_system(letters), block, 30)
        return _validate([c.strip().split(" ")[0] for c in re.split(r"[,;]", raw) if c.strip()])[:3]


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


def classify_and_store(conn, schema: str, cfg: Config | None = None, limit: int = 0,
                       only_missing: bool = False,
                       commit_every: int = 200, word_ids: list[int] | None = None) -> dict:
    """Classify every word in {schema}.word and write tags to word_category.
    Idempotent for the LLM-sourced rows (cleared and rewritten each run).

    Commits every `commit_every` words, so a crash partway through an
    hours-long run keeps what it finished and a restart with only_missing
    resumes near where it stopped (chunking doesn't change any word's result).
    Each chunk re-checks which of its word ids still exist before use, since
    a word can be deleted mid-run. History:
    docs/decisions/0002-classify-chunked-commits.md"""
    ssch = db._safe_schema(schema)
    with conn.cursor() as cur:
        where = (" WHERE NOT EXISTS (SELECT 1 FROM " + ssch + ".word_category wc WHERE wc.word_id=word.id)"
                 if only_missing else "")
        params: tuple = ()
        if word_ids is not None:
            where = " WHERE word.id = ANY(%s)"     # a targeted re-classify: only these words
            params = (list(word_ids),)
        cur.execute(f"SELECT id, lemma, part_of_speech, definition, sentence FROM {ssch}.word word"
                    + where + (f" LIMIT {int(limit)}" if limit else ""), params)
        rows = cur.fetchall()
        cur.execute(f"SELECT code, id FROM {ssch}.category WHERE taxonomy='usas'")
        code_id = dict(cur.fetchall())

    items = [{"word": r[1], "pos": r[2] or "", "definition": r[3] or "",
              "sentence": r[4] or "", "_id": r[0]} for r in rows]
    bare_pointer_senses(conn, items)
    clf = Classifier(cfg)  # loaded once, reused for every chunk below

    stats = {"words": len(items), "classified": 0, "assignments": 0, "vanished": 0}

    # One-time clear, before any chunk's writes -- only_missing's WHERE
    # already guarantees these words have no rows to clear; a --limit run
    # clears just its own row ids up front so the chunk loop below only
    # ever needs to INSERT, never worry about stale rows from a prior run
    # over the same ids.
    with conn.cursor() as cur:
        if only_missing or word_ids is not None:
            pass      # targeted re-classify replaces per word below, only when new codes come back
        elif limit:
            cur.execute(f"DELETE FROM {ssch}.word_category WHERE source IN ('llm','wnd+llm') "
                        "AND word_id = ANY(%s)", ([r[0] for r in rows],))
        else:
            cur.execute(f"DELETE FROM {ssch}.word_category WHERE source IN ('llm','wnd+llm')")
    conn.commit()

    for start in range(0, len(items), max(1, commit_every)):
        chunk = items[start:start + commit_every]

        # Re-verify against the live table rather than trusting the
        # snapshot -- see this function's own docstring for the crash this
        # is fixing. Filtered BEFORE classify() so a vanished word doesn't
        # also cost an LLM call for nothing.
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {ssch}.word WHERE id = ANY(%s)", ([it["_id"] for it in chunk],))
            still_present = {r[0] for r in cur.fetchall()}
        vanished = [it for it in chunk if it["_id"] not in still_present]
        if vanished:
            stats["vanished"] += len(vanished)
            chunk = [it for it in chunk if it["_id"] in still_present]
        if not chunk:
            continue

        tags = clf.classify(chunk)
        with conn.cursor() as cur:
            for it in chunk:
                codes = tags.get(it["word"].lower(), [])
                if not codes:
                    continue      # keep whatever the word already has rather than leave it untagged
                if word_ids is not None:
                    cur.execute(f"DELETE FROM {ssch}.word_category WHERE word_id = %s "
                                "AND source IN ('llm','wnd+llm')", (it["_id"],))
                stats["classified"] += 1
                prior_fields = {c[0] for c in wndomains.usas_prior_for_sense(
                    it["word"], classification_gloss(it["definition"]))}
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
        print(f"  ...{min(start + commit_every, len(items))}/{len(items)} words classified "
              f"({stats['classified']} tagged, {stats['vanished']} vanished mid-run)")
    # Deterministic release, not left to implicit GC timing -- `maintain`
    # chains straight into the next GPU-loading step (normalize-pos is
    # cheap, but quizdef further down loads its own fresh multi-GB model)
    # immediately after this returns. See fill_definitions' matching comment
    # for the live crash this pattern is fixing.
    clf.close()
    return stats


def bare_pointer_senses(conn, items: list[dict], wikt_schema: str = "vocab") -> None:
    """For each item whose definition is a gloss-less pointer, set
    item["pointer_sense"] to the first sense of the pointed-to word in the
    local Wiktionary (first entry in dump order, first ';'-separated sense).
    Wiktionary lists a word's primary sense first, so a bare "form of faggot"
    gets "a bundle of sticks", not the slur. That sense is itself run through
    classification_gloss, so a chain of pointers yields nothing rather than
    another spelling. Mutates `items`."""
    targets = {it["_id"]: t for it in items if (t := bare_pointer_target(it.get("definition")))}
    if not targets:
        return
    with conn.cursor() as cur:
        cur.execute(f"""SELECT DISTINCT ON (lower(term)) lower(term), definition
                        FROM {db._safe_schema(wikt_schema)}.wiktionary
                        WHERE lower(term) = ANY(%s) ORDER BY lower(term), id""",
                    (sorted(set(targets.values())),))
        first = {term: classification_gloss((d or "").split(";")[0]) for term, d in cur.fetchall()}
    for it in items:
        sense = first.get(targets.get(it["_id"], ""), "")
        if sense:
            it["pointer_sense"] = sense


def sense_affected_word_ids(conn, schema: str) -> list[int]:
    """Active words whose classifier input changes under the
    classification_gloss / usas_prior_for_sense rules: a definition that
    cross-references another spelling, or a WordNet-Domains hint whose
    senses disagree (so the old merged hint may have pointed at the wrong
    sense). The set a targeted re-classify should cover."""
    ssch = db._safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, coalesce(definition, '') FROM {ssch}.word WHERE active")
        rows = cur.fetchall()
    senses = wndomains.load_senses()
    out = []
    for wid, lemma, definition in rows:
        s = senses.get(lemma.strip().lower())
        polysemous = bool(s) and len({doms for doms, _ in s}) > 1 and any(doms for doms, _ in s)
        if polysemous or classification_gloss(definition) != definition.strip():
            out.append(wid)
    return out
