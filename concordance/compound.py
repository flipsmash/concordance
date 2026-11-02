"""Compound transparency: does a compound's meaning follow from its parts?

`difficulty` eases a derivative toward its root (unbuttoned -> button). A
compound of two common words (woodchopper, mudbrick) deserves the same ease,
but splitting at letter boundaries is a coin flip: of a 45-word sample of
"both halves are common words" splits, only about half were genuinely
transparent. The rest were coincidences (deflower is not def + lower,
kinchin is German Kindchen), wrong split points (eyeshot is not eyes + hot),
or real compounds whose parts mislead (ratline, dogwatch; downland's "down"
is Old English dun, a hill). Wiktionary's own compound etymologies fix the
split point but not the transparency -- ashcake (a johnnycake), killcrop (a
changeling) and titman (the runt of a litter) are all tagged compounds.

So each candidate gets one local-model verdict, cached per lemma in
compound_verdict and never re-asked:

  transparent   the everyday senses of the parts give the meaning
                -> blended halfway toward the rarer part's zipf
  partly        one part literal, the other figurative or specialised
                -> the flat transparency ease only
  opaque        the parts don't lead to the meaning -> no ease
  not_compound  none of the offered splits is how the word was formed

Words `difficulty` already eases toward a root (validity_score._morph_root,
wikt.affix_root) are never candidates: transparency is credited once.
Difficulty only -- nothing here touches the ingest frequency floor, where an
inflated zipf could drop a real word.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from wordfreq import zipf_frequency

from .config import Config
from .crossref import classification_gloss

VERDICTS = ("transparent", "partly", "opaque", "not_compound")

# A letter-boundary split is offered only when both halves are words an
# ordinary reader knows (the same bar the original 45-word sample used).
# Wiktionary-attested and hyphenated splits are offered regardless: the
# split itself is not in doubt there, only the meaning.
_MIN_PART_ZIPF = 3.5
_MIN_PART_LEN = 3

_SYSTEM = (
    "You judge English compound words. Imagine an ordinary adult reader who knows the "
    "everyday meanings of the parts but has never met the compound. Could they work out "
    "what it means?\n\n"
    "You get the word, its dictionary definition, its etymology when known, and one or "
    "more candidate splits. First pick the split that is how the word was actually formed, "
    "or none if no candidate is (a coincidental split: 'deflower' is not 'def + lower'; "
    "'kinchin' comes from German, not 'kin + chin').\n\n"
    "Then rate that split:\n"
    "transparent - the everyday senses of the parts give the meaning (woodchopper: a person "
    "who chops wood; mudbrick: a brick of dried mud; hogskin: leather from a hog's skin).\n"
    "partly - the reader would guess roughly the right idea, but one part is figurative or "
    "specialised, or the word means something narrower than the parts say (mainsheet: a "
    "sail's control rope; redcoat: a British soldier; bullneck: a thick-necked person).\n"
    "opaque - the reader would guess wrong or have no idea: the meaning is a metaphor or "
    "idiom the parts don't reveal, a name for a particular thing, or a part is used in a rare "
    "or obsolete sense (ashcake: a johnnycake; killcrop: a changeling; titman: the runt of a "
    "litter; blackleg: a strikebreaker; dogwatch: a short shift on a ship; downland: rolling hills, from Old "
    "English dun 'hill').\n\n"
    "Judge from the definition, not from how plausible the parts sound together.\n\n"
    'Answer with JSON only, e.g. {"split": "wood+chopper", "verdict": "transparent"} '
    'or {"split": null, "verdict": "not_compound"}.'
)


def candidate_splits(lemma: str, wikt_parts: list[str] | None = None) -> list[list[str]]:
    """The ways `lemma` could be a compound, best evidence first. Wiktionary's
    attested parts, when it has them, are the only split offered; a
    hyphenated word splits at its hyphens; otherwise every letter boundary
    where both halves are common words (eyeshot -> eye+shot AND eyes+hot --
    the model picks)."""
    w = lemma.strip().lower()
    if wikt_parts:
        return [list(wikt_parts)]
    if "-" in w:
        parts = w.split("-")
        return [parts] if len(parts) >= 2 and all(re.fullmatch(r"[a-z']+", p) for p in parts) else []
    if not re.fullmatch(r"[a-z]+", w):
        return []
    return [[w[:i], w[i:]] for i in range(_MIN_PART_LEN, len(w) - _MIN_PART_LEN + 1)
            if zipf_frequency(w[:i], "en") >= _MIN_PART_ZIPF
            and zipf_frequency(w[i:], "en") >= _MIN_PART_ZIPF]


def rarer_part_zipf(parts: list[str]) -> float:
    """The zipf a transparent compound blends toward: its weakest link."""
    return min(zipf_frequency(p, "en") for p in parts)


def _user_block(it: dict) -> str:
    lines = [f"word: {it['word']}",
             f"definition: {it.get('definition') or '(none)'}"]
    if it.get("etymology"):
        lines.append(f"etymology: {it['etymology']}")
    lines.append("candidate splits: " + "; ".join("+".join(s) for s in it["splits"]))
    return "\n".join(lines)


def parse_verdict(text: str, splits: list[list[str]]) -> tuple[list[str] | None, str] | None:
    """(chosen parts or None, verdict) from the model's answer, or None when
    it is unusable -- a split that wasn't offered, or an unknown verdict.
    An unusable answer is never stored, so the word is asked again next run."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        return None
    if verdict == "not_compound":
        return None, "not_compound"
    split = obj.get("split")
    if not split:
        return None
    chosen = [p.strip().lower() for p in str(split).split("+")]
    offered = {tuple(s) for s in splits}
    if tuple(chosen) not in offered:
        return None
    return chosen, verdict


class Judge:
    def __init__(self, cfg: Config | None = None, model_path: str | None = None):
        from llama_cpp import Llama
        cfg = cfg or Config()
        mp = model_path or cfg.model_path
        if not mp or not Path(mp).exists():
            raise RuntimeError(f"compound judge model not found: {mp!r}")
        self.model = Path(mp).name
        self.llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    def close(self) -> None:
        """Free the GPU memory now, not at garbage collection -- `maintain`
        loads the next model straight after (see classify.Classifier.close)."""
        self.llm.close()

    def judge(self, it: dict) -> tuple[list[str] | None, str] | None:
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": _user_block(it)}],
            temperature=0.0, max_tokens=40)
        return parse_verdict(out["choices"][0]["message"]["content"] or "", it["splits"])


def candidates(conn, schema: str, refresh: bool = False, lemmas: list[str] | None = None,
               wikt_schema: str = "wikt", vocab_schema: str = "vocab") -> list[dict]:
    """Active words that could be compounds and have no cached verdict (all
    of them with refresh), each with its splits, definition gloss and
    etymology. `lemmas` restricts to those words (a trial run)."""
    from .db import _safe_schema
    from .validity_score import _morph_root
    s, g, v = _safe_schema(schema), _safe_schema(wikt_schema), _safe_schema(vocab_schema)
    with conn.cursor() as cur:
        params: list = []
        where = "w.active"
        if lemmas is not None:
            where += " AND lower(w.lemma) = ANY(%s)"
            params.append([x.lower() for x in lemmas])
        if not refresh:
            where += f" AND NOT EXISTS (SELECT 1 FROM {s}.compound_verdict cv WHERE cv.lemma_lc = lower(w.lemma))"
        cur.execute(f"SELECT DISTINCT ON (lower(w.lemma)) lower(w.lemma), w.definition "
                    f"FROM {s}.word w WHERE {where} ORDER BY lower(w.lemma), w.id", params)
        rows = cur.fetchall()
        affix, wikt = set(), {}
        cur.execute("SELECT to_regclass(%s), to_regclass(%s)", (f"{g}.affix_root", f"{g}.compound"))
        has_affix, has_compound = cur.fetchone()
        terms = [r[0] for r in rows]
        if has_affix:
            cur.execute(f"SELECT term FROM {g}.affix_root WHERE term = ANY(%s)", (terms,))
            affix = {r[0] for r in cur.fetchall()}
        if has_compound:
            cur.execute(f"SELECT term, parts, etymology FROM {g}.compound WHERE term = ANY(%s)", (terms,))
            wikt = {t: (p, e) for t, p, e in cur.fetchall()}

    items = []
    for lemma, definition in rows:
        if lemma in affix or _morph_root(lemma):
            continue                      # already eased toward a root
        parts, ety = wikt.get(lemma, (None, None))
        splits = candidate_splits(lemma, parts)
        if splits:
            items.append({"word": lemma, "splits": splits, "etymology": ety or "",
                          "definition": (classification_gloss(definition) or definition or "")[:300]})

    # Etymology for the rest from the local Wiktionary, where it has one --
    # often exactly what vetoes a false split (sippet: "diminutive of sop").
    need = [it["word"] for it in items if not it["etymology"]]
    if need:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"{v}.wiktionary",))
            if cur.fetchone()[0] is not None:
                cur.execute(f"""SELECT DISTINCT ON (lower(term)) lower(term), etymology
                                FROM {v}.wiktionary
                                WHERE lower(term) = ANY(%s) AND coalesce(etymology, '') <> ''
                                ORDER BY lower(term), id""", (need,))
                ety = dict(cur.fetchall())
                for it in items:
                    if not it["etymology"] and it["word"] in ety:
                        it["etymology"] = ety[it["word"]][:300]
    return items


def judge_and_store(conn, schema: str, cfg: Config | None = None, limit: int = 0,
                    refresh: bool = False, lemmas: list[str] | None = None,
                    commit_every: int = 100, progress: bool = True) -> dict:
    """`concordance compounds`: judge every candidate without a cached verdict
    and store the answers. Commits in chunks, so an interrupted run keeps
    what it finished and simply resumes."""
    from .db import _safe_schema
    s = _safe_schema(schema)
    items = candidates(conn, schema, refresh=refresh, lemmas=lemmas)
    if limit:
        items = items[:limit]
    stats = {"candidates": len(items), "unparsed": 0, **{v: 0 for v in VERDICTS}}
    if not items:
        return stats
    judge = Judge(cfg)
    try:
        for start in range(0, len(items), max(1, commit_every)):
            chunk = items[start:start + commit_every]
            with conn.cursor() as cur:
                for it in chunk:
                    res = judge.judge(it)
                    if res is None:
                        stats["unparsed"] += 1
                        continue
                    parts, verdict = res
                    stats[verdict] += 1
                    cur.execute(
                        f"""INSERT INTO {s}.compound_verdict (lemma_lc, parts, verdict, model)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (lemma_lc) DO UPDATE SET parts = EXCLUDED.parts,
                                verdict = EXCLUDED.verdict, model = EXCLUDED.model, judged_at = now()""",
                        (it["word"], parts, verdict, judge.model))
            conn.commit()
            if progress:
                print(f"  ...{min(start + commit_every, len(items))}/{len(items)} compounds judged")
    finally:
        judge.close()
    return stats


def verdicts(conn, schema: str) -> dict[str, tuple[list[str] | None, str]]:
    """lemma_lc -> (parts, verdict) for every cached verdict ({} before the
    table exists)."""
    from .db import _safe_schema
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{s}.compound_verdict",))
        if cur.fetchone()[0] is None:
            return {}
        cur.execute(f"SELECT lemma_lc, parts, verdict FROM {s}.compound_verdict")
        return {l: (p, v) for l, p, v in cur.fetchall()}
