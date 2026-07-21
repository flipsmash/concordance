"""Stage 8 — the interestingness judge (§03.8).

Given the words that survived the validity gate, decide which are worth
learning. Two implementations behind one interface:

  * StubJudge  — no model; passes survivors through so the pipeline runs
                 end-to-end. Used until a .gguf is configured.
  * LlamaJudge — local llama.cpp (Qwen2.5 / Llama-3.1) via llama-cpp-python.
                 Judges in batches, returns keep/skip + a one-line reason, and
                 carries the "reject names" backstop from §04.

Selecting a model is a config choice, never a paid API call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .config import Config
from .model import Candidate, RejectReason, Verdict

# The rubric handed to the model — this is where "interesting = any word I don't
# personally know or want to flag" is encoded, along with the proper-noun backstop.
#
# Lessons paid for in blood on the Shadow of the Torturer run: a weak local model
# (a) OMITS most words when asked for a verbose per-word object, and every omitted
# word then defaults to keep — so the output format is deliberately tiny (one
# {"w","k"} per word, no free-text reason) and completeness is enforced by re-query
# (see LlamaJudge._judge_batch); and (b) rationalizes common words as "literary"
# unless the bar is nailed down with concrete examples spanning the frequency band.
RUBRIC = (
    "You build a study list of RARE vocabulary for an advanced, well-read English "
    "reader. For each word decide keep=true ONLY if a typical native-speaking college "
    "graduate would likely NOT know its meaning — genuinely obscure, archaic, "
    "technical, or literary-rare words. Set keep=false for any word that is part of "
    "ordinary educated vocabulary, however vivid or literary, and for proper nouns, "
    "names, places, and misspellings. "
    "REJECT (keep=false) words like: stink, whisper, growl, murmur, radiant, avalanche, "
    "tremble, stumble, nostril, eyelid, footstep, gloomy, clumsy, shriek, tendril, "
    "curtsy, contort, scimitar. "
    "KEEP (keep=true) words like: refectory, cangue, armiger, bartizan, effluvium, "
    "destrier, fuligin, abacination. "
    "A per-word frequency hint (common / uncommon / rare) is advisory only — it is NOT "
    "a hard cut, but almost always reject freq=common. Be aggressive: on a typical "
    "page most words are keep=false. When genuinely torn, reject."
)


def _freq_band(zipf: float) -> str:
    """Turn a wordfreq Zipf value into a coarse hint for the judge. Advisory only
    — it steadies the model's rarity sense; it is NOT a hard frequency cut."""
    if zipf >= 3.0:
        return "common"
    if zipf >= 2.0:
        return "uncommon"
    return "rare"


class Judge(Protocol):
    def judge(self, candidates: list[Candidate]) -> None: ...


def get_judge(cfg: Config) -> Judge:
    if cfg.model_path and Path(cfg.model_path).exists():
        try:
            return LlamaJudge(cfg)
        except Exception as exc:  # noqa: BLE001 — fall back rather than crash a run
            print(f"[judge] could not load model ({exc}); using stub.")
    return StubJudge()


def _survivors(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.verdict in (Verdict.KEEP, Verdict.UNSURE)]


class StubJudge:
    """No model available: keep every survivor, rarest first. Lets the walking
    skeleton produce real output without an LLM."""

    def judge(self, candidates: list[Candidate]) -> None:
        for c in _survivors(candidates):
            if not c.interesting_reason:
                c.interesting_reason = "(stub judge — no model configured)"


class LlamaJudge:
    def __init__(self, cfg: Config):
        from llama_cpp import Llama

        self.cfg = cfg
        self.llm = Llama(
            model_path=cfg.model_path,
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=cfg.n_ctx,
            verbose=False,
        )

    # Re-query words the model omitted, up to this many passes, before falling back
    # to the keep-biased default. Omission was the single biggest source of junk in
    # the output — a weak model silently drops most words and they all default keep.
    _MAX_PASSES = 3

    def judge(self, candidates: list[Candidate]) -> None:
        survivors = _survivors(candidates)
        batch = self.cfg.judge_batch
        for i in range(0, len(survivors), batch):
            self._judge_batch(survivors[i : i + batch])

    def _judge_batch(self, batch: list[Candidate]) -> None:
        """Judge a batch, re-querying any word the model left out until every word
        has an explicit verdict or we run out of passes."""
        verdicts: dict[str, dict] = {}
        pending = list(batch)
        for _ in range(self._MAX_PASSES):
            parsed = _parse_verdicts(self._query(pending)) or []
            for v in parsed:
                w = _verdict_word(v)
                if w:
                    verdicts[w] = v
            pending = [c for c in batch if c.lemma.lower() not in verdicts]
            if not pending:
                break
        apply_verdicts(batch, list(verdicts.values()))

    def _query(self, batch: list[Candidate]) -> str:
        # `interesting_reason` already carries validity.py's own suspicion for
        # an UNSURE candidate ("likely misspelling of 'abject'", "recurs but
        # sits in a non-English context") -- forwarded as `hint` so the model
        # judges with that evidence instead of guessing blind from the bare
        # lemma, the same pattern classify.py already uses for its WND prior.
        # A model call can't verify a real dictionary distinction (abject vs.
        # abiect) from the string alone; the pipeline already worked that out.
        items = [
            {"word": c.lemma, "freq": _freq_band(c.zipf), **({"hint": c.interesting_reason}
                                                               if c.interesting_reason else {})}
            for c in batch
        ]
        instructions = (
            'A "hint" field, when present, is a suspicion from an earlier pipeline stage '
            '(e.g. a likely misspelling of a named word, or a non-English context) -- treat it '
            "as strong evidence for keep=false unless you have good reason to overrule it. "
            'Output ONLY a JSON array, one object per input word IN THE SAME ORDER, '
            'no prose and no code fences: [{"w": "<word>", "k": <true|false>}]. '
            "Include EVERY input word exactly once."
        )
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": f"{RUBRIC}\n\n{instructions}"},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=len(batch) * 16 + 64,
        )
        return out["choices"][0]["message"]["content"]


def _verdict_word(v) -> str:
    """Word key from a verdict object — compact 'w' or legacy 'word'."""
    if not isinstance(v, dict):
        return ""
    return str(v.get("w", v.get("word", ""))).strip().lower()


def _verdict_keep(v: dict) -> bool:
    """Keep flag from a verdict object — compact 'k' or legacy 'keep'. Defaults to
    keep so a malformed entry never silently drops a word."""
    return bool(v.get("k", v.get("keep", True)))


def apply_verdicts(batch: list[Candidate], verdicts: list | None) -> None:
    """Apply the model's keep/skip decisions to a batch. Pure and side-effecting
    on the candidates; kept separate from model I/O so it can be unit-tested
    without the (nondeterministic) LLM. Keep-biased: a missing or unparseable
    verdict leaves the word on the shortlist rather than dropping it.

    A keep=false verdict drops the word regardless of whether validity.py
    handed it a KEEP or an UNSURE ("recurs but resembles a misspelling" /
    "unattested but recurs") -- UNSURE was designed as "send to human
    review," but `ingest` has no review step, so treating it as veto-proof
    silently converted every UNSURE into a permanent keep no matter what the
    judge said. The judge's own rubric already instructs it to reject
    misspellings and unremarkable words; UNSURE candidates need that
    scrutiny even more than plain KEEPs, not less."""
    if verdicts is None:
        for c in batch:
            c.interesting_reason = c.interesting_reason or "(judge parse error — kept)"
        return
    by_word = {w: v for v in verdicts if (w := _verdict_word(v))}
    for c in batch:
        v = by_word.get(c.lemma.lower())
        if not v:
            continue
        if not _verdict_keep(v):
            c.verdict = Verdict.DROP
            c.reject_reason = RejectReason.NOT_INTERESTING
        else:
            c.interesting_reason = str(v.get("reason", ""))[:60]


def _parse_verdicts(text: str) -> list | None:
    """Tolerantly pull the verdict list out of a model reply — handles code
    fences, an object wrapper, or a bare array."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1 :] if "\n" in text else text
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
    if start == -1:
        return None
    snippet = text[start:]
    for end in range(len(snippet), 0, -1):  # trim trailing junk until it parses
        try:
            data = json.loads(snippet[:end])
            break
        except json.JSONDecodeError:
            continue
    else:
        return None
    if isinstance(data, dict):
        return data.get("words") or data.get("results") or None
    return data if isinstance(data, list) else None
