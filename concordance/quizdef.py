"""Quiz-safe definitions (§ quizzing).

~37% of definitions leak the target word's root ("audaciously" -> "in an audacious
manner"), which makes a recall quiz trivial. This produces a separate
`quiz_definition` that preserves meaning without the giveaway:

  clean     the definition already doesn't leak -> use as-is (the majority);
  rewritten an LLM *paraphrase* (not invention) of the leaking definition, then
            machine-validated to actually be leak-free;
  redacted  fallback when the rewrite still leaks or the model balked — the leaking
            span is blanked.

Built to scale: the detector is cheap/streamable and the LLM rewrite is batched and
meant to run resumably over an arbitrarily large corpus (only leakers are rewritten).
"""

from __future__ import annotations

import json
import re

from nltk.stem import SnowballStemmer

from .config import Config
from .validity_score import _PREFIXES

_st = SnowballStemmer("english")


# --- leak detection (no model) --------------------------------------------

def _shared_root(a: str, b: str) -> bool:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    if n >= 4 and n >= 0.55 * min(len(a), len(b)):
        return True
    # A word formed by adding a prefix ("premeditate" = "pre" + "meditate",
    # "irremovable" = "ir" + "removable") shares NO leading characters with
    # its base, so the prefix-alignment check above can never catch it --
    # check explicitly for "one string is exactly a known prefix glued onto
    # the other" instead. Targeted at a specific prefix list (shared with
    # validity_score's own morphology check) rather than a generic
    # shared-suffix scan, which would false-positive constantly on
    # coincidental shared endings between unrelated words (e.g. "nation" and
    # "creation" both ending in "-ation").
    for word, other in ((a, b), (b, a)):
        for p in _PREFIXES:
            if word.startswith(p) and word[len(p):] == other and len(other) >= 4:
                return True
    return False


def leaking_tokens(word: str, definition: str) -> list[str]:
    """Def tokens that share the word's root/stem (case-insensitive)."""
    wl = word.strip().lower()
    ws = _st.stem(wl)
    out = []
    for tok in re.findall(r"[a-z]+", (definition or "").lower()):
        if len(tok) < 3:
            continue
        if tok == wl or _st.stem(tok) == ws or _shared_root(wl, tok):
            out.append(tok)
    return out


def has_leak(word: str, definition: str) -> bool:
    return bool(leaking_tokens(word, definition))


def redact(word: str, definition: str) -> str:
    """Blank the leaking tokens — the always-available fallback."""
    leaks = set(leaking_tokens(word, definition))
    if not leaks:
        return definition

    def repl(m):
        return "—" if m.group(0).lower() in leaks else m.group(0)

    return re.sub(r"[A-Za-z]+", repl, definition)


# --- LLM rewrite (batched, validated) -------------------------------------

_SYSTEM = (
    "You rewrite dictionary definitions into quiz clues. For each item, rewrite the "
    "definition so it means the SAME thing but does NOT contain the target word or ANY "
    "word sharing its root (for 'audaciously' avoid audacious/audacity — say 'bold, "
    "daring'). Keep it a concise, natural clue; do not add facts. "
    'Output ONLY JSON: [{"w":"<word>","d":"<rewrite>"}], every input word exactly once, no prose.'
)


class Rewriter:
    _MAX_PASSES = 3

    def __init__(self, cfg: Config | None = None, model_path: str | None = None):
        from pathlib import Path
        from llama_cpp import Llama
        cfg = cfg or Config()
        mp = model_path or cfg.model_path
        if not mp or not Path(mp).exists():
            raise RuntimeError(f"rewriter model not found: {mp!r}")
        self.llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)
        self.batch = 10

    def rewrite(self, items: list[dict]) -> dict[str, tuple[str, str]]:
        """items: {word, definition} (leakers only). Returns word -> (quiz_def, source)."""
        result: dict[str, tuple[str, str]] = {}
        for i in range(0, len(items), self.batch):
            self._batch(items[i : i + self.batch], result)
        # anything the model never returned or that still leaks -> redact
        for it in items:
            w = it["word"].lower()
            if w not in result or has_leak(it["word"], result[w][0]):
                result[w] = (redact(it["word"], it["definition"]), "redacted")
        return result

    def _batch(self, batch: list[dict], result: dict) -> None:
        pending = list(batch)
        for _ in range(self._MAX_PASSES):
            for obj in self._query(pending):
                w = str(obj.get("w", "")).strip().lower()
                d = str(obj.get("d", "")).strip()
                if not w or not d:
                    continue
                src_word = next((it["word"] for it in batch if it["word"].lower() == w), w)
                if not has_leak(src_word, d):
                    result[w] = (d, "rewritten")
            pending = [it for it in batch if it["word"].lower() not in result]
            if not pending:
                break

    def _query(self, items: list[dict]) -> list:
        payload = json.dumps([{"word": it["word"], "definition": (it.get("definition") or "")[:220]}
                              for it in items], ensure_ascii=False)
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": payload}],
            temperature=0.2, max_tokens=len(items) * 60 + 128)
        return _parse(out["choices"][0]["message"]["content"])


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


# --- quiz suitability -----------------------------------------------------

# Definitions that just point at another word (the "answer" isn't real vocabulary).
_VARIANT_RE = re.compile(
    r"\b(form|spelling|inflection|tense|participle|plural|abbreviation|initialism) of\b"
    r"|\b(first|second|third)-person (singular|plural)\b", re.IGNORECASE)

# A morphological derivative whose root is at least this common is trivially
# inferable from the root (reveller<-revel); a rare root (abacination<-abacinate)
# is not, so the word stays quizzable. wordfreq Zipf is corpus-independent.
_COMMON_ROOT_ZIPF = 3.0


# TODO(quizzable-derivative-false-positives): the common-root rule is purely
# ORTHOGRAPHIC — it excludes any word whose _morph_root strips to a common root,
# even when the suffix shifted the meaning so the word is NOT inferable from the
# root. e.g. `battlement` (indented parapet) is dropped as a "derivative of
# 'battle'"; same risk for any word whose sense drifted from its root form.
# ~762 words are excluded by this rule, so the blast radius is non-trivial.
# Proposed fix: only exclude a derivative when the definition ALSO literally
# leaks the root, i.e. gate on `has_leak(word, definition)` in addition to the
# morphology+zipf match. That keeps semantically-drifted derivatives quizzable
# (their gloss won't contain the root) while still dropping the truly
# transparent ones (reveller -> "one who revels"). Needs the surface word passed
# in, which compute_quizzable already has.
def quizzable(definition: str, morph_root: str | None = None,
              root_zipf: float | None = None) -> tuple[bool, str]:
    """(quizzable, reason). False when the answer is trivially inferable."""
    if _VARIANT_RE.search(definition or ""):
        return False, "grammatical/variant form"
    if morph_root and root_zipf is not None and root_zipf >= _COMMON_ROOT_ZIPF:
        return False, f"transparent derivative of common root '{morph_root}'"
    return True, ""
