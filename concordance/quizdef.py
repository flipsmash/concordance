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

_st = SnowballStemmer("english")


# --- leak detection (no model) --------------------------------------------

def _shared_root(a: str, b: str) -> bool:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n >= 4 and n >= 0.55 * min(len(a), len(b))


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
