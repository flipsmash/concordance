"""Web-search + grounded LLM extraction — the last-resort definition tier.

For a word no dictionary defines, search the open web (keyless, via DuckDuckGo)
and have the local model EXTRACT a definition that is actually present in the
result snippets. The model is explicitly forbidden from using its own knowledge:
it either finds the meaning in the provided text or answers NONE. This keeps the
project's rule that the model never *invents* a definition — it only reads.

Used only when asked (`concordance define --web`) and only for words the validity
scorer didn't already flag as likely artifacts, since a web search for OCR noise
is wasted effort.
"""

from __future__ import annotations

import re

from .model import Candidate

_MAX_RESULTS = 6
_SYSTEM = (
    "You extract a dictionary definition ONLY from the provided web-search snippets. "
    "You are given a WORD and SNIPPETS. If the snippets actually state what the WORD "
    "means, reply with ONE short definition (<=20 words) drawn from that text. If the "
    "snippets merely mention the word, are about a different word, or do not define it, "
    "reply with exactly NONE. Never use any knowledge beyond the snippets; never guess."
)


def search_snippets(word: str, max_results: int = _MAX_RESULTS) -> list[str]:
    """Real web-result snippets for the word, or [] on any failure."""
    try:
        from ddgs import DDGS
    except ImportError:                       # older package name
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return []
    out: list[str] = []
    try:
        with DDGS() as d:
            for r in d.text(f'{word} word meaning definition', max_results=max_results):
                body = (r.get("body") or "").strip()
                title = (r.get("title") or "").strip()
                if body:
                    out.append(f"{title} — {body}" if title else body)
    except Exception:
        return out
    return out


def extract_definition(word: str, snippets: list[str], llm) -> str:
    """Ask the model to pull a definition out of the snippets; '' if none present."""
    if not snippets:
        return ""
    joined = "\n".join(f"- {s}" for s in snippets[:_MAX_RESULTS])
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"WORD: {word}\nSNIPPETS:\n{joined}"},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    text = (out["choices"][0]["message"]["content"] or "").strip()
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    if not text or text.strip(".").upper() == "NONE" or len(text) < 4:
        return ""
    # Guard against the model echoing the instruction or refusing.
    if text.upper().startswith(("NONE", "I ", "SORRY", "THE SNIPPETS")):
        return ""
    return text


def define_via_web(cand: Candidate, llm) -> bool:
    gloss = extract_definition(cand.lemma, search_snippets(cand.lemma), llm)
    if not gloss:
        return False
    cand.definition = gloss
    cand.definition_source = "Web (LLM-extracted)"
    return True
