"""Admin "suggest a new word" API -- lets an admin add a word directly,
without waiting for it to turn up in a book (§ admin suggest-word plan).

Two things this deliberately does NOT need to build, because the rest of
the codebase already provides them for free:

  - "Tracked as normal" for a future book that uses this word: sync_book_results
    (concordance/db.py) upserts via ON CONFLICT (lemma_lc) DO UPDATE and
    already handles attaching a word_book row to a book-less word the first
    time any book uses it -- import_defined_words proves the same pattern.
  - "Proof positive for backend vetting": fetch_known_verdicts treats ANY
    word row with active=true as a cached "keep", skipping the LLM judge
    for every future book -- regardless of how the word got there. Nothing
    admin-suggestion-specific is needed beyond inserting with active=true
    (the column default).

Search deliberately does NOT reuse resolve.py's stop-at-first-hit cascade:
that cascade is built to answer "what's the ONE best definition," but this
flow wants every source's own independent answer shown side by side so the
admin can compare and pick. Each source's own private per-source function
is called directly (not the public enrich()/deep_enrich() wrappers, which
themselves stop at the first hit and would silently collapse two sources
into one candidate).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from concordance import deepdef, dictionary, localdict, mw
from concordance.model import Candidate, RejectReason, junk_pos_reason, normalize_pos
from concordance.resolve import _pace_wordnik
from webapp.backend import main as _main

router = APIRouter()


# --- request/response models -------------------------------------------------

class SuggestSearchResult(BaseModel):
    lemma: str
    exists: bool
    word_id: int | None = None
    active: bool | None = None
    candidates: list[dict] = []
    web_search_unavailable: bool = False


class SuggestFinalizeRequest(BaseModel):
    lemma: str
    definition: str = ""
    part_of_speech: str = ""
    ipa: str = ""
    etymology: str = ""
    synonyms: list[str] = []
    definition_source: str = ""


class SuggestFinalizeResult(BaseModel):
    id: int
    lemma: str
    definition: str | None


def _candidate_dict(cand: Candidate, source: str) -> dict:
    return {
        "source": source,
        "definition": cand.definition,
        "part_of_speech": normalize_pos(cand.part_of_speech),
        "ipa": cand.ipa,
        "etymology": cand.etymology,
        "synonyms": list(cand.synonyms),
    }


def _gather_candidates(conn, lemma: str) -> tuple[list[dict], bool]:
    """Query every source independently and return (candidates, web_search_unavailable).
    A source with no hit is simply omitted, not returned as an empty entry."""
    candidates: list[dict] = []
    session = dictionary.make_session()

    # Local Wiktionary can hold multiple senses for one lemma -- surface each
    # as its own card rather than collapsing to one, since it's the richest
    # multi-candidate source available (see localdict.lookup_one's own
    # docstring: it returns list[Entry] directly, not a mutated Candidate).
    entries = localdict.lookup_one(conn, lemma)
    for i, (pos, definition, ipa, etymology, _is_archaic, _is_obsolete) in enumerate(entries):
        label = "Local Wiktionary" if i == 0 else f"Local Wiktionary (sense {i + 1})"
        candidates.append({
            "source": label,
            "definition": definition.split(";")[0].strip(),
            "part_of_speech": normalize_pos(pos),
            "ipa": ipa,
            "etymology": etymology,
            "synonyms": [],
        })

    cand = Candidate(lemma=lemma, pos="")
    if dictionary._from_freedict(cand, session):
        candidates.append(_candidate_dict(cand, "Free Dictionary API"))

    cand = Candidate(lemma=lemma, pos="")
    if dictionary._from_wiktionary(cand, session):
        candidates.append(_candidate_dict(cand, "Wiktionary"))

    key = deepdef.wordnik_key()
    if key:
        cand = Candidate(lemma=lemma, pos="")
        _pace_wordnik()
        if deepdef._from_wordnik(cand, session, key):
            candidates.append(_candidate_dict(cand, cand.definition_source or "Wordnik"))

    cand = Candidate(lemma=lemma, pos="")
    if deepdef._from_yourdictionary(cand, session):
        candidates.append(_candidate_dict(cand, "yourdictionary.com"))

    mw_key = mw.mw_api_key()
    if mw_key and not mw.quota_exhausted():
        entries = mw.exact_matches(mw.lookup_api(lemma, mw_key, session), lemma)
        for i, e in enumerate(entries):
            label = "Merriam-Webster" if i == 0 else f"Merriam-Webster ({e.part_of_speech})"
            resolved_pos = normalize_pos(e.part_of_speech)
            # is_foreign_pos checks the RAW (pre-normalize_pos) string -- MW's
            # "<Language> noun" foreign-loanword tag is a capitalized demonym,
            # a signal normalize_pos's lowercasing destroys (see db.py's own
            # mw_backfill, which applies this exact same check the same way).
            reason = junk_pos_reason(resolved_pos) or (
                RejectReason.FOREIGN_LANGUAGE if mw.is_foreign_pos(e.part_of_speech) else None)
            candidates.append({
                "source": label,
                "definition": "; ".join(e.definitions),
                "part_of_speech": resolved_pos,
                "ipa": e.pronunciations[0].respelling if e.pronunciations else "",
                "etymology": e.etymology,
                "synonyms": [],
                "junk_pos_warning": reason.value if reason else None,
            })

    web_search_unavailable = False
    if not candidates:
        # Last resort, only tried when every deterministic source above
        # missed -- and, in practice, expected to be unavailable whenever a
        # bulk maintain/ingest job already has the GPU's VRAM in use (the
        # common state on this box, not a rare edge case). llm=None before
        # the try so the finally below never calls .close() on a name that
        # was never bound; the try wraps construction itself, since that is
        # where a GPU-busy failure actually raises, not somewhere later.
        llm = None
        try:
            from concordance.config import Config
            cfg = Config()
            if cfg.model_path:
                from pathlib import Path
                if Path(cfg.model_path).exists():
                    from llama_cpp import Llama
                    llm = Llama(model_path=cfg.model_path, n_gpu_layers=cfg.n_gpu_layers,
                                n_ctx=cfg.n_ctx, verbose=False)
            if llm is not None:
                from concordance import websearch
                cand = Candidate(lemma=lemma, pos="")
                if websearch.define_via_web(cand, llm):
                    candidates.append(_candidate_dict(cand, "Web search + local model"))
            else:
                web_search_unavailable = True
        except Exception:
            web_search_unavailable = True
        finally:
            # Best-effort: a partially-constructed llm's own .close() raising
            # must not turn an otherwise-successful (or already-degraded)
            # response into a 500.
            if llm is not None:
                try:
                    llm.close()
                except Exception:
                    pass

    # Informational only, per the plan's non-goal: this flow doesn't block a
    # symbol/proper-noun-only resolved sense on finalize (the admin's call is
    # final, unlike accept_rejected's hard gate) -- but it's still worth
    # surfacing the same junk_pos_reason verdict as a hint on each candidate
    # card so the admin can see it before choosing. The MW branch above
    # already set a stronger verdict (it also catches the RAW-string
    # foreign-loanword case normalize_pos would otherwise destroy) --
    # setdefault leaves that alone rather than clobbering it with a plain
    # junk_pos_reason recheck against the now-normalized POS.
    for c in candidates:
        if "junk_pos_warning" not in c:
            reason = junk_pos_reason(c["part_of_speech"])
            c["junk_pos_warning"] = reason.value if reason else None

    return candidates, web_search_unavailable


@router.get("/api/admin/suggest-word/search", response_model=SuggestSearchResult)
def search_suggest_word(lemma: str, _: dict = Depends(_main.require_admin)) -> SuggestSearchResult:
    lemma = lemma.strip()
    if not lemma or " " in lemma:
        raise HTTPException(status_code=422, detail="enter a single word, not a phrase")

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT id, active FROM {_main.SCHEMA}.word WHERE lemma_lc = lower(%s)", (lemma,))
        row = cur.fetchone()
        if row is not None:
            return SuggestSearchResult(lemma=lemma, exists=True, word_id=row[0], active=row[1])

        candidates, web_search_unavailable = _gather_candidates(conn, lemma)

    return SuggestSearchResult(
        lemma=lemma, exists=False, candidates=candidates,
        web_search_unavailable=web_search_unavailable,
    )


@router.post("/api/admin/suggest-word/finalize", response_model=SuggestFinalizeResult)
def finalize_suggest_word(
    body: SuggestFinalizeRequest, user: dict = Depends(_main.require_admin),
) -> SuggestFinalizeResult:
    lemma = body.lemma.strip()
    if not lemma or " " in lemma:
        raise HTTPException(status_code=422, detail="enter a single word, not a phrase")

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {_main.SCHEMA}.word WHERE lemma_lc = lower(%s)", (lemma,))
        if cur.fetchone() is not None:
            raise HTTPException(status_code=409, detail=f"{lemma!r} was already added, possibly by another admin")

        username = user.get("username") or "admin"
        cur.execute(
            f"""INSERT INTO {_main.SCHEMA}.word
                (lemma, definition, part_of_speech, ipa, synonyms, etymology,
                 definition_source, first_added, active,
                 admin_suggested, admin_suggested_by, admin_suggested_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE, true, true, %s, now())
                RETURNING id""",
            (lemma, body.definition, normalize_pos(body.part_of_speech), body.ipa,
             list(body.synonyms), body.etymology, body.definition_source, username),
        )
        word_id = cur.fetchone()[0]
        conn.commit()

    return SuggestFinalizeResult(id=word_id, lemma=lemma, definition=body.definition or None)
