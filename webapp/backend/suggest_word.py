"""Admin "suggest a new word" API -- lets an admin add a word directly,
without waiting for it to turn up in a book (§ admin suggest-word plan).

Two things this deliberately does NOT need to build, because the rest of
the codebase already provides them for free:

  - "Tracked as normal" for a future book that uses this word: sync_book_results
    (concordance/db) upserts via ON CONFLICT (lemma_lc) DO UPDATE and
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

import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from concordance import deepdef, dictionary, localdict, mw
from concordance.oed import definitions as oed_definitions
from concordance.model import Candidate, RejectReason, junk_pos_reason, normalize_pos
from concordance.resolve import _pace_wordnik
from webapp.backend import main as _main

router = APIRouter()


# --- request/response models -------------------------------------------------

class RejectedIn(BaseModel):
    book_title: str
    reason: str | None
    detail: str | None


class SuggestSearchResult(BaseModel):
    lemma: str
    exists: bool
    word_id: int | None = None
    active: bool | None = None
    # for an existing word: what it says now and, if inactive, why it was cast out
    definition: str | None = None
    inactive_reason: str | None = None
    inactive_note: str | None = None
    # books whose ingest rejected this lemma, and why (a new word only)
    rejected_in: list[RejectedIn] = []
    rejected_book_count: int = 0
    candidates: list[dict] = []
    web_search_unavailable: bool = False


class SuggestReactivateRequest(BaseModel):
    word_id: int


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
    linked_books: int = 0


# A word or a short phrase: letters (any script's), spaces, hyphens,
# apostrophes; at most 4 words. Phrases are real vocabulary too (in
# medias res, ad hominem), and every source takes them.
_MAX_WORDS = 4


def _clean_lemma(raw: str) -> str:
    lemma = re.sub(r"\s+", " ", (raw or "").replace("\u2019", "'")).strip()
    if (not lemma or len(lemma) > 60 or len(lemma.split(" ")) > _MAX_WORDS
            or not all(ch.isalpha() or ch in " -'" for ch in lemma) or not lemma[0].isalpha()):
        raise HTTPException(status_code=422,
                            detail=f"enter a word or a short phrase (up to {_MAX_WORDS} words; "
                                   "letters, spaces, hyphens and apostrophes only)")
    return lemma


def _rejections(cur, lemma: str, limit: int = 8) -> tuple[list[RejectedIn], int]:
    cur.execute(f"""SELECT b.title, r.reason, r.detail, count(*) OVER ()
                    FROM {_main.SCHEMA}.rejected_word r JOIN {_main.SCHEMA}.book b ON b.id = r.book_id
                    WHERE r.lemma_lc = lower(%s) ORDER BY b.title LIMIT %s""", (lemma, limit))
    rows = cur.fetchall()
    return [RejectedIn(book_title=t, reason=r, detail=d) for t, r, d, _ in rows], (rows[0][3] if rows else 0)


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

    # 0 Dict (stored as definition_source "OED", same as the ingest cascade):
    # one card per homograph entry, like the local Wiktionary senses above.
    for i, sense in enumerate(oed_definitions.definition_lexicon(conn, {lemma.lower()}).get(lemma.lower(), [])):
        candidates.append({
            "source": "0 Dict" if i == 0 else f"0 Dict (entry {i + 1})",
            "definition_source": "OED",
            "definition": sense.definition,
            "part_of_speech": normalize_pos(sense.part_of_speech),
            "ipa": "",
            "etymology": sense.etymology,
            "synonyms": [],
        })

    # One source failing (a timeout, a page that doesn't take a phrase)
    # never costs the admin the others' answers.
    def _try(fetch, source):
        cand = Candidate(lemma=lemma, pos="")
        try:
            if fetch(cand):
                candidates.append(_candidate_dict(cand, source(cand)))
        except Exception:  # noqa: BLE001
            pass

    _try(lambda c: dictionary._from_freedict(c, session), lambda c: "Free Dictionary API")
    _try(lambda c: dictionary._from_wiktionary(c, session), lambda c: "Wiktionary")

    key = deepdef.wordnik_key()
    if key:
        def _wordnik(c):
            _pace_wordnik()
            return deepdef._from_wordnik(c, session, key)
        _try(_wordnik, lambda c: c.definition_source or "Wordnik")

    _try(lambda c: deepdef._from_yourdictionary(c, session), lambda c: "yourdictionary.com")

    mw_key = mw.mw_api_key()
    if mw_key and not mw.quota_exhausted():
        try:
            entries = mw.exact_matches(mw.lookup_api(lemma, mw_key, session), lemma)
        except Exception:  # noqa: BLE001
            entries = []
        for i, e in enumerate(entries):
            label = "Merriam-Webster" if i == 0 else f"Merriam-Webster ({e.part_of_speech})"
            resolved_pos = normalize_pos(e.part_of_speech)
            # is_foreign_pos checks the RAW (pre-normalize_pos) string -- MW's
            # "<Language> noun" foreign-loanword tag is a capitalized demonym,
            # a signal normalize_pos's lowercasing destroys (see db's own
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
    lemma = _clean_lemma(lemma)

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT id, active, definition, variant_flag_reason,
                               coalesce(variant_flag_note, validity_notes)
                        FROM {_main.SCHEMA}.word WHERE lemma_lc = lower(%s)""", (lemma,))
        row = cur.fetchone()
        if row is not None:
            wid, active, definition, reason, note = row
            return SuggestSearchResult(
                lemma=lemma, exists=True, word_id=wid, active=active, definition=definition,
                inactive_reason=None if active else (reason or "pruned"),
                inactive_note=None if active else note)

        rejected_in, rejected_count = _rejections(cur, lemma)
        candidates, web_search_unavailable = _gather_candidates(conn, lemma)

    return SuggestSearchResult(
        lemma=lemma, exists=False, rejected_in=rejected_in, rejected_book_count=rejected_count,
        candidates=candidates, web_search_unavailable=web_search_unavailable,
    )


@router.post("/api/admin/suggest-word/reactivate", response_model=SuggestFinalizeResult)
def reactivate_suggest_word(
    body: SuggestReactivateRequest, user: dict = Depends(_main.require_admin),
) -> SuggestFinalizeResult:
    """Bring a pruned / cast-out word back. It's marked admin_suggested, which
    the automated sweeps respect, so it isn't cast out again; the old
    cast-out reason stays in variant_flag_* as history."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""UPDATE {_main.SCHEMA}.word SET active=true, admin_suggested=true,
                    admin_suggested_by=%s, admin_suggested_at=now(), updated_at=now()
                WHERE id=%s RETURNING id, lemma, definition""",
            (user.get("username") or "admin", body.word_id))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="word not found")
        conn.commit()
    return SuggestFinalizeResult(id=row[0], lemma=row[1], definition=row[2] or None)


@router.post("/api/admin/suggest-word/finalize", response_model=SuggestFinalizeResult)
def finalize_suggest_word(
    body: SuggestFinalizeRequest, user: dict = Depends(_main.require_admin),
) -> SuggestFinalizeResult:
    lemma = _clean_lemma(body.lemma)

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
        # The word really occurs in any book whose ingest rejected it: link
        # those books and drop the rejections, as accepting a rejected word
        # from the Rejected tab does.
        cur.execute(
            f"""WITH gone AS (DELETE FROM {_main.SCHEMA}.rejected_word
                              WHERE lemma_lc = lower(%s) RETURNING book_id)
                INSERT INTO {_main.SCHEMA}.word_book (word_id, book_id)
                SELECT DISTINCT %s, book_id FROM gone ON CONFLICT DO NOTHING""",
            (lemma, word_id))
        linked = cur.rowcount
        conn.commit()

    return SuggestFinalizeResult(id=word_id, lemma=lemma, definition=body.definition or None,
                                 linked_books=linked)
