"""`concordance add-pointer-targets`: give the spelling a definition points
at its own entry.

"soubadar" is defined as "Alternative spelling of subadar." -- useful only
if subadar has a page to link to. Ingest only ever adds words a book uses,
so the target of a pointer often isn't in the vocabulary at all. This adds
it, held to ingest's own standards:

  - single words only ("thorn apple", "pea-souper" stay as text -- the
    definition linker matches single tokens);
  - the frequency floor (effective_zipf < cfg.min_zipf): a pointer at a
    common word (myopy -> myopia) is no reason to add myopia;
  - earlier verdicts stand: a target the judge already rejected in some
    book (not_interesting, not_a_word, misspelling...) or a human pruned
    stays out, as does one a cleanup sweep cast out. The one exception is a
    proper_noun rejection -- that only means a book capitalized it, and the
    dictionary's own cross-reference says it's a common word;
  - the LLM judge, for every target without a verdict, exactly as ingest
    runs it on a new book's words;
  - a definition from ingest's own cascade (LOCAL -> FREE -> 0 Dict -> MW),
    then the same post-lookup cast-outs (foreign, respelling of a common
    word, junk POS) and review flags (pipeline.post_enrichment_checks). A
    target whose only definition is 0 Dict scan text is left out: those
    are often OCR-mangled ("Now only Anat. and Zoo/. Also 7 -uble"), and a
    page that exists to explain a pointer needs a readable definition.

The validity gate is skipped: a dictionary's cross-reference already vouches
that the target is a real English word.

Every word added records pointer_target_of (the pointing words) and
pointer_target_added_at. It is added book-less: a book whose ingest rejected
the lemma as a proper noun keeps that rejection (unlike Suggest Word's
finalize, which links such books) -- it usually really was the name:
Tartuffe the play, Heth the biblical figure, Sewel the surname.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from . import crossref, judge, localdict, mw
from .config import Config
from .db import _safe_schema
from .dictionary import make_session
from .model import Candidate, Verdict
from .oed import definitions as oed_definitions
from .validity_score import effective_zipf

_SINGLE_WORD = re.compile(r"[^\W\d_]+")

# word.part_of_speech -> the tagger's coarse tag the resolvers pick senses by.
# A spelling variant shares its source word's part of speech.
_COARSE = {"noun": "NOUN", "verb": "VERB", "adjective": "ADJ", "adverb": "ADV"}

_OED_SOURCE = "OED"


def find_targets(conn, schema: str = "concordance") -> dict[str, dict]:
    """Single-word pointer targets of active words that aren't themselves
    active: target -> {"sources": [source lemmas], "pos": coarse tag}."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT lemma_lc, definition, part_of_speech FROM {s}.word
                        WHERE active AND definition IS NOT NULL AND definition <> ''
                        ORDER BY lemma_lc""")
        rows = cur.fetchall()
        cur.execute(f"SELECT lemma_lc FROM {s}.word WHERE active")
        active = {r[0] for r in cur.fetchall()}

    out: dict[str, dict] = {}
    for lemma_lc, definition, pos in rows:
        if not crossref.mentions_pointer(definition):
            continue
        for target in crossref.pointer_target_phrases(definition):
            if target == lemma_lc or target in active or not _SINGLE_WORD.fullmatch(target):
                continue
            entry = out.setdefault(target, {"sources": [], "pos": ""})
            entry["sources"].append(lemma_lc)
            entry["pos"] = entry["pos"] or _COARSE.get((pos or "").lower(), "")
    return out


def _prior_verdicts(conn, s: str, lemmas: list[str]) -> dict[str, str]:
    """lemma -> why it's already out: "inactive" (pruned or cast out -- any
    word row that isn't active), a rejected_word reason other than
    proper_noun, or an earlier run's own rejection."""
    out: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT lemma_lc FROM {s}.word WHERE NOT active AND lemma_lc = ANY(%s)", (lemmas,))
        out.update((r[0], "inactive") for r in cur.fetchall())
        cur.execute(f"""SELECT DISTINCT ON (lemma_lc) lemma_lc, reason FROM {s}.rejected_word
                        WHERE lemma_lc = ANY(%s) AND reason IS DISTINCT FROM 'proper_noun'
                        ORDER BY lemma_lc, reason""", (lemmas,))
        for lemma, reason in cur.fetchall():
            out.setdefault(lemma, reason or "rejected")
        cur.execute(f"SELECT lemma_lc, reason FROM {s}.pointer_target_rejection WHERE lemma_lc = ANY(%s)",
                    (lemmas,))
        for lemma, reason in cur.fetchall():
            out.setdefault(lemma, reason)
    return out


def add_pointer_targets(conn, schema: str = "concordance", cfg: Config | None = None, *,
                        dry_run: bool = False, limit: int = 0, judge_obj=None, progress=None) -> dict:
    """Add every eligible pointer target; see the module docstring. Returns
    counts plus the "added" lemmas (and, on a dry run, what would be)."""
    from .pipeline import _enrich_one, post_enrichment_checks

    cfg = cfg or Config()
    s = _safe_schema(schema)
    targets = find_targets(conn, schema)
    stats = {"targets": len(targets), "too_common": 0, "prior_verdict": 0, "judged": 0,
             "judge_rejected": 0, "undefined": 0, "oed_only": 0, "cast_out": 0, "flagged": 0,
             "added": []}

    rare = {t: info for t, info in sorted(targets.items()) if effective_zipf(t) < cfg.min_zipf}
    stats["too_common"] = len(targets) - len(rare)
    prior = _prior_verdicts(conn, s, list(rare))
    stats["prior_verdict"] = sum(t in prior for t in rare)
    # KEEP is what the (skipped) validity gate would hand the judge for a
    # real word; the judge only looks at KEEP/UNSURE candidates.
    cands = [Candidate(lemma=t, pos=info["pos"], zipf=effective_zipf(t), verdict=Verdict.KEEP)
             for t, info in rare.items() if t not in prior]
    if limit:
        cands = cands[:limit]
    if not cands:
        return stats

    # --- the judge, as ingest runs it on a new book's words ---------------
    stats["judged"] = len(cands)
    (judge_obj or judge.get_judge(cfg)).judge(cands)
    rejections = [(c.lemma, "not_interesting") for c in cands if c.verdict not in (Verdict.KEEP, Verdict.UNSURE)]
    cands = [c for c in cands if c.verdict in (Verdict.KEEP, Verdict.UNSURE)]
    stats["judge_rejected"] = stats["judged"] - len(cands)

    # --- definitions, exactly as ingest looks them up ---------------------
    if cands:
        lexicon = localdict.build_lexicon(conn, {c.lemma for c in cands})
        localdict.expand_lexicon_for_stubs(conn, lexicon)
        oed_lexicon = oed_definitions.definition_lexicon(conn, {c.lemma for c in cands})
        session = make_session()
        mw_api_key = mw.mw_api_key() if cfg.mw_enrichment else ""
        if mw_api_key and mw.quota_exhausted():
            mw_api_key = ""
        with ThreadPoolExecutor(max_workers=max(1, cfg.enrichment_workers)) as pool:
            for i, _ in enumerate(pool.map(
                    lambda c: _enrich_one(c, lexicon=lexicon, oed_lexicon=oed_lexicon,
                                          session=session, mw_api_key=mw_api_key), cands), 1):
                if progress and i % 100 == 0:
                    progress(i, len(cands))
        lookup_cast_out = sum(c.verdict is Verdict.DROP for c in cands)   # MW's foreign signal
        for c in cands:
            if c.verdict is Verdict.DROP:
                continue
            if not c.definition:
                c.verdict = Verdict.DROP   # nothing to show on its page
                stats["undefined"] += 1
            elif c.definition_source == _OED_SOURCE:
                c.verdict = Verdict.DROP
                stats["oed_only"] += 1
        before = sum(c.verdict is Verdict.DROP for c in cands)
        _, stats["flagged"] = post_enrichment_checks(
            conn, [c for c in cands if c.verdict is not Verdict.DROP], cfg)
        stats["cast_out"] = lookup_cast_out + sum(c.verdict is Verdict.DROP for c in cands) - before
    keep = [c for c in cands if c.verdict is not Verdict.DROP]
    stats["added"] = [c.lemma for c in keep]
    if dry_run:
        return stats

    for c in cands:
        if c.verdict is Verdict.DROP:
            rejections.append((c.lemma, (c.reject_reason.value if c.reject_reason
                                         else "only_0_dict_definition" if c.definition else "undefined")))

    with conn.cursor() as cur:
        cur.executemany(f"""INSERT INTO {s}.pointer_target_rejection (lemma_lc, reason) VALUES (%s, %s)
                            ON CONFLICT (lemma_lc) DO UPDATE SET reason = EXCLUDED.reason,
                                                                 rejected_at = now()""", rejections)
        for c in keep:
            cur.execute(
                f"""INSERT INTO {s}.word
                    (lemma, definition, part_of_speech, ipa, synonyms, etymology, definition_source,
                     first_added, variant_flag_reason, variant_flag_note, variant_flagged_at,
                     pointer_target_of, pointer_target_added_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE,
                            NULLIF(%s,''), NULLIF(%s,''), CASE WHEN %s <> '' THEN now() END,
                            %s, now())
                    ON CONFLICT (lemma_lc) DO NOTHING""",
                (c.lemma, c.definition, c.part_of_speech, c.ipa, list(c.synonyms), c.etymology,
                 c.definition_source, c.variant_flag_reason, c.variant_flag_note, c.variant_flag_reason,
                 ", ".join(rare[c.lemma]["sources"])))
    conn.commit()
    return stats

