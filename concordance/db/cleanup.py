"""Sweeps that cast out non-vocabulary: script/dialect/archaic respellings, foreign words,
non-English contexts, and clearing stale review flags."""

from __future__ import annotations

from pathlib import Path

from .core import DEFAULT_SCHEMA, _safe_schema
from .reference import english_reference_terms, foreign_only_langs


def clean_script_variants(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False) -> dict:
    """`concordance clean-script-variants`: sweep ACTIVE words with a non-ASCII
    lemma -- the one place junk concentrates among the words Google Books
    never shows in recent print (Greek quotations, þ/ȝ Middle English, poetic
    accents on common words, accented duplicates of words already present).

    Per word, one of:
      - cast out (active=false) when validity_score.script_reject_reason says
        the spelling alone proves it isn't vocabulary (the same gate ingest
        now applies up front);
      - dedupe when its accent/ligature-folded spelling is ALREADY an active
        word (vicuña/vicuna): one survives, the other is cast out as
        `script_duplicate`, and the loser's book links are copied onto the
        survivor so book pages/stats don't lose the word. The ASCII row
        survives unless only the accented one carries quiz history or a
        definition;
      - otherwise flagged `script_review` (NOT deactivated): a real rare word
        in a variant spelling (mélange, uræus, crispèd) -- keep-bias says a
        human decides, via the review list's flag filter.

    Every cast-out records its kind in variant_flag_reason (distinct
    script_* values, so the deleted-as-difficulty-signal dataset can exclude
    them -- these weren't pruned for being easy). Soft and reversible like
    every other removal here. Idempotent: words already carrying a script_*
    flag are skipped. Dry run unless `apply`."""
    from ..config import Config
    from ..validity_score import fold_spelling, script_reject_reason
    s = _safe_schema(schema)
    min_zipf = Config().min_zipf
    history_sql = " + ".join(
        f"(SELECT count(*) FROM {s}.{tbl} x WHERE x.word_id = w.id)"
        for tbl in ("quiz_answer", "word_review_schedule", "word_set_item"))
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma, coalesce(w.definition,'') <> '', {history_sql}
                        FROM {s}.word w WHERE w.active AND NOT w.admin_suggested AND NOT w.lemma ~ '^[\\x01-\\x7f]*$'
                          -- already swept: an active script_* word is either
                          -- awaiting review or was reactivated by a human --
                          -- never re-cast-out behind their back
                          AND coalesce(w.variant_flag_reason, '') NOT LIKE 'script%%'
                        ORDER BY w.id""")
        rows = cur.fetchall()
        actions: list[tuple] = []          # (kind, loser_id, lemma, note, survivor_id|None)
        for wid, lemma, has_def, hist in rows:
            script = script_reject_reason(lemma, min_zipf)
            if script:
                actions.append((script[0], wid, lemma, script[1], None))
                continue
            folded = fold_spelling(lemma)
            cur.execute(f"""SELECT w.id, coalesce(w.definition,'') <> '', {history_sql}
                            FROM {s}.word w WHERE w.active AND w.lemma_lc = %s""", (folded,))
            twin = cur.fetchone()
            if twin:
                tid, t_def, t_hist = twin
                keep_accented = (hist and not t_hist) or (has_def and not t_def and not t_hist)
                if keep_accented:
                    actions.append(("script_duplicate", tid, folded,
                                    f"duplicate of accented spelling '{lemma}'", wid))
                else:
                    actions.append(("script_duplicate", wid, lemma,
                                    f"accented/ligature spelling of '{folded}'", tid))
            else:
                actions.append(("script_review", wid, lemma, f"variant spelling of '{folded}'?", None))

        if apply:
            for kind, wid, _lemma, note, survivor in actions:
                if kind == "script_review":
                    cur.execute(f"""UPDATE {s}.word SET variant_flag_reason=%s, variant_flag_note=%s,
                                        variant_flagged_at=now()
                                    WHERE id=%s AND variant_flag_reason IS NULL""", (kind, note, wid))
                    continue
                _cast_out_variant(cur, s, wid, kind, note, survivor)
            conn.commit()

    from collections import Counter
    return {"scanned": len(rows), "counts": dict(Counter(a[0] for a in actions)), "actions": actions}


def _cast_out_variant(cur, s: str, wid: int, kind: str, note: str, survivor: int | None) -> None:
    """Soft-delete a spelling variant, recording why in variant_flag_*; when
    it duplicates a surviving word, copy its book links onto the survivor
    first so book pages/stats don't lose the word."""
    if survivor is not None:
        cur.execute(f"""INSERT INTO {s}.word_book (word_id, book_id)
                        SELECT %s, book_id FROM {s}.word_book WHERE word_id=%s
                        ON CONFLICT DO NOTHING""", (survivor, wid))
    cur.execute(f"""UPDATE {s}.word SET active=false, variant_flag_reason=%s,
                        variant_flag_note=%s, variant_flagged_at=now(), updated_at=now()
                    WHERE id=%s""", (kind, note, wid))


def clean_dialect_spellings(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False) -> dict:
    """`concordance clean-dialect-spellings`: active words whose definition is
    purely a dialect/eye-dialect respelling cross-reference (bettah ->
    "Pronunciation spelling of better.") -- see
    crossref.dialect_respelling_target, which detects by definition,
    never by word shape. Kinds dialect_common_variant / dialect_duplicate
    (cast out) and dialect_review (flag only) -- see _sweep_respellings."""
    from ..crossref import dialect_respelling_target
    return _sweep_respellings(
        conn, schema, "dialect", apply=apply,
        prefilter="definition ~* '(spelling|form) of'",
        detect=lambda lemma, definition, min_zipf: (dialect_respelling_target(definition), "definition"))


def clean_archaic_spellings(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False) -> dict:
    """`concordance clean-archaic-spellings`: archaic inflections of a verb
    (thinketh, findest -- crossref.archaic_inflection_target, by
    definition + ending) and early-printing u-for-v spellings (reuelation,
    nerue -- early_modern_uv_target) aren't distinct vocabulary; the modern
    word is (design rule 3). Kinds archaic_common_variant /
    archaic_duplicate (cast out) and archaic_review (flag only). Also any
    word whose whole definition is "Obsolete/Archaic spelling|form of X"
    (crossref.obsolete_spelling_target)."""
    from ..crossref import archaic_inflection_target, obsolete_spelling_target
    from ..validity_score import early_modern_uv_target
    return _sweep_respellings(
        conn, schema, "archaic", apply=apply,
        prefilter=("(lemma_lc ~ '(eth|est|th|st)$' OR lemma_lc ~ '[aeioulr]u[aeiou]'"
                   " OR definition ~* '(obsolete|archaic)[a-z ]* (spelling|form) of')"),
        detect=lambda lemma, definition, min_zipf: (
            (t, "definition") if (t := archaic_inflection_target(lemma, definition)
                                  or obsolete_spelling_target(definition))
            else (early_modern_uv_target(lemma, min_zipf), "spelling")))


# Definition sources that fuzzy-match and can return a different word's
# gloss; their "X spelling of Y" claims are review-only (see _sweep_respellings).
_UNTRUSTED_DEF_SOURCES = frozenset({"datamuse", "Web (LLM-extracted)", "corpus"})

# Real respelling pairs score >= ~0.73 (palkee/palki, polliwig/pollywog);
# a bogus gloss's target scores far lower (shakester/shiksa 0.40).
_TWIN_MIN_SIMILARITY = 0.6


def _sweep_respellings(conn, schema: str, prefix: str, *, apply: bool, prefilter: str, detect) -> dict:
    """Shared core of the clean-*-spellings sweeps. `detect(lemma, definition,
    min_zipf)` returns (standard word an active word merely respells | None,
    evidence) where evidence is "definition" or "spelling". Definition
    evidence from a fuzzy-lookup source (_UNTRUSTED_DEF_SOURCES) is never
    enough to cast out -- datamuse hands back a NEARBY word's gloss
    (spurcidical -> "Obsolete form of suicidal", phytosophy -> "...of
    philosophy") -- so those always go to <prefix>_review. Per respelling:
      - <prefix>_common_variant  cast out: the standard word is common enough
                                 to sit above the frequency floor (rule 3);
      - <prefix>_duplicate       cast out: the standard word is itself an
                                 active, non-respelling entry (yander ->
                                 yonder) spelled recognizably alike; book
                                 links move to it. Dissimilar pairs go to
                                 review instead: a bogus source gloss
                                 (shakester -> "shiksa") must not move book
                                 links onto an unrelated word;
      - <prefix>_review          flagged only: a rare standard word not in the
                                 list -- keep-bias, a human decides.
    Soft/reversible, recorded in variant_flag_*; idempotent (<prefix>_* flags
    skipped); dry run unless `apply`."""
    from difflib import SequenceMatcher
    from wordfreq import zipf_frequency
    from ..config import Config
    s = _safe_schema(schema)
    min_zipf = Config().min_zipf
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma, definition, coalesce(definition_source, '') FROM {s}.word
                        WHERE active AND NOT admin_suggested AND {prefilter}
                          AND coalesce(variant_flag_reason, '') NOT LIKE %s
                        ORDER BY id""", (f"{prefix}%",))
        found = []
        for wid, lemma, d, src in cur.fetchall():
            tg, evidence = detect(lemma.lower(), d, min_zipf)
            if tg:
                found.append((wid, lemma, tg, evidence == "definition" and src in _UNTRUSTED_DEF_SOURCES, src))
        respellings = {lemma.lower() for _, lemma, *_ in found}
        label = "dialect" if prefix == "dialect" else prefix
        actions: list[tuple] = []
        for wid, lemma, target, untrusted, src in found:
            z = zipf_frequency(target, "en")
            if untrusted:
                actions.append((f"{prefix}_review", wid, lemma,
                                f"{label} spelling of '{target}'? (per {src} -- unverified)", None))
                continue
            if z >= min_zipf:
                actions.append((f"{prefix}_common_variant", wid, lemma,
                                f"{label} spelling of common '{target}' (zipf {z:.1f})", None))
                continue
            twin = None
            if (target not in respellings     # never "keep" a word that's itself a respelling
                    and SequenceMatcher(None, lemma.lower(), target).ratio() >= _TWIN_MIN_SIMILARITY):
                cur.execute(f"SELECT id FROM {s}.word WHERE active AND lemma_lc = %s", (target,))
                twin = cur.fetchone()
            if twin:
                actions.append((f"{prefix}_duplicate", wid, lemma, f"{label} spelling of '{target}'", twin[0]))
            else:
                actions.append((f"{prefix}_review", wid, lemma, f"{label} spelling of '{target}'", None))

        if apply:
            for kind, wid, _lemma, note, survivor in actions:
                if kind.endswith("_review"):
                    cur.execute(f"""UPDATE {s}.word SET variant_flag_reason=%s, variant_flag_note=%s,
                                        variant_flagged_at=now()
                                    WHERE id=%s""", (kind, note, wid))
                else:
                    _cast_out_variant(cur, s, wid, kind, note, survivor)
            conn.commit()

    from collections import Counter
    return {"scanned": len(found), "counts": dict(Counter(a[0] for a in actions)), "actions": actions}


def clear_stale_foreign_flags(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False) -> dict:
    """`concordance clear-foreign-flags`: the old cross-language-Zipf
    heuristic (validity_score.foreign_language_hint) flagged ~2k active words
    variant_flag_reason='foreign_language', mostly real English (haft,
    glaive). Clear the flag wherever validity_score.english_evidence finds
    any sign of English use; the rest keep it for review. Dry run unless
    `apply`."""
    return _clear_stale_flags(conn, schema, "foreign_language", apply=apply, strict=False)


def clear_stale_misspelling_flags(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False) -> dict:
    """`concordance clear-misspelling-flags`: the SymSpell near-neighbor
    heuristic (validity_score.unambiguous_dominant_neighbor) flagged ~9k
    active words 'misspelling', mostly real words (titlark -> "titular",
    gasalier -> "cavalier"). Clear the flag where a CURATED source vouches
    for the word -- English Wiktionary (not a "misspelling of" gloss), 0
    Dict, an English dictionary definition, Webster list, WordNet. wordfreq
    doesn't count: common misspellings have web footprints."""
    return _clear_stale_flags(conn, schema, "misspelling", apply=apply, strict=True)


def _clear_stale_flags(conn, schema: str, reason: str, *, apply: bool, strict: bool) -> dict:
    from ..validity_score import english_evidence
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma_lc, coalesce(definition_source,''), coalesce(definition,'')
                        FROM {s}.word WHERE active AND variant_flag_reason = %s""", (reason,))
        rows = cur.fetchall()
    ref = english_reference_terms(conn, [l for _, l, _, _ in rows], exclude_misspelling_glosses=strict)
    cleared = [(wid, l, ev) for wid, l, src, d in rows
               if (ev := english_evidence(l, src, l in ref, use_wordfreq=not strict, definition=d))]
    if apply:
        with conn.cursor() as cur:
            cur.execute(f"""UPDATE {s}.word SET variant_flag_reason=NULL, variant_flag_note=NULL
                            WHERE id = ANY(%s) AND variant_flag_reason = %s""",
                        ([wid for wid, _, _ in cleared], reason))
        conn.commit()
    return {"flagged": len(rows), "cleared": len(cleared), "kept": len(rows) - len(cleared),
            "actions": cleared}


def clean_foreign_words(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False,
                        wikt_schema: str = "wikt") -> dict:
    """`concordance clean-foreign-words`: cast out active words that are
    foreign and not used in English (db.foreign_only_langs +
    validity_score.foreign_cast_out_reason; Latin and Greek never qualify).
    Cast out = active=false, validity_label='likely-artifact', and
    variant_flag_reason='foreign_word' with the languages in the note (a
    distinct reason, so the deleted-as-difficulty dataset can exclude them).
    Soft/reversible; idempotent (only active words, and never one flagged
    foreign_review -- a kept exception); dry run unless `apply`."""
    from ..validity_score import foreign_cast_out_reason
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        # foreign_review = a human (or a reviewed exception) decided to keep
        # it despite the evidence -- never re-cast-out behind their back
        cur.execute(f"""SELECT id, lemma_lc, coalesce(definition_source,'') FROM {s}.word
                        WHERE active AND NOT admin_suggested
                          AND coalesce(variant_flag_reason, '') <> 'foreign_review'""")
        rows = cur.fetchall()
    langs = foreign_only_langs(conn, [l for _, l, _ in rows], wikt_schema)
    actions = [(wid, lemma, note) for wid, lemma, src in rows
               if lemma in langs and (note := foreign_cast_out_reason(lemma, langs[lemma], src))]
    if apply:
        with conn.cursor() as cur:
            for wid, _lemma, note in actions:
                cur.execute(f"""UPDATE {s}.word SET active=false, validity_label='likely-artifact',
                                    variant_flag_reason='foreign_word', variant_flag_note=%s,
                                    variant_flagged_at=now(), updated_at=now()
                                WHERE id=%s""", (note, wid))
        conn.commit()
    return {"candidates": len(langs), "cast_out": len(actions), "actions": actions}


_CTX_CLASSIFIER = None


def _ctx_init(historic_terms):
    global _CTX_CLASSIFIER
    from ..context_lang import ContextClassifier
    _CTX_CLASSIFIER = ContextClassifier(historic_terms)


def _ctx_scan_book(item):
    """Worker: one book -> {word id: [(kind, detail), ...]} for its target words."""
    from ..context_lang import occurrences
    path, words = item
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return {wid: [_CTX_CLASSIFIER.classify(sent, words[wid]) for sent in sents]
            for wid, sents in occurrences(text, words).items()}


def clean_non_english_context(conn, schema: str = DEFAULT_SCHEMA, *, apply: bool = False,
                              labels=("likely-artifact", "uncertain"), workers: int = 8,
                              wikt_schema: str = "wikt") -> dict:
    """`concordance clean-context-language`: for active words with one of
    `labels`, find every use in their source books (book.archive_path) and
    classify each sentence (context_lang.ContextClassifier). A word is cast
    out only when it was found, at least one sentence was classifiable, and
    NONE was modern English -- i.e. every use is Old/Middle English or a
    foreign language. Too-short/ambiguous sentences never count either way;
    a word not found in its books is kept. Cast out = active=false,
    variant_flag_reason='non_english_context' with the languages seen.
    Dry run unless `apply`."""
    from collections import Counter, defaultdict
    from multiprocessing import Pool
    from ..context_lang import target_forms
    s, g = _safe_schema(schema), _safe_schema(wikt_schema)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{g}.historic_term",))
        if cur.fetchone()[0] is None:
            raise RuntimeError(f"{g}.historic_term missing -- run `concordance wiktionary-langs` first")
        cur.execute(f"SELECT term FROM {g}.historic_term")
        historic = {r[0] for r in cur.fetchall()}
        cur.execute(f"""SELECT w.id, w.lemma_lc, w.as_seen, b.archive_path
                        FROM {s}.word w
                        JOIN {s}.word_book wb ON wb.word_id = w.id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND NOT w.admin_suggested AND w.validity_label = ANY(%s) AND b.archive_path IS NOT NULL""",
                    (list(labels),))
        by_book: dict[str, dict[int, set[str]]] = defaultdict(dict)
        lemmas: dict[int, str] = {}
        for wid, lemma, seen, path in cur.fetchall():
            by_book[path][wid] = target_forms(lemma, seen)
            lemmas[wid] = lemma
    found: dict[int, list] = defaultdict(list)
    with Pool(workers, initializer=_ctx_init, initargs=(historic,)) as pool:
        for result in pool.imap_unordered(_ctx_scan_book, list(by_book.items()), chunksize=8):
            for wid, kinds in result.items():
                found[wid].extend(kinds)
    actions = []
    for wid, kinds in found.items():
        determinate = [(k, d) for k, d in kinds if k != "unknown"]
        if determinate and all(k != "english" for k, _ in determinate):
            langs = Counter("Middle/Old English" if k == "middle_english" else d for k, d in determinate)
            note = (f"all {len(determinate)} classifiable use(s) in source books are non-modern-English: "
                    + ", ".join(f"{lang} ×{n}" for lang, n in langs.most_common()))
            actions.append((wid, lemmas[wid], note))
    if apply:
        with conn.cursor() as cur:
            for wid, _lemma, note in actions:
                cur.execute(f"""UPDATE {s}.word SET active=false, variant_flag_reason='non_english_context',
                                    variant_flag_note=%s, variant_flagged_at=now(), updated_at=now()
                                WHERE id=%s""", (note, wid))
        conn.commit()
    return {"words": len(lemmas), "found_in_books": len(found), "cast_out": len(actions),
            "actions": actions}
