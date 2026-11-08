"""Repair definitions with words missing from the middle.

The local Wiktionary table (vocab.wiktionary, carried over from the earlier
vocab project) was built by a parser that dropped linked words instead of
unwrapping them: statal's "Pertaining to a [[state]] of the American Union"
became "Pertaining to a of the American Union", prologist's "The [[author]]
or [[speaker]] of a [[prologue]]" became "The or of a.". Every word whose
definition came from that table can carry the gap.

The current Wiktionary dump (data/wiktextract-en.jsonl.gz) has the full
glosses. A clause is repaired only when it is one of the word's dump glosses
with words missing from the middle -- the parser's signature. A clause that
is a gloss with words only added at the end is a later Wiktionary edit, not
damage, and is left alone; so is anything that matches no gloss.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

from .db import _safe_schema
from .db.sync import _invalidate_definition_dependents

_TOKEN = re.compile(r"[\w'’-]+|[^\w\s]")
_SPLIT = re.compile(r"(; | — )")
_MAX_MISSING = 8
_DANGLING = {"a", "an", "the", "of", "to", "by", "in", "with", "for", "from", "and", "or", "on", "at", "as"}


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


_ARTICLES = {"a", "an", "the"}
# the dump now glosses a linked word inline: "a wharf (“artificial landing
# place for ships…”)" -- an aside the old text never had, so it's dropped
_ASIDE = re.compile(r"\s*\(“[^”]*”\)")


def _inner_gap(short: list[str], full: list[str], links: frozenset[str]) -> bool:
    """`short` is `full` with 1.._MAX_MISSING of its LINKED words removed (an
    article may go with them), at least one between two words `short` keeps
    -- the old parser's signature. Words missing that weren't links mean a
    later rewording of the gloss, not damage."""
    missing = len(full) - len(short)
    if not 0 < missing <= _MAX_MISSING or not short or (len(short) == 1 and short[0] not in _ARTICLES):
        return False
    i = 0
    inner = False
    dropped_link = False
    for tok in full:
        if i < len(short) and tok == short[i]:
            i += 1
            continue
        if tok in links:
            dropped_link = True
        elif tok not in _ARTICLES:
            return False
        if 0 < i < len(short):
            inner = True
    if not dropped_link:
        return False
    # a gloss can't end on an article or preposition ("A member of a."):
    # a gap at the very end counts too
    return i == len(short) and (inner or (missing and short[-1] in _DANGLING))


def repair(definition: str, glosses: list[tuple[str, frozenset[str]]]) -> str | None:
    """`definition` with each gapped clause replaced by the full gloss it
    came from, or None when nothing needed repair."""
    parts = _SPLIT.split(definition)
    texts, seps = parts[0::2], parts[1::2]
    candidates = [(g, _tokens(g.rstrip(".")), links)
                  for g, links in ((_ASIDE.sub("", g0), links0) for g0, links0 in glosses)]
    changed = False
    out: list[str] = []
    i = 0
    while i < len(texts):
        fixed = None
        # a gloss may itself contain "; ", so try runs of up to 3 clauses
        for span in (1, 2, 3):
            if i + span > len(texts):
                break
            text = "".join(t + (seps[i + k] if k < span - 1 else "") for k, t in enumerate(texts[i:i + span]))
            short = _tokens(text.rstrip("."))
            match = next((g for g, full, links in candidates if _inner_gap(short, full, links)), None)
            if match:
                trail = "." if text.rstrip().endswith(".") and not match.rstrip().endswith(".") else ""
                fixed = (match.rstrip() + trail, span)
                break
        if fixed:
            out.append(fixed[0])
            changed = True
            i += fixed[1]
        else:
            out.append(texts[i])
            i += 1
        if i < len(texts):
            out.append(seps[i - 1])
    return "".join(out) if changed else None


# --- the data ------------------------------------------------------------------

def load_glosses(dump_path: str | Path, needed: set[str],
                 progress=None) -> dict[str, list[tuple[str, frozenset[str]]]]:
    """Every English gloss in the dump for each (lowercased) word in `needed`,
    with the words its sense links. One pass over the 2.7GB dump, minutes."""
    out: dict[str, list[tuple[str, frozenset[str]]]] = {}
    with gzip.open(dump_path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if '"lang_code": "en"' not in line:
                continue
            obj = json.loads(line)
            word = (obj.get("word") or "").lower()
            if obj.get("lang_code") != "en" or word not in needed:
                continue
            seen = out.setdefault(word, [])
            for sense in obj.get("senses", []):
                links = frozenset(t for link in sense.get("links") or [] if link
                                  for t in _tokens(str(link[0])))
                for g in sense.get("glosses") or []:
                    seen.append((g, links))
            if progress and n % 1_000_000 == 0:
                progress(n)
    return out


def find_repairs(conn, schema: str, glosses: dict[str, list[str]]) -> dict:
    """{"table": [(row id, term, old, new)], "words": [(word id, lemma, old, new)]}:
    the local Wiktionary rows and the active words' definitions to repair. A
    word's definition is matched against its own glosses and those of any
    word it points at ("Variant spelling of abnormality — The or quality…").
    Hand-edited definitions are left alone."""
    from . import crossref
    s = _safe_schema(schema)
    out = {"table": [], "words": []}
    with conn.cursor() as cur:
        cur.execute("SELECT id, term, definition FROM vocab.wiktionary WHERE coalesce(definition, '') <> ''")
        for rid, term, old in cur.fetchall():
            new = repair(old, glosses.get(term.lower(), []))
            if new:
                out["table"].append((rid, term, old, new))
        cur.execute(f"""SELECT id, lemma, definition FROM {s}.word
                        WHERE active AND coalesce(definition, '') <> '' AND definition_edited_by IS NULL""")
        for wid, lemma, old in cur.fetchall():
            pool = list(glosses.get(lemma.lower(), []))
            for target in crossref.pointer_target_phrases(old):
                pool += glosses.get(target, [])
            new = repair(old, pool)
            if new:
                out["words"].append((wid, lemma, old, new))
    return out


def apply_repairs(conn, schema: str, repairs: dict, backup_csv: Path) -> None:
    """Write the repairs (old values saved to `backup_csv` first) and clear
    what each changed word derived from its old text (quiz definition,
    categories, definition embedding -- sync._invalidate_definition_dependents).
    Commits."""
    s = _safe_schema(schema)
    backup_csv.parent.mkdir(parents=True, exist_ok=True)
    with backup_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["where", "id", "term", "old_definition", "new_definition"])
        for rid, term, old, new in repairs["table"]:
            w.writerow(["vocab.wiktionary", rid, term, old, new])
        for wid, lemma, old, new in repairs["words"]:
            w.writerow([f"{s}.word", wid, lemma, old, new])
    with conn.cursor() as cur:
        # a repaired row can come out identical to a complete row the table
        # already has (term, part of speech, definition are unique): then the
        # damaged copy just goes
        for rid, _, _, new in repairs["table"]:
            cur.execute("""SELECT 1 FROM vocab.wiktionary o JOIN vocab.wiktionary r ON r.id = %s
                           WHERE o.id <> r.id AND o.term = r.term AND o.definition = %s
                             AND o.part_of_speech IS NOT DISTINCT FROM r.part_of_speech""", (rid, new))
            if cur.fetchone():
                cur.execute("DELETE FROM vocab.wiktionary WHERE id = %s", (rid,))
            else:
                cur.execute("UPDATE vocab.wiktionary SET definition = %s WHERE id = %s", (new, rid))
        for wid, _, _, new in repairs["words"]:
            cur.execute(f"UPDATE {s}.word SET definition = %s, updated_at = now() WHERE id = %s", (new, wid))
            _invalidate_definition_dependents(cur, s, wid)
    conn.commit()


# --- what the dump can't repair: a review list ----------------------------------

# Article/preposition directly before punctuation or another function word --
# how a dropped link reads ("a of the", "the,"); not after an apostrophe (Qur'an.)
_GAP_SHAPE = re.compile(
    r"(?<!['’\w])(a|an|the)\s+(of|in|to|for|with|from|by|on|at|or|and|rather|than|that|which|as|is|are|was|were)\b"
    r"|(?<!['’\w])(a|an|the)\s*[.,;:)]|\(\s*\)|\(\s*[,;]|,\s*,", re.I)
_STOP = {"a", "an", "the", "of", "to", "in", "or", "and", "for", "with", "by", "on", "at", "as", "from", "that", "which"}


def review_rows(conn, schema: str, glosses: dict) -> list[dict]:
    """Active words whose local-Wiktionary definition still has a gap after
    repair -- Wiktionary has reworded the sense since, so nothing matches
    mechanically -- each with the closest current gloss as a proposal (blank
    when nothing is close). For data/review/ and `apply-reviewed`."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma, definition FROM {s}.word
                        WHERE active AND coalesce(definition, '') <> '' AND definition_edited_by IS NULL
                          AND definition_source ILIKE %s ORDER BY lemma_lc""", ("%wiktionary (db)%",))
        rows = cur.fetchall()
    out = []
    for wid, lemma, definition in rows:
        if not _GAP_SHAPE.search(definition):
            continue
        words = {t for t in _tokens(definition) if t.isalpha() and t not in _STOP}
        best, score = "", 0.0
        for g, _ in glosses.get(lemma.lower(), []):
            g = _ASIDE.sub("", g)
            gw = {t for t in _tokens(g) if t.isalpha() and t not in _STOP}
            sim = len(words & gw) / len(words | gw) if words | gw else 0.0
            if sim > score:
                best, score = g, sim
        out.append({"word_id": wid, "word": lemma, "damaged_definition": definition,
                    "proposed_definition": best if score >= 0.2 else "", "approve (y/n)": ""})
    return out


def write_review(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word_id", "word", "damaged_definition", "proposed_definition",
                                          "approve (y/n)"])
        w.writeheader()
        w.writerows(rows)
