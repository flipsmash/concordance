"""Book genre classifier -- tags each book with 1+ labels from a fixed list,
using its title/author plus Gutenberg RDF subject/bookshelf hints (see
archive_metadata.genre_hints_from_rdf) as LLM input. Structured like
classify.py's USAS word classifier: a fixed assignable-label set, a
"hint" injected as a candidate the model prunes/confirms against rather
than treats as authoritative, retry-on-omission, temperature 0, and
every returned label validated against the fixed list so a hallucinated
tag never lands in the DB.

The one rule classify.py's word tagger didn't need: a book should not
carry BOTH a generic content tag (Fiction/Nonfiction) and a more specific
tag on the same axis (Science Fiction implies fiction; don't also add
"Fiction"). That's enforced twice -- as a prompt instruction (to bias the
model away from redundant output) AND, authoritatively, as a deterministic
post-filter (apply_redundancy_filter) -- because a prompt instruction
alone is not reliable enough to skip the code-level check, same reasoning
classify.py's own _validate exists despite the prompt already asking for
"fewest codes". Format (Comics/Graphic Novels/Manga/Poetry), audience
(Children's/Young Adult), and canon-status (Classics) tags are a book's
own book_similarity-vector or shelf status, not a fiction/nonfiction
lens, ("Persepolis" is a graphic memoir) so they're never touched by the filter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import db
from .archive_metadata import extract_gutenberg_id, fetch_gutenberg_rdf, year_to_era
from .config import Config

GENRE_LIST = (
    "Art", "Biography", "Business", "Chick Lit", "Children's", "Christian", "Classics",
    "Comics", "Contemporary", "Cookbooks", "Crime", "Drama", "Fantasy", "Fiction",
    "Gay and Lesbian", "Graphic Novels", "Historical Fiction", "History", "Horror",
    "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal",
    "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction",
    "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult",
)
_CANONICAL = {g.lower(): g for g in GENRE_LIST}

# The one axis the redundancy rule applies to. Christian/Gay and Lesbian/Music
# are deliberately left out of both sets -- each spans fiction AND nonfiction
# in practice (Christian fiction vs. Christian devotional; a memoir vs. a
# novel with LGBTQ leads), so there's no safe generic label to drop for them.
_FICTION_SPECIFIC = frozenset({
    "Chick Lit", "Contemporary", "Crime", "Drama", "Fantasy", "Historical Fiction",
    "Horror", "Humor and Comedy", "Mystery", "Paranormal", "Romance", "Science Fiction",
    "Suspense", "Thriller",
})
_NONFICTION_SPECIFIC = frozenset({
    "Art", "Biography", "Business", "Cookbooks", "History", "Memoir", "Philosophy",
    "Psychology", "Religion", "Science", "Self Help", "Spirituality", "Sports", "Travel",
})

_REFERENCE = "\n".join(GENRE_LIST)
_SYSTEM = (
    "You tag books with genre labels from this fixed list ONLY:\n\n" + _REFERENCE + "\n\n"
    "Assign every label that clearly applies to the book -- a book may have "
    "several if it genuinely spans genres (e.g. both Science Fiction and "
    "Fantasy). Do NOT also assign the generic 'Fiction' or 'Nonfiction' label "
    "when a more specific label from the list already applies (Science "
    "Fiction implies fiction; do not also add 'Fiction'). Only fall back to "
    "the bare 'Fiction'/'Nonfiction' label when nothing more specific fits. "
    "Comics, Graphic Novels, Manga, Poetry, Children's, Young Adult, and "
    "Classics describe format/audience/canon status, not fiction-vs-"
    "nonfiction, so they may combine freely with anything else on the list. "
    "'hints' are subject/shelf tags from a library catalog, not "
    "authoritative -- keep what fits the book, ignore what doesn't (a "
    "cataloging shelf like \"Category: Novels\" is not one of the allowed "
    "labels), and add labels the hints miss. If nothing on the list clearly "
    "applies, return an empty list rather than guessing.\n"
    'Output ONLY a JSON array, one object per input book IN ORDER, no prose: '
    '[{"b":<id>,"g":["Label",...]}]. Include EVERY input book exactly once.'
)


def apply_redundancy_filter(genres: list[str]) -> list[str]:
    """Drops the generic Fiction/Nonfiction tag when a same-axis specific
    tag is also present. Order-preserving, safe to call on already-clean
    input (a no-op if neither generic tag is present)."""
    tags = set(genres)
    if "Fiction" in tags and tags & _FICTION_SPECIFIC:
        tags.discard("Fiction")
    if "Nonfiction" in tags and tags & _NONFICTION_SPECIFIC:
        tags.discard("Nonfiction")
    return [g for g in genres if g in tags]


def _validate(raw_genres) -> list[str]:
    """Case-insensitive match against GENRE_LIST, normalized to canonical
    casing; unrecognized labels dropped, duplicates removed, order kept."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_genres if isinstance(raw_genres, list) else []:
        canonical = _CANONICAL.get(str(raw).strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _prompt_items(items: list[dict]) -> list[dict]:
    return [
        {"b": it["_id"], "title": it["title"], "author": it.get("author") or "",
         "hints": (it.get("hints") or [])[:12]}
        for it in items
    ]


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


class GenreClassifier:
    _MAX_PASSES = 3

    def __init__(self, cfg: Config | None = None, model_path: str | None = None):
        from llama_cpp import Llama
        cfg = cfg or Config()
        mp = model_path or cfg.model_path
        if not mp or not Path(mp).exists():
            raise RuntimeError(f"genre classifier model not found: {mp!r}")
        self.llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)
        self.batch = 8  # smaller than classify.py's 15 -- each item carries up to 12 hint strings + title/author

    def close(self) -> None:
        self.llm.close()

    def classify(self, items: list[dict]) -> dict[int, list[str]]:
        """book id -> validated, redundancy-filtered genre tags."""
        result: dict[int, list[str]] = {}
        for i in range(0, len(items), self.batch):
            self._classify_batch(items[i : i + self.batch], result)
        return result

    def _classify_batch(self, batch: list[dict], result: dict) -> None:
        pending = list(batch)
        for _ in range(self._MAX_PASSES):
            got = self._query(pending)
            for book_id, tags in got.items():
                result[book_id] = tags
            pending = [it for it in batch if it["_id"] not in result]
            if not pending:
                break
        for it in pending:                       # unresolved after retries: leave empty
            result.setdefault(it["_id"], [])

    def _query(self, items: list[dict]) -> dict[int, list[str]]:
        payload = json.dumps(_prompt_items(items), ensure_ascii=False)
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": payload}],
            temperature=0.0, max_tokens=len(items) * 60 + 128)
        parsed = _parse(out["choices"][0]["message"]["content"])
        got = {}
        for obj in parsed:
            try:
                book_id = int(obj.get("b"))
            except (TypeError, ValueError):
                continue
            got[book_id] = apply_redundancy_filter(_validate(obj.get("g", [])))
        return got


def classify_and_store_genres(conn, schema: str, cfg: Config | None = None, limit: int = 0,
                              only_missing: bool = False, batch: int | None = None,
                              commit_every: int = 200, fetch_rdf: bool = True,
                              delay: float = 0.3) -> dict:
    """Classify every book in {schema}.book and write tags to book_genre.
    Idempotent for the LLM-sourced rows (cleared and rewritten each run,
    same as classify_and_store's word_category handling), chunked commits
    and a per-chunk still-exists re-check for the identical crash-safety
    and concurrent-delete reasons documented on classify_and_store.

    Also backfills publication_year/era (db.set_book_publication_info,
    NULL-only) for any book whose RDF gets fetched here anyway -- one
    network round trip serving both purposes, see
    archive_metadata.fetch_gutenberg_rdf's own docstring for why this
    matters at the historical-backlog's scale (most of it has never had
    ANY successful Gutenberg RDF lookup)."""
    ssch = db._safe_schema(schema)
    with conn.cursor() as cur:
        where = (" WHERE NOT EXISTS (SELECT 1 FROM " + ssch + ".book_genre bg WHERE bg.book_id=book.id)"
                 if only_missing else "")
        cur.execute(f"SELECT id, title, author, archive_path, publication_year FROM {ssch}.book book"
                    + where + " ORDER BY id" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    items = [{"_id": r[0], "title": r[1], "author": r[2] or "", "archive_path": r[3],
              "publication_year": r[4], "hints": []} for r in rows]
    clf = GenreClassifier(cfg)  # loaded once, reused for every chunk below
    if batch:
        clf.batch = batch

    stats = {"books": len(items), "classified": 0, "assignments": 0, "vanished": 0,
             "rdf_fetched": 0, "publication_info_backfilled": 0}

    with conn.cursor() as cur:
        if only_missing:
            pass
        elif limit:
            cur.execute(f"DELETE FROM {ssch}.book_genre WHERE source='llm' AND book_id = ANY(%s)",
                        ([r[0] for r in rows],))
        else:
            cur.execute(f"DELETE FROM {ssch}.book_genre WHERE source='llm'")
    conn.commit()

    for start in range(0, len(items), max(1, commit_every)):
        chunk = items[start:start + commit_every]

        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {ssch}.book WHERE id = ANY(%s)", ([it["_id"] for it in chunk],))
            still_present = {r[0] for r in cur.fetchall()}
        vanished = [it for it in chunk if it["_id"] not in still_present]
        if vanished:
            stats["vanished"] += len(vanished)
            chunk = [it for it in chunk if it["_id"] in still_present]
        if not chunk:
            continue

        if fetch_rdf:
            # One cursor for the whole fetch+update loop, no commit here --
            # the chunk-level commit below covers these UPDATEs too, so a
            # crash mid-chunk can't leave publication_year written for a
            # book whose genre rows never made it (see
            # db.set_book_publication_info's own docstring).
            with conn.cursor() as cur:
                for it in chunk:
                    path = it["archive_path"]
                    if not path or not path.lower().endswith(".txt") or not Path(path).is_file():
                        continue
                    raw = Path(path).read_text(encoding="utf-8", errors="replace")
                    gutenberg_id = extract_gutenberg_id(raw)
                    if not gutenberg_id:
                        continue
                    meta = fetch_gutenberg_rdf(gutenberg_id)
                    stats["rdf_fetched"] += 1
                    it["hints"] = meta["genre_hints"]
                    if it["publication_year"] is None and (meta["publication_year"] or meta["publication_era"]):
                        era = meta["publication_era"] or (
                            year_to_era(meta["publication_year"]) if meta["publication_year"] else None)
                        db.set_book_publication_info(cur, it["_id"], meta["publication_year"], era, schema)
                        stats["publication_info_backfilled"] += 1
                    time.sleep(delay)

        tags = clf.classify(chunk)
        with conn.cursor() as cur:
            for it in chunk:
                # apply_redundancy_filter also runs inside GenreClassifier._query,
                # but enforced again here (idempotent) so the rule holds
                # structurally for whatever lands in book_genre, not just
                # for output that happened to pass through _query.
                genres_for_book = apply_redundancy_filter(tags.get(it["_id"], []))
                if not genres_for_book:
                    continue
                stats["classified"] += 1
                for g in genres_for_book:
                    cur.execute(
                        f"""INSERT INTO {ssch}.book_genre (book_id, genre, source, confidence)
                            VALUES (%s,%s,'llm',%s)
                            ON CONFLICT (book_id, genre) DO UPDATE SET
                                source=EXCLUDED.source, confidence=EXCLUDED.confidence""",
                        (it["_id"], g, 0.8))
                    stats["assignments"] += 1
        conn.commit()
        print(f"  ...{min(start + commit_every, len(items))}/{len(items)} books classified "
              f"({stats['classified']} tagged, {stats['rdf_fetched']} RDF fetches, "
              f"{stats['vanished']} vanished mid-run)")
    clf.close()  # deterministic release -- see classify_and_store's matching comment
    return stats
