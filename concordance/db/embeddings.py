"""Definition and fastText embeddings."""

from __future__ import annotations

from .core import DEFAULT_SCHEMA, _safe_schema


def compute_definition_embeddings(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                                  limit: int = 0, batch: int = 64) -> dict:
    """Embed definition_text(definition, synonyms, sentence) into
    word_embedding.definition_vector for every active word. Resumable via
    only_missing (scale-ready — see embed.py's module docstring for why this
    is per-word/incremental rather than a full-corpus recompute)."""
    from pgvector.psycopg import register_vector
    from .. import embed as _embed
    s = _safe_schema(schema)
    register_vector(conn)
    where = (f"NOT EXISTS (SELECT 1 FROM {s}.word_embedding e "
             f"WHERE e.word_id = w.id AND e.definition_vector IS NOT NULL) AND ") if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT w.id, w.lemma, w.definition, w.synonyms, w.sentence "
                    f"FROM {s}.word w WHERE {where}w.active" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"words": len(rows), "embedded": 0, "skipped_no_text": 0}
    resolved = []
    for wid, lemma, definition, synonyms, sentence in rows:
        text = _embed.definition_text(definition, synonyms, sentence)
        if text is None:
            stats["skipped_no_text"] += 1
            continue
        resolved.append((wid, *text))
    if not resolved:
        return stats

    embedder = _embed.DefinitionEmbedder()
    with conn.cursor() as cur:
        for i in range(0, len(resolved), batch):
            chunk = resolved[i : i + batch]
            vectors = embedder.encode([text for _, text, _ in chunk])
            for (wid, _text, source), vec in zip(chunk, vectors):
                cur.execute(
                    f"""INSERT INTO {s}.word_embedding (word_id, definition_vector, definition_model, definition_source, updated_at)
                        VALUES (%s,%s,%s,%s, now())
                        ON CONFLICT (word_id) DO UPDATE SET
                            definition_vector=EXCLUDED.definition_vector,
                            definition_model=EXCLUDED.definition_model,
                            definition_source=EXCLUDED.definition_source,
                            updated_at=now()""",
                    (wid, vec, embedder.model_name, source))
                stats["embedded"] += 1
            conn.commit()
            print(f"  ...{stats['embedded']}/{len(resolved)} embedded")
    # Deterministic release, not left to implicit GC timing -- same reasoning
    # as fill_definitions' matching comment, but for sentence-transformers'
    # torch-backed model instead of a llama-cpp one: dropping the reference
    # alone doesn't return its CUDA memory pool to the driver, empty_cache()
    # does. compute_fasttext_embeddings (the step right after this one in
    # `maintain`) is CPU-only (fasttext has no GPU support at all), so this
    # is really about not leaving VRAM held for whatever runs after that.
    del embedder
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return stats


def compute_fasttext_embeddings(conn, schema: str = DEFAULT_SCHEMA, model_path: str = "",
                                only_missing: bool = True, limit: int = 0) -> dict:
    """Compute word_embedding.fasttext_vector for every active word via a
    trained FastText model (see `concordance train-fasttext`). Unlike
    definition embedding, this never skips a word for lack of text — FastText
    composes a vector from any lemma's subwords, including words never seen
    during training."""
    from pgvector.psycopg import register_vector
    from .. import embed as _embed
    s = _safe_schema(schema)
    register_vector(conn)
    where = (f"NOT EXISTS (SELECT 1 FROM {s}.word_embedding e "
             f"WHERE e.word_id = w.id AND e.fasttext_vector IS NOT NULL) AND ") if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT w.id, w.lemma FROM {s}.word w WHERE {where}w.active" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"words": len(rows), "embedded": 0}
    if not rows:
        return stats

    embedder = _embed.FastTextEmbedder(model_path)
    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(rows, 1):
            vec = embedder.vector(lemma)
            cur.execute(
                f"""INSERT INTO {s}.word_embedding (word_id, fasttext_vector, fasttext_model, updated_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        fasttext_vector=EXCLUDED.fasttext_vector,
                        fasttext_model=EXCLUDED.fasttext_model,
                        updated_at=now()""",
                (wid, vec, embedder.model_path))
            stats["embedded"] += 1
            if i % 500 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} embedded")
    conn.commit()
    return stats
