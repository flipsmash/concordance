"""Stage: semantic-distance vectors (§ post-ingestion).

Two independent per-word signals, both stored in ``word_embedding`` (one row
per word, two vector columns — see db.py's ``_VECTOR_DDL``) and queried on
demand via a pgvector HNSW index for nearest-neighbor lookups. Deliberately
**not** an all-pairs distance matrix: at 20k+ active words (and growing every
batch ingest) that's already 200M+ pairs, and only gets worse — a fixed-size
vector per word plus an ANN index turns "who's near word X" into an O(log N)
query instead of an O(N^2) precompute.

  * **definition embedding** — embeds the word's dictionary *gloss* (modern
    English text), not the rare headword itself. This is what makes the
    corpus's archaic/obsolete vocabulary (*cangue*, *bartizan*) tractable at
    all: the word is rare, but its definition is written in ordinary English
    any sentence-embedding model handles well. Falls back to synonyms, then
    the book sentence the word was found in, when a definition is missing —
    a word with none of the three simply stays un-embedded on this channel
    (most such words are OCR/junk candidates already surfaced by
    flagged_undefined/deepen, not content worth a manufactured vector).
  * **FastText subword vectors** — trained from scratch on this project's own
    archived book texts (not a generic pretrained web-crawl binary), so its
    character n-gram units are learned from the same archaic/literary English
    these words come from. Because it composes a vector from subwords, it
    produces one for *every* lemma regardless of whether a definition exists
    — including words no dictionary could define at all.

These are two genuinely different signals (meaning vs. form) — both are
stored and queried independently. Fusing them into one score, and the actual
distractor-selection heuristic built on top of this, are future work.
"""

from __future__ import annotations

from pathlib import Path

DEFINITION_MODEL = "BAAI/bge-small-en-v1.5"
DEFINITION_DIM = 384
FASTTEXT_DIM = 300


def definition_text(definition: str | None, synonyms: list[str] | None,
                    sentence: str | None) -> tuple[str, str] | None:
    """(text, source_label) to embed for a word, or None if nothing usable
    exists. Order matters: a real dictionary gloss beats a synonym list beats
    a single example sentence, but any of the three is fine to embed from."""
    if definition and definition.strip():
        return definition.strip(), "definition"
    if synonyms:
        joined = ", ".join(s.strip() for s in synonyms if s.strip())
        if joined:
            return joined, "synonyms"
    if sentence and sentence.strip():
        return sentence.strip(), "sentence"
    return None


class DefinitionEmbedder:
    """Batched sentence-embedding over definition/synonym/sentence text.
    Loaded once per CLI pass, not per word — model load is the expensive part."""

    def __init__(self, model_name: str = DEFINITION_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


def build_fasttext_corpus(archive_dir: Path, out_path: Path) -> int:
    """Concatenate every archived book's cleaned plain text into one training
    file for FastText — reuses the same extract/clean stages the ingestion
    pipeline already runs, rather than re-implementing text extraction.
    Returns the number of source files successfully read. Files extract.py
    doesn't recognize (e.g. leftover .csv artifacts from the pre-DB CSV
    workflow) are skipped, not fatal — a training corpus missing a few files
    is fine; failing the whole build over one bad file is not."""
    from . import clean, extract

    archive_dir = Path(archive_dir)
    out_path = Path(out_path)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted(archive_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                chapters = extract.extract(path)
            except Exception:  # noqa: BLE001 — skip anything extract.py can't handle
                continue
            for ch in chapters:
                out.write(clean.clean(ch.text))
                out.write("\n")
            n += 1
    return n


def train_fasttext(corpus_path: Path, model_path: Path, dim: int = FASTTEXT_DIM) -> None:
    """Unsupervised skip-gram training over the whole corpus file. Holistic,
    not incremental — must see the full corpus at once, so this is an
    occasional/periodic operation (re-run as the archive grows), not a
    per-word or per-book step."""
    import fasttext

    model = fasttext.train_unsupervised(str(corpus_path), model="skipgram", dim=dim)
    model.save_model(str(model_path))


class FastTextEmbedder:
    """Loads a trained FastText model and produces a vector for any lemma —
    including words never seen during training, via subword composition.
    That OOV-by-construction behavior is the entire reason FastText is used
    here rather than a word2vec/GloVe-style whole-word lookup table."""

    def __init__(self, model_path: Path):
        import fasttext
        self.model = fasttext.load_model(str(model_path))
        self.model_path = str(model_path)

    def vector(self, word: str) -> list[float]:
        return self.model.get_word_vector(word).tolist()
