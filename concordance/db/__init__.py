"""Sync the master vocabulary list into PostgreSQL (§07 follow-on).

The CSV stays the working format; this mirrors it into a database so a future web
app (and eventual integration with the related project) has a real, queryable
store. Tables live in their own schema (default ``concordance``) so they can share
a database with other projects without name clashes.

Normalisation vs the flat CSV: the ``source_book`` cell (a "BookA; BookB" list) is
split into a proper many-to-many via ``word_book``; ``synonyms`` becomes a text[].
Everything is upsert-based and idempotent — re-running ``sync-db`` reconciles the
DB with the current CSV.

Connection comes from ``DATABASE_URL`` (env or a git-ignored .env), e.g.
    DATABASE_URL=postgresql://user:pass@host:5432/dbname

The code is split by domain into the modules of this package; every name
is re-exported here, so callers keep using ``db.<name>``. A test that
monkeypatches a module-level value must patch the submodule that owns it
(e.g. ``db.schema.MIGRATIONS``), since that's where the code looks it up.
"""

from __future__ import annotations

from .core import (  # noqa: F401
    DEFAULT_SCHEMA,
    _IDENT,
    database_url,
    _safe_schema,
    connect,
)
from .schema import (  # noqa: F401
    _SCHEMA_DDL,
    _TRGM_DDL,
    _REJECTED_LEMMA_INDEX_DDL,
    _VECTOR_DDL,
    _SCHEMA_VERSION_DDL,
    _schema_version,
    _trgm_present,
    apply_schema,
    _migration_0001_baseline,
    MIGRATIONS,
    refresh_rejected_lemma_index,
)
from .sync import (  # noqa: F401
    _synonyms,
    _books,
    _read_master_rows,
    sync_master,
    _invalidate_definition_dependents,
    sync_book_results,
    fetch_known_verdicts,
    normalize_word_pos,
)
from .definitions import (  # noqa: F401
    _POS_TO_TAGGER,
    fill_definitions,
    refill_definitions,
    deepen_definitions,
    mw_backfill,
    import_defined_words,
    dedupe_plural_definitions,
    _SYNONYM_OF_RE,
    _CSS_JUNK_RE,
    expand_synonym_definitions,
    compute_definition_links,
)
from .reference import (  # noqa: F401
    load_taxonomy,
    load_gazetteer,
    fetch_gazetteer_names,
    load_ngram_bulk,
    fetch_ngrams,
    _ENGLISH_FAMILY_LANGS,
    load_wiktionary_langs,
    foreign_only_langs,
    english_reference_terms,
)
from .cleanup import (  # noqa: F401
    clean_script_variants,
    _cast_out_variant,
    clean_dialect_spellings,
    clean_archaic_spellings,
    _UNTRUSTED_DEF_SOURCES,
    _TWIN_MIN_SIMILARITY,
    _sweep_respellings,
    clear_stale_foreign_flags,
    clear_stale_misspelling_flags,
    _clear_stale_flags,
    clean_foreign_words,
    _CTX_CLASSIFIER,
    _ctx_init,
    _ctx_scan_book,
    clean_non_english_context,
)
from .scoring import (  # noqa: F401
    compute_archaic,
    compute_difficulty,
    compute_quiz_definitions,
    compute_quizzable,
    compute_personal_difficulty,
)
from .books import (  # noqa: F401
    get_book_by_title,
    backfill_publication_era,
    update_book_archive_metadata,
    set_book_publication_info,
    fill_publication_years,
    compute_book_similarity,
    PLACEHOLDER_AUTHORS,
    compute_author_similarity,
    _FAME_EVIDENCE_FAILURE_MIN_SAMPLE,
    _FAME_EVIDENCE_FAILURE_THRESHOLD,
    _no_usable_author_evidence,
    _no_usable_book_evidence,
    _load_fame_llm,
    compute_author_stats,
    compute_author_fame,
    compute_book_stats,
    compute_book_fame,
    upsert_book_merge_group,
    mark_book_merge_compiled,
    mark_book_merge_merged,
    merge_book_group,
)
from .clustering import (  # noqa: F401
    _linkage_to_tree,
    compute_author_clustering,
    compute_book_clustering,
)
from .embeddings import (  # noqa: F401
    compute_definition_embeddings,
    compute_fasttext_embeddings,
)
from .audio import (  # noqa: F401
    fetch_wordnik_pronunciations,
    search_commons_direct,
    compute_ipa,
    backfill_ipa_from_oed,
    download_commons_direct_finds,
    compute_audio,
    synthesize_unverified_guesses,
)
from .maintain import (  # noqa: F401
    MAINTAIN_STEPS,
    maintain_status,
)
