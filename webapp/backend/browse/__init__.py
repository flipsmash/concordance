"""End-user word-browsing API (§ word browsing) -- author/book/domain/
difficulty facets, freely combinable in any order, plus an A-Z jump, an
archaic-ness filter, text search, and a random-word picker.

Imports `main` as a module and always accesses `_main.SCHEMA`/
`_main.get_conn()`/`_main.require_viewer` via dotted attribute lookup, not a
bare `from ... import`, for the same reason quiz.py does: tests monkeypatch
`main.SCHEMA` after import, and a bare import would freeze the value before
that monkeypatch runs. Registered into `app` at the bottom of main.py, after
get_conn/SCHEMA/require_viewer are all defined, for the same ordering reason
quiz.py's router is.

Every endpoint here is `require_viewer` -- this is the end-user browsing
surface, distinct from /api/words's `require_admin` curation view (different
audience, different response shape: no `rescued_from_reject` etc.).

--- The dedup rule (see the word-browsing plan) ---

word_book and word_category are both many-to-many against word (62% of words
appear in more than one book). Two situations, two different SQL shapes:

  - FILTERING through the junction table (author/book/domain narrow which
    words qualify, in common._build_word_filters): always EXISTS(...), never
    JOIN ... ON book_id = ANY(%s). A JOIN produces one row per matching
    (word, book) pair, silently duplicating a multi-book word in a paginated
    list and inflating `total`. EXISTS collapses that to true/false per word.
  - AGGREGATING through the junction table (authors/books listings, where
    book/author IS the thing being counted): a JOIN ... GROUP BY is correct
    here -- the fan-out is the point -- with count(DISTINCT w.id) so a word
    isn't double-counted for an author just because it's in two of their
    books.

Mixing these up in either direction is the bug to avoid.

The endpoints are split by what they browse into the modules of this
package, all registering on the one `router` defined in common.py.
"""

from __future__ import annotations

from .common import (  # noqa: F401
    router,
    _TOP_CODES,
    _TOP_CODE_NAMES,
    _ALL_CODES,
    _ARCHIVE_ROOT,
    _WORD_SORT_COLUMNS,
    _BOOK_SORT_COLUMNS,
    _AUTHOR_SORT_COLUMNS,
    _NULLABLE_SORTS,
    _SORT_TITLE_EXPR,
    _subtree_or_sql,
    _build_word_filters,
    _UNIQUE_WORD_BUCKETS,
    _unique_word_bucket_label,
    _unique_word_bucket_filter,
    _OVERALL_DIFFICULTY_BAND_WIDTH,
    _OVERALL_DIFFICULTY_UNSCORED_LABEL,
    _overall_difficulty_band_filter,
    _unique_word_bucket_range,
)
from .authors import (  # noqa: F401
    _browse_authors_from_stats,
    AuthorRow,
    AuthorPage,
    browse_authors,
    AuthorGraphNode,
    AuthorGraphEdge,
    AuthorRelatedResponse,
    AuthorRelatednessGraph,
    author_related,
    author_shared_words,
    authors_relatedness,
    AuthorMapNode,
    AuthorMapResponse,
    authors_map,
    AuthorMatrixCell,
    AuthorMatrixResponse,
    authors_matrix,
    DendrogramNode,
    AuthorDendrogramResponse,
    authors_dendrogram,
)
from .books import (  # noqa: F401
    BookRow,
    BookPage,
    browse_genres,
    GenreCount,
    GenreOverlapCell,
    GenreOverlap,
    browse_genre_overlap,
    _browse_books_from_stats,
    browse_books,
    book_text,
    BookGraphNode,
    BookGraphEdge,
    BookRelatedResponse,
    book_related,
    SharedWord,
    SharedWordsResponse,
    book_shared_words,
    BookRelatednessGraph,
    books_relatedness,
    BookMapNode,
    BookMapResponse,
    books_map,
    BookMatrixEntry,
    BookMatrixCell,
    BookMatrixResponse,
    books_matrix,
    BookDendrogramNode,
    BookDendrogramResponse,
    books_dendrogram,
)
from .words import (  # noqa: F401
    BrowseWordRow,
    BrowseWordPage,
    browse_words,
    browse_pos_values,
)
from .stats import (  # noqa: F401
    DomainBucketCount,
    _bucket_counts,
    browse_domains,
    DomainSummary,
    browse_domain_summary,
    DifficultyBandCount,
    browse_difficulty_bands,
    UniqueWordBucket,
    browse_unique_word_histogram,
    _FAME_SCORE_LABELS,
    _FAME_UNSCORED_LABEL,
    browse_fame_histogram,
    DailyCount,
    browse_growth,
    browse_overall_difficulty_histogram,
)
from .categories import (  # noqa: F401
    CategoryCount,
    _child_subtree_counts,
    browse_category_counts,
    CategoryOverlapCell,
    CategoryOverlap,
    _overlap_matrix,
    browse_category_overlap,
    CategoryLeaderRow,
    CategoryLeaderPage,
    browse_category_leaders,
    DomainMapNode,
    DomainMapResponse,
    _domain_vectors_to_map,
    browse_domain_map,
    _CATEGORY_TREE_WORDS,
    _CATEGORY_TREE_TTL,
    _category_tree_cache,
    CategoryTreeWord,
    CategoryTreeLeaf,
    CategoryDendrogramNode,
    CategoryDendrogramResponse,
    _category_dendrogram_compute,
    categories_dendrogram,
)
from .trends import (  # noqa: F401
    _TREND_MAX_TERMS,
    NgramTrendSeries,
    NgramTrend,
    ngram_trend,
)
