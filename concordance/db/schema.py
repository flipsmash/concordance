"""DDL for the concordance schema and the versioned migrations that apply it."""

from __future__ import annotations

import psycopg

from .core import DEFAULT_SCHEMA, _safe_schema


_SCHEMA_DDL = """
CREATE SCHEMA IF NOT EXISTS {s};

CREATE TABLE IF NOT EXISTS {s}.book (
    id          serial PRIMARY KEY,
    title       text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word (
    id                serial PRIMARY KEY,
    lemma             text NOT NULL,
    lemma_lc          text GENERATED ALWAYS AS (lower(lemma)) STORED UNIQUE,
    as_seen           text,
    definition        text,
    part_of_speech    text,
    ipa               text,
    sentence          text,
    chapter           text,
    synonyms          text[] NOT NULL DEFAULT '{{}}',
    etymology         text,
    definition_source text,
    first_added       date,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word_book (
    word_id  integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    book_id  integer NOT NULL REFERENCES {s}.book(id) ON DELETE CASCADE,
    PRIMARY KEY (word_id, book_id)
);

-- `concordance link-definitions` -- one row per (source, target) pair when a
-- word's OWN definition text uses another word that's also in this app's
-- vocabulary (lemma-matched, see db.compute_definition_links), so the
-- webapp can render a real clickable link instead of inert prose. `surface`
-- is the literal inflected form found in the text ("proscribed"), not
-- necessarily target's own headword spelling ("proscribe") -- the frontend
-- highlights that exact substring. The UNIQUE constraint's own index
-- already serves "every link for word X" lookups (source_word_id leads).
CREATE TABLE IF NOT EXISTS {s}.word_definition_link (
    id               serial PRIMARY KEY,
    source_word_id   integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    target_word_id   integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    surface          text NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    CHECK (source_word_id != target_word_id),
    UNIQUE (source_word_id, target_word_id)
);

CREATE TABLE IF NOT EXISTS {s}.category (
    id          serial PRIMARY KEY,
    taxonomy    text NOT NULL DEFAULT 'usas',
    code        text NOT NULL,
    name        text NOT NULL,
    parent_id   integer REFERENCES {s}.category(id) ON DELETE CASCADE,
    level       integer NOT NULL DEFAULT 0,
    assignable  boolean NOT NULL DEFAULT true,
    UNIQUE (taxonomy, code)
);

-- Curated names/places, checked as a DISQUALIFYING signal in validity.py
-- (not a vouch -- see DESIGN.md's "still-mostly-unclosed gap" section for
-- why: SymSpell/WordNet/wordfreq are all frequency-derived from general web
-- text, so a real name with any web footprint still looks "attested" to
-- every one of them). (name_lc, kind) as the key rather than name_lc alone
-- -- a name can legitimately be more than one kind (Washington: both
-- surname and place), and re-running one source's loader independently
-- shouldn't fight over the same row a different source already owns.
CREATE TABLE IF NOT EXISTS {s}.gazetteer_name (
    name_lc text NOT NULL,
    kind    text NOT NULL,   -- 'given_name' | 'surname' | 'place'
    PRIMARY KEY (name_lc, kind)
);

CREATE TABLE IF NOT EXISTS {s}.word_category (
    word_id     integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    category_id integer NOT NULL REFERENCES {s}.category(id) ON DELETE CASCADE,
    confidence  real,
    source      text,          -- 'usas-tagger' | 'wordnet' | 'llm' | 'dict-label'
    is_primary  boolean NOT NULL DEFAULT false,
    PRIMARY KEY (word_id, category_id)
);

-- One row per (book, genre) tag -- see concordance/genre.py for the fixed
-- tag vocabulary and the fiction/nonfiction-redundancy rule enforced in
-- Python before a row ever lands here (not a DB CHECK -- this schema
-- validates enum-like strings in code, not constraints; see
-- word_difficulty.archaic below for the same convention). `source`
-- distinguishes the LLM classifier from a future admin correction, same
-- provenance pattern as word_category.source.
CREATE TABLE IF NOT EXISTS {s}.book_genre (
    book_id     integer NOT NULL REFERENCES {s}.book(id) ON DELETE CASCADE,
    genre       text NOT NULL,
    source      text NOT NULL DEFAULT 'llm',   -- 'llm' | 'manual'
    confidence  real,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (book_id, genre)
);

CREATE TABLE IF NOT EXISTS {s}.word_difficulty (
    word_id           integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    archaic             text,          -- current | dated | archaic | obsolete
    archaic_evidence    text,
    archaic_confidence  double precision,
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word_ngram (
    word_id        integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    peak           double precision,
    recent         double precision,
    recency_ratio  double precision,
    peak_year      integer,
    fetched_at     timestamptz NOT NULL DEFAULT now()
);

-- Each book's top-k most vocabulary-related books (IDF-weighted cosine
-- similarity over shared active words -- see compute_book_similarity),
-- NOT a full all-pairs matrix: storing only the top-k neighbors per book
-- keeps this O(k*n_books), flat as the corpus grows, matching this
-- project's own no-fixed-corpus-scale principle. Both directions are
-- stored (a related to b AND b related to a as separate rows) so "book
-- X's related books" is always a single indexed WHERE book_a_id=X, no
-- UNION/OR needed -- the same "one row per lookup direction" shape
-- sessions(token) already uses for the identical reason.
CREATE TABLE IF NOT EXISTS {s}.book_similarity (
    book_a_id          integer NOT NULL REFERENCES {s}.book(id) ON DELETE CASCADE,
    book_b_id          integer NOT NULL REFERENCES {s}.book(id) ON DELETE CASCADE,
    score              double precision NOT NULL,
    shared_word_count  integer NOT NULL,
    updated_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (book_a_id, book_b_id)
);
CREATE INDEX IF NOT EXISTS book_similarity_rank_idx
    ON {s}.book_similarity (book_a_id, score DESC);

-- Same shape as book_similarity, one level up: each author's top-k most
-- vocabulary-related authors. Originally shipped as an on-demand,
-- compute-every-request query (the relatedness-visualization plan's own
-- reasoning: "authors are dozens today, full O(n^2) pairwise at request
-- time is cheap") -- that premise was wrong the moment it met real data
-- (~3,500 authors, not dozens; full-corpus timing came back at ~39s), so
-- this was precomputed instead, matching book_similarity's pattern rather
-- than trying to make the on-demand query fast enough for an HTTP request.
-- author_a/author_b are plain text (book.author has no own table), not an
-- integer FK.
CREATE TABLE IF NOT EXISTS {s}.author_similarity (
    author_a           text NOT NULL,
    author_b           text NOT NULL,
    score              double precision NOT NULL,
    shared_word_count  integer NOT NULL,
    updated_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (author_a, author_b)
);
CREATE INDEX IF NOT EXISTS author_similarity_rank_idx
    ON {s}.author_similarity (author_a, score DESC);

-- Precomputed replacement for the aggregate stats webapp/backend/browse's
-- browse_authors used to recompute live, on every request, over the WHOLE
-- corpus (mean_difficulty/density/overall_difficulty/unique_word_count) --
-- same "didn't survive contact with the real corpus" story as
-- author_similarity above, found 2026-09-13 once the corpus reached 29k
-- books / 4.5M word_book rows (browse_authors' author_word_ids alone, a
-- corpus-wide SELECT DISTINCT (author, word_id), measured at ~3.2s). Only
-- valid for the UNFILTERED population (no domain/difficulty/archaic/pos/
-- quizzable_only/book_id facet active) -- see compute_author_stats and
-- browse_authors' own use of it for why a faceted request still falls back
-- to live computation. fame_score/fame_reasoning stay on their own
-- independently-refreshed author_fame table, not duplicated here.
CREATE TABLE IF NOT EXISTS {s}.author_stats (
    author              text PRIMARY KEY,
    book_count          integer NOT NULL,
    word_count          integer NOT NULL,
    scored_word_count   integer NOT NULL,
    mean_difficulty     double precision,
    stddev_difficulty   double precision,
    density             double precision,
    unique_word_count   integer NOT NULL,
    overall_difficulty  double precision,
    computed_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS author_stats_overall_difficulty_idx
    ON {s}.author_stats (overall_difficulty DESC NULLS LAST);

-- Hierarchical clustering + 2D projection over the top-N (by book count)
-- authors -- see compute_author_clustering. Unlike author_similarity
-- (every author, top-k neighbors only), this covers a bounded, smaller set
-- in full: cluster_id/mds_x/mds_y are meaningful only relative to everyone
-- else in the SAME computed run, so this table always holds exactly one
-- run's worth of authors, truncated and repopulated wholesale each time.
CREATE TABLE IF NOT EXISTS {s}.author_cluster (
    author       text PRIMARY KEY,
    cluster_id   integer NOT NULL,
    mds_x        double precision NOT NULL,
    mds_y        double precision NOT NULL,
    book_count   integer NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now()
);

-- Singleton (id always 1): the full pairwise similarity grid and the
-- dendrogram tree are read as whole blobs (the /matrix and /dendrogram
-- endpoints never query a single pair or subtree server-side), and both
-- must come from the exact same computation pass as author_cluster above --
-- a blob avoids three separately-writable tables silently drifting apart
-- (e.g. a crash between writing clusters and writing the tree would
-- otherwise leave the map and matrix reflecting different runs).
CREATE TABLE IF NOT EXISTS {s}.author_cluster_run (
    id           integer PRIMARY KEY CHECK (id = 1),
    leaf_order   text[] NOT NULL,   -- seriated author order, for matrix display
    grid         jsonb NOT NULL,    -- NxN [[score, shared_word_count], ...] in leaf_order
    tree_json    jsonb NOT NULL,    -- nested linkage tree, for the dendrogram
    computed_at  timestamptz NOT NULL DEFAULT now()
);

-- Same idea as author_cluster/author_cluster_run, one level down -- see
-- compute_book_clustering. book_cluster_run.leaf_order is jsonb, not
-- text[]: an author is uniquely identified by their name alone (the
-- string IS the display label AND the navigation key), but a book needs
-- id (navigation, e.g. two books can share a title), title (display),
-- AND author (to build a /app/authors/:author/:bookId link) together, so
-- a flat string array isn't enough here the way it is for authors.
CREATE TABLE IF NOT EXISTS {s}.book_cluster (
    book_id      integer PRIMARY KEY REFERENCES {s}.book(id) ON DELETE CASCADE,
    title        text NOT NULL,
    author       text,
    cluster_id   integer NOT NULL,
    mds_x        double precision NOT NULL,
    mds_y        double precision NOT NULL,
    word_count   integer NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.book_cluster_run (
    id           integer PRIMARY KEY CHECK (id = 1),
    leaf_order   jsonb NOT NULL,    -- [{{"id","title","author"}}, ...] in seriated order
    grid         jsonb NOT NULL,    -- NxN [[score, shared_word_count], ...] in leaf_order
    tree_json    jsonb NOT NULL,    -- nested linkage tree, for the dendrogram
    computed_at  timestamptz NOT NULL DEFAULT now()
);

-- Identical shape to author_cluster/author_cluster_run and
-- book_cluster/book_cluster_run above, one selection criterion swapped:
-- these hold a run selected by fame_score threshold (compute_author_
-- clustering/compute_book_clustering called with min_fame set) instead of
-- top-N by volume -- a second, independently-computed lens on the corpus
-- ("the most historically important authors/books" rather than "the
-- most-represented ones"), not a replacement. Separate tables rather than
-- a scope column on the originals because both runs need to coexist and
-- be read independently (the frontend's Volume/Fame toggle switches
-- between them by fetching a different endpoint, not by filtering one
-- shared table), and because the two runs are recomputed independently
-- and on different schedules (see compute_author_clustering's min_fame
-- docstring for why the fame variant isn't part of routine `maintain`).
CREATE TABLE IF NOT EXISTS {s}.author_cluster_fame (
    author       text PRIMARY KEY,
    cluster_id   integer NOT NULL,
    mds_x        double precision NOT NULL,
    mds_y        double precision NOT NULL,
    book_count   integer NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.author_cluster_fame_run (
    id           integer PRIMARY KEY CHECK (id = 1),
    leaf_order   text[] NOT NULL,
    grid         jsonb NOT NULL,
    tree_json    jsonb NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.book_cluster_fame (
    book_id      integer PRIMARY KEY REFERENCES {s}.book(id) ON DELETE CASCADE,
    title        text NOT NULL,
    author       text,
    cluster_id   integer NOT NULL,
    mds_x        double precision NOT NULL,
    mds_y        double precision NOT NULL,
    word_count   integer NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.book_cluster_fame_run (
    id           integer PRIMARY KEY CHECK (id = 1),
    leaf_order   jsonb NOT NULL,
    grid         jsonb NOT NULL,
    tree_json    jsonb NOT NULL,
    computed_at  timestamptz NOT NULL DEFAULT now()
);

-- Absolute (NOT corpus-relative) fame/historical-importance score, 1-10,
-- LLM-judged against a fixed external rubric anchored on real reference
-- figures (Shakespeare=10) -- see fame.py. author is plain text, same
-- keying as author_similarity/author_cluster above: there is no author
-- table to reference. checked_at is bumped on every attempt (hit or miss,
-- same sticky-resumability convention as word.mw_checked_at) so a rerun
-- never re-spends a real LLM call + several network round-trips on a row
-- already attempted; computed_at is set only when a real score landed.
CREATE TABLE IF NOT EXISTS {s}.author_fame (
    author         text PRIMARY KEY,
    fame_score     double precision,
    fame_reasoning text,
    fame_factors   jsonb,
    computed_at    timestamptz,
    checked_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS author_fame_score_idx ON {s}.author_fame (fame_score DESC NULLS LAST);

-- Same shape one level down. book_id is a real FK (book.id exists, unlike
-- author) so ON DELETE CASCADE is correct here even though author_fame
-- can't have an equivalent.
CREATE TABLE IF NOT EXISTS {s}.book_fame (
    book_id        integer PRIMARY KEY REFERENCES {s}.book(id) ON DELETE CASCADE,
    fame_score     double precision,
    fame_reasoning text,
    fame_factors   jsonb,
    computed_at    timestamptz,
    checked_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS book_fame_score_idx ON {s}.book_fame (fame_score DESC NULLS LAST);

-- Same shape one level down from author_stats -- see that table's own
-- CREATE TABLE comment for the backstory. browse_books' unique_word_count
-- (which words in a book appear NOWHERE else in the corpus) turned out to
-- need the identical fix: computing it live means a full scan of every
-- word_book row, on every single request, found 2026-09-14 once book
-- browsing was still slow after browse_books' own count(*) fix. genres/
-- fame_score/fame_reasoning stay off this table for the same reason they
-- stay off author_stats -- book_genre/book_fame are already cheap joins
-- (~10-20ms measured), duplicating them here would just be another
-- staleness source for no speed benefit.
CREATE TABLE IF NOT EXISTS {s}.book_stats (
    book_id             integer PRIMARY KEY REFERENCES {s}.book(id) ON DELETE CASCADE,
    word_count          integer NOT NULL,
    scored_word_count   integer NOT NULL,
    mean_difficulty     double precision,
    stddev_difficulty   double precision,
    density             double precision,
    unique_word_count   integer NOT NULL,
    overall_difficulty  double precision,
    computed_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS book_stats_overall_difficulty_idx
    ON {s}.book_stats (overall_difficulty DESC NULLS LAST);

-- concordance book-merge's manifest of detected multi-part-book groups (see
-- concordance/book_merge.py). Unlike checked_at elsewhere in this schema,
-- checked_at here is NOT a skip gate -- a group excluded today for a gap
-- becomes eligible the moment the missing volume is ingested, so every run
-- re-detects fresh from `book` and overwrites part_book_ids/part_labels/
-- skip_reason/gap_detail/checked_at unconditionally. Only compiled_at/
-- merged_at are terminal: they gate the actual expensive/destructive steps
-- (writing the combined file, folding the DB records together) so a killed
-- run resumes exactly where it left off instead of redoing either.
CREATE TABLE IF NOT EXISTS {s}.book_merge_group (
    id                serial PRIMARY KEY,
    title_base        text NOT NULL,
    author            text NOT NULL,
    part_count        integer NOT NULL,
    part_book_ids     integer[] NOT NULL,
    part_labels       jsonb NOT NULL,
    survivor_book_id  integer,
    compiled_path     text,
    skip_reason       text,
    gap_detail        jsonb,
    compiled_at       timestamptz,
    merged_at         timestamptz,
    checked_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (title_base, author)
);

CREATE TABLE IF NOT EXISTS {s}.word_audio (
    word_id      integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    source       text,          -- 'commons' | 'mw' | 'azure' | 'piper' | 'azure_guess' (legacy) | 'none' (looked up, nothing found)
    file_path    text,
    ipa_used     text,          -- the exact phoneme string sent to the synthesizer (azure only)
    voice        text,          -- azure voice name, or the Commons source URL
    license_note text,
    generated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word_commons_search (
    word_id      integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    found_title  text,          -- Commons "File:..." title of an exact English match, or NULL
    download_url text,
    checked_at   timestamptz NOT NULL DEFAULT now()
);

-- One row per (book, lemma) rejection, deliberately NOT deduped across books
-- like word/word_book is: the same lemma can be rejected for different
-- reasons in different books (e.g. the coinage/UNSURE call depends on
-- per-book recurrence count), so each book's ingestion run keeps its own
-- verdict rather than merging into a single global history.
CREATE TABLE IF NOT EXISTS {s}.rejected_word (
    id          serial PRIMARY KEY,
    book_id     integer NOT NULL REFERENCES {s}.book(id) ON DELETE CASCADE,
    lemma       text NOT NULL,
    lemma_lc    text GENERATED ALWAYS AS (lower(lemma)) STORED,
    reason      text,          -- frequency_floor | proper_noun | misspelling | not_a_word | not_interesting
    detail      text,
    count       integer,
    zipf        double precision,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (book_id, lemma_lc)
);

CREATE INDEX IF NOT EXISTS rejected_word_lemma_idx ON {s}.rejected_word (lemma_lc);

-- Backs fetch_known_verdicts's `SELECT DISTINCT lemma_lc, reason ... WHERE
-- reason IN (...)` scan (re-run once per book during ingestion). lemma_lc is
-- a real key column here (not just INCLUDE'd) so the index is already sorted
-- by (reason, lemma_lc) -- exactly the DISTINCT's grouping order -- letting
-- the planner satisfy the query with a plain index-only scan + streaming
-- Unique instead of a HashAggregate/sort over however many rows currently
-- match. Superseded rejected_word_reason_lemma_idx ((reason) INCLUDE
-- (lemma_lc)), which was already too expensive as the corpus grew (measured:
-- ~5.9s/full scan vs ~1.5s/index-only scan at ~40M rows, gap widens with
-- table size) and, without DISTINCT on the query side, was no defense
-- against rejected_word's deliberate one-row-per-(book,lemma) duplication
-- (see its own table comment) -- fetch_known_verdicts shipped 28M duplicate
-- rows to Python for what collapsed to 862K distinct lemmas at 106M total
-- rejected_word rows, OOM-killing a live ingest run on 2026-08-16.
CREATE INDEX IF NOT EXISTS rejected_word_reason_lemma_key_idx ON {s}.rejected_word (reason, lemma_lc);

-- App-level accounts, separate from Cloudflare Access (which gates the admin
-- curation UI at the network edge). is_admin distinguishes the curation-side
-- role from an ordinary browsing/study account.
CREATE TABLE IF NOT EXISTS {s}.users (
    id             serial PRIMARY KEY,
    username       text NOT NULL,
    username_lc    text GENERATED ALWAYS AS (lower(username)) STORED UNIQUE,
    password_hash  text NOT NULL,
    is_admin       boolean NOT NULL DEFAULT false,
    created_at     timestamptz NOT NULL DEFAULT now(),
    last_login_at  timestamptz
);

-- token is the cookie value itself (no separate id/lookup indirection) --
-- session validation is one indexed WHERE token=%s.
CREATE TABLE IF NOT EXISTS {s}.sessions (
    token       text PRIMARY KEY,
    user_id     integer NOT NULL REFERENCES {s}.users(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON {s}.sessions (user_id);
CREATE INDEX IF NOT EXISTS sessions_expires_at_idx ON {s}.sessions (expires_at);

-- Invite-only signup: admin generates a one-time link carrying `token`;
-- registering consumes it (sets used_at/used_by_user_id) so it can't be reused.
CREATE TABLE IF NOT EXISTS {s}.invite_tokens (
    id                 serial PRIMARY KEY,
    token              text NOT NULL UNIQUE,
    label              text,
    created_at         timestamptz NOT NULL DEFAULT now(),
    expires_at         timestamptz NOT NULL,
    used_at            timestamptz,
    used_by_user_id    integer REFERENCES {s}.users(id) ON DELETE SET NULL
);

-- Generic global key/value settings so future admin-configurable toggles
-- don't need a new table/migration each time. Currently just one key,
-- 'quiz_feedback_timing' (value {{"mode": "immediate"|"end_of_test"}}).
CREATE TABLE IF NOT EXISTS {s}.app_settings (
    key         text PRIMARY KEY,
    value       jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.quiz_session (
    id                serial PRIMARY KEY,
    user_id           integer NOT NULL REFERENCES {s}.users(id) ON DELETE CASCADE,
    config            jsonb NOT NULL,
    feedback_timing   text NOT NULL,   -- snapshot of app_settings at start time, so a
                                        -- mid-quiz admin change never mutates a session
                                        -- already in progress
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    score_pct         double precision
);
CREATE INDEX IF NOT EXISTS quiz_session_user_idx ON {s}.quiz_session (user_id);

CREATE TABLE IF NOT EXISTS {s}.quiz_question (
    id              serial PRIMARY KEY,
    session_id      integer NOT NULL REFERENCES {s}.quiz_session(id) ON DELETE CASCADE,
    seq             integer NOT NULL,        -- 1-based order within the session, also the
                                              -- test-length budget unit (a matching set is
                                              -- still exactly 1 here even though it holds
                                              -- multiple word/definition pairs)
    question_type   text NOT NULL,           -- 'mc' | 'true_false' | 'matching'
    target_word_ids integer[] NOT NULL,      -- 1 word for mc/tf, N for a matching set
    payload         jsonb NOT NULL,          -- type-specific, includes the answer key --
                                              -- stripped before any client-facing response
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);

CREATE TABLE IF NOT EXISTS {s}.quiz_answer (
    id              serial PRIMARY KEY,
    question_id     integer NOT NULL REFERENCES {s}.quiz_question(id) ON DELETE CASCADE,
    word_id         integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
                                              -- one row per matching pair (per-pair credit),
                                              -- exactly one row for mc/tf
    response        jsonb NOT NULL,
    is_correct      boolean NOT NULL,
    answered_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS quiz_answer_question_idx ON {s}.quiz_answer (question_id);
CREATE INDEX IF NOT EXISTS quiz_answer_word_idx ON {s}.quiz_answer (word_id);

-- Lightweight priority re-exposure for spaced repetition -- NOT full SM-2,
-- NOT a mastery-tracking system (that's explicitly deferred). Updated on
-- every quiz_answer regardless of whether the session that produced it had
-- spaced repetition turned on, so enabling it later immediately benefits
-- from all prior history rather than starting cold.
CREATE TABLE IF NOT EXISTS {s}.word_review_schedule (
    user_id           integer NOT NULL REFERENCES {s}.users(id) ON DELETE CASCADE,
    word_id           integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    streak            integer NOT NULL DEFAULT 0,
    last_seen_at      timestamptz,
    next_eligible_at  timestamptz,
    correct_count     integer NOT NULL DEFAULT 0,
    incorrect_count   integer NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, word_id)
);
CREATE INDEX IF NOT EXISTS word_review_schedule_eligible_idx
    ON {s}.word_review_schedule (user_id, next_eligible_at);

-- Personalized difficulty calibration (see concordance/calibration.py and
-- compute_personal_difficulty) -- an adjustment to the ex-ante
-- word_difficulty.difficulty score, from THIS user's own first exposure to
-- the word in a quiz. Deliberately NOT a population-level IRT calibration
-- and deliberately NOT written into word_difficulty.difficulty itself: with
-- one dominant rater, response data only ever tells you that rater's own
-- relative gaps, never identifies "true" item difficulty the way a real
-- multi-rater calibration would -- folding it into the shared, all-users-
-- facing difficulty column would silently distort a future second user's
-- experience with the first user's idiosyncratic blind spots. Same
-- PRIMARY KEY (user_id, word_id) shape as word_review_schedule, which
-- exists for the same "per-person view of a word" reason.
CREATE TABLE IF NOT EXISTS {s}.word_personal_difficulty (
    user_id              integer NOT NULL REFERENCES {s}.users(id) ON DELETE CASCADE,
    word_id              integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    item_rating          double precision NOT NULL,   -- logit scale
    personal_difficulty  double precision NOT NULL,   -- 0-100, same scale as word_difficulty.difficulty
    based_on_correct     boolean NOT NULL,             -- the first-exposure outcome that produced this value
    calibrated_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, word_id)
);

-- User-curated flashcard sets (§ word sets) -- a persistent, user-named
-- collection of words, distinct from quiz_session's ephemeral per-session
-- word selection. word_review_schedule (spaced-repetition bias for
-- quizzing) is deliberately NOT reused for "mastered" here -- its own
-- comment is explicit that it's not a mastery-tracking system, just a
-- continuously-updated re-exposure bias with no per-set notion at all.
CREATE TABLE IF NOT EXISTS {s}.word_set (
    id          serial PRIMARY KEY,
    user_id     integer NOT NULL REFERENCES {s}.users(id) ON DELETE CASCADE,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, name)
);
CREATE INDEX IF NOT EXISTS word_set_user_idx ON {s}.word_set (user_id);

-- One row per (set, word) -- `mastered` is a plain sticky flag, toggled by
-- the user (flashcard-run "Mastered" button, or the summary page's
-- checkbox), not computed from any response history the way
-- word_review_schedule's streak is. A mastered word is simply excluded
-- from that set's next flashcard deck (see word_sets.py's flashcards
-- endpoint) until un-toggled.
CREATE TABLE IF NOT EXISTS {s}.word_set_item (
    set_id       integer NOT NULL REFERENCES {s}.word_set(id) ON DELETE CASCADE,
    word_id      integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    mastered     boolean NOT NULL DEFAULT false,
    mastered_at  timestamptz,
    added_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (set_id, word_id)
);

-- Analogy quiz questions (§ analogies, concordance/analogies.py). Every term that
-- can appear in an analogy relation edge, vocab word or not -- word_id is set iff
-- this term IS a vocab word. One row per distinct term rather than two nullable
-- word columns per edge, so vocab-vocab, vocab-ordinary, and ordinary-ordinary
-- edges all share the same word_relation_edge shape below with no per-row
-- branching on which side is which.
CREATE TABLE IF NOT EXISTS {s}.wn_relation_term (
    id             serial PRIMARY KEY,
    word_id        integer REFERENCES {s}.word(id) ON DELETE CASCADE,
    lemma          text NOT NULL,
    lemma_lc       text GENERATED ALWAYS AS (lower(lemma)) STORED,
    wn_pos         text NOT NULL,                    -- 'n' | 'v' | 'a' | 'r'
    synset_name    text,                              -- canonical sense, e.g. 'cangue.n.01';
                                                        -- NULL for a vocab word with no WordNet
                                                        -- synset at all (definition-pattern-only)
    gloss          text,                               -- WordNet gloss (ordinary term) or
                                                        -- word.definition (vocab term) -- feeds
                                                        -- ONLY the LLM verification prompt, never
                                                        -- shown in the quiz UI itself
    synonym_lemmas text[] NOT NULL DEFAULT '{{}}',       -- other lemma_names sharing synset_name --
                                                        -- the "D's own synonyms" ambiguity exclusion
    zipf           double precision,                   -- wordfreq zipf_frequency(lemma, "en")
    is_common      boolean NOT NULL DEFAULT false,      -- zipf >= 4.0 (same "plainly frequent" bar
                                                        -- validity_score.py already uses) -- anchor
                                                        -- (ordinary-term) eligibility for style B
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS wn_relation_term_word_id_idx
    ON {s}.wn_relation_term (word_id) WHERE word_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS wn_relation_term_lemma_pos_idx
    ON {s}.wn_relation_term (lemma_lc, wn_pos) WHERE word_id IS NULL;
CREATE INDEX IF NOT EXISTS wn_relation_term_common_idx
    ON {s}.wn_relation_term (wn_pos) WHERE is_common;

-- _find_term_id and the trap-lemma lookup in analogy_select.py both resolve
-- a bare lemma_lc (+ optional wn_pos) to a term id without knowing in advance
-- whether word_id is NULL, so neither of the two partial indexes above can
-- serve them -- every such lookup was a full sequential scan (measured:
-- ~22ms/call at ~63k rows, called 400k+ times total per pg_stat_user_tables,
-- and this is the hottest query pattern against this table).
CREATE INDEX IF NOT EXISTS wn_relation_term_lemma_lc_idx
    ON {s}.wn_relation_term (lemma_lc, wn_pos);

-- Resumability marker, one row per term once its relation edges have been
-- extracted -- same "we looked and found nothing" shape as word_commons_search
-- (a term with zero WordNet/definition-pattern relations produces zero rows in
-- word_relation_edge and would be rescanned every run without this).
CREATE TABLE IF NOT EXISTS {s}.wn_relation_scan (
    term_id      integer PRIMARY KEY REFERENCES {s}.wn_relation_term(id) ON DELETE CASCADE,
    scanned_at   timestamptz NOT NULL DEFAULT now(),
    edges_found  integer NOT NULL DEFAULT 0,   -- raw (pre-verification) candidates found with
                                                -- this term as term_a, across every relation type
    method       text NOT NULL                 -- 'wordnet' | 'definition_pattern' | 'both'
);

-- One row per candidate relation pair -- vocab-vocab, vocab-ordinary, or
-- ordinary-ordinary all share this shape. verification_status defaults to
-- 'pending' and an edge is NEVER usable in a quiz until 'verified' -- an
-- unverified pair shipping means a live question with two right answers, so
-- every quiz-time query filters WHERE verification_status = 'verified'
-- (see word_relation_edge_verified_idx).
CREATE TABLE IF NOT EXISTS {s}.word_relation_edge (
    id                   serial PRIMARY KEY,
    term_a_id            integer NOT NULL REFERENCES {s}.wn_relation_term(id) ON DELETE CASCADE,
    term_b_id            integer NOT NULL REFERENCES {s}.wn_relation_term(id) ON DELETE CASCADE,
    relation_type        text NOT NULL,   -- 'hypernym' | 'holonym_part' | 'holonym_member' |
                                           -- 'holonym_substance' | 'antonym' | 'similar_to' |
                                           -- 'derivationally_related' | 'attribute' |
                                           -- 'definition_pattern_kind_of' |
                                           -- 'definition_pattern_agent' |
                                           -- 'definition_pattern_part_of' |
                                           -- 'definition_pattern_purpose' |
                                           -- 'definition_pattern_relates_to' |
                                           -- 'definition_pattern_resembling' |
                                           -- 'definition_pattern_characterized_by' --
                                           -- the last three are definition-text-mined
                                           -- (concordance/analogies.py's _build_matchers),
                                           -- added because WordNet's own hypernym/holonym/
                                           -- purpose/agentive relations are almost entirely
                                           -- absent for adjective synsets
    relation_family      text NOT NULL,   -- 'is_a' | 'part_of' | 'opposite' | 'similar' |
                                           -- 'derived' | 'agentive' | 'purpose' | 'attribute' |
                                           -- 'relates_to' | 'resembling' --
                                           -- the bucket used to pair this edge with a DIFFERENT
                                           -- edge as the item's anchor (A:B) leg
    pos_a                text NOT NULL,   -- canonical POS (model.normalize_pos) of term_a
    pos_b                text NOT NULL,   -- canonical POS of term_b
    source               text NOT NULL,   -- 'wordnet_hypernym' | 'wordnet_holonym_part' | ... |
                                           -- 'definition_pattern_kind_of' | ...
    verification_status  text NOT NULL DEFAULT 'pending',  -- 'pending' | 'verified' | 'rejected'
    verification_note    text,
    verifier_model        text,
    verified_at            timestamptz,
    created_at              timestamptz NOT NULL DEFAULT now(),
    UNIQUE (term_a_id, term_b_id, relation_type)
);
CREATE INDEX IF NOT EXISTS word_relation_edge_family_idx
    ON {s}.word_relation_edge (relation_family, verification_status);
CREATE INDEX IF NOT EXISTS word_relation_edge_term_a_idx ON {s}.word_relation_edge (term_a_id);
CREATE INDEX IF NOT EXISTS word_relation_edge_term_b_idx ON {s}.word_relation_edge (term_b_id);
CREATE INDEX IF NOT EXISTS word_relation_edge_verified_idx
    ON {s}.word_relation_edge (verification_status) WHERE verification_status = 'verified';

-- The FULL (non-vocab-restricted), transitive-closure-where-applicable WordNet
-- target set for (term acting as term_a, relation_type) -- populated regardless
-- of verification, since its only job is the ambiguity exclusion set and
-- trap-distractor sourcing at quiz-assembly time (concordance/analogy_select.py),
-- never shown as quiz content itself. Also carries the synthetic relation_type
-- 'sibling_of_hypernym_parent' (a term's co-hyponyms under its own immediate
-- parent), used only for one-hard-term distractor plausibility.
CREATE TABLE IF NOT EXISTS {s}.wn_relation_fanout (
    id            serial PRIMARY KEY,
    term_id       integer NOT NULL REFERENCES {s}.wn_relation_term(id) ON DELETE CASCADE,
    relation_type text NOT NULL,
    target_lemma  text NOT NULL,
    target_pos    text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (term_id, relation_type, target_lemma, target_pos)
);
CREATE INDEX IF NOT EXISTS wn_relation_fanout_lookup_idx
    ON {s}.wn_relation_fanout (term_id, relation_type);
"""


# pg_trgm powers future fuzzy "did-you-mean" lookups; optional because CREATE
# EXTENSION needs privileges a managed role may lack. (rejected_lemma_index's
# own search uses a plain prefix LIKE, not trigram similarity -- confirmed
# live that trigram against it returns mostly coincidental-trigram noise at
# this table's scale -- so it needs no trgm index of its own here.)
_TRGM_DDL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS word_lemma_trgm ON {s}.word USING gin (lemma gin_trgm_ops);
"""

# Precomputed distinct-lemma view over rejected_word -- the Rejected
# curation tab (RejectedView) is meant to be reviewed once per WORD, not
# once per (book, lemma) instance: rejected_word is deliberately many-rows-
# per-lemma by design (~25M rows, ~468k distinct lemmas at real corpus
# scale -- see rejected_word's own schema comment, the same lemma can be
# rejected for different reasons in different books), which makes it both
# the wrong grain for a review list AND too big to search/count directly:
# an exact COUNT(*) with any filter is a multi-second scan REGARDLESS of
# indexing (trigram or plain prefix LIKE) -- Postgres's planner won't use
# an index for an aggregate over that many rows. This view is the review
# grain: one row per lemma, aggregated across every book it was rejected in.
#
# `NOT EXISTS (... w.active)` is what makes a promoted word actually
# disappear from the next refresh -- accept_rejected only deletes the ONE
# rejected_word row it was invoked on (see its own docstring), so without
# this the lemma would keep resurfacing via its other book instances even
# after being accepted. Same "word wins over rejected_word" rule
# classify.py's verdict cache already applies, just enforced here too --
# and the reason this view can only be created once word.active exists,
# hence its own DDL block below rather than living in _SCHEMA_DDL (which
# runs before any ALTER-added column exists, and would fail on a brand-new
# schema where `word` was just CREATEd bare).
#
# rep_id is a representative rejected_word.id (arbitrary but stable choice
# via min()) -- accept_rejected's existing by-id, single-book-link, single-
# row-delete behavior is intentionally left as-is (see its own docstring);
# this just gives the distinct-lemma UI an id to call it with, rather than
# rewriting accept to promote/link/delete across every book at once, a
# materially different and riskier operation nobody asked for.
#
# A real MATERIALIZED VIEW (this project's first), not a hand-maintained
# table kept in sync at insert time: nothing would otherwise notice a
# lemma's last active instance being accepted or its book deleted, so an
# incrementally-maintained copy would accumulate phantom rows forever.
# Refreshed periodically (concordance refresh-rejected-index / maintain),
# not on every write -- "as of a bit ago" is an acceptable staleness window
# for curation review/search, unlike the enrichment tables that have to
# reflect the very latest ingest.
_REJECTED_LEMMA_INDEX_DDL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS {s}.rejected_lemma_index AS
    SELECT r.lemma_lc,
           min(r.lemma) AS lemma,
           min(r.id) AS rep_id,
           count(DISTINCT r.book_id) AS book_count,
           sum(r.count) AS total_count,
           max(r.zipf) AS zipf,
           array_agg(DISTINCT r.reason) FILTER (WHERE r.reason IS NOT NULL) AS reasons
    FROM {s}.rejected_word r
    WHERE NOT EXISTS (
        SELECT 1 FROM {s}.word w WHERE w.lemma_lc = r.lemma_lc AND w.active
    )
    GROUP BY r.lemma_lc;
CREATE UNIQUE INDEX IF NOT EXISTS rejected_lemma_index_pkey ON {s}.rejected_lemma_index (lemma_lc);
"""

# One row per word, two independent per-word vectors (not an all-pairs distance
# matrix — see embed.py's module docstring for why that doesn't scale). hnsw
# over ivfflat deliberately: ivfflat's `lists` parameter must be re-tuned as
# the table grows, which is exactly the "baking in today's corpus size"
# mistake this project avoids elsewhere; hnsw's parameters are corpus-size-
# independent and support incremental inserts natively. Optional for the same
# privileges reason as pg_trgm above.
_VECTOR_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {s}.word_embedding (
    word_id            integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    definition_vector  vector(384),
    definition_model   text,
    definition_source  text,
    fasttext_vector    vector(300),
    fasttext_model     text,
    updated_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS word_embedding_def_hnsw_idx
    ON {s}.word_embedding USING hnsw (definition_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS word_embedding_ft_hnsw_idx
    ON {s}.word_embedding USING hnsw (fasttext_vector vector_cosine_ops);
"""


# --- versioned migrations -----------------------------------------------------
#
# apply_schema used to re-run ~55 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
# statements on EVERY call -- every CLI command and every web-server start.
# Even a no-op ALTER takes an ACCESS EXCLUSIVE lock, so a web restart during
# `maintain` waited on the job's locks (85 s on 2026-09-25; minutes before).
# Now a per-schema schema_version table records what has been applied: when
# current, apply_schema is one plain SELECT (no locks); otherwise pending
# migrations run once, in order, under an advisory lock so two processes
# never migrate the same schema concurrently. Future schema changes are new
# numbered functions appended to MIGRATIONS -- never edits to an old one.

_SCHEMA_VERSION_DDL = """CREATE TABLE IF NOT EXISTS {s}.schema_version (
    version    integer PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now())"""


def _schema_version(conn: psycopg.Connection, s: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{s}.schema_version",))
        if cur.fetchone()[0] is None:
            return 0
        cur.execute(f"SELECT coalesce(max(version), 0) FROM {s}.schema_version")
        return cur.fetchone()[0]


def _trgm_present(conn: psycopg.Connection, s: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{s}.word_lemma_trgm",))
        return cur.fetchone()[0] is not None


def apply_schema(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA) -> bool:
    """Bring `schema` up to the latest migration. Returns True if the pg_trgm
    lemma index exists (False if privileges didn't allow it -- the rest still
    works). Lock-free when already current (see the section comment)."""
    s = _safe_schema(schema)
    latest = MIGRATIONS[-1][0]
    if _schema_version(conn, s) >= latest:
        present = _trgm_present(conn, s)
        conn.commit()               # end the read-only transaction; don't sit idle-in-transaction
        return present
    lock_key = f"concordance-schema:{s}"
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (lock_key,))
    try:
        current = _schema_version(conn, s)       # another process may have migrated meanwhile
        for version, migrate in MIGRATIONS:
            if version <= current:
                continue
            migrate(conn, s)
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_VERSION_DDL.format(s=s))
                cur.execute(f"INSERT INTO {s}.schema_version (version) VALUES (%s) "
                            "ON CONFLICT (version) DO NOTHING", (version,))
            conn.commit()
        return _trgm_present(conn, s)
    except BaseException:
        conn.rollback()             # an aborted transaction would refuse the unlock below
        raise
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (lock_key,))
        conn.commit()


def _migration_0001_baseline(conn: psycopg.Connection, s: str) -> None:
    """Everything apply_schema did before versioning, unchanged: every
    statement is idempotent (IF NOT EXISTS / ON CONFLICT DO NOTHING), so on an
    existing database this is a no-op and simply stamps version 1."""
    from psycopg.types.json import Json

    from .. import calibration

    with conn.cursor() as cur:
        cur.execute(_SCHEMA_DDL.format(s=s))
        # rejected_word_reason_lemma_key_idx (created above) supersedes this --
        # same leading column, plus lemma_lc as a real sort key instead of just
        # an INCLUDE payload. IF NOT EXISTS above can't drop a pre-existing,
        # differently-named index, so evolve it explicitly here, same as the
        # ADD COLUMN IF NOT EXISTS lines below.
        cur.execute(f"DROP INDEX IF EXISTS {s}.rejected_word_reason_lemma_idx")
        # idempotent column additions (CREATE TABLE IF NOT EXISTS won't alter an
        # existing table, so evolve columns explicitly)
        cur.execute(f"ALTER TABLE {s}.book ADD COLUMN IF NOT EXISTS author text")
        # See concordance/archive_metadata.py -- word_count/distinct_nonstop_
        # word_count are computed locally from each book's own archive/ text
        # (Gutenberg boilerplate stripped first); publication_year is only
        # ever populated when Gutenberg's catalog metadata states an exact
        # year (rare -- confirmed live: 0/30 in a random corpus sample),
        # publication_era is the far-more-common free-text fallback (e.g.
        # "early 20th century") for when it doesn't. archive_path is repo-
        # relative (e.g. "archive/1601 -- Twain, Mark.txt"), not absolute --
        # portable across any checkout that also has archive/ populated.
        cur.execute(f"ALTER TABLE {s}.book ADD COLUMN IF NOT EXISTS publication_year integer")
        cur.execute(f"ALTER TABLE {s}.book ADD COLUMN IF NOT EXISTS publication_era text")
        cur.execute(f"ALTER TABLE {s}.book ADD COLUMN IF NOT EXISTS word_count integer")
        cur.execute(f"ALTER TABLE {s}.book ADD COLUMN IF NOT EXISTS distinct_nonstop_word_count integer")
        cur.execute(f"ALTER TABLE {s}.book ADD COLUMN IF NOT EXISTS archive_path text")
        # word_book's PK (word_id, book_id) serves word-id-leading lookups (does
        # this word belong to book X) for free, but the browse feature's author/
        # book listing endpoints join book -> word_book on book_id, a direction
        # the PK doesn't cover -- a full scan of the link table without this.
        cur.execute(f"CREATE INDEX IF NOT EXISTS word_book_book_id_idx ON {s}.word_book (book_id)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS book_author_idx ON {s}.book (author)")
        # so "un-rejecting" a word in the review webapp can produce a word row
        # with the same context a normally-kept word has, not a bare stub
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS pos text")
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS as_seen text")
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS sentence text")
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS chapter text")
        cur.execute(f"ALTER TABLE {s}.word_difficulty "
                    "ADD COLUMN IF NOT EXISTS archaic_confidence double precision")
        cur.execute(f"ALTER TABLE {s}.word_difficulty "
                    "ADD COLUMN IF NOT EXISTS difficulty double precision")
        cur.execute(f"ALTER TABLE {s}.word_difficulty "
                    "ADD COLUMN IF NOT EXISTS difficulty_factors jsonb")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS quiz_definition text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS quiz_def_source text")
        cur.execute(f"ALTER TABLE {s}.word_difficulty ADD COLUMN IF NOT EXISTS quizzable boolean")
        cur.execute(f"ALTER TABLE {s}.word_difficulty ADD COLUMN IF NOT EXISTS quizzable_reason text")
        # Raw Wordnik pronunciation, stored separately from ipa: fetching is a slow
        # rate-limited pass (~1 word/6s observed), converting to IPA is fast and
        # iterable — keeping them apart means a converter fix never costs a re-fetch.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS wordnik_pron_raw text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS wordnik_pron_type text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS wordnik_checked_at timestamptz")
        # Provenance of the IPA string currently in `ipa` -- 'kaikki'/'wordnik'/
        # 'local_wiktionary'/'oed', kept in sync on every write to `ipa` (see
        # compute_ipa, compute_audio, backfill_ipa_from_oed). NULL means legacy
        # (predates this column) and is always treated as US dialect, matching
        # every existing source's actual behavior -- see audio.ipa_dialect_for_source.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS ipa_source text")
        # When every IPA source was checked and came up empty, `ipa` stays ''
        # (the historical convention, not NULL -- see the one-time migration
        # note below) but this timestamp is what actually distinguishes
        # "confirmed nothing anywhere" from "never checked": compute_ipa
        # should only overwrite ipa='' when this is NULL or stale, never
        # re-walk the whole cascade on every run for words that already came
        # up empty. Same shape as wordnik_checked_at, one level up (covers
        # the whole cascade, not just the Wordnik tier).
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS ipa_checked_at timestamptz")
        # soft-delete flag for the review-and-prune web UI: pruned words stay in
        # place (history/audio/etc. intact) but drop out of every downstream view
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true")
        cur.execute(f"CREATE INDEX IF NOT EXISTS word_active_idx ON {s}.word (active)")
        # tracks words the pipeline itself rejected but a human rescued via the
        # review webapp's Rejected tab — distinct from words the pipeline kept
        # on its own, so this history survives even though rejected_word
        # (which had the original reason/detail) is deleted once promoted
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS rescued_from_reject boolean NOT NULL DEFAULT false")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS rescued_at timestamptz")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS rescued_reason text")
        # tracks a word an admin added directly via the "suggest a new word"
        # flow (webapp/backend/suggest_word.py) rather than one the ingest
        # pipeline discovered in a book — purely provenance/audit (a badge on
        # WordDetail, a future filter), NOT load-bearing for judge-skip
        # behavior: fetch_known_verdicts already treats ANY active=true word
        # as a cached "keep" for future books, admin-suggested or not.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS admin_suggested boolean NOT NULL DEFAULT false")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS admin_suggested_by text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS admin_suggested_at timestamptz")
        # tracks a word that came from the legacy vocab.defined bootstrap
        # (import_defined_words) -- purely provenance, same rationale as
        # admin_suggested above. definition_source is deliberately left alone
        # for these words (it keeps vocab.defined's own per-row source label,
        # e.g. "datamuse"/"phrontistery" -- confirmed with the user those are
        # more informative than a blanket tag), so this is the only place
        # that origin is recorded.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS vocab1_import boolean NOT NULL DEFAULT false")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS vocab1_import_at timestamptz")
        # audit trail for an admin hand-editing `definition` directly (PATCH
        # /api/words/{id}/definition) -- edited_by/edited_at note WHO and
        # WHEN, previous_definition holds the value it replaced. Deliberately
        # only the immediately-prior value, not a full history table: the
        # feature request asked for "the former definition," singular, and a
        # second edit simply overwrites this with what was live just before
        # it. Never surfaced on the frontend by design -- the edit itself
        # isn't indicated anywhere a reader would see it, only queryable
        # directly against the DB for admin review.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS previous_definition text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS definition_edited_by text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS definition_edited_at timestamptz")
        # persistent audit marker: this word was ever accepted with no dictionary
        # able to define it (a weaker validity signal than a normal keep — worth
        # a human glance). Sticky by design: never cleared even if `refill`
        # later finds a definition, so the history survives.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS flagged_undefined boolean NOT NULL DEFAULT false")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS flagged_undefined_at timestamptz")
        cur.execute(f"CREATE INDEX IF NOT EXISTS word_flagged_undefined_idx ON {s}.word (flagged_undefined)")
        # `deepen` writes these for a word that STILL has no definition after
        # every dictionary source (local + Free Dictionary/Wiktionary + Wordnik/
        # yourdictionary) has been tried — the DB-native version of deepen.py's
        # <book>.undefined.csv report, since ingest has no CSV to write one to.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS validity_label text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS validity_score double precision")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS validity_notes text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS suggested_correction text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS validity_checked_at timestamptz")
        # mw_backfill's own sticky "already attempted" marker -- set the moment a
        # word is checked against Merriam-Webster (hit OR miss), so a daily cron
        # never re-spends API quota / re-scrapes the same word twice. Never
        # cleared, same permanent-marker convention as flagged_undefined/
        # validity_checked_at above. first_known_use has no other home in this
        # schema (MW's own field; not attempted by any other source here).
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS mw_checked_at timestamptz")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS first_known_use text")
        # A human-review queue, not an auto-reject: validity_score.variant_reject_reason
        # (foreign-language / archaic-spelling-variant detection) was tried as a
        # hard cast-out gate and found to flag ~21% of the live vocabulary with
        # mostly false positives at real scale (haft/glaive/thurible/discomfit
        # all wrongly flagged) -- edit-distance similarity and cross-language
        # zipf comparison are both too weak a signal to auto-drop on. Flagging
        # here instead: the word is accepted/defined normally, but marked for a
        # human to glance at and manually prune via the review webapp if it's
        # really junk. Never cleared automatically, same sticky-marker pattern
        # as flagged_undefined.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS variant_flag_reason text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS variant_flag_note text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS variant_flagged_at timestamptz")
        cur.execute(f"CREATE INDEX IF NOT EXISTS word_variant_flag_idx ON {s}.word (variant_flag_reason) "
                    f"WHERE variant_flag_reason IS NOT NULL")
        # guessing_floor is load-bearing for compute_personal_difficulty's
        # response-probability model (see calibration.py); question_type/
        # direction are cheap diagnostic/future-proofing columns, not yet
        # consumed by anything, matching this project's existing habit of
        # capturing explainability data (difficulty_factors, validity_notes)
        # before it's actively used.
        cur.execute(f"ALTER TABLE {s}.quiz_answer ADD COLUMN IF NOT EXISTS guessing_floor double precision")
        cur.execute(f"ALTER TABLE {s}.quiz_answer ADD COLUMN IF NOT EXISTS question_type text")
        cur.execute(f"ALTER TABLE {s}.quiz_answer ADD COLUMN IF NOT EXISTS direction text")
        cur.execute(
            f"""INSERT INTO {s}.app_settings (key, value) VALUES ('quiz_feedback_timing', '{{"mode": "immediate"}}')
                ON CONFLICT (key) DO NOTHING""")
        # Hand-tuned, not auto-fit -- see calibration.py's module docstring
        # for why there isn't enough independent (multi-rater) data to fit
        # these from response data itself.
        cur.execute(
            f"""INSERT INTO {s}.app_settings (key, value) VALUES ('calibration_eta', %s)
                ON CONFLICT (key) DO NOTHING""", (Json({"value": calibration.DEFAULT_ETA}),))
        cur.execute(
            f"""INSERT INTO {s}.app_settings (key, value) VALUES ('calibration_scale', %s)
                ON CONFLICT (key) DO NOTHING""", (Json({"value": calibration.DEFAULT_SCALE}),))
        # Only valid once word.active exists (added a few lines up in this
        # same block) -- see _REJECTED_LEMMA_INDEX_DDL's own comment.
        cur.execute(_REJECTED_LEMMA_INDEX_DDL.format(s=s))
    # Commit the core schema BEFORE the privilege-sensitive extension steps:
    # their failure path rolls back, which used to discard every table and
    # column created above in the same still-open transaction.
    conn.commit()
    try:
        with conn.cursor() as cur:
            cur.execute(_TRGM_DDL.format(s=s))
        conn.commit()
    except psycopg.Error:
        conn.rollback()
    try:
        with conn.cursor() as cur:
            cur.execute(_VECTOR_DDL.format(s=s))
        conn.commit()
    except psycopg.Error:
        conn.rollback()


MIGRATIONS: list[tuple[int, object]] = [
    (1, _migration_0001_baseline),
]


def refresh_rejected_lemma_index(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA) -> None:
    """`concordance refresh-rejected-index` -- refreshes rejected_lemma_index
    (see its own schema comment) from the current state of rejected_word.
    CONCURRENTLY so RejectedView's search/letter-jump keep working against
    the old data while this runs, rather than blocking readers for however
    long the GROUP BY over rejected_word takes (~15-20s at real corpus
    scale) -- requires the unique index on lemma_lc already created in
    schema DDL. Meant to run on its own periodic schedule (daily cron), not
    on every write and not gated to `maintain`'s cadence -- curation search
    tolerates being a bit stale in a way `maintain`'s enrichment steps don't."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {s}.rejected_lemma_index")
    conn.commit()
