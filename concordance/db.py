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
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import csv

import psycopg
import requests

from .deepdef import _load_dotenv
from .model import RejectReason, normalize_pos

DEFAULT_SCHEMA = os.environ.get("CONCORDANCE_DB_SCHEMA", "concordance")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def database_url(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if "DATABASE_URL" not in os.environ:
        _load_dotenv(Path(".env"))
    return os.environ.get("DATABASE_URL", "").strip()


def _safe_schema(schema: str) -> str:
    if not _IDENT.match(schema):
        raise ValueError(f"unsafe schema name: {schema!r}")
    return schema


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

CREATE TABLE IF NOT EXISTS {s}.word_category (
    word_id     integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    category_id integer NOT NULL REFERENCES {s}.category(id) ON DELETE CASCADE,
    confidence  real,
    source      text,          -- 'usas-tagger' | 'wordnet' | 'llm' | 'dict-label'
    is_primary  boolean NOT NULL DEFAULT false,
    PRIMARY KEY (word_id, category_id)
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
    source       text,          -- 'commons' | 'azure' | 'none' (looked up, nothing found)
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

-- Backs load_verdict_cache's `WHERE reason IN (...)` scan (re-run once per
-- book during ingestion): without this the planner has no way to avoid a
-- sequential scan of the whole table, which only gets more expensive as the
-- corpus grows (measured: ~5.9s/full scan vs ~1.5s/index-only scan at ~40M
-- rows, and the gap widens with table size).
CREATE INDEX IF NOT EXISTS rejected_word_reason_lemma_idx ON {s}.rejected_word (reason) INCLUDE (lemma_lc);

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


def connect(url: str | None = None) -> psycopg.Connection:
    resolved = database_url(url)
    if not resolved:
        raise RuntimeError("no DATABASE_URL set (env or .env)")
    return psycopg.connect(resolved)


def apply_schema(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA) -> bool:
    """Create schema/tables if absent. Returns True if the pg_trgm index was
    created (False if privileges didn't allow it — the rest still works)."""
    from psycopg.types.json import Json

    from . import calibration

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_DDL.format(s=s))
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
    trgm = True
    try:
        with conn.cursor() as cur:
            cur.execute(_TRGM_DDL.format(s=s))
    except psycopg.Error:
        conn.rollback()
        trgm = False
    try:
        with conn.cursor() as cur:
            cur.execute(_VECTOR_DDL.format(s=s))
    except psycopg.Error:
        conn.rollback()
    conn.commit()
    return trgm


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


def _synonyms(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def _books(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def _read_master_rows(path: Path) -> list[dict]:
    """master_vocab.csv is tool-written with a full MASTER_COLUMNS header (it is not
    hand-edited in Excel like the per-book files), so a plain DictReader keeps every
    column — crucially date_added and source_book, which the vocab-only reader drops."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if (r.get("word") or "").strip()]


def sync_master(csv_path: Path, conn: psycopg.Connection,
                schema: str = DEFAULT_SCHEMA) -> dict:
    """Upsert every row of master_vocab.csv into the DB. Idempotent."""
    s = _safe_schema(schema)
    rows = _read_master_rows(Path(csv_path))
    stats = {"words": 0, "books": 0, "links": 0, "rows": len(rows)}
    seen_books: dict[str, int] = {}

    with conn.cursor() as cur:
        for r in rows:
            word = (r.get("word") or "").strip()
            if not word:
                continue
            definition = r.get("definition") or ""
            is_blank = not definition.strip()

            cur.execute(f"SELECT definition FROM {s}.word WHERE lemma_lc = lower(%s)", (word,))
            prior = cur.fetchone()
            old_definition = (prior[0] or "").strip() if prior else None

            cur.execute(
                f"""INSERT INTO {s}.word
                    (lemma, as_seen, definition, part_of_speech, ipa, sentence,
                     chapter, synonyms, etymology, definition_source, first_added,
                     flagged_undefined, flagged_undefined_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NULLIF(%s,'')::date,
                            %s, CASE WHEN %s THEN now() ELSE NULL END)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        as_seen=EXCLUDED.as_seen,
                        definition=COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition),
                        part_of_speech=EXCLUDED.part_of_speech,
                        ipa=COALESCE(NULLIF(EXCLUDED.ipa,''), {s}.word.ipa),
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter,
                        synonyms=CASE WHEN cardinality(EXCLUDED.synonyms) > 0
                                      THEN EXCLUDED.synonyms ELSE {s}.word.synonyms END,
                        etymology=COALESCE(NULLIF(EXCLUDED.etymology,''), {s}.word.etymology),
                        definition_source=COALESCE(NULLIF(EXCLUDED.definition_source,''),
                                                    {s}.word.definition_source),
                        first_added=LEAST(
                            {s}.word.first_added,
                            COALESCE(EXCLUDED.first_added, {s}.word.first_added)),
                        flagged_undefined={s}.word.flagged_undefined OR
                            (COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''),
                        flagged_undefined_at=CASE
                            WHEN {s}.word.flagged_undefined THEN {s}.word.flagged_undefined_at
                            WHEN COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''
                                THEN now()
                            ELSE {s}.word.flagged_undefined_at
                        END,
                        updated_at=now()
                    RETURNING id, definition""",
                (word, r.get("as_seen"), definition, normalize_pos(r.get("part_of_speech")),
                 r.get("ipa"), r.get("sentence"), r.get("chapter"), _synonyms(r.get("synonyms", "")),
                 r.get("etymology"), r.get("source"), (r.get("date_added") or ""),
                 is_blank, is_blank),
            )
            word_id, new_definition = cur.fetchone()
            stats["words"] += 1

            if old_definition and (new_definition or "").strip() != old_definition:
                _invalidate_definition_dependents(cur, s, word_id)

            for title in _books(r.get("source_book", "")):
                if title not in seen_books:
                    cur.execute(
                        f"""INSERT INTO {s}.book (title) VALUES (%s)
                            ON CONFLICT (title) DO UPDATE SET title=EXCLUDED.title
                            RETURNING id""", (title,))
                    seen_books[title] = cur.fetchone()[0]
                    stats["books"] += 1
                cur.execute(
                    f"""INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s,%s)
                        ON CONFLICT DO NOTHING""", (word_id, seen_books[title]))
                if cur.rowcount:
                    stats["links"] += 1
    conn.commit()
    return stats


def _invalidate_definition_dependents(cur, s: str, word_id: int) -> None:
    """Clear the downstream artifacts computed FROM word.definition text whose
    recompute is only-missing/NOT-EXISTS gated -- i.e. the ones that would
    otherwise silently go stale and never get revisited once this word's
    definition changes (e.g. the same lemma resolving to a different
    dictionary sense when a later book re-ingests it -- see the "changeful"
    bug this was written for: its quiz_definition was a redaction of an
    earlier, longer definition no longer stored anywhere).

    Deliberately NOT touched here: archaic, difficulty, and quizzable. All
    three fully recompute every row unconditionally whenever their command
    runs (no only-missing filter), so they self-correct on the next
    maintenance pass with no help -- invalidating them would just be a
    no-op that adds noise."""
    cur.execute(f"UPDATE {s}.word SET quiz_definition=NULL, quiz_def_source=NULL WHERE id=%s", (word_id,))
    cur.execute(f"DELETE FROM {s}.word_category WHERE word_id=%s", (word_id,))
    cur.execute(
        f"""UPDATE {s}.word_embedding SET definition_vector=NULL, definition_model=NULL, definition_source=NULL
            WHERE word_id=%s""",
        (word_id,))


def sync_book_results(conn, book_title: str, kept: list, rejected: list,
                       schema: str = DEFAULT_SCHEMA, author: str | None = None) -> dict:
    """Upsert one book's ingestion results straight into Postgres — no CSV, no
    hand-edit, no `finalize`. KEEP/UNSURE candidates go into word/word_book
    exactly like sync_master; DROPped ones go into rejected_word, one row per
    (book, lemma). Review/pruning happens afterward in the review webapp
    (word.active) rather than before promotion. Idempotent: re-running the
    same book updates both tables in place. `author` is COALESCEd on conflict
    so re-ingesting a book without a parsed author never blanks a known one."""
    s = _safe_schema(schema)
    stats = {"kept": 0, "rejected": 0, "cast_out": 0}

    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {s}.book (title, author) VALUES (%s, %s)
                ON CONFLICT (title) DO UPDATE SET title=EXCLUDED.title,
                    author=COALESCE(EXCLUDED.author, {s}.book.author)
                RETURNING id""", (book_title, author))
        book_id = cur.fetchone()[0]

        for c in kept:
            rep = c.representative
            definition = c.definition or ""
            is_blank = not definition.strip()

            # Fetched before the upsert so it reflects the pre-upsert value --
            # needed to tell "this lemma's definition just changed" apart from
            # "first time seeing this lemma" / "same value again", the only
            # case _invalidate_definition_dependents needs to fire for.
            cur.execute(f"SELECT definition FROM {s}.word WHERE lemma_lc = lower(%s)", (c.lemma,))
            prior = cur.fetchone()
            old_definition = (prior[0] or "").strip() if prior else None

            cur.execute(
                f"""INSERT INTO {s}.word
                    (lemma, as_seen, definition, part_of_speech, ipa, sentence,
                     chapter, synonyms, etymology, definition_source, first_added,
                     flagged_undefined, flagged_undefined_at,
                     variant_flag_reason, variant_flag_note, variant_flagged_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE,
                            %s, CASE WHEN %s THEN now() ELSE NULL END,
                            NULLIF(%s,''), NULLIF(%s,''), CASE WHEN %s <> '' THEN now() ELSE NULL END)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        as_seen=EXCLUDED.as_seen,
                        definition=COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition),
                        part_of_speech=EXCLUDED.part_of_speech,
                        ipa=COALESCE(NULLIF(EXCLUDED.ipa,''), {s}.word.ipa),
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter,
                        synonyms=CASE WHEN cardinality(EXCLUDED.synonyms) > 0
                                      THEN EXCLUDED.synonyms ELSE {s}.word.synonyms END,
                        etymology=COALESCE(NULLIF(EXCLUDED.etymology,''), {s}.word.etymology),
                        definition_source=COALESCE(NULLIF(EXCLUDED.definition_source,''),
                                                    {s}.word.definition_source),
                        flagged_undefined={s}.word.flagged_undefined OR
                            (COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''),
                        flagged_undefined_at=CASE
                            WHEN {s}.word.flagged_undefined THEN {s}.word.flagged_undefined_at
                            WHEN COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition, '') = ''
                                THEN now()
                            ELSE {s}.word.flagged_undefined_at
                        END,
                        variant_flag_reason=COALESCE(EXCLUDED.variant_flag_reason, {s}.word.variant_flag_reason),
                        variant_flag_note=COALESCE(EXCLUDED.variant_flag_note, {s}.word.variant_flag_note),
                        variant_flagged_at=COALESCE(EXCLUDED.variant_flagged_at, {s}.word.variant_flagged_at),
                        updated_at=now()
                    RETURNING id, definition""",
                (c.lemma, rep.surface if rep else "", definition,
                 normalize_pos(c.part_of_speech or c.pos), c.ipa,
                 rep.sentence if rep else "", rep.chapter if rep else "",
                 list(c.synonyms), c.etymology,
                 c.definition_source or ", ".join(c.validity_sources),
                 is_blank, is_blank,
                 c.variant_flag_reason, c.variant_flag_note, c.variant_flag_reason))
            word_id, new_definition = cur.fetchone()
            stats["kept"] += 1

            if old_definition and (new_definition or "").strip() != old_definition:
                _invalidate_definition_dependents(cur, s, word_id)

            cur.execute(
                f"""INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s,%s)
                    ON CONFLICT DO NOTHING""", (word_id, book_id))

        for c in rejected:
            rep = c.representative
            cur.execute(
                f"""INSERT INTO {s}.rejected_word
                    (book_id, lemma, reason, detail, count, zipf, pos, as_seen, sentence, chapter)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (book_id, lemma_lc) DO UPDATE SET
                        reason=EXCLUDED.reason, detail=EXCLUDED.detail,
                        count=EXCLUDED.count, zipf=EXCLUDED.zipf,
                        pos=EXCLUDED.pos, as_seen=EXCLUDED.as_seen,
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter""",
                (book_id, c.lemma, c.reject_reason.value if c.reject_reason else None,
                 c.interesting_reason or None, c.count, c.zipf,
                 c.pos, rep.surface if rep else None,
                 rep.sentence if rep else None, rep.chapter if rep else None))
            stats["rejected"] += 1

            # A symbol/proper-noun rejection can happen for a lemma that's
            # already an active word from an earlier book (pipeline.py's
            # post-enrichment junk-POS check now applies on every
            # re-encounter, not just the first) -- cast it out here too, same
            # as refill/deepen already do for their own junk-POS
            # resolutions. A no-op UPDATE (0 rows) for a lemma with no
            # existing word row, so this is safe to run unconditionally
            # rather than needing to first check whether one exists.
            if c.reject_reason in (RejectReason.PROPER_NOUN, RejectReason.NUMERIC_OR_SYMBOL):
                cur.execute(
                    f"""UPDATE {s}.word SET active=false, updated_at=now()
                        WHERE lemma_lc = lower(%s) AND active""",
                    (c.lemma,))
                stats["cast_out"] += cur.rowcount

    conn.commit()
    return stats


_POS_TO_TAGGER = {"noun": "NOUN", "verb": "VERB", "adjective": "ADJ", "adverb": "ADV"}


def fill_definitions(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                     use_web: bool = False, model_path: str | None = None,
                     recheck_after_days: int = 14) -> dict:
    """The single definition-acquisition pass for words whose definition is
    still blank: one candidate SELECT, one lexicon build, one per-row trip
    through resolve.resolve_definition at whatever depth `use_web` allows
    (YOURDICT without it, WEB with it) -- replaces what used to be two
    separate passes (refill_definitions then deepen_definitions) each
    re-entering the cascade at Tier LOCAL, the second one's local/free
    attempts always redundant with the first's on the same lemma.

    A word that resolves to a symbol/proper-noun-only sense (see
    model.junk_pos_reason -- the same gate ingest's pipeline.process()
    applies) is cast out (active=false) instead of being filled in: these
    words were ACCEPTED with no definition at all, so this is the first
    real evidence of what they actually are. Never clears flagged_undefined
    -- that flag is a permanent "this one needed a second look" marker, not
    a live status (see apply_schema).

    Whatever's still undefined after the full cascade gets a
    validity_score.estimate() written to word.validity_* -- the DB-native
    version of deepen.py's <book>.undefined.csv report, so a word that's
    both flagged_undefined AND scored likely-artifact is an obvious prune
    candidate, not silent noise in the accepted list. WEB (when use_web) is
    tried for EVERY word nothing else defined, regardless of that estimate
    -- there used to be a pre-gate skipping WEB for anything already scored
    likely-artifact, on the theory that a web search for OCR noise was
    wasted effort; dropped because that same "probably not a real word"
    signal is exactly the rare/archaic vocabulary this project's judge
    rubric exists to prize, and a word simply not matching any of the
    dictionaries checked earlier is not strong enough evidence to skip the
    one source most likely to catch what they all missed.

    `recheck_after_days`: a word already scored by validity_score recently
    is skipped entirely rather than re-run through the full cascade (Wordnik
    pacing included) again -- without this, every `maintain` run would
    re-grind the entire permanently-undefined tail through Wordnik/web-search
    forever, not just the first time it's ever seen."""
    from . import deepdef, localdict, resolve, validity_score
    from .config import Config
    from .dictionary import make_session
    from .model import Candidate, Occurrence, junk_pos_reason

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, part_of_speech, sentence, chapter, as_seen
                FROM {s}.word
                WHERE coalesce(definition,'') = ''
                  AND (validity_checked_at IS NULL
                       OR validity_checked_at < now() - (%s * interval '1 day'))
                ORDER BY flagged_undefined_at NULLS LAST, lemma""" +
            (f" LIMIT {int(limit)}" if limit else ""), (recheck_after_days,))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "defined": 0, "still_undefined": 0, "cast_out": 0}
    if not rows:
        return stats

    lexicon = localdict.build_lexicon(conn, {lemma.lower() for _, lemma, *_ in rows})
    session = make_session()
    key = deepdef.wordnik_key()
    max_tier = resolve.Tier.WEB if use_web else resolve.Tier.YOURDICT

    llm = None
    if use_web:
        cfg = Config()
        mp = model_path or cfg.model_path
        if mp and Path(mp).exists():
            from llama_cpp import Llama
            llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    with conn.cursor() as cur:
        for i, (wid, lemma, pos, sentence, chapter, as_seen) in enumerate(rows, 1):
            cand = Candidate(lemma=lemma, pos=_POS_TO_TAGGER.get((pos or "").lower(), ""))
            if sentence:
                cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "",
                                                    surface=as_seen or lemma))
            # llm=None here even when a model is loaded: max_tier already
            # includes WEB when use_web is set, but resolve_definition would
            # try it before validity_score ever runs -- deliberately not
            # skipped here (the likely-artifact pre-gate was removed; WEB is
            # now the true last resort, tried for anything nothing else
            # defined), just sequenced so validity_score's estimate() always
            # gets computed and is available to write if WEB also misses.
            est = None
            found = resolve.resolve_definition(
                cand, max_tier=max_tier, lexicon=lexicon, session=session,
                wordnik_key=key, llm=None) is not None
            if not found:
                est = validity_score.estimate(lemma, session=session, sentence=sentence or "")
                if llm is not None:
                    from . import websearch
                    found = websearch.define_via_web(cand, llm)
                    if found:
                        resolve.apply_pos_repair(cand, lexicon)

            # validity_score.variant_reject_reason (foreign-word / archaic-
            # spelling-variant detection) is a human-review flag here too,
            # same as pipeline.py: NOT a hard cast-out (a real-scale dry-run
            # sweep against the live word table found it flags ~21% of
            # already-accepted vocabulary, mostly genuine rare words --
            # haft, glaive, thurible, discomfit, kickshaw -- rather than the
            # junk it was built to catch) but still worth recording so a
            # human can review + prune via the webapp.
            reason = junk_pos_reason(cand.part_of_speech) if found else None
            variant = validity_score.variant_reject_reason(lemma) if (found and not reason) else None
            if reason:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=%s, active=false, updated_at=now()
                        WHERE id=%s""",
                    (cand.definition, cand.definition_source,
                     normalize_pos(cand.part_of_speech), wid))
                stats["cast_out"] += 1
            elif found:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=COALESCE(NULLIF(%s,''), part_of_speech),
                            ipa=COALESCE(NULLIF(%s,''), ipa),
                            etymology=COALESCE(NULLIF(%s,''), etymology),
                            synonyms=CASE WHEN %s THEN %s ELSE synonyms END,
                            variant_flag_reason=COALESCE(%s, variant_flag_reason),
                            variant_flag_note=COALESCE(%s, variant_flag_note),
                            variant_flagged_at=CASE WHEN %s::text IS NOT NULL THEN now() ELSE variant_flagged_at END,
                            updated_at=now()
                        WHERE id=%s""",
                    (cand.definition, cand.definition_source, normalize_pos(cand.part_of_speech),
                     cand.ipa, cand.etymology, bool(cand.synonyms), list(cand.synonyms),
                     variant[0].value if variant else None, variant[1] if variant else None,
                     variant[0].value if variant else None, wid))
                stats["defined"] += 1
            else:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            validity_label=%s, validity_score=%s, validity_notes=%s,
                            suggested_correction=%s, validity_checked_at=now()
                        WHERE id=%s""",
                    (est.label, est.score, est.notes, est.suggestion or None, wid))
                stats["still_undefined"] += 1
            # Committed every word, not batched every 200: each iteration's
            # slow network call (Wordnik/yourdictionary, rate-limited) can
            # itself take longer than the whole old batch interval, so a
            # 200-row batch left one transaction open for tens of minutes at
            # a time -- long enough to block a webapp restart's schema-check
            # ALTER TABLE, which needs an ACCESS EXCLUSIVE lock on this same
            # table and would otherwise queue behind it. Per-word commits cap
            # any held lock at one row's write.
            conn.commit()
            if i % 25 == 0:
                print(f"  ...{i}/{len(rows)} words attempted "
                      f"({stats['defined']} defined, {stats['still_undefined']} still undefined)")
    return stats


def refill_definitions(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Standalone `concordance refill`: the cheap/free tiers only (LOCAL,
    FREE), never Wordnik/yourdictionary/web -- a thin wrapper around
    fill_definitions for the independent, human-scheduled command. Doesn't
    write validity_score (that's specifically deepen/fill_definitions'
    deep-pass signal; a word cheap tiers missed hasn't earned an artifact
    verdict yet, it just hasn't been tried deeply). Returns refill's
    historical stat vocabulary (filled/still_missing) rather than
    fill_definitions' (defined/still_undefined) for backward compatibility
    with existing callers/scripts."""
    from . import localdict, resolve, validity_score
    from .dictionary import make_session
    from .model import Candidate, Occurrence, junk_pos_reason

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, part_of_speech, sentence, chapter, as_seen
                FROM {s}.word WHERE coalesce(definition,'') = ''
                ORDER BY flagged_undefined_at NULLS LAST, lemma""" +
            (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "filled": 0, "still_missing": 0, "cast_out": 0}
    if not rows:
        return stats

    lexicon = localdict.build_lexicon(conn, {lemma.lower() for _, lemma, *_ in rows})
    session = make_session()

    with conn.cursor() as cur:
        for i, (wid, lemma, pos, sentence, chapter, as_seen) in enumerate(rows, 1):
            cand = Candidate(lemma=lemma, pos=_POS_TO_TAGGER.get((pos or "").lower(), ""))
            if sentence:
                cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "",
                                                    surface=as_seen or lemma))
            resolve.resolve_definition(cand, max_tier=resolve.Tier.FREE, lexicon=lexicon, session=session)
            reason = junk_pos_reason(cand.part_of_speech)
            variant = validity_score.variant_reject_reason(lemma) if (cand.definition and not reason) else None
            if reason:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=%s, active=false, updated_at=now()
                        WHERE id=%s""",
                    (cand.definition, cand.definition_source,
                     normalize_pos(cand.part_of_speech), wid))
                stats["cast_out"] += 1
            elif cand.definition:
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s,
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=COALESCE(NULLIF(%s,''), part_of_speech),
                            ipa=COALESCE(NULLIF(%s,''), ipa),
                            etymology=COALESCE(NULLIF(%s,''), etymology),
                            synonyms=CASE WHEN %s THEN %s ELSE synonyms END,
                            variant_flag_reason=COALESCE(%s, variant_flag_reason),
                            variant_flag_note=COALESCE(%s, variant_flag_note),
                            variant_flagged_at=CASE WHEN %s::text IS NOT NULL THEN now() ELSE variant_flagged_at END,
                            updated_at=now()
                        WHERE id=%s""",
                    (cand.definition, cand.definition_source, normalize_pos(cand.part_of_speech),
                     cand.ipa, cand.etymology, bool(cand.synonyms), list(cand.synonyms),
                     variant[0].value if variant else None, variant[1] if variant else None,
                     variant[0].value if variant else None, wid))
                stats["filled"] += 1
            else:
                stats["still_missing"] += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return stats


def deepen_definitions(conn, schema: str = DEFAULT_SCHEMA, use_web: bool = False,
                       model_path: str | None = None, limit: int = 0) -> dict:
    """Standalone `concordance deepen`: a thin wrapper around fill_definitions
    with no cooldown (recheck_after_days=0) -- an explicit, human-invoked
    deepen run should always retry the undefined tail regardless of when it
    was last checked; the cooldown exists to stop `maintain`'s *automatic*
    re-grinding, not to gate a deliberate one-off command."""
    return fill_definitions(conn, schema, limit=limit, use_web=use_web,
                            model_path=model_path, recheck_after_days=0)


def mw_backfill(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                use_scrape: bool = True, headless: bool = False,
                scrape_timeout_ms: int = 10000) -> dict:
    """Standalone `concordance mw-backfill`: check Merriam-Webster (the
    scripts/lookup_mw.py cascade -- API first, then a Playwright site-scrape
    fallback for words the API misses) for every accepted word that's still
    undefined AND not already written off as likely-artifact -- exactly the
    words fill_definitions'/deepen's own cascade (Free Dictionary/Wiktionary/
    Wordnik/yourdictionary/web-search) couldn't resolve, where MW's own
    Collegiate coverage sometimes succeeds anyway.

    `scrape_timeout_ms` (default 10s, half lookup_mw.py's own 20s default):
    a genuine miss on the live site still costs the full page-load wait --
    confirmed empirically (MW's "isn't in the dictionary" suggestions page
    never satisfies the entry-container selector, so it always times out
    rather than returning fast) -- and most candidates reaching this scrape
    tier already failed Wordnik/yourdictionary/web-search too, so misses here
    are the common case, not the exception. A warmed-up profile with a
    valid cleared cookie loads in well under a second; 20s was sized for
    lookup_mw.py's interactive one-or-few-word use, where patience costs
    nothing, not a batch scan that may hit this tier hundreds of times.

    `mw_checked_at` is a STICKY marker (never cleared) set the moment a word
    is attempted here, hit or miss -- re-querying MW for the same word
    tomorrow is very unlikely to produce a different answer, so this is a
    permanent "already tried" flag, same convention as flagged_undefined,
    not a recheck-after-N-days cooldown. A repeated daily run just keeps
    working through whatever's left.

    Only definition/part_of_speech/etymology/definition_source/
    first_known_use are ever written -- NOT ipa. MW's pronunciation field is
    its own proprietary respelling, not true IPA (no ahd.py-style converter
    exists for it yet), and word.ipa is trusted elsewhere (audio.py's Azure
    TTS synthesis) to actually contain IPA; writing MW's respelling there
    would silently corrupt that pipeline. Never overwrites an existing
    non-blank value in any of the columns it does write (COALESCE(NULLIF(...))
    guard), same as fill_definitions.

    The API's free tier caps out at 1000 queries/day (tracked in
    concordance/mw.py's own on-disk cache, shared with lookup_mw.py -- a word
    either tool already looked up today never costs a second query). Once
    that cap is hit, THE WHOLE RUN STOPS (not just the API tier) -- remaining
    candidates are left untouched (mw_checked_at not set) for tomorrow's run,
    rather than falling through to an unbounded scrape-only tail that would
    hammer the live site far harder than the polite, quota-capped API path.

    Commits every word (not batched), same lock-safety rationale as
    fill_definitions: a long-running batch holding one open transaction can
    block a webapp restart's schema-check ALTER TABLE, which needs an ACCESS
    EXCLUSIVE lock on this same table."""
    from contextlib import ExitStack

    from . import mw as mw_module
    from .dictionary import make_session
    from .model import junk_pos_reason

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, part_of_speech
                FROM {s}.word
                WHERE active
                  AND coalesce(definition,'') = ''
                  AND (validity_label IS NULL OR validity_label IN ('uncertain','likely-valid'))
                  AND mw_checked_at IS NULL
                ORDER BY lemma""" +
            (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"attempted": 0, "defined": 0, "cast_out": 0, "no_entry": 0,
             "quota_stopped": False, "remaining": 0}
    if not rows:
        return stats

    api_key = mw_module.mw_api_key()
    session = make_session()
    scraper = None

    with conn.cursor() as cur, ExitStack() as stack:
        for i, (wid, lemma, pos) in enumerate(rows, 1):
            if api_key and mw_module.quota_exhausted():
                stats["quota_stopped"] = True
                stats["remaining"] = len(rows) - i + 1
                break
            stats["attempted"] += 1

            # exact_matches: MW's search is fuzzy and will return a same-
            # ballpark idiom for a query that isn't a real headword at all
            # (confirmed on live data -- see exact_matches' own docstring),
            # so a returned entry only counts here if its own headword
            # literally is this word. If the API's fuzzy hit doesn't survive
            # that filter, still give the scrape tier its own chance (the
            # site's ranking isn't guaranteed identical) before calling it a
            # genuine miss.
            entries = mw_module.exact_matches(
                mw_module.lookup_api(lemma, api_key, session) if api_key else [], lemma)
            if not entries and use_scrape:
                if scraper is None:
                    from . import mw_scrape
                    scraper = stack.enter_context(mw_scrape.MWScraper(headless=headless))
                entries = mw_module.exact_matches(
                    scraper.lookup(lemma, timeout_ms=scrape_timeout_ms), lemma)

            if not entries:
                cur.execute(f"UPDATE {s}.word SET mw_checked_at=now() WHERE id=%s", (wid,))
                stats["no_entry"] += 1
                conn.commit()
                if i % 25 == 0:
                    print(f"  ...{i}/{len(rows)} words attempted ({stats['defined']} defined, "
                          f"{stats['no_entry']} no MW entry)")
                continue

            tagger_pos = _POS_TO_TAGGER.get((pos or "").lower(), "")
            entry = mw_module.pick_entry(entries, tagger_pos)
            definition = "; ".join(entry.definitions)
            resolved_pos = normalize_pos(entry.part_of_speech)
            # is_foreign_pos checks the RAW (pre-normalize_pos) string --
            # MW's "<Language> noun" foreign-loanword tag is a capitalized
            # demonym, a signal normalize_pos's lowercasing would destroy.
            reason = junk_pos_reason(resolved_pos) or (
                RejectReason.FOREIGN_LANGUAGE if mw_module.is_foreign_pos(entry.part_of_speech) else None)

            if reason:
                # Same safety net as fill_definitions: an ACCEPTED word whose
                # only resolvable sense turns out to be a symbol/proper-noun/
                # foreign-language entry gets cast out now that there's real
                # evidence of what it is, rather than sitting active with a
                # junk definition.
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=%s, definition_source=%s, part_of_speech=%s,
                            active=false, mw_checked_at=now(), updated_at=now()
                        WHERE id=%s""",
                    (definition, entry.source, resolved_pos, wid))
                stats["cast_out"] += 1
            else:
                # COALESCE(NULLIF(%s,''), column) -- new value preferred, existing
                # kept only if the new one is blank -- same direction as
                # fill_definitions' own UPDATE. Not the reverse: definition_source
                # (and, in principle, the others) can carry a stale non-blank
                # value from history even while definition itself is blank (the
                # candidate filter only guarantees the latter), so getting this
                # backwards silently keeps old metadata under a brand-new
                # definition instead of recording MW as its real source --
                # caught empirically on a live word ("aglance") during testing.
                cur.execute(
                    f"""UPDATE {s}.word SET
                            definition=COALESCE(NULLIF(%s,''), definition),
                            definition_source=COALESCE(NULLIF(%s,''), definition_source),
                            part_of_speech=COALESCE(NULLIF(%s,''), part_of_speech),
                            etymology=COALESCE(NULLIF(%s,''), etymology),
                            first_known_use=COALESCE(NULLIF(%s,''), first_known_use),
                            mw_checked_at=now(), updated_at=now()
                        WHERE id=%s""",
                    (definition, entry.source, resolved_pos, entry.etymology,
                     entry.first_known_use, wid))
                stats["defined"] += 1
            conn.commit()
            if i % 25 == 0:
                print(f"  ...{i}/{len(rows)} words attempted ({stats['defined']} defined, "
                      f"{stats['no_entry']} no MW entry, {stats['cast_out']} cast out)")
    return stats


def import_defined_words(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                         commit_every: int = 500) -> dict:
    """One-time/occasional bootstrap: pull genuinely-new terms from the
    legacy `vocab.defined` table (a predecessor project's term/POS/definition
    list, collected outside any book) into `word` as book-less words -- no
    `word_book` row, since there's no book occurrence to attach. They pick up
    all the normal `maintain` processing (classify, difficulty, quizdef,
    etc.) the next time it runs, same as any book-sourced word; only the
    three purely book/author-relatedness computations
    (compute_book_similarity/compute_author_similarity/compute_author_clustering)
    read exclusively FROM word_book and so simply won't see these words --
    a correct no-op, not something this import needs to handle.

    Excludes: phrases (the `phrase` flag column, confirmed to match 100% of
    space-containing terms -- skipped outright per instruction, not just
    deprioritized), rows flagged `bad=1`, terms already in `word`, and terms
    ever rejected in ANY book for ANY reason (not just "hard" rejection
    reasons -- confirmed with the user). Multiple `vocab.defined` rows per
    term (different senses/POS) are collapsed to the single richest row
    before insert, since `word.lemma_lc` is UNIQUE.

    `vocab.defined` has no ipa/etymology/synonyms columns. fill_definitions'
    gate (`WHERE coalesce(definition,'') = ''`) will never revisit these
    words to backfill ipa/etymology since they arrive with a non-blank
    definition, so this reuses localdict.build_lexicon (the same
    vocab.wiktionary lookup the ingestion pipeline already does) once for
    the whole batch to best-effort fill those two; synonyms stays blank (no
    source has it).

    Commits every `commit_every` words, not once at the end -- same
    crash-safety rationale as classify_and_store (a run over ~11k candidates
    that dies partway through should keep whatever it already inserted)."""
    from . import localdict

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('vocab.defined')")
        if cur.fetchone()[0] is None:
            return {"available": False, "candidates": 0, "imported": 0, "skipped_conflict": 0}

        cur.execute(
            f"""SELECT DISTINCT ON (lower(d.term))
                    d.term, d.part_of_speech,
                    COALESCE(NULLIF(d.corrected_definition,''), d.definition),
                    d.definition_source
                FROM vocab.defined d
                WHERE d.phrase IS DISTINCT FROM 1
                  AND position(' ' in d.term) = 0
                  AND COALESCE(d.bad,0) != 1
                  AND NOT EXISTS (SELECT 1 FROM {s}.word w WHERE w.lemma_lc = lower(d.term))
                  AND NOT EXISTS (SELECT 1 FROM {s}.rejected_word r WHERE r.lemma_lc = lower(d.term))
                ORDER BY lower(d.term),
                    (d.part_of_speech IS NOT NULL AND upper(d.part_of_speech) NOT IN ('', 'TBD')) DESC,
                    length(COALESCE(NULLIF(d.corrected_definition,''), d.definition)) DESC,
                    d.id"""
            + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"available": True, "candidates": len(rows), "imported": 0, "skipped_conflict": 0}
    if not rows:
        return stats

    lexicon = localdict.build_lexicon(conn, {term.lower() for term, *_ in rows})

    with conn.cursor() as cur:
        for i, (term, pos, definition, def_source) in enumerate(rows, 1):
            raw_pos = "" if (pos or "").strip().upper() == "TBD" else (pos or "")
            norm_pos = normalize_pos(raw_pos)

            ipa = etymology = ""
            entries = lexicon.get(term.lower())
            if entries:
                match = next((e for e in entries if normalize_pos(e[0]) == norm_pos), entries[0])
                ipa, etymology = match[2], match[3]

            cur.execute(
                f"""INSERT INTO {s}.word
                        (lemma, as_seen, definition, part_of_speech, ipa, etymology,
                         definition_source, first_added)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE)
                    ON CONFLICT (lemma_lc) DO NOTHING""",
                (term, term, definition, norm_pos, ipa, etymology,
                 def_source or "vocab.defined import"))
            if cur.rowcount:
                stats["imported"] += 1
            else:
                stats["skipped_conflict"] += 1
            if i % commit_every == 0:
                conn.commit()
    conn.commit()
    return stats


_PLURAL_OF_RE = re.compile(
    r"^(?:alternative |archaic |dialectal |obsolete )?plural (?:form )?of (\S+?)\.?$",
    re.IGNORECASE,
)


def dedupe_plural_definitions(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                              use_web: bool = True, model_path: str | None = None) -> dict:
    """`concordance dedupe-plurals`: a definition that just says "plural of X"
    isn't real vocabulary content -- the word IS real (a dictionary vouched
    for it as its own headword), but it's redundant scaffolding once X exists
    as its own properly-defined entry. quizdef.quizzable() already excludes
    these from quizzes (_VARIANT_RE matches "plural of"), so this isn't a
    correctness fix -- it's consolidation: for every such word, resolve its
    singular X and soft-delete the plural (active=false, same reversible
    pattern as every other removal in this codebase -- never a hard delete).

    Only considers currently-active words -- an already-pruned plural isn't
    cluttering anything and doesn't need reprocessing. Idempotent: a plural
    already deactivated by an earlier run won't be selected again.

    Three outcomes per plural, tracked separately:
      - `linked`    the singular already exists and is active -- just needed
                     the plural deactivated.
      - `left_inactive` the singular exists but is currently inactive.
                     DELIBERATELY left untouched, whatever the reason it's
                     inactive -- checked against real data before building
                     this: every one of the handful of cases found already
                     has a real definition (not a blank/unresolved one),
                     meaning "inactive" here is near-certainly a deliberate
                     decision (a human prune via the review webapp, or a
                     justified automated cast-out) that a plural merely
                     existing is not good evidence to override.
      - `created`    the singular didn't exist at all -- a new word row,
                     resolved through the full cascade (same as any newly
                     ingested word), inheriting the plural's own
                     sentence/chapter context since there's no literal book
                     occurrence of the singular form to draw from. Cast out
                     (active=false) rather than accepted if the resolution
                     itself reveals a symbol/proper-noun sense
                     (junk_pos_reason) -- same gate every other
                     definition-acceptance path applies. Otherwise still
                     gets flagged_undefined if the cascade can't define it,
                     same as any other word -- refill/deepen will keep
                     trying on later runs."""
    from . import deepdef, localdict, resolve
    from .config import Config
    from .dictionary import make_session
    from .model import Candidate, Occurrence, junk_pos_reason

    s = _safe_schema(schema)
    # Broad SQL prefilter (plain substring, case-insensitive) + precise
    # Python-side regex match below -- NOT a direct `~*` on
    # _PLURAL_OF_RE.pattern. Postgres's regex dialect is POSIX ERE, which
    # doesn't support Python re's non-greedy `+?`, so the exact same pattern
    # string silently matches a different (smaller) row set in each engine
    # -- confirmed empirically. A plain literal substring has no such
    # quantifiers so it's safe to run directly in Postgres as a superset
    # filter; _PLURAL_OF_RE.match() (needed anyway, to parse the singular
    # out) does the real, precise matching in Python.
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, definition, part_of_speech, sentence, chapter, as_seen
                FROM {s}.word
                WHERE active AND definition ~* 'plural of'
                ORDER BY id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "linked": 0, "left_inactive": 0, "created": 0,
             "cast_out": 0, "still_undefined": 0, "unparsed": 0}
    if not rows:
        return stats

    parsed = []
    for wid, lemma, defn, pos, sentence, chapter, as_seen in rows:
        m = _PLURAL_OF_RE.match((defn or "").strip())
        if not m:
            stats["unparsed"] += 1
            continue
        singular = m.group(1).strip(".,;").lower()
        parsed.append((wid, lemma, pos, sentence, chapter, as_seen, singular))
    if not parsed:
        return stats

    lexicon = localdict.build_lexicon(conn, {sing for *_, sing in parsed})
    session = make_session()
    key = deepdef.wordnik_key()
    max_tier = resolve.Tier.WEB if use_web else resolve.Tier.YOURDICT

    llm = None
    if use_web:
        cfg = Config()
        mp = model_path or cfg.model_path
        if mp and Path(mp).exists():
            from llama_cpp import Llama
            llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    with conn.cursor() as cur:
        for plural_id, plural_lemma, plural_pos, sentence, chapter, as_seen, singular in parsed:
            cur.execute(f"SELECT id, active FROM {s}.word WHERE lemma_lc = %s", (singular,))
            existing = cur.fetchone()

            if existing and existing[1]:
                stats["linked"] += 1

            elif existing:
                stats["left_inactive"] += 1

            else:
                cand = Candidate(lemma=singular, pos=_POS_TO_TAGGER.get((plural_pos or "").lower(), ""))
                if sentence:
                    cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "",
                                                        surface=singular))
                found = resolve.resolve_definition(
                    cand, max_tier=max_tier, lexicon=lexicon, session=session,
                    wordnik_key=key, llm=llm) is not None
                reason = junk_pos_reason(cand.part_of_speech) if found else None
                is_blank = not found
                cur.execute(
                    f"""INSERT INTO {s}.word
                            (lemma, as_seen, definition, part_of_speech, sentence, chapter,
                             definition_source, first_added, active, flagged_undefined, flagged_undefined_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE, %s, %s, CASE WHEN %s THEN now() ELSE NULL END)
                        ON CONFLICT (lemma_lc) DO UPDATE SET active=EXCLUDED.active, updated_at=now()""",
                    (singular, singular, cand.definition, normalize_pos(cand.part_of_speech),
                     sentence or "", chapter or "", cand.definition_source,
                     not reason, is_blank, is_blank))
                if reason:
                    stats["cast_out"] += 1
                else:
                    stats["created"] += 1
                    if is_blank:
                        stats["still_undefined"] += 1

            cur.execute(f"UPDATE {s}.word SET active=false, updated_at=now() WHERE id=%s", (plural_id,))
            conn.commit()
    return stats


# "Synonym of X" from a source that embedded a real gloss right there --
# either quoted ("...") or bare -- e.g. 'Synonym of nithing ("a coward...").'
_SYNONYM_OF_RE = re.compile(
    r"^synonym of ([^(\n]+?)\s*(?:\(\s*[“\"]?(.+?)[”\"]?\s*\))?\.?\s*$", re.IGNORECASE)
# Wiktionary REST occasionally leaves a raw CSS rule trailing a gloss --
# ".mw-parser-output .defdate{font-size:smaller}" -- a copy-through of the
# page's own stylesheet class, not content.
_CSS_JUNK_RE = re.compile(r"\.mw-parser-output[^{]*\{[^}]*\}")


def expand_synonym_definitions(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                               use_web: bool = True, model_path: str | None = None) -> dict:
    """`concordance expand-synonyms`: a definition that just says "synonym of
    X" is a real data-quality problem, not merely a quizzability one (unlike
    "plural of X", quizdef._VARIANT_RE doesn't even exclude these from
    quizzing today -- "synonym" was never in its word list). But the fix is
    the OPPOSITE of dedupe-plurals': a synonym is a genuinely distinct
    headword worth keeping on its own, not redundant scaffolding for another
    surface form of the same word -- so this never deletes/deactivates the
    word carrying the "synonym of X" definition. It replaces that
    definition with real content, and separately assesses X (the synonym
    target) for inclusion in the corpus, mirroring dedupe-plurals' handling
    of a plural's singular.

    Three ways a word's definition gets upgraded:
      - the source already embedded a real gloss right in the cross-
        reference -- 'Synonym of nithing ("a coward...").' -- extracted
        directly, no lookup needed.
      - the source put the real definition on a later line after the
        "Synonym of X." sentence (seen in a couple of live rows) -- used
        as-is.
      - bare 'Synonym of X.' with nothing else -- X's OWN definition is
        reused (or freshly resolved through the same cascade every other
        definition-acceptance path uses, creating X as a new word if it
        doesn't exist yet). Never done if X exists but is currently
        inactive -- checked against real data before building this: the one
        live case is inactive WITH a real definition already, meaning
        "inactive" here is (as with dedupe-plurals) near-certain evidence of
        a deliberate earlier decision that a bare synonym pointer is not
        good reason to override, and definitely not good reason to import
        that same word's content into a DIFFERENT word's definition.
        Likewise never done if a fresh resolution of X reveals a
        symbol/proper-noun sense (junk_pos_reason) -- X still gets created,
        cast out (same as dedupe-plurals), but W's definition is left
        untouched rather than "upgraded" with content that isn't real
        vocabulary.

    Whenever a word's own definition text actually changes, its stale
    downstream artifacts (quiz_definition, USAS categories, definition
    embedding) are invalidated via _invalidate_definition_dependents --
    the same "changeful"-bug fix sync_book_results already applies, needed
    here for the identical reason (this is a direct word.definition write,
    not going through that upsert path)."""
    from . import deepdef, localdict, resolve
    from .config import Config
    from .dictionary import make_session
    from .model import Candidate, Occurrence, junk_pos_reason

    s = _safe_schema(schema)
    # Same POSIX-ERE-vs-Python-re caveat as dedupe_plural_definitions: a
    # plain literal substring for the SQL prefilter, precise parsing here.
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, definition, part_of_speech, sentence, chapter, as_seen
                FROM {s}.word
                WHERE active AND definition ~* 'synonym of'
                ORDER BY id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"attempted": len(rows), "extracted": 0, "reused_existing": 0, "target_created": 0,
             "target_cast_out": 0, "target_inactive": 0, "target_still_undefined": 0, "unparsed": 0}
    if not rows:
        return stats

    def _parse(raw: str) -> tuple[str, str | None] | None:
        """(target, gloss_or_None) if `raw` cleanly parses, else None."""
        d = _CSS_JUNK_RE.sub("", raw or "").strip()
        if "\n" in d:
            first, rest = d.split("\n", 1)
            if rest.strip():
                return "", rest.strip()  # real content on a later line -- no target needed
            d = first.strip()
        m = _SYNONYM_OF_RE.match(d)
        if not m:
            return None
        target = m.group(1).strip().rstrip(".")
        gloss = m.group(2)
        return target, (gloss.strip() if gloss and len(gloss.strip()) >= 4 else None)

    parsed = []
    for wid, lemma, defn, pos, sentence, chapter, as_seen in rows:
        result = _parse(defn)
        if result is None:
            stats["unparsed"] += 1
            continue
        target, gloss = result
        parsed.append((wid, lemma, pos, sentence, chapter, target.lower() if target else "", gloss))

    bare_targets = {t for *_, t, gloss in parsed if t and gloss is None}
    lexicon = localdict.build_lexicon(conn, bare_targets)
    session = make_session()
    key = deepdef.wordnik_key()
    max_tier = resolve.Tier.WEB if use_web else resolve.Tier.YOURDICT

    llm = None
    if use_web:
        cfg = Config()
        mp = model_path or cfg.model_path
        if mp and Path(mp).exists():
            from llama_cpp import Llama
            llm = Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)

    with conn.cursor() as cur:
        for wid, lemma, pos, sentence, chapter, target, gloss in parsed:
            if gloss is not None:
                # Already-embedded content -- straight extraction, no lookup.
                cur.execute(f"UPDATE {s}.word SET definition=%s, updated_at=now() WHERE id=%s",
                            (gloss, wid))
                _invalidate_definition_dependents(cur, s, wid)
                stats["extracted"] += 1
                conn.commit()
                continue

            cur.execute(f"SELECT active, coalesce(definition,''), definition_source "
                        f"FROM {s}.word WHERE lemma_lc = %s", (target,))
            existing = cur.fetchone()

            if existing and not existing[0]:
                stats["target_inactive"] += 1
                conn.commit()
                continue

            if existing and existing[1]:
                _, target_def, target_src = existing
                cur.execute(f"UPDATE {s}.word SET definition=%s, "
                            f"definition_source=%s, updated_at=now() WHERE id=%s",
                            (target_def, f"{target_src} (synonym of '{target}')", wid))
                _invalidate_definition_dependents(cur, s, wid)
                stats["reused_existing"] += 1
                conn.commit()
                continue

            # Target doesn't exist, or exists active with a blank definition
            # (existing is (True, '', ...) at this point -- the inactive and
            # active-with-content cases were both already handled above) --
            # resolve it fresh, same cascade as any other definition-
            # acceptance path. ON CONFLICT DO UPDATE actually fills in the
            # definition/POS this time (not just active/updated_at) -- a
            # plain no-op update here would silently drop a genuine
            # resolution for an existing-but-blank target.
            cand = Candidate(lemma=target, pos=_POS_TO_TAGGER.get((pos or "").lower(), ""))
            if sentence:
                cand.occurrences.append(Occurrence(sentence=sentence, chapter=chapter or "", surface=target))
            found = resolve.resolve_definition(
                cand, max_tier=max_tier, lexicon=lexicon, session=session,
                wordnik_key=key, llm=llm) is not None
            reason = junk_pos_reason(cand.part_of_speech) if found else None
            is_blank = not found

            cur.execute(
                f"""INSERT INTO {s}.word
                        (lemma, as_seen, definition, part_of_speech, sentence, chapter,
                         definition_source, first_added, active, flagged_undefined, flagged_undefined_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE, %s, %s, CASE WHEN %s THEN now() ELSE NULL END)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        definition=COALESCE(NULLIF(EXCLUDED.definition,''), {s}.word.definition),
                        part_of_speech=COALESCE(NULLIF(EXCLUDED.part_of_speech,''), {s}.word.part_of_speech),
                        definition_source=COALESCE(NULLIF(EXCLUDED.definition_source,''), {s}.word.definition_source),
                        active=EXCLUDED.active, updated_at=now()""",
                (target, target, cand.definition, normalize_pos(cand.part_of_speech),
                 sentence or "", chapter or "", cand.definition_source,
                 not reason, is_blank, is_blank))

            if reason:
                stats["target_cast_out"] += 1
            elif is_blank:
                stats["target_still_undefined"] += 1
            else:
                cur.execute(f"UPDATE {s}.word SET definition=%s, "
                            f"definition_source=%s, updated_at=now() WHERE id=%s",
                            (cand.definition, f"{cand.definition_source} (synonym of '{target}')", wid))
                _invalidate_definition_dependents(cur, s, wid)
                stats["target_created"] += 1
            conn.commit()
    return stats


def fetch_known_verdicts(conn, schema: str = DEFAULT_SCHEMA) -> dict[str, str]:
    """Map lemma_lc -> a cached verdict from EARLIER books, so the (expensive)
    LLM judge is only ever run on lemmas whose verdict isn't already known.

    The judge's input for a word is purely (lemma, its wordfreq band) — no
    book/sentence/POS context — and it runs at temp 0, so a given lemma's
    verdict is the same in every book. Re-judging "refectory" from scratch in
    every book of a shared-vocabulary corpus is pure waste; this is the cache
    that eliminates it.

      'keep'    -> in `word`, active    (judge kept it; human hasn't pruned)
      'pruned'  -> in `word`, inactive  (human manually pruned via the webapp)
      <reason>  -> in `rejected_word`, one of 'not_interesting', 'numeric_or_symbol',
                                        or 'proper_noun' -- the specific reason, not a
                                        generic 'reject', so pipeline.py's _VERDICT_MAP
                                        can restore the true original reason on a cached
                                        hit (judge, or the post-enrichment junk-POS gate,
                                        rejected it before — both are purely lemma-derived,
                                        like the judge verdict, so caching them is exactly
                                        as safe: see pipeline.py's junk_pos_reason gate)

    `word` wins over `rejected_word` for a lemma present in both: a promoted
    row is authoritative and its `active` flag reflects the human's latest
    call. Re-fetched per book (cheap, indexed) so book N sees the new keeps
    that books 1..N-1 added earlier in the same batch."""
    s = _safe_schema(schema)
    verdicts: dict[str, str] = {}
    with conn.cursor() as cur:
        # The specific reason (not a generic "reject") so pipeline.py's
        # _VERDICT_MAP can restore the true original reason on a cached hit
        # instead of relabeling every cached reject as not_interesting.
        cur.execute(f"""SELECT lemma_lc, reason FROM {s}.rejected_word
                        WHERE reason IN ('not_interesting', 'numeric_or_symbol', 'proper_noun')""")
        for lemma, reason in cur.fetchall():
            verdicts[lemma] = reason
        cur.execute(f"SELECT lemma_lc, active FROM {s}.word")
        for lemma, active in cur.fetchall():
            verdicts[lemma] = "keep" if active else "pruned"   # word overrides rejected_word
    return verdicts


def normalize_word_pos(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Clean up word.part_of_speech in place: folds abbreviations/case variants
    (adj, adv, pron, adp, sconj, num, Noun, Adjective, ...) accumulated from
    older write paths down to the canonical vocabulary via normalize_pos().
    Idempotent — safe to re-run any time a new inconsistency creeps in.
    Always recomputes every word in scope (no only_missing gate): the source
    column is mutable and there's no separate signal to gate a re-check on,
    so freezing a word's normalized POS after the one time this ran would
    silently stop it from self-correcting if part_of_speech changes later."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, part_of_speech FROM {s}.word ORDER BY id" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        changed = 0
        for wid, pos in rows:
            new_pos = normalize_pos(pos)
            if new_pos != (pos or ""):
                cur.execute(f"UPDATE {s}.word SET part_of_speech = %s WHERE id = %s", (new_pos, wid))
                changed += 1
    conn.commit()
    return {"words": len(rows), "changed": changed}


def load_taxonomy(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA,
                  taxonomy: str = "usas") -> dict:
    """Upsert the USAS category tree into {schema}.category. Idempotent."""
    from . import usas
    s = _safe_schema(schema)
    cats = usas.categories()
    code_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        # pass 1: upsert nodes (parent set in pass 2 once every id is known)
        for c in cats:
            cur.execute(
                f"""INSERT INTO {s}.category (taxonomy, code, name, level, assignable)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (taxonomy, code) DO UPDATE SET
                        name=EXCLUDED.name, level=EXCLUDED.level, assignable=EXCLUDED.assignable
                    RETURNING id""",
                (taxonomy, c["code"], c["name"], c["level"], c["assignable"]))
            code_to_id[c["code"]] = cur.fetchone()[0]
        # pass 2: wire parents
        for c in cats:
            pid = code_to_id.get(c["parent_code"]) if c["parent_code"] else None
            cur.execute(f"UPDATE {s}.category SET parent_id=%s WHERE id=%s",
                        (pid, code_to_id[c["code"]]))
    conn.commit()
    return {"categories": len(cats), "top_level": sum(1 for c in cats if c["parent_code"] is None)}


def compute_archaic(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Set the archaic-currency ordinal on word_difficulty for every word. Uses the
    definition register-label + (if present) vocab.wiktionary is_archaic/is_obsolete.
    Always recomputes every word in scope (no only_missing gate) -- definition
    text and ngram data can both change after the first run, and there's no
    signal to gate a re-check on other than just running it again."""
    from collections import Counter
    from . import archaic as _archaic
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute("select to_regclass('vocab.wiktionary')")
        have_wik = cur.fetchone()[0] is not None
    join = ("LEFT JOIN (select lower(term) t, bool_or(is_archaic) arc, bool_or(is_obsolete) obs "
            "from vocab.wiktionary group by lower(term)) k on k.t = lower(w.lemma)") if have_wik else ""
    cols = "coalesce(k.arc,false), coalesce(k.obs,false)" if have_wik else "false, false"
    dist: Counter = Counter()
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.definition, {cols}, g.peak, g.recency_ratio
                        FROM {s}.word w {join}
                        LEFT JOIN {s}.word_ngram g ON g.word_id = w.id
                        ORDER BY w.id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        for wid, defn, arc, obs, peak, ratio in rows:
            flag, evid, conf = _archaic.classify(defn, arc, obs, peak, ratio)
            dist[flag] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, archaic, archaic_evidence, archaic_confidence, updated_at)
                    VALUES (%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        archaic=EXCLUDED.archaic, archaic_evidence=EXCLUDED.archaic_evidence,
                        archaic_confidence=EXCLUDED.archaic_confidence, updated_at=now()""",
                (wid, flag, evid, conf))
    conn.commit()
    return dict(dist)


def fetch_ngrams(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                 limit: int = 0, delay: float = 0.3) -> dict:
    """Fetch + cache Google Books Ngram features for words. Returns counts."""
    import time
    from . import ngram
    s = _safe_schema(schema)
    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_ngram g WHERE g.word_id=w.id)"
             if only_missing else "")
    with conn.cursor() as cur:
        cur.execute(f"SELECT w.id, w.lemma FROM {s}.word w{where}" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (concordance vocab tool)"})
    stats = {"words": len(rows), "fetched": 0, "in_corpus": 0, "failed": 0}
    with conn.cursor() as cur:
        for wid, lemma in rows:
            f = ngram.fetch(lemma, session)
            if f is None:
                stats["failed"] += 1
                time.sleep(delay); continue
            if f["peak"]:
                stats["in_corpus"] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_ngram (word_id, peak, recent, recency_ratio, peak_year, fetched_at)
                    VALUES (%s,%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET peak=EXCLUDED.peak, recent=EXCLUDED.recent,
                        recency_ratio=EXCLUDED.recency_ratio, peak_year=EXCLUDED.peak_year, fetched_at=now()""",
                (wid, f["peak"], f["recent"], f["recency_ratio"], f["peak_year"]))
            stats["fetched"] += 1
            if stats["fetched"] % 200 == 0:
                conn.commit()
                print(f"  ...{stats['fetched']}/{len(rows)} fetched ({stats['in_corpus']} in corpus, "
                      f"{stats['failed']} failed)")
            time.sleep(delay)
    conn.commit()
    return stats


def compute_difficulty(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Compute the ex-ante difficulty scalar (+ factor breakdown) for every word.
    Always recomputes every word in scope (no only_missing gate) -- ngram,
    archaic, and domain data are all mutable upstream inputs with no signal
    to gate a re-check on."""
    import statistics
    from psycopg.types.json import Json
    from . import difficulty as _diff
    from .validity_score import _morph_root, effective_zipf
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT w.id, w.lemma, g.peak, d.archaic, d.archaic_confidence, coalesce(dom.fields,'')
            FROM {s}.word w
            LEFT JOIN {s}.word_ngram g ON g.word_id = w.id
            LEFT JOIN {s}.word_difficulty d ON d.word_id = w.id
            LEFT JOIN (SELECT wc.word_id, string_agg(DISTINCT left(c.code,1), '') fields
                       FROM {s}.word_category wc JOIN {s}.category c ON c.id = wc.category_id
                       GROUP BY wc.word_id) dom ON dom.word_id = w.id
            ORDER BY w.id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        scores = []
        for wid, lemma, peak, archaic, aconf, fields in rows:
            zipf = effective_zipf(lemma)
            has_domain = any(f in _diff.DOMAIN_FIELDS for f in fields)
            morph = _morph_root(lemma) is not None
            sc, factors = _diff.score(zipf, peak, archaic or "current", aconf, has_domain, morph)
            scores.append(sc)
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, difficulty, difficulty_factors, updated_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        difficulty=EXCLUDED.difficulty, difficulty_factors=EXCLUDED.difficulty_factors,
                        updated_at=now()""",
                (wid, sc, Json(factors)))
    conn.commit()
    return {"words": len(scores),
            "mean": round(statistics.mean(scores), 1) if scores else 0,
            "median": statistics.median(scores) if scores else 0}


def compute_quiz_definitions(conn, schema: str = DEFAULT_SCHEMA, cfg=None,
                             only_missing: bool = True, limit: int = 0) -> dict:
    """Set quiz_definition/quiz_def_source. Clean defs pass through free; leakers are
    LLM-rewritten (validated) or redacted. Resumable via only_missing (scale-ready)."""
    from collections import Counter
    from . import quizdef
    s = _safe_schema(schema)
    where = "quiz_definition IS NULL AND " if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, definition FROM {s}.word "
                    f"WHERE {where}coalesce(definition,'') <> ''" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    clean = [(i, l, d) for i, l, d in rows if not quizdef.has_leak(l, d)]
    leakers = [(i, l, d) for i, l, d in rows if quizdef.has_leak(l, d)]
    stats = Counter()

    with conn.cursor() as cur:
        for wid, lemma, defn in clean:                       # free — no model
            cur.execute(f"UPDATE {s}.word SET quiz_definition=%s, quiz_def_source='clean' WHERE id=%s",
                        (defn, wid))
            stats["clean"] += 1
        conn.commit()

    if leakers:
        rw = quizdef.Rewriter(cfg)
        res = rw.rewrite([{"word": l, "definition": d} for _, l, d in leakers])
        with conn.cursor() as cur:
            for wid, lemma, defn in leakers:
                qd, src = res.get(lemma.lower(), (quizdef.redact(lemma, defn), "redacted"))
                cur.execute(f"UPDATE {s}.word SET quiz_definition=%s, quiz_def_source=%s WHERE id=%s",
                            (qd, src, wid))
                stats[src] += 1
        conn.commit()
    return {"words": len(rows), "clean": stats["clean"],
            "rewritten": stats["rewritten"], "redacted": stats["redacted"]}


def compute_quizzable(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0) -> dict:
    """Set the quizzable flag (+ reason) on word_difficulty for every word.
    Always recomputes every word in scope (no only_missing gate) -- definition
    and quiz_definition are both mutable upstream inputs with no signal to
    gate a re-check on."""
    from collections import Counter
    from wordfreq import zipf_frequency
    from . import quizdef
    from .validity_score import _morph_root
    s = _safe_schema(schema)
    dist: Counter = Counter()
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, definition, quiz_definition, quiz_def_source "
                    f"FROM {s}.word WHERE coalesce(definition,'') <> '' ORDER BY id" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
        for wid, lemma, defn, quiz_defn, quiz_def_source in rows:
            root = _morph_root(lemma)
            rz = zipf_frequency(root, "en") if root else None
            ok, reason = quizdef.quizzable(defn, root, rz, quiz_defn, quiz_def_source)
            dist["quizzable" if ok else "excluded"] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, quizzable, quizzable_reason, updated_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        quizzable=EXCLUDED.quizzable, quizzable_reason=EXCLUDED.quizzable_reason, updated_at=now()""",
                (wid, ok, reason or None))
    conn.commit()
    return dict(dist)


def compute_personal_difficulty(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0) -> dict:
    """`concordance calibrate-difficulty` / `maintain`'s calibration step: a
    per-(user, word) personalized adjustment to the ex-ante difficulty
    score, from that user's own FIRST exposure to the word in a quiz --
    see concordance/calibration.py's module docstring for the model and
    why this is deliberately NOT written into the shared, all-users-facing
    word_difficulty.difficulty column (one dominant rater's response data
    never identifies population-level item difficulty, no matter how much
    of it accumulates -- see calibration.py).

    Only a word's FIRST quiz exposure per user counts (a window-function
    row_number() = 1 filter, below) -- a later re-exposure of the same
    word is evidence the person is LEARNING it (word_review_schedule's own
    reason for existing), not independent evidence about a fixed item
    difficulty; folding repeat exposures in would read "he learned it" as
    "it got easier," a confound that gets worse, not better, as the same
    user answers the same words repeatedly over a long time.

    KNOWN GAP: "first" here means first quiz_answer row WITH a guessing_floor
    (the WHERE below), not first ever. guessing_floor didn't exist before
    this feature shipped, so a (user, word) pair quizzed pre-migration and
    then answered again post-migration gets that later answer treated as
    rn=1 -- a real repeat exposure miscounted as a first one, the exact
    confound the paragraph above is trying to avoid. Accepted rather than
    fixed: pre-migration rows have no guessing_floor to build a response-
    probability model from, so they're unusable as an anchor regardless: the
    alternative (skip any pair with prior history, migration-era or not)
    trades this confound for discarding real data. Revisit if it turns out
    to matter in practice.

    Always recomputes every first-exposure row in scope on every run (no
    only-missing gate) -- cheap, pure-local arithmetic, and it must re-run
    whenever the underlying ex-ante difficulty changes upstream anyway,
    same "recompute is fine, it's cheap" reasoning as archaic/difficulty/
    quizzable. Truncates the whole table before repopulating on an
    unqualified (limit=0) run rather than a targeted delete -- see
    compute_book_similarity's own comment on this: a targeted delete only
    reaches rows still in scope THIS run, so a (user, word) pair that drops
    out of scope (its quiz_answer/quiz_question/quiz_session deleted, say)
    would otherwise keep a stale row forever."""
    from . import calibration as calib

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT value FROM {s}.app_settings WHERE key = 'calibration_eta'")
        row = cur.fetchone()
        eta = (row[0] or {}).get("value", calib.DEFAULT_ETA) if row else calib.DEFAULT_ETA
        cur.execute(f"SELECT value FROM {s}.app_settings WHERE key = 'calibration_scale'")
        row = cur.fetchone()
        scale = (row[0] or {}).get("value", calib.DEFAULT_SCALE) if row else calib.DEFAULT_SCALE

        cur.execute(f"""
            WITH first_exposure AS (
                SELECT qa.word_id, qs.user_id, qa.is_correct, qa.guessing_floor,
                       row_number() OVER (PARTITION BY qs.user_id, qa.word_id
                                           ORDER BY qa.answered_at) AS rn
                FROM {s}.quiz_answer qa
                JOIN {s}.quiz_question qq ON qq.id = qa.question_id
                JOIN {s}.quiz_session  qs ON qs.id = qq.session_id
                WHERE qa.guessing_floor IS NOT NULL
            )
            SELECT word_id, user_id, is_correct, guessing_floor
            FROM first_exposure WHERE rn = 1""" + (f" LIMIT {int(limit)}" if limit else ""))
        exposures = cur.fetchall()

        if not exposures:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"words": 0}

        word_ids = list({r[0] for r in exposures})
        cur.execute(f"""SELECT word_id, difficulty FROM {s}.word_difficulty
                        WHERE word_id = ANY(%s) AND difficulty IS NOT NULL""", (word_ids,))
        difficulty_by_word = dict(cur.fetchall())

    stored = 0
    skipped_no_baseline = 0
    with conn.cursor() as cur:
        if limit:
            # A composite (user_id, word_id) = ANY(%s) isn't a portable psycopg
            # parameter binding (would need an actual Postgres row-type array,
            # not a plain Python list of tuples) -- exposures is small whenever
            # limit is set anyway (that's the point of limit), so a per-pair
            # delete is simpler and just as correct.
            for word_id, user_id, *_ in exposures:
                cur.execute(f"""DELETE FROM {s}.word_personal_difficulty
                                WHERE user_id = %s AND word_id = %s""", (user_id, word_id))
        else:
            cur.execute(f"TRUNCATE {s}.word_personal_difficulty")

        for i, (word_id, user_id, is_correct, c_q) in enumerate(exposures, 1):
            base_difficulty = difficulty_by_word.get(word_id)
            if base_difficulty is None:   # no ex-ante score yet -- nothing to anchor a nudge to
                skipped_no_baseline += 1
                continue
            b0 = calib.difficulty_to_logit(base_difficulty, scale)
            b_new = calib.update_rating(b0, is_correct, c_q, eta)
            personal_difficulty = calib.logit_to_difficulty(b_new, scale)
            cur.execute(
                f"""INSERT INTO {s}.word_personal_difficulty
                        (user_id, word_id, item_rating, personal_difficulty, based_on_correct, calibrated_at)
                    VALUES (%s,%s,%s,%s,%s, now())
                    ON CONFLICT (user_id, word_id) DO UPDATE SET
                        item_rating=EXCLUDED.item_rating, personal_difficulty=EXCLUDED.personal_difficulty,
                        based_on_correct=EXCLUDED.based_on_correct, calibrated_at=now()""",
                (user_id, word_id, b_new, personal_difficulty, is_correct))
            stored += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return {"words": stored, "skipped_no_baseline": skipped_no_baseline}


def get_book_by_title(conn, title: str, schema: str = DEFAULT_SCHEMA) -> tuple[int, str | None] | None:
    """(book_id, existing archive_path) for a title, or None if no such book
    -- `concordance archive-metadata` uses this to match an archive/
    filename's parsed title (see cli.py's _parse_incoming_name) to its row."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, archive_path FROM {s}.book WHERE title = %s", (title,))
        return cur.fetchone()


def backfill_publication_era(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """Derives publication_era from publication_year (archive_metadata.year_to_era)
    for every book that has an exact year but no free-text era hedge -- a
    pure computation from data already in `book`, no network involved, so
    this is cheap enough to run unconditionally at the start of
    `archive-metadata` rather than needing its own command. Covers both
    books processed before year_to_era existed and any RDF summary that
    stated a year without phrasing a matching century hedge."""
    from .archive_metadata import year_to_era

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, publication_year FROM {s}.book
                        WHERE publication_year IS NOT NULL AND coalesce(publication_era,'') = ''""")
        rows = cur.fetchall()
        for book_id, year in rows:
            cur.execute(f"UPDATE {s}.book SET publication_era=%s WHERE id=%s",
                        (year_to_era(year), book_id))
    conn.commit()
    return {"backfilled": len(rows)}


def update_book_archive_metadata(conn, book_id: int, *, archive_path: str, word_count: int,
                                  distinct_nonstop_word_count: int, publication_year: int | None,
                                  publication_era: str | None, schema: str = DEFAULT_SCHEMA) -> None:
    """Writes one book's concordance/archive_metadata.py-computed stats --
    see that module's own docstring for what each field means and why
    publication date is two columns, not one."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""UPDATE {s}.book SET archive_path=%s, word_count=%s,
                    distinct_nonstop_word_count=%s, publication_year=%s, publication_era=%s
                WHERE id=%s""",
            (archive_path, word_count, distinct_nonstop_word_count, publication_year, publication_era, book_id),
        )
    conn.commit()


def compute_book_similarity(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                            top_k: int = 20, min_shared_words: int = 3,
                            max_df_fraction: float = 0.5) -> dict:
    """`concordance book-similarity` / `maintain`'s book-similarity step:
    each book's top-k most vocabulary-related books, by IDF-weighted cosine
    similarity over shared ACTIVE words -- lexical usage overlap, not
    semantic similarity (that's the existing word_embedding graph's job; a
    different axis, deliberately not duplicated here).

    Always recomputes everything in scope (no only-missing gate, same
    reasoning as archaic/difficulty/quizzable): IDF weights are corpus-wide,
    so they shift whenever ANY book's word_book membership changes, not
    just the book being looked at.

    Why cosine, not raw Jaccard: an earlier bug in this same file
    (browse_books/browse_authors, see their docstrings) is exactly what
    unweighted overlap reproduces -- a word shared by nearly every book
    (the/said/table) counts the same as a shared "cangue", so common words
    would dominate every score. IDF weighting fixes that; cosine (rather
    than a weighted Jaccard) also avoids penalizing a short book for having
    a small vocabulary relative to a long one it otherwise overlaps with
    almost entirely, since cosine normalizes each book's own vector
    magnitude away.

    `max_df_fraction` (default 0.5): words appearing in more than half of
    all books are excluded from the similarity computation entirely, not
    just down-weighted. Not merely a performance shortcut (though it is
    one -- without it, a self-join for computing shared-word contributions
    is combinatorial in how many books each word appears in, and a handful
    of ubiquitous words would dominate the join's cost) -- ln(N/df) for
    such a word is already close to zero, so this is a near-lossless
    approximation of the same math, expressed as a scale-independent
    fraction rather than a fixed count so it stays correct as the corpus
    grows. `shared_word_count` (an explainability field, not used for
    ranking) only counts words that passed this same filter -- "N shared
    RARE words" is a more honest, more on-brand number to show a user here
    than a raw count dominated by function words."""
    import math
    from collections import defaultdict

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT wb.book_id) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id WHERE w.active""")
        n_books = cur.fetchone()[0]
        if n_books < 2:
            # Closing the cursor does NOT end the connection's transaction --
            # a bare SELECT still opens one in the default isolation level,
            # and an early `return` here without an explicit commit leaves
            # `conn` sitting "idle in transaction" indefinitely, holding
            # locks that block anything needing DDL (a schema drop, another
            # connection's ALTER TABLE) until the caller happens to touch
            # this same connection again. Found live: a 2-book test schema
            # hit this path, and a completely separate connection's DROP
            # SCHEMA hung for 10+ minutes waiting on it.
            conn.commit()
            return {"books": n_books, "pairs_stored": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT wb.book_id) AS df
                        FROM {s}.word_book wb JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active GROUP BY wb.word_id""")
        max_df = max_df_fraction * n_books
        idf = {wid: math.log(n_books / df) for wid, df in cur.fetchall() if df <= max_df}

        if not idf:
            conn.commit()  # same reasoning as the n_books < 2 early return above
            return {"books": n_books, "pairs_stored": 0}

        cur.execute(f"""SELECT wb.word_id, wb.book_id FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active AND wb.word_id = ANY(%s)""", (list(idf.keys()),))
        books_by_word: dict[int, list[int]] = defaultdict(list)
        for wid, bid in cur.fetchall():
            books_by_word[wid].append(bid)

    norm_sq: dict[int, float] = defaultdict(float)
    for wid, books in books_by_word.items():
        w = idf[wid] ** 2
        for bid in books:
            norm_sq[bid] += w
    norm = {bid: math.sqrt(v) for bid, v in norm_sq.items()}

    dot: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    shared: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for wid, books in books_by_word.items():
        w2 = idf[wid] ** 2
        for i, a in enumerate(books):
            for b in books[i + 1:]:
                dot[a][b] += w2
                dot[b][a] += w2
                shared[a][b] += 1
                shared[b][a] += 1

    book_ids = list(norm.keys())
    if limit:
        book_ids = book_ids[: int(limit)]

    stored = 0
    with conn.cursor() as cur:
        # Unqualified (limit=0, the normal/maintain case): truncate the whole
        # table, not just WHERE book_a_id = ANY(book_ids) -- a book that HAD
        # rows last run but is no longer in scope this run (word_book fully
        # emptied, an exclusion filter added, etc.) would otherwise keep its
        # stale rows forever, since it'd never again appear on either side of
        # a targeted delete. Only skip the full wipe when --limit narrows
        # this run to a deliberate subset, where nuking everyone else's rows
        # would be destructive instead of correct.
        if limit:
            cur.execute(f"DELETE FROM {s}.book_similarity WHERE book_a_id = ANY(%s)", (book_ids,))
        else:
            cur.execute(f"DELETE FROM {s}.book_similarity")
        for i, a in enumerate(book_ids, 1):
            candidates = [
                (b, dot[a][b] / (norm[a] * norm[b]), shared[a][b])
                for b in dot[a] if shared[a][b] >= min_shared_words and norm[b] > 0
            ]
            candidates.sort(key=lambda t: t[1], reverse=True)
            for b, score, shared_count in candidates[:top_k]:
                cur.execute(
                    f"""INSERT INTO {s}.book_similarity (book_a_id, book_b_id, score, shared_word_count, updated_at)
                        VALUES (%s,%s,%s,%s, now())
                        ON CONFLICT (book_a_id, book_b_id) DO UPDATE SET
                            score=EXCLUDED.score, shared_word_count=EXCLUDED.shared_word_count, updated_at=now()""",
                    (a, b, score, shared_count))
                stored += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return {"books": len(book_ids), "pairs_stored": stored}


# book.author values that are an aggregation label, not an actual author --
# an anthology's shared vocabulary owes nothing to any individual writer, so
# treating "Various"/"Unknown Author"/etc. as a real author in the
# relatedness graph produces spurious, meaningless similarity scores (a
# high-book-count phantom author that ends up "related" to nearly everyone).
# Checked against the real corpus (2026-07-23): these four cover every
# non-name book.author value.
PLACEHOLDER_AUTHORS = frozenset({"Various", "Unknown Author", "Unknown", "Anonymous"})


def compute_author_similarity(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                              top_k: int = 20, min_shared_words: int = 3,
                              max_df_fraction: float = 0.5) -> dict:
    """`concordance author-similarity` / `maintain`'s author-similarity step:
    each author's top-k most vocabulary-related authors, by IDF-weighted
    cosine similarity over shared ACTIVE words -- same metric shape as
    compute_book_similarity, one level up (an author's vector is the union
    of their books' word sets).

    Originally shipped as an on-demand, compute-per-request query in
    browse.py, on the plan's own reasoning that "authors are dozens today,
    full O(n^2) pairwise at request time is cheap." That premise didn't
    survive contact with the real corpus: ~3,500 authors, and a full-corpus
    timing came back at ~39s for a SINGLE request -- unusable behind an
    HTTP endpoint, let alone one a "See full relatedness graph" link would
    hit on every click. Precomputed here instead, exactly like books.

    IDF is *author*-document-frequency (ln(N_authors / df_authors)), NOT
    book-level df: a word spread across 30 books all by one author has high
    book-df (looks common) but low author-df (df=1) -- it's a distinctive
    marker of that one author, and book-df would wash out exactly the
    signal that matters at this granularity. See compute_book_similarity's
    own docstring for the cosine-over-Jaccard and max_df_fraction reasoning,
    which applies identically here.

    An author with several books containing the same word must count once
    per word, not once per book -- the DISTINCT below is load-bearing, not
    decorative: without it, an author with many books sharing a word would
    have that word's weight (and every pair involving them) inflated by
    however many of their own books happen to contain it."""
    import math
    from collections import defaultdict

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT b.author) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))""", (placeholders,))
        n_authors = cur.fetchone()[0]
        if n_authors < 2:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"authors": n_authors, "pairs_stored": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT b.author) AS df
                        FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                        GROUP BY wb.word_id""", (placeholders,))
        max_df = max_df_fraction * n_authors
        idf = {wid: math.log(n_authors / df) for wid, df in cur.fetchall() if df <= max_df}

        if not idf:
            conn.commit()
            return {"authors": n_authors, "pairs_stored": 0}

        cur.execute(f"""SELECT DISTINCT wb.word_id, b.author FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                          AND wb.word_id = ANY(%s)""", (placeholders, list(idf.keys())))
        authors_by_word: dict[int, list[str]] = defaultdict(list)
        for wid, author in cur.fetchall():
            authors_by_word[wid].append(author)

    norm_sq: dict[str, float] = defaultdict(float)
    for wid, authors in authors_by_word.items():
        w = idf[wid] ** 2
        for a in authors:
            norm_sq[a] += w
    norm = {a: math.sqrt(v) for a, v in norm_sq.items()}

    dot: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    shared: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for wid, authors in authors_by_word.items():
        w2 = idf[wid] ** 2
        for i, a in enumerate(authors):
            for b in authors[i + 1:]:
                dot[a][b] += w2
                dot[b][a] += w2
                shared[a][b] += 1
                shared[b][a] += 1

    author_names = list(norm.keys())
    if limit:
        author_names = author_names[: int(limit)]

    stored = 0
    with conn.cursor() as cur:
        # See compute_book_similarity's identical comment: a targeted delete
        # only reaches authors still in scope THIS run, so an author who
        # drops out of scope (PLACEHOLDER_AUTHORS gaining an entry, their
        # last book losing its words, etc.) would keep stale rows forever.
        # Found live: adding "Anonymous" et al. to PLACEHOLDER_AUTHORS left
        # their old author_similarity rows behind under the targeted delete.
        if limit:
            cur.execute(f"DELETE FROM {s}.author_similarity WHERE author_a = ANY(%s)", (author_names,))
        else:
            cur.execute(f"DELETE FROM {s}.author_similarity")
        for i, a in enumerate(author_names, 1):
            candidates = [
                (b, dot[a][b] / (norm[a] * norm[b]), shared[a][b])
                for b in dot[a] if shared[a][b] >= min_shared_words and norm[b] > 0
            ]
            candidates.sort(key=lambda t: t[1], reverse=True)
            for b, score, shared_count in candidates[:top_k]:
                cur.execute(
                    f"""INSERT INTO {s}.author_similarity (author_a, author_b, score, shared_word_count, updated_at)
                        VALUES (%s,%s,%s,%s, now())
                        ON CONFLICT (author_a, author_b) DO UPDATE SET
                            score=EXCLUDED.score, shared_word_count=EXCLUDED.shared_word_count, updated_at=now()""",
                    (a, b, score, shared_count))
                stored += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return {"authors": len(author_names), "pairs_stored": stored}


# --- fame scoring -------------------------------------------------------------

# If at least this many items have been attempted AND the running
# no-usable-evidence rate crosses this fraction, stop the whole run rather
# than grind through the rest effectively blind -- same "stop, don't
# silently degrade" instinct as mw_backfill's quota-stop, gated on evidence
# quality instead of an API cap. The minimum-sample guard keeps a handful of
# unlucky early misses from tripping this by chance.
_FAME_EVIDENCE_FAILURE_MIN_SAMPLE = 20
_FAME_EVIDENCE_FAILURE_THRESHOLD = 0.30


def _no_usable_author_evidence(factors: dict) -> bool:
    ngram = factors.get("ngram") or {}
    wikidata = factors.get("wikidata") or {}
    ngram_ok = not ngram.get("failed") and not ngram.get("skipped")
    wikidata_ok = bool(wikidata.get("sitelinks")) and wikidata.get("corroborated")
    snippets_ok = not factors.get("snippets_failed")
    return not (ngram_ok or wikidata_ok or snippets_ok)


def _no_usable_book_evidence(factors: dict) -> bool:
    ngram = factors.get("ngram") or {}
    ngram_ok = not ngram.get("failed") and not ngram.get("skipped")
    snippets_ok = not factors.get("snippets_failed")
    return not (ngram_ok or snippets_ok)


def _load_fame_llm():
    from pathlib import Path

    from .config import Config
    cfg = Config()
    if not (cfg.model_path and Path(cfg.model_path).exists()):
        raise RuntimeError(
            f"no local model available (model_path {cfg.model_path!r} missing) -- "
            "fame scoring needs a real LLM; pass dry_run=True to only gather evidence")
    from llama_cpp import Llama
    return Llama(model_path=cfg.model_path, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)


def compute_author_fame(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                        stale_days: int = 0, llm=None, dry_run: bool = False) -> dict:
    """`concordance author-fame`: an ABSOLUTE (not corpus-relative) 1-10
    historical/cultural importance score per author, LLM-judged against a
    fixed external rubric (see concordance/fame.py's module docstring for
    why absolute over corpus-relative percentile, and for the evidence
    sources). Excludes PLACEHOLDER_AUTHORS -- same reasoning as
    compute_author_similarity: an aggregation label has no individual fame
    to score.

    `checked_at` is a STICKY marker (bumped on every attempt, hit or miss)
    so a rerun with stale_days=0 only touches never-scored authors -- this
    is a genuinely expensive job (several network round-trips + one real
    LLM generation per author, realistically 5-15s each), not a quick
    backfill, so resumability is load-bearing. `dry_run=True` only gathers
    and prints evidence (no LLM call, no DB write, checked_at untouched) --
    for sanity-checking evidence quality before committing to a real run.

    Commits after EVERY author, never batched -- a long-held transaction
    here would block a webapp restart's schema-check ALTER TABLE (this
    already happened once in production). Stops the whole run early if the
    running no-usable-evidence rate crosses
    _FAME_EVIDENCE_FAILURE_THRESHOLD, rather than grinding through
    thousands of effectively-blind LLM guesses."""
    from . import fame
    from .dictionary import make_session

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    stale_clause = (
        "(af.checked_at IS NULL OR af.checked_at < now() - (%s * interval '1 day'))"
        if stale_days else "af.checked_at IS NULL"
    )
    extra_params = [stale_days] if stale_days else []
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT b.author FROM {s}.book b
                LEFT JOIN {s}.author_fame af ON af.author = b.author
                WHERE b.author IS NOT NULL AND b.author <> ''
                  AND NOT (b.author = ANY(%s))
                  AND {stale_clause}
                ORDER BY b.author""" + (f" LIMIT {int(limit)}" if limit else ""),
            (placeholders, *extra_params))
        authors = [r[0] for r in cur.fetchall()]

    stats = {"attempted": 0, "scored": 0, "failed_evidence": 0, "failed_parse": 0, "errors": 0,
             "stopped_early": False, "remaining": 0}
    if not authors:
        return stats

    if not dry_run and llm is None:
        llm = _load_fame_llm()

    session = make_session()
    no_evidence_count = 0

    with conn.cursor() as cur:
        for i, author in enumerate(authors, 1):
            try:
                factors = fame.gather_author_evidence(author, session)
                stats["attempted"] += 1
                if _no_usable_author_evidence(factors):
                    no_evidence_count += 1
                    stats["failed_evidence"] += 1

                if dry_run:
                    print(f"[dry-run] {author}: {json.dumps(factors, default=str)[:300]}")
                else:
                    score, why = fame.score_author(llm, author, factors)
                    if score is None:
                        stats["failed_parse"] += 1
                    else:
                        stats["scored"] += 1
                    cur.execute(
                        f"""INSERT INTO {s}.author_fame
                                (author, fame_score, fame_reasoning, fame_factors, computed_at, checked_at)
                            VALUES (%s,%s,%s,%s, CASE WHEN %s THEN now() ELSE NULL END, now())
                            ON CONFLICT (author) DO UPDATE SET
                                fame_score=EXCLUDED.fame_score, fame_reasoning=EXCLUDED.fame_reasoning,
                                fame_factors=EXCLUDED.fame_factors, computed_at=EXCLUDED.computed_at,
                                checked_at=now()""",
                        (author, score, why, json.dumps(factors, default=str), score is not None))
                    conn.commit()
            except Exception as exc:  # noqa: BLE001 -- one poisoned item must not kill a multi-hour run
                conn.rollback()
                stats["errors"] += 1
                print(f"  [author-fame] {author!r} raised {exc!r} -- skipped, left unattempted for a future run")
                continue

            if i % 25 == 0:
                print(f"  ...{i}/{len(authors)} authors attempted "
                      f"({stats['scored']} scored, {stats['failed_evidence']} no usable evidence)")

            if (stats["attempted"] >= _FAME_EVIDENCE_FAILURE_MIN_SAMPLE
                    and no_evidence_count / stats["attempted"] > _FAME_EVIDENCE_FAILURE_THRESHOLD):
                stats["stopped_early"] = True
                stats["remaining"] = len(authors) - i
                break
    return stats


def compute_book_fame(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                      stale_days: int = 0, llm=None, dry_run: bool = False) -> dict:
    """`concordance book-fame`: same shape one level down from
    compute_author_fame, scoring the SPECIFIC WORK rather than its author.
    LEFT JOINs author_fame on book.author (NULL-tolerant by design -- a
    first run, or a book by a not-yet-scored or placeholder author, simply
    has no prior; concordance/fame.py's BOOK_RUBRIC already tells the model
    to treat that as "no prior available", not as evidence of obscurity).
    The exact author-fame snapshot shown (score + computed_at, or None) is
    recorded in fame_factors.author_fame_seen so a later author-fame rerun
    never makes an existing book's reasoning unverifiable against what it
    actually saw.

    Run author-fame first for the best results, but this does not require
    it. Ordered by word_count DESC so an interrupted multi-day run banks
    the highest-traffic books first (see book.word_count; NULLS LAST for
    any book that hasn't been through classify.py's counting pass)."""
    from . import fame
    from .dictionary import make_session

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    stale_clause = (
        "(bf.checked_at IS NULL OR bf.checked_at < now() - (%s * interval '1 day'))"
        if stale_days else "bf.checked_at IS NULL"
    )
    extra_params = [stale_days] if stale_days else []
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT b.id, b.title, b.author, af.fame_score, af.fame_reasoning, af.computed_at
                FROM {s}.book b
                LEFT JOIN {s}.book_fame bf ON bf.book_id = b.id
                LEFT JOIN {s}.author_fame af
                    ON af.author = b.author AND NOT (b.author = ANY(%s))
                WHERE {stale_clause}
                ORDER BY b.word_count DESC NULLS LAST, b.title""" +
            (f" LIMIT {int(limit)}" if limit else ""),
            (placeholders, *extra_params))
        rows = cur.fetchall()

    stats = {"attempted": 0, "scored": 0, "failed_evidence": 0, "failed_parse": 0, "errors": 0,
             "stopped_early": False, "remaining": 0}
    if not rows:
        return stats

    if not dry_run and llm is None:
        llm = _load_fame_llm()

    session = make_session()
    no_evidence_count = 0

    with conn.cursor() as cur:
        for i, (book_id, title, author, a_score, a_reasoning, a_computed_at) in enumerate(rows, 1):
            try:
                author_fame = (
                    {"fame_score": a_score, "fame_reasoning": a_reasoning, "computed_at": a_computed_at}
                    if a_score is not None else None
                )
                factors = fame.gather_book_evidence(title, author or "", author_fame, session)
                stats["attempted"] += 1
                if _no_usable_book_evidence(factors):
                    no_evidence_count += 1
                    stats["failed_evidence"] += 1

                if dry_run:
                    print(f"[dry-run] {title!r}: {json.dumps(factors, default=str)[:300]}")
                else:
                    score, why = fame.score_book(llm, title, author or "", factors)
                    if score is None:
                        stats["failed_parse"] += 1
                    else:
                        stats["scored"] += 1
                    cur.execute(
                        f"""INSERT INTO {s}.book_fame
                                (book_id, fame_score, fame_reasoning, fame_factors, computed_at, checked_at)
                            VALUES (%s,%s,%s,%s, CASE WHEN %s THEN now() ELSE NULL END, now())
                            ON CONFLICT (book_id) DO UPDATE SET
                                fame_score=EXCLUDED.fame_score, fame_reasoning=EXCLUDED.fame_reasoning,
                                fame_factors=EXCLUDED.fame_factors, computed_at=EXCLUDED.computed_at,
                                checked_at=now()""",
                        (book_id, score, why, json.dumps(factors, default=str), score is not None))
                    conn.commit()
            except Exception as exc:  # noqa: BLE001 -- one poisoned item must not kill a multi-day run
                conn.rollback()
                stats["errors"] += 1
                print(f"  [book-fame] {title!r} raised {exc!r} -- skipped, left unattempted for a future run")
                continue

            if i % 25 == 0:
                print(f"  ...{i}/{len(rows)} books attempted "
                      f"({stats['scored']} scored, {stats['failed_evidence']} no usable evidence)")

            if (stats["attempted"] >= _FAME_EVIDENCE_FAILURE_MIN_SAMPLE
                    and no_evidence_count / stats["attempted"] > _FAME_EVIDENCE_FAILURE_THRESHOLD):
                stats["stopped_early"] = True
                stats["remaining"] = len(rows) - i
                break
    return stats


# --- book-merge: multi-part-book detection manifest + DB fold-together ------

def upsert_book_merge_group(conn, schema: str, *, title_base: str, author: str,
                            part_book_ids: list[int], part_labels: list[dict],
                            survivor_book_id: int | None, skip_reason: str | None,
                            gap_detail: list[int] | None) -> int:
    """Record (or refresh) one detected group -- see book_merge_group's own
    schema comment for why checked_at/skip_reason/part_book_ids are
    unconditionally overwritten on every call (re-detection is cheap and a
    group's eligibility can change as the corpus changes) while
    compiled_at/merged_at are preserved via COALESCE (they're the only
    terminal, "don't redo this" markers)."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {s}.book_merge_group
                    (title_base, author, part_count, part_book_ids, part_labels,
                     survivor_book_id, skip_reason, gap_detail, checked_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (title_base, author) DO UPDATE SET
                    part_count=EXCLUDED.part_count, part_book_ids=EXCLUDED.part_book_ids,
                    part_labels=EXCLUDED.part_labels, survivor_book_id=EXCLUDED.survivor_book_id,
                    skip_reason=EXCLUDED.skip_reason, gap_detail=EXCLUDED.gap_detail,
                    checked_at=now()
                RETURNING id""",
            (title_base, author, len(part_book_ids), part_book_ids, json.dumps(part_labels),
             survivor_book_id, skip_reason,
             json.dumps(gap_detail) if gap_detail else None))
        return cur.fetchone()[0]


def mark_book_merge_compiled(conn, schema: str, group_id: int, compiled_path: str) -> None:
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {s}.book_merge_group SET compiled_path=%s, compiled_at=now() WHERE id=%s",
                    (compiled_path, group_id))
    conn.commit()


def mark_book_merge_merged(conn, schema: str, group_id: int) -> None:
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {s}.book_merge_group SET merged_at=now() WHERE id=%s", (group_id,))
    conn.commit()


def merge_book_group(conn, schema: str, survivor_book_id: int, other_book_ids: list[int], *,
                     title: str, author: str, archive_path: str,
                     word_count: int, distinct_nonstop_word_count: int) -> dict:
    """Folds other_book_ids into survivor_book_id -- one transaction (unlike
    mw_backfill/compute_book_fame's per-item commit: a group can't safely
    be left half-merged). Tolerates other_book_ids that no longer exist at
    all (a rerun after a successful merge is a no-op, not an error -- every
    statement below is already a no-op on an empty match).

    word_book/rejected_word are REPOINTED (with dedup against their own
    unique constraints, never violating them), not deleted -- the point of
    a merge is that this vocabulary still belongs to the compiled book.
    book_similarity/book_cluster/book_fame are DELETED for the whole group
    (survivor included): these are precomputed, periodically-wholesale-
    regenerated derived data with no meaning for a differently-sized
    compiled work, so the right move is deleting and letting the next
    book-similarity/book-clustering/book-fame run rescope the compiled
    whole from scratch rather than hand-merging stale per-volume numbers.

    Explicitly accepted scope boundary: word_book keeps reflecting whatever
    each part's own extraction pipeline already found -- this does NOT
    re-run extraction over the newly compiled text. publication_year/
    publication_era are left untouched on the survivor: they're a
    Gutenberg-catalog lookup keyed on ONE volume's own eBook id, with no
    principled single answer for which volume's info the compiled work
    should inherit."""
    s = _safe_schema(schema)
    all_ids = [survivor_book_id, *other_book_ids]
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {s}.word_book (word_id, book_id)
                SELECT word_id, %s FROM {s}.word_book WHERE book_id = ANY(%s)
                ON CONFLICT DO NOTHING""",
            (survivor_book_id, other_book_ids))

        cur.execute(
            f"""INSERT INTO {s}.rejected_word
                    (book_id, lemma, reason, detail, count, zipf, pos, as_seen, sentence, chapter)
                SELECT DISTINCT ON (lemma_lc) %s, lemma, reason, detail, count, zipf, pos, as_seen, sentence, chapter
                FROM {s}.rejected_word WHERE book_id = ANY(%s)
                ORDER BY lemma_lc, book_id
                ON CONFLICT (book_id, lemma_lc) DO NOTHING""",
            (survivor_book_id, other_book_ids))

        cur.execute(f"DELETE FROM {s}.book_similarity WHERE book_a_id = ANY(%s) OR book_b_id = ANY(%s)",
                    (all_ids, all_ids))
        cur.execute(f"DELETE FROM {s}.book_cluster WHERE book_id = ANY(%s)", (all_ids,))
        cur.execute(f"DELETE FROM {s}.book_fame WHERE book_id = ANY(%s)", (all_ids,))

        cur.execute(f"DELETE FROM {s}.book WHERE id = ANY(%s)", (other_book_ids,))

        cur.execute(
            f"""UPDATE {s}.book SET title=%s, author=%s, archive_path=%s,
                    word_count=%s, distinct_nonstop_word_count=%s
                WHERE id=%s""",
            (title, author, archive_path, word_count, distinct_nonstop_word_count, survivor_book_id))
    conn.commit()
    return {"survivor_book_id": survivor_book_id, "merged_count": len(other_book_ids)}


def _linkage_to_tree(Z, leaf_data: list[dict]) -> dict:
    """scipy.cluster.hierarchy.linkage's output (an (n-1)x4 array: each row
    [left, right, distance, size], where left/right index either an
    original leaf (0..n-1) or a previously-built internal node (n..)) into
    a nested, JSON-serializable tree for the frontend dendrogram -- one
    bottom-up pass, no recursion needed since Z is already in merge order.

    leaf_data[i] is spread directly into leaf i's dict (e.g. {"author": name}
    for compute_author_clustering, {"id", "title", "author"} for
    compute_book_clustering) -- generic over what identifies a leaf rather
    than assuming a single string label, since a book (unlike an author)
    needs more than one field to be both displayable and navigable."""
    n = len(leaf_data)
    nodes: dict[int, dict] = {i: {**leaf_data[i], "size": 1} for i in range(n)}
    for i, (a, b, dist, size) in enumerate(Z):
        nodes[n + i] = {
            "left": nodes[int(a)],
            "right": nodes[int(b)],
            "distance": float(dist),
            "size": int(size),
        }
    return nodes[n + len(Z) - 1]


def compute_author_clustering(conn, schema: str = DEFAULT_SCHEMA, *, top_n: int = 200,
                              max_df_fraction: float = 0.5, n_clusters: int = 12) -> dict:
    """`concordance author-clustering` / `maintain`'s clustering step: the
    data behind the cluster map, similarity matrix, and dendrogram views for
    the top `top_n` authors by book count.

    Reuses the exact corpus-wide author-df IDF setup compute_author_similarity
    uses (same n_authors/df/max_df_fraction computation over ALL authors,
    not just the top_n) -- so "the why" (author_similarity's scores) and
    "the map" (cluster positions here) share one consistent notion of
    similarity; only WHICH authors enter the pairwise/clustering step is
    restricted to top_n, not how a shared word is weighted.

    PLACEHOLDER_AUTHORS ("Various", "Unknown Author", ...) are filtered in
    the SAME WHERE clause as the top_n ORDER BY/LIMIT, not afterward --
    they dwarf every real author by book count (Various alone: 1193 books
    vs. ~125 for the top real author), so a post-hoc filter would silently
    burn top_n slots on aggregation labels instead of real authors.

    Distance is `sqrt(2 * (1 - cosine))`, a proper Euclidean distance for
    L2-normalized vectors -- not raw `1 - cosine`, which fails the triangle
    inequality and produces a non-PSD Gram matrix (forcing lossy eigenvalue
    clipping in the MDS step below). This one distance definition feeds
    both `ward` linkage (which assumes squared-Euclidean input -- not valid
    for raw cosine distance) and classical MDS, rather than being decided
    independently in two places.

    n_clusters=12 (via `fcluster(..., criterion='maxclust')`) is a starting
    default, not a permanent one -- validate cluster-size distribution
    against the real corpus (not all-one-cluster, not all-singletons)
    before treating it as final.

    Classical (Torgerson) MDS is computed directly via a single
    numpy.linalg.eigh on the double-centered squared-distance Gram matrix
    (no sklearn) -- fast and deterministic at this scale, unlike sklearn's
    default iterative SMACOF. eigh's eigenvector sign is otherwise
    arbitrary and can flip between runs on near-identical input (the same
    instability class as the force-graph bugs already found and fixed
    elsewhere in this project), so each axis's sign is pinned deterministically.
    Authors landing on the exact same point (identical qualifying word
    sets) get a small, deterministic (name-hash-seeded) jitter so they stay
    individually clickable.

    Writes author_cluster and the singleton author_cluster_run in one
    transaction at the end -- no partial/interleaved commits (unlike
    compute_book_similarity's every-200-rows batching): top_n=200 is small
    enough that one clean transaction is trivial, and a partial write here
    would be worse than there, since the map/matrix/dendrogram must never
    disagree with each other -- they all derive from one computation pass."""
    import hashlib
    import math

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)

    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT b.author) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))""", (placeholders,))
        n_authors = cur.fetchone()[0]
        if n_authors < 3:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"authors": 0, "clusters": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT b.author) AS df
                        FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                        GROUP BY wb.word_id""", (placeholders,))
        max_df = max_df_fraction * n_authors
        idf = {wid: math.log(n_authors / df) for wid, df in cur.fetchall() if df <= max_df}
        if not idf:
            conn.commit()
            return {"authors": 0, "clusters": 0}

        cur.execute(f"""SELECT b.author, count(DISTINCT b.id) AS book_count
                        FROM {s}.book b
                        WHERE b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                        GROUP BY b.author
                        ORDER BY book_count DESC
                        LIMIT %s""", (placeholders, top_n))
        top_authors = cur.fetchall()
        top_author_names = [r[0] for r in top_authors]
        book_count_by_author = {r[0]: r[1] for r in top_authors}

        if len(top_author_names) < 3:
            conn.commit()
            return {"authors": 0, "clusters": 0}

        cur.execute(f"""SELECT DISTINCT wb.word_id, b.author FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author = ANY(%s)
                          AND wb.word_id = ANY(%s)""",
                    (top_author_names, list(idf.keys())))
        rows = cur.fetchall()

    n = len(top_author_names)
    author_index = {a: i for i, a in enumerate(top_author_names)}
    word_index: dict[int, int] = {}
    row_idx, col_idx, weighted_data = [], [], []
    for wid, author in rows:
        j = word_index.setdefault(wid, len(word_index))
        row_idx.append(author_index[author])
        col_idx.append(j)
        weighted_data.append(idf[wid])

    matrix = csr_matrix((weighted_data, (row_idx, col_idx)), shape=(n, len(word_index)))
    binary = csr_matrix(([1] * len(row_idx), (row_idx, col_idx)), shape=(n, len(word_index)))
    shared_counts = (binary @ binary.T).toarray().astype(int)

    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).reshape(-1)
    norms[norms == 0] = 1.0  # an author with zero qualifying words (all excluded by max_df) -- avoid /0
    normalized = matrix.multiply(1 / norms[:, None]).tocsr()
    cosine = (normalized @ normalized.T).toarray()
    np.clip(cosine, -1.0, 1.0, out=cosine)  # guard float round-off past 1 on a self-pair

    dist = np.sqrt(np.clip(2 * (1 - cosine), 0, None))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    tree = linkage(condensed, method="ward", optimal_ordering=True)
    leaf_order_idx = leaves_list(tree)
    leaf_order = [top_author_names[i] for i in leaf_order_idx]
    cluster_ids = fcluster(tree, t=n_clusters, criterion="maxclust")

    # Classical (Torgerson) MDS: double-center the squared-distance matrix,
    # take the top-2 eigenvectors of the resulting Gram matrix.
    d2 = dist ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ d2 @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    top2 = np.argsort(eigvals)[::-1][:2]
    coords = eigvecs[:, top2] * np.sqrt(np.clip(eigvals[top2], 0, None))

    for axis in range(coords.shape[1]):
        col = coords[:, axis]
        if col[np.argmax(np.abs(col))] < 0:
            coords[:, axis] = -col

    seen_points: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (round(float(coords[i, 0]), 6), round(float(coords[i, 1]), 6))
        seen_points.setdefault(key, []).append(i)
    spread = max(float(np.abs(coords).max()), 1.0)
    for idxs in seen_points.values():
        if len(idxs) < 2:
            continue
        for k, i in enumerate(idxs):
            h = int(hashlib.sha256(top_author_names[i].encode()).hexdigest(), 16)
            angle = (h % 360) * math.pi / 180
            radius = 0.02 * (k + 1) * spread
            coords[i, 0] += radius * math.cos(angle)
            coords[i, 1] += radius * math.sin(angle)

    grid = [[[float(cosine[i, j]), int(shared_counts[i, j])] for j in leaf_order_idx] for i in leaf_order_idx]
    tree_json = _linkage_to_tree(tree, [{"author": name} for name in top_author_names])

    from psycopg.types.json import Json

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {s}.author_cluster")
        for i, author in enumerate(top_author_names):
            cur.execute(
                f"""INSERT INTO {s}.author_cluster (author, cluster_id, mds_x, mds_y, book_count, computed_at)
                    VALUES (%s,%s,%s,%s,%s, now())""",
                (author, int(cluster_ids[i]), float(coords[i, 0]), float(coords[i, 1]),
                 book_count_by_author[author]),
            )
        cur.execute(
            f"""INSERT INTO {s}.author_cluster_run (id, leaf_order, grid, tree_json, computed_at)
                VALUES (1, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    leaf_order=EXCLUDED.leaf_order, grid=EXCLUDED.grid,
                    tree_json=EXCLUDED.tree_json, computed_at=now()""",
            (leaf_order, Json(grid), Json(tree_json)),
        )
    conn.commit()
    return {"authors": n, "clusters": int(cluster_ids.max()) if n else 0}


def compute_book_clustering(conn, schema: str = DEFAULT_SCHEMA, *, top_n: int = 200,
                            max_df_fraction: float = 0.5, n_clusters: int = 12) -> dict:
    """`concordance book-clustering` / `maintain`'s clustering step, one
    level down from compute_author_clustering: the data behind the cluster
    map, similarity matrix, and dendrogram views for the top `top_n` books
    by (extracted-vocabulary) word count.

    Reuses the exact corpus-wide book-df IDF setup compute_book_similarity
    uses (same n_books/df/max_df_fraction computation over ALL books, not
    just the top_n) -- so "the why" (book_similarity's scores) and "the
    map" (cluster positions here) share one consistent notion of
    similarity; only WHICH books enter the pairwise/clustering step is
    restricted to top_n, not how a shared word is weighted. Unlike
    compute_author_clustering, there's no PLACEHOLDER_AUTHORS filter here --
    that's an author-level aggregation-label concept ("Various" isn't a
    real writer), and doesn't disqualify an individual book from having a
    real, clusterable vocabulary of its own; a book by a placeholder author
    just carries that through to book_cluster.author as-is (nullable,
    same as book.author itself).

    See compute_author_clustering's own docstring for the distance
    definition (sqrt(2*(1-cosine)), a proper Euclidean distance for
    L2-normalized vectors), classical MDS technique (direct eigh, no
    sklearn, deterministic axis-sign pinning), and jitter reasoning
    (identical qualifying word sets landing on the same point) -- all
    reused verbatim here, just keyed by book_id instead of author name.

    Writes book_cluster and the singleton book_cluster_run in one
    transaction at the end, same all-or-nothing reasoning as
    compute_author_clustering (top_n=200 is small enough that one clean
    transaction is trivial, and the map/matrix/dendrogram must never
    disagree with each other)."""
    import hashlib
    import math

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix

    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT wb.book_id) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id WHERE w.active""")
        n_books = cur.fetchone()[0]
        if n_books < 3:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"books": 0, "clusters": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT wb.book_id) AS df
                        FROM {s}.word_book wb JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active GROUP BY wb.word_id""")
        max_df = max_df_fraction * n_books
        idf = {wid: math.log(n_books / df) for wid, df in cur.fetchall() if df <= max_df}
        if not idf:
            conn.commit()
            return {"books": 0, "clusters": 0}

        cur.execute(f"""SELECT b.id, b.title, b.author, count(DISTINCT w.id) AS word_count
                        FROM {s}.book b
                        JOIN {s}.word_book wb ON wb.book_id = b.id
                        JOIN {s}.word w ON w.id = wb.word_id AND w.active
                        GROUP BY b.id, b.title, b.author
                        ORDER BY word_count DESC
                        LIMIT %s""", (top_n,))
        top_books = cur.fetchall()
        top_book_ids = [r[0] for r in top_books]
        title_by_id = {r[0]: r[1] for r in top_books}
        author_by_id = {r[0]: r[2] for r in top_books}
        word_count_by_id = {r[0]: r[3] for r in top_books}

        if len(top_book_ids) < 3:
            conn.commit()
            return {"books": 0, "clusters": 0}

        cur.execute(f"""SELECT DISTINCT wb.word_id, wb.book_id FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active AND wb.book_id = ANY(%s)
                          AND wb.word_id = ANY(%s)""",
                    (top_book_ids, list(idf.keys())))
        rows = cur.fetchall()

    n = len(top_book_ids)
    book_index = {bid: i for i, bid in enumerate(top_book_ids)}
    word_index: dict[int, int] = {}
    row_idx, col_idx, weighted_data = [], [], []
    for wid, bid in rows:
        j = word_index.setdefault(wid, len(word_index))
        row_idx.append(book_index[bid])
        col_idx.append(j)
        weighted_data.append(idf[wid])

    matrix = csr_matrix((weighted_data, (row_idx, col_idx)), shape=(n, len(word_index)))
    binary = csr_matrix(([1] * len(row_idx), (row_idx, col_idx)), shape=(n, len(word_index)))
    shared_counts = (binary @ binary.T).toarray().astype(int)

    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).reshape(-1)
    norms[norms == 0] = 1.0  # a book with zero qualifying words (all excluded by max_df) -- avoid /0
    normalized = matrix.multiply(1 / norms[:, None]).tocsr()
    cosine = (normalized @ normalized.T).toarray()
    np.clip(cosine, -1.0, 1.0, out=cosine)  # guard float round-off past 1 on a self-pair

    dist = np.sqrt(np.clip(2 * (1 - cosine), 0, None))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    tree = linkage(condensed, method="ward", optimal_ordering=True)
    leaf_order_idx = leaves_list(tree)
    leaf_order_ids = [top_book_ids[i] for i in leaf_order_idx]
    cluster_ids = fcluster(tree, t=n_clusters, criterion="maxclust")

    # Classical (Torgerson) MDS: double-center the squared-distance matrix,
    # take the top-2 eigenvectors of the resulting Gram matrix.
    d2 = dist ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ d2 @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    top2 = np.argsort(eigvals)[::-1][:2]
    coords = eigvecs[:, top2] * np.sqrt(np.clip(eigvals[top2], 0, None))

    for axis in range(coords.shape[1]):
        col = coords[:, axis]
        if col[np.argmax(np.abs(col))] < 0:
            coords[:, axis] = -col

    seen_points: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (round(float(coords[i, 0]), 6), round(float(coords[i, 1]), 6))
        seen_points.setdefault(key, []).append(i)
    spread = max(float(np.abs(coords).max()), 1.0)
    for idxs in seen_points.values():
        if len(idxs) < 2:
            continue
        for k, i in enumerate(idxs):
            h = int(hashlib.sha256(str(top_book_ids[i]).encode()).hexdigest(), 16)
            angle = (h % 360) * math.pi / 180
            radius = 0.02 * (k + 1) * spread
            coords[i, 0] += radius * math.cos(angle)
            coords[i, 1] += radius * math.sin(angle)

    grid = [[[float(cosine[i, j]), int(shared_counts[i, j])] for j in leaf_order_idx] for i in leaf_order_idx]
    leaf_order = [{"id": bid, "title": title_by_id[bid], "author": author_by_id[bid]} for bid in leaf_order_ids]
    tree_json = _linkage_to_tree(
        tree, [{"id": bid, "title": title_by_id[bid], "author": author_by_id[bid]} for bid in top_book_ids]
    )

    from psycopg.types.json import Json

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {s}.book_cluster")
        for i, bid in enumerate(top_book_ids):
            cur.execute(
                f"""INSERT INTO {s}.book_cluster (book_id, title, author, cluster_id, mds_x, mds_y,
                                                    word_count, computed_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, now())""",
                (bid, title_by_id[bid], author_by_id[bid], int(cluster_ids[i]),
                 float(coords[i, 0]), float(coords[i, 1]), word_count_by_id[bid]),
            )
        cur.execute(
            f"""INSERT INTO {s}.book_cluster_run (id, leaf_order, grid, tree_json, computed_at)
                VALUES (1, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    leaf_order=EXCLUDED.leaf_order, grid=EXCLUDED.grid,
                    tree_json=EXCLUDED.tree_json, computed_at=now()""",
            (Json(leaf_order), Json(grid), Json(tree_json)),
        )
    conn.commit()
    return {"books": n, "clusters": int(cluster_ids.max()) if n else 0}


def compute_definition_embeddings(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                                  limit: int = 0, batch: int = 64) -> dict:
    """Embed definition_text(definition, synonyms, sentence) into
    word_embedding.definition_vector for every active word. Resumable via
    only_missing (scale-ready — see embed.py's module docstring for why this
    is per-word/incremental rather than a full-corpus recompute)."""
    from pgvector.psycopg import register_vector
    from . import embed as _embed
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
    return stats


def compute_fasttext_embeddings(conn, schema: str = DEFAULT_SCHEMA, model_path: str = "",
                                only_missing: bool = True, limit: int = 0) -> dict:
    """Compute word_embedding.fasttext_vector for every active word via a
    trained FastText model (see `concordance train-fasttext`). Unlike
    definition embedding, this never skips a word for lack of text — FastText
    composes a vector from any lemma's subwords, including words never seen
    during training."""
    from pgvector.psycopg import register_vector
    from . import embed as _embed
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


def fetch_wordnik_pronunciations(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                                  limit: int = 0, delay: float = 0.1) -> dict:
    """Fetch RAW pronunciation strings from Wordnik (ahd-5 diacritic respelling,
    arpabet, or gcide-diacritical — whichever it has) and store them as-is, with
    no IPA conversion here. Rate-limited (~1 word per several seconds observed on
    the free tier) but that cost is paid once: wordnik_checked_at gates re-fetch,
    so converting to IPA later is a separate, fast, freely-iterable pass that never
    re-triggers this fetch. only_missing also skips inactive words and anything
    that already has a valid ipa — those wouldn't gain anything from a Wordnik
    round trip, and at several seconds/word skipping them saves real hours."""
    import time
    from collections import Counter
    from . import deepdef
    s = _safe_schema(schema)
    key = deepdef.wordnik_key()
    if not key:
        return {"error": "no WORDNIK_API_KEY in .env"}

    where = (f" WHERE wordnik_checked_at IS NULL AND active"
             f" AND (ipa IS NULL OR ipa = '')") if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word{where}" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    import requests
    from .dictionary import _get
    session = requests.Session()
    dist: Counter = Counter()
    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(rows, start=1):
            r = _get(session, f"https://api.wordnik.com/v4/word.json/{lemma}/pronunciations",
                     {"api_key": key, "limit": 5})
            raw, rtype = None, None
            if r is not None and r.status_code == 200:
                data = r.json()
                if data:
                    raw, rtype = data[0].get("raw"), data[0].get("rawType")
                    dist[rtype or "unknown"] += 1
            if raw is None:
                dist["none"] += 1
            cur.execute(f"UPDATE {s}.word SET wordnik_pron_raw=%s, wordnik_pron_type=%s, "
                        "wordnik_checked_at=now() WHERE id=%s", (raw, rtype, wid))
            if i % 25 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} checked")
            time.sleep(delay)
    conn.commit()
    return {"words": len(rows), **dist}


def search_commons_direct(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                           only_missing: bool = True, limit: int = 0, delay: float = 2.5) -> dict:
    """Second-pass Commons search for words kaikki's dump reported no audio for
    (confirmed empirically to under-count: kaikki missed real, exact-match English
    recordings for words like "unpeople"/"enkindle"). Stores only the search
    result (title + constructed URL) — actually downloading is a separate,
    fast, freely-retriable step. Deliberately slow (Commons rate-limits hard);
    meant to run for hours unattended."""
    import time
    from collections import Counter
    from . import commons_search, wiktextract
    s = _safe_schema(schema)

    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_commons_search c WHERE c.word_id=w.id)"
             if only_missing else "")
    with conn.cursor() as cur:
        cur.execute(f"SELECT w.id, w.lemma FROM {s}.word w{where}" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    if not rows:
        return {"candidates": 0}

    # skip words kaikki already solved — only worth the slow search for real gaps
    lemmas = {lemma.strip().lower() for _, lemma in rows}
    dump_path = dump_path or wiktextract.DEFAULT_DUMP_PATH
    lexicon = wiktextract.build_lexicon(
        dump_path, lemmas, progress_cb=lambda n: print(f"  ...{n} lines scanned"))
    candidates = [(wid, lemma) for wid, lemma in rows
                  if not lexicon.get(lemma.strip().lower(), {}).get("audio")]

    dist: Counter = Counter(total=len(rows), skipped_kaikki_has_audio=len(rows) - len(candidates))
    session = requests.Session()
    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(candidates, start=1):
            titles = commons_search.search_word(lemma, session)
            match = commons_search.best_english_exact_match(titles, lemma)
            url = commons_search.download_url(match) if match else None
            cur.execute(
                f"""INSERT INTO {s}.word_commons_search (word_id, found_title, download_url, checked_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET found_title=EXCLUDED.found_title,
                        download_url=EXCLUDED.download_url, checked_at=now()""",
                (wid, match, url))
            dist["found"] += 1 if match else 0
            dist["not_found"] += 0 if match else 1
            if i % 20 == 0:
                conn.commit()
                print(f"  ...{i}/{len(candidates)} searched")
            time.sleep(delay)
        # words skipped because kaikki already has audio still need a checked_at
        # row so a re-run doesn't re-parse the dump for them pointlessly
        for wid, lemma in rows:
            if (wid, lemma) not in candidates:
                cur.execute(
                    f"""INSERT INTO {s}.word_commons_search (word_id, found_title, download_url, checked_at)
                        VALUES (%s, NULL, NULL, now()) ON CONFLICT (word_id) DO NOTHING""", (wid,))
    conn.commit()
    return dict(dist)


def compute_ipa(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                 only_missing: bool = True, limit: int = 0) -> dict:
    """Backfill + clean word.ipa. Sources tried in order per word: (1) kaikki's
    Wiktextract dump; (2) Wordnik's raw pronunciation (already fetched by
    `wordnik-pron`), converted via the matching notation converter — direct
    IPA as-is, ARPAbet or AHD respellings through their own deterministic
    converters (gcide-diacritical has no converter yet, lowest yield, skipped);
    (3) the local vocab.wiktionary dump's us_pronunciation column — low yield
    (it's the same Wiktionary data kaikki's dump already draws from, just a
    different snapshot, so it only rescues the handful of words where the two
    dumps disagree) but free, since the DB connection is already open.
    Also NULLs out any existing transcription that fails the English-language
    sanity check (the pre-existing ad hoc scrape occasionally grabbed a
    cross-referenced foreign cognate's IPA instead of the word's own — e.g.
    "murmurer" had the French verb's transcription). Idempotent: with
    only_missing=True (default), only words with an empty or invalid ipa are
    candidates, so a re-run after everything's resolved does no dump parsing
    at all and is a no-op."""
    from collections import Counter
    from . import ahd, arpabet, audio, localdict, wiktextract
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, ipa, wordnik_pron_raw, wordnik_pron_type "
                    f"FROM {s}.word ORDER BY id")
        all_rows = cur.fetchall()

    def is_valid(ipa):
        return bool(ipa) and audio.looks_like_english_ipa(ipa)

    candidates = all_rows if not only_missing else [r for r in all_rows if not is_valid(r[2])]
    dist: Counter = Counter(total=len(all_rows), already_valid=len(all_rows) - len(candidates))
    # `limit` slices the already-filtered candidate set, not the raw fetch --
    # applying it beforehand (the original bug) could silently hand back
    # fewer than `limit` words, or zero, depending on where the first N rows
    # in scan order happened to already be valid. already_valid above is
    # computed from the full filtered set, before this slice, so it still
    # reflects the whole table regardless of `limit`.
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        return dict(dist)

    lemmas = {lemma.strip().lower() for _, lemma, _, _, _ in candidates}
    dump_path = dump_path or wiktextract.DEFAULT_DUMP_PATH
    lexicon = wiktextract.build_lexicon(
        dump_path, lemmas, progress_cb=lambda n: print(f"  ...{n} lines scanned"))
    local_lexicon = localdict.build_lexicon(conn, lemmas)

    def wordnik_ipa(raw, rtype):
        if not raw:
            return None
        if rtype == "IPA":
            converted = raw
        elif rtype == "arpabet":
            converted = arpabet.to_ipa(raw)
        elif rtype == "ahd-5":
            converted = ahd.to_ipa(raw)
        else:
            return None  # gcide-diacritical: no converter yet
        return converted if converted and audio.looks_like_english_ipa(converted) else None

    def local_wiktionary_ipa(lemma):
        for _pos, _definition, ipa, *_rest in local_lexicon.get(lemma, []):
            if ipa and audio.looks_like_english_ipa(ipa):
                return ipa
        return None

    with conn.cursor() as cur:
        for i, (wid, lemma, existing_ipa, wn_raw, wn_type) in enumerate(candidates, 1):
            if i % 5000 == 0:
                conn.commit()
                print(f"  ...{i}/{len(candidates)} words checked")
            had_valid_existing = is_valid(existing_ipa)
            lemma_lc = lemma.strip().lower()
            entry = lexicon.get(lemma_lc, {})
            kaikki_ipa = wiktextract.best_ipa(entry.get("ipa", []))
            if kaikki_ipa and not audio.looks_like_english_ipa(kaikki_ipa):
                kaikki_ipa = None
            wn_ipa = wordnik_ipa(wn_raw, wn_type)
            replacement = kaikki_ipa or wn_ipa or local_wiktionary_ipa(lemma_lc)
            source = "kaikki" if kaikki_ipa else ("wordnik" if wn_ipa else ("local_wiktionary" if replacement else None))

            if had_valid_existing and not replacement:
                dist["already_valid"] += 1  # nothing to do, no change
                continue
            if not (existing_ipa or "").strip() and replacement:
                cur.execute(f"UPDATE {s}.word SET ipa=%s WHERE id=%s", (replacement, wid))
                dist[f"backfilled_{source}"] += 1
            elif (existing_ipa or "").strip() and not had_valid_existing and replacement:
                cur.execute(f"UPDATE {s}.word SET ipa=%s WHERE id=%s", (replacement, wid))
                dist[f"corrected_{source}"] += 1
            elif (existing_ipa or "").strip() and not had_valid_existing:
                cur.execute(f"UPDATE {s}.word SET ipa=NULL WHERE id=%s", (wid,))
                dist["cleared_no_replacement"] += 1
            else:
                dist["unresolved"] += 1
    conn.commit()
    return dict(dist)


def download_commons_direct_finds(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0,
                                   delay: float = 4.0) -> dict:
    """Download the real recordings `commons-search` confirmed exist, upgrading
    any word currently on 'azure' or 'none' to the real recording. Split out
    from `compute_audio` because interleaving Commons downloads with fast Azure
    calls exhausted Commons' upload-CDN rate limit mid-run (429s that the
    per-request backoff wasn't patient enough for — this earlier in the session
    took over a minute to clear even at near-zero request volume). Paced like
    `commons-search` itself: slow, meant to run unattended."""
    import time
    from collections import Counter
    from . import audio
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT w.id, w.lemma, cs.download_url, a.source
            FROM {s}.word w
            JOIN {s}.word_commons_search cs ON cs.word_id = w.id
            LEFT JOIN {s}.word_audio a ON a.word_id = w.id
            WHERE cs.found_title IS NOT NULL AND (a.source IS NULL OR a.source <> 'commons')
        """ + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter(candidates=len(rows))
    if not rows:
        return dict(dist)
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma, url, prior_source) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
            if audio.fetch_commons_audio(url, dest, tries=6):
                cur.execute(
                    f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice, license_note, generated_at)
                        VALUES (%s,'commons',%s,NULL,%s,%s, now())
                        ON CONFLICT (word_id) DO UPDATE SET source='commons', file_path=EXCLUDED.file_path,
                            ipa_used=NULL, voice=EXCLUDED.voice, license_note=EXCLUDED.license_note, generated_at=now()""",
                    (wid, str(dest), url,
                     "Wikimedia Commons recording (direct search — kaikki's dump missed it); "
                     "verify per-file license before public reuse"))
                dist["downloaded"] += 1
                dist[f"upgraded_from_{prior_source}"] += 1 if prior_source else 0
            else:
                dist["failed"] += 1
            if i % 20 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} downloaded")
            time.sleep(delay)
    conn.commit()
    return dict(dist)


def compute_audio(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                   only_missing: bool = True, limit: int = 0, delay: float = 0.3) -> dict:
    """Fill in word_audio: real Commons recordings where kaikki/Wiktextract has
    one, else a real recording the direct Commons search found that kaikki
    missed, else Azure IPA-guided synthesis where a transcription is known
    (ours, kaikki's, or Wordnik's — backfilling word.ipa along the way), else
    a 'none' placeholder so re-runs don't keep re-parsing the dump for words
    with nothing to find."""
    import time
    from collections import Counter
    from . import audio, wiktextract
    s = _safe_schema(schema)

    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_audio a WHERE a.word_id=w.id)"
             if only_missing else "")
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma, w.ipa, cs.download_url
                        FROM {s}.word w
                        LEFT JOIN {s}.word_commons_search cs ON cs.word_id = w.id{where}""" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter()
    if not rows:
        return {"candidates": 0, **dist}

    lemmas = {lemma.strip().lower() for _, lemma, _, _ in rows}
    dump_path = dump_path or wiktextract.DEFAULT_DUMP_PATH
    lexicon = wiktextract.build_lexicon(
        dump_path, lemmas, progress_cb=lambda n: print(f"  ...{n} lines scanned"))

    key, region = audio.azure_credentials()
    if not (key and region):
        print("  (no AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in .env — skipping synthesis, Commons-only pass)")
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma, existing_ipa, direct_search_url) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            entry = lexicon.get(lemma_lc, {})

            existing_ipa = existing_ipa if audio.looks_like_english_ipa(existing_ipa or "") else None

            kaikki_ipa = wiktextract.best_ipa(entry.get("ipa", []))
            if kaikki_ipa and not audio.looks_like_english_ipa(kaikki_ipa):
                kaikki_ipa = None
            if kaikki_ipa and not (existing_ipa or "").strip():
                cur.execute(f"UPDATE {s}.word SET ipa=%s WHERE id=%s", (kaikki_ipa, wid))
                existing_ipa = kaikki_ipa

            # tries=2 (not fetch_commons_audio's default 4-6): this loop needs to
            # move fast through many candidates and has Azure as a good fallback.
            # A sustained Commons rate-limit block turned a handful of slow
            # downloads into an hours-long stall here — `commons-download` is the
            # dedicated, patient (tries=6) pass for real recovery, run separately.
            best_recording = wiktextract.best_audio(entry.get("audio", []))
            row = None
            if best_recording:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                if audio.fetch_commons_audio(best_recording["url"], dest, tries=1):
                    row = ("commons", str(dest), None, best_recording["url"],
                           "Wikimedia Commons recording; verify per-file license before public reuse")
                    dist["commons"] += 1
            if row is None and direct_search_url:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                if audio.fetch_commons_audio(direct_search_url, dest, tries=1):
                    row = ("commons", str(dest), None, direct_search_url,
                           "Wikimedia Commons recording (direct search — kaikki's dump missed it); "
                           "verify per-file license before public reuse")
                    dist["commons_direct_search"] += 1
            if row is None and (existing_ipa or "").strip() and key and region:
                clip = audio.synthesize_azure(lemma, existing_ipa, key, region)
                if clip:
                    dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                    dest.write_bytes(clip)
                    row = ("azure", str(dest), audio.normalize_ipa(existing_ipa), audio.AZURE_VOICE, None)
                    dist["azure"] += 1
            if row is None:
                row = ("none", None, None, None, None)
                dist["none"] += 1

            cur.execute(
                f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice, license_note, generated_at)
                    VALUES (%s,%s,%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET source=EXCLUDED.source, file_path=EXCLUDED.file_path,
                        ipa_used=EXCLUDED.ipa_used, voice=EXCLUDED.voice, license_note=EXCLUDED.license_note,
                        generated_at=now()""",
                (wid, *row))
            if i % 50 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} words processed")
            time.sleep(delay)
    conn.commit()
    return {"candidates": len(rows), **dist}


def synthesize_unverified_guesses(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0,
                                   delay: float = 0.3) -> dict:
    """Last resort for words with no real recording and no IPA anywhere: Azure
    guesses pronunciation from spelling alone, same as any TTS would. Recorded
    with source='azure_guess' — deliberately distinct from 'azure' (IPA-guided)
    so the quiz app can flag these as unverified rather than presenting a guess
    with the same confidence as a verified pronunciation."""
    import time
    from collections import Counter
    from . import audio
    s = _safe_schema(schema)

    key, region = audio.azure_credentials()
    if not (key and region):
        return {"error": "no AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in .env"}

    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma FROM {s}.word w
                        JOIN {s}.word_audio a ON a.word_id = w.id
                        WHERE a.source = 'none'""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter(candidates=len(rows))
    if not rows:
        return dict(dist)
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            clip = audio.synthesize_azure_guess(lemma, key, region)
            if clip:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                dest.write_bytes(clip)
                cur.execute(
                    f"""UPDATE {s}.word_audio SET source='azure_guess', file_path=%s, ipa_used=NULL,
                        voice=%s, license_note='unverified: no IPA available, Azure guessed from spelling',
                        generated_at=now() WHERE word_id=%s""",
                    (str(dest), audio.AZURE_VOICE, wid))
                dist["synthesized"] += 1
            else:
                dist["failed"] += 1
            if i % 50 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} words processed")
            time.sleep(delay)
    conn.commit()
    return dict(dist)


# --- maintain progress (§ maintain-status) -------------------------------------
#
# Read-only: safe to run at any time, including against a schema another
# `maintain`/`backfill-analogies` process is actively writing to (it never
# takes a lock beyond a plain SELECT). Exists because `maintain` chains ~16
# steps and, for a run kicked off hours ago in a terminal nobody's watching
# anymore, its own step-by-step console output is long gone -- this
# reconstructs an equivalent picture from DB state instead.
#
# Each step's "done" fraction is a proxy, not a real percentage the step
# itself reports -- most steps loop over "words not yet touched" and this
# just counts touched-vs-total the same way. A few are legitimately capped
# below 100% forever (e.g. fill-definitions: some words are permanently
# undefined and cast out; quizdef/quizzable are similar) -- that's not a
# stalled step, just a step whose target was never "every active word."
MAINTAIN_STEPS = [
    "fill-definitions", "classify", "normalize-pos", "ngram", "archaic",
    "difficulty", "quizdef", "quizzable", "calibrate-difficulty",
    "book-similarity", "author-similarity", "author-clustering",
    "wordnik-pron", "ipa", "embed", "backfill-analogies",
]


def maintain_status(conn, schema: str = DEFAULT_SCHEMA) -> list[dict]:
    """One dict per MAINTAIN_STEPS entry, in maintain's own run order:
    {name, done, total, note}. `done`/`total` are None for a step with no
    meaningful fraction (normalize-pos, quizzable, calibrate-difficulty,
    book-clustering, author-clustering -- each recomputes fully in one fast
    pass rather than incrementally skipping already-touched rows, so there's nothing to be
    partway through between runs)."""
    s = _safe_schema(schema)
    steps: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active")
        active_total = cur.fetchone()[0]

        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND coalesce(definition,'') <> ''")
        steps.append({"name": "fill-definitions", "done": cur.fetchone()[0], "total": active_total,
                      "note": "some words are permanently undefined (cast out) -- won't reach 100%"})

        cur.execute(f"""SELECT count(DISTINCT wc.word_id) FROM {s}.word_category wc
                        JOIN {s}.word w ON w.id = wc.word_id WHERE w.active""")
        steps.append({"name": "classify", "done": cur.fetchone()[0], "total": active_total,
                      "note": "a word can legitimately classify to zero categories -- "
                              "won't reach 100% either way"})

        steps.append({"name": "normalize-pos", "done": None, "total": None,
                      "note": "recomputes fully each run -- fast, no meaningful fraction"})

        cur.execute(f"SELECT count(*) FROM {s}.word_ngram")
        steps.append({"name": "ngram", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"SELECT count(*) FROM {s}.word_difficulty WHERE archaic_confidence IS NOT NULL")
        steps.append({"name": "archaic", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"SELECT count(*) FROM {s}.word_difficulty WHERE difficulty IS NOT NULL")
        steps.append({"name": "difficulty", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND quiz_definition IS NOT NULL")
        steps.append({"name": "quizdef", "done": cur.fetchone()[0], "total": active_total,
                      "note": "only words with a definition to build one from are in scope"})

        steps.append({"name": "quizzable", "done": None, "total": None,
                      "note": "recomputes fully each run -- fast, no meaningful fraction"})

        steps.append({"name": "calibrate-difficulty", "done": None, "total": None,
                      "note": "driven by real quiz answers, not a fixed word-count target"})

        cur.execute(f"SELECT count(*) FROM {s}.book_similarity")
        steps.append({"name": "book-similarity", "done": None, "total": cur.fetchone()[0],
                      "note": "row count only -- pairwise, no fixed target to divide by"})

        steps.append({"name": "book-clustering", "done": None, "total": None,
                      "note": "recomputes fully each run -- fast, no meaningful fraction"})

        cur.execute(f"SELECT count(*) FROM {s}.author_similarity")
        steps.append({"name": "author-similarity", "done": None, "total": cur.fetchone()[0],
                      "note": "row count only -- pairwise, no fixed target to divide by"})

        steps.append({"name": "author-clustering", "done": None, "total": None,
                      "note": "recomputes fully each run -- fast, no meaningful fraction"})

        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND wordnik_checked_at IS NOT NULL")
        steps.append({"name": "wordnik-pron", "done": cur.fetchone()[0], "total": active_total,
                      "note": "rate-limited (~seconds/word on the free tier) -- typically the slowest step"})

        # ipa IS NOT NULL alone overcounts: a prior compute_ipa pass clears an
        # unresolvable word to '' (empty string, not NULL) rather than leaving
        # a stale bad value -- see compute_ipa's cleared_no_replacement branch.
        # Matching the same "still needs work" test the pipeline itself uses
        # (fetch_wordnik_pronunciations' and compute_ipa's own WHERE clauses).
        cur.execute(f"SELECT count(*) FROM {s}.word WHERE active AND ipa IS NOT NULL AND ipa <> ''")
        steps.append({"name": "ipa", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"""SELECT count(*) FROM {s}.word_embedding e
                        JOIN {s}.word w ON w.id = e.word_id
                        WHERE w.active AND e.definition_vector IS NOT NULL""")
        steps.append({"name": "embed", "done": cur.fetchone()[0], "total": active_total, "note": ""})

        cur.execute(f"""SELECT count(*) FROM {s}.word w JOIN {s}.word_difficulty wd ON wd.word_id = w.id
                        WHERE w.active AND wd.quizzable = true AND w.part_of_speech IS NOT NULL
                          AND w.definition IS NOT NULL""")
        analogy_total = cur.fetchone()[0]
        cur.execute(f"SELECT count(*) FROM {s}.wn_relation_scan")
        steps.append({"name": "backfill-analogies", "done": cur.fetchone()[0], "total": analogy_total,
                      "note": "eligible = active, quizzable, WordNet-mappable POS"})

    return steps
