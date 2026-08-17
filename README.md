# Concordance

Extract interesting vocabulary from books you read (EPUB, text PDF, `.txt`)
using a **local** LLM — no paid API required. Rare, worth-learning words are
surfaced; common words and junk are filtered out. Results land straight in
Postgres, reviewed afterward in a web app with quizzing, visualizations, and
user accounts built on top.

Design is deliberately **keep-biased**: a genuine rarity should survive to
your review even at the cost of a little noise, never the reverse. See
[DESIGN.md](DESIGN.md) for the full rationale behind this and every other
non-obvious choice below — this document only covers what things do and how
to run them.

## Pipeline

```
extract → clean → tokenize → frequency-floor → cross-book verdict cache
        → strip-proper-nouns → validity-gate → LLM-judge → dictionary-lookup
        → Postgres (ingest) → maintain's fill-definitions (refill/deepen)
```

- **frequency floor** — a stop-word-style cut of common words (a floor only, never a rarity ceiling).
- **cross-book verdict cache** — a lemma already kept/pruned/judge-rejected in an earlier book is pre-marked and never re-judged, since the judge's input is purely `(lemma, frequency band)` and its verdict at temp 0 is always the same for a given lemma. Judge cost tracks *distinct new rare words*, not corpus size.
- **validity gate** — a word is real if *any* authority vouches for it (local Wiktionary dump, SymSpell, WordNet, NLTK's word corpus), checked cheapest-first. Misspelling is considered last, by relative frequency, with a recurrence escape hatch.
- **LLM judge** — a local model decides what's worth learning; frequency alone can't do this (*tendril* is rarer than *refectory* but everyone knows it).
- **dictionary lookup** — one shared cascade (`concordance/resolve.py`) used by every enrichment path: local Wiktionary → Free Dictionary API → online Wiktionary → OED reference dataset → Merriam-Webster → Wordnik → yourdictionary.com → web search + grounded local-LLM extraction. `ingest`'s own inline cascade stops after Merriam-Webster (network throughput); anything still blank gets the rest of the depth from `maintain`.
- **review** — prune too-common/easy terms afterward in the [web app](#web-app-webapp) — a soft delete, `word.active = false`, nothing is ever hard-deleted.
- Every cut is logged to `rejected_word`, one row per (book, lemma) — except a deterministic reason like frequency-floor, which doesn't vary by book and isn't worth duplicating per occurrence. See DESIGN.md.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

```bash
pip install -e ".[web]"          # + web app backend deps
pip install -e ".[embed]"        # + sentence-transformers/pgvector/fasttext, for `embed`/`train-fasttext`
pip install -e ".[scrape]"       # + Playwright, for the Merriam-Webster site-scrape fallback
playwright install chromium      # only if you installed [scrape]
(cd webapp/frontend && npm install)   # web app frontend
```

NLTK's `wordnet`/`words` corpora download automatically on first run.

### Environment

Set these in the environment or a git-ignored `.env` at the repo root:

| Variable | Required for | Notes |
|---|---|---|
| `DATABASE_URL` | Everything | `postgresql://user:pass@host:5432/dbname`. The validity gate and enrichment both need the local `vocab.wiktionary` dump loaded into this DB — see below. |
| `CONCORDANCE_DB_SCHEMA` | — | Overrides the default `concordance` schema name. Optional. |
| `MW_DICTIONARY_API_KEY` | `ingest`'s MW enrichment step, `mw-backfill`, `lookup_mw.py`, `deepen` | Free tier: 1000 queries/day. |
| `WORDNIK_API_KEY` | `deepen`'s Wordnik tier | Falls back to yourdictionary-only without it. |
| `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` | `audio`, `audio-guess` | Azure TTS for IPA-guided/guessed pronunciation synthesis. |

`vocab.wiktionary` (~500k terms) is checked first everywhere because it's
free and — unlike every frequency-derived authority — carries no "Proper
noun" POS category to be confused by real names. Load this dump into your
Postgres instance before running `ingest` for the first time (see the
related-project docs for that dump if you don't already have it loaded).

## Quick start

```bash
# 1. Ingest a book straight into Postgres
concordance ingest "Ulysses -- Joyce, James.txt"

# 2. Or process everything sitting in incoming/ (filenames "Title -- Author.ext")
concordance ingest

# 3. Stand up the review/quiz web app
pip install -e ".[web]"
(cd webapp/frontend && npm install)
concordance create-admin yourname     # first admin account, prompts for a password
./webapp/dev.sh                       # http://localhost:5173
```

Review and prune happens in the web app afterward — see
[Web app](#web-app-webapp) below.

## Ingesting books

```bash
concordance ingest "some book.epub"                 # one file
concordance ingest                                    # every .epub/.pdf/.txt in incoming/
concordance ingest --model models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

Runs extract → filter → judge → enrich and writes straight into Postgres.
Kept words upsert into `word`/`word_book`; dropped candidates go to
`rejected_word`. Idempotent — re-ingesting the same book updates rows in
place, never clobbers a filled field with a blank one from a re-run.

Batch mode (no argument) processes every `.epub`/`.pdf`/`.txt` in
`incoming/`, parsing `[Title] -- [Author]` from each filename (no delimiter
found → whole filename becomes the title, blank author). Each file moves to
`archive/` after processing. The judge model, spaCy, and the validity gate's
dictionaries load once for the whole batch, not once per book.

Key flags: `--min-zipf` (frequency floor; higher = rarer only, default 3.5),
`--limit`, `--no-lookup`, `--model`/`-m`, `--stub` (force the no-model
fallback judge), `--workers` (enrichment thread pool, default 4), `--no-mw`,
`--schema`, `--oed-schema` (default `oed`), `--database-url`, `--no-archive`
(batch mode only). `concordance run` is the same pipeline but writes a
`.vocab.csv` for hand-editing instead of the DB — a legacy path, no longer
the primary workflow; `finalize`/`sync-db`/`define` remain for anyone still
on it.

## Running the local judge model (RTX 3060, 12GB example)

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
# or, if no prebuilt wheel matches your setup:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python

pip install huggingface-hub
huggingface-cli download bartowski/Qwen2.5-14B-Instruct-GGUF \
  Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir models
```

The judge defaults to that 14B model; if the file is missing it silently
falls back to a no-model stub judge (keeps every survivor, so the pipeline
still runs end-to-end) — pass `--stub` to force the stub even when the model
is present. `Config.n_gpu_layers = -1` offloads as many layers to the GPU as
VRAM allows; reduce it if you hit out-of-memory. A 7B/8B model at Q5_K_M is a
faster, lower-quality alternative for big books.

## Post-ingest maintenance

```bash
concordance maintain
```

Runs the whole enrichment/scoring/pronunciation/embedding chain in
dependency order in one command: `fill-definitions → classify →
normalize-pos → ngram → archaic → difficulty → quizdef → quizzable →
calibrate-difficulty → wordnik-pron → ipa → embed → backfill-analogies`.
Every step runs incrementally (only-missing), so a re-run after catching up
is fast — it only touches the newest batch. The *first* run against a real
backlog is not: `classify`, `quizdef`, and `backfill-analogies` each load a
local LLM and call it per word, so catching up thousands of words can take
hours. `--skip-<step>` (one flag per step, e.g. `--skip-classify`,
`--skip-fill-definitions`) defers a slow step to run separately; `--limit`
caps words processed per step, for chunking a big backlog into resumable
pieces. `--recheck-after-days` (default 14) skips re-grinding a word whose
last resolution attempt failed recently.

`book-similarity`/`book-clustering`/`author-similarity`/`author-clustering`
and `book-fame`/`author-fame` are deliberately **not** part of this chain —
each is a full-corpus recompute with no incremental mode, so folding one in
would mean `maintain` could never have a cheap catch-up run. Run those on
their own schedule (see their own sections below). `load-taxonomy` and
`train-fasttext` are one-time/occasional setup, also excluded; so are the
Commons/Azure audio steps, since Commons rate-limits hard and is meant to
run for hours unattended on its own.

```bash
concordance maintain-status              # one-shot progress snapshot
concordance maintain-status --watch 30   # re-render every 30s
```

Read-only — reconstructs each step's progress from DB state (words touched
vs. eligible), safe to run at any time including alongside a live `maintain`.
Useful for checking on an unattended multi-hour run.

## Definitions

```bash
concordance refill                # cheap sources only: local Wiktionary, Free Dictionary API, online Wiktionary
concordance deepen                 # + OED, Merriam-Webster, Wordnik, yourdictionary, web-search+LLM
concordance deepen --no-web        # skip the web-search/LLM tier (faster, no model load)
concordance mw-backfill            # Merriam-Webster (API + Playwright scrape) for what deepen still missed
```

`maintain`'s `fill-definitions` step runs `refill` then `deepen` as one
combined pass per word now (`--skip-fill-definitions`, or the deprecated
`--skip-refill`/`--skip-deepen` aliases, to skip it). `refill`/`deepen`
remain as separate standalone commands for running just the cheap pass, or
re-running the deep pass on demand. Neither overwrites an existing
definition — both only fill blanks.

A word kept without ever getting a definition is durably marked
`word.flagged_undefined` (sticky — never cleared, even once defined later; a
permanent "needed a second look" audit trail). Whatever `deepen`/
`fill-definitions` still can't define gets a deterministic validity estimate
(`word.validity_label`: `likely-valid`/`uncertain`/`likely-artifact`,
`validity_score` 0–1, `validity_notes`) — cross-reference
`flagged_undefined = true AND validity_label = 'likely-artifact'` for your
prune queue.

A separate flag, `word.variant_flag_reason`, marks a word that *was*
successfully defined but looks like a foreign word or an archaic spelling of
a common one — never an auto-reject, see DESIGN.md for why. Query it directly:

```sql
SELECT lemma, variant_flag_reason, variant_flag_note
FROM word WHERE variant_flag_reason IS NOT NULL ORDER BY lemma;
```

`scripts/sweep_variant_rejects.py` (`--apply` to write) runs the same check
retroactively against words already active before the flag existed.

```bash
concordance dedupe-plurals         # consolidate "warrs" (-> plural of warr) into "warr"
concordance expand-synonyms        # replace "Synonym of X" definitions with real content
concordance link-definitions       # cross-link a definition's mention of another active word
```

Both `dedupe-plurals`/`expand-synonyms` default to `--web` (full cascade
depth, `--no-web` to stay keyless/local-only) and accept `--schema`,
`--limit`, `--database-url`. `link-definitions` is a full recompute every
run (cheap, no LLM) — not part of `maintain`, safe to re-run any time a
definition changes.

Standalone lookup tools:

```bash
python scripts/lookup_mw.py concordance          # print a full MW entry to stdout
python scripts/lookup_mw.py concordance --no-fallback   # API tier only, skip the site scrape
```

## Enrichment & scoring

```bash
concordance load-taxonomy          # once: load the USAS category tables
concordance load-gazetteer         # once: load the names/places gazetteer (validity gate's proper-noun check)
concordance classify               # tag every word with 1-3 USAS domain codes
concordance normalize-pos          # fold part_of_speech into one clean vocabulary
concordance ngram                  # cache Google Books Ngram rarity/recency per word
concordance archaic                # set current/dated/archaic/obsolete + confidence
concordance difficulty             # 0-100 ex-ante difficulty scalar + factor breakdown
concordance quizdef                # quiz-safe definitions (rewrite ones that leak the target word)
concordance quizzable               # flag variant/inferable-derivative words as unquizzable
concordance calibrate-difficulty   # per-user difficulty nudge from quiz response data
```

Run in roughly this order the first time (`maintain` already does). All
accept `--schema`, `--limit`, `--database-url`; most take `--only-missing` or
a `--refresh`/`--refetch` toggle for incremental vs. full recompute — see
each command's `--help`. `archaic` needs `ngram` to have run first.
`normalize-pos`/`archaic`/`difficulty`/`quizzable` always recompute every row
in scope (capped by `--limit`, not gated by only-missing) since they read
mutable upstream columns with no separate staleness signal.
`calibrate-difficulty` writes to a **separate** `word_personal_difficulty`
table, never the shared `word_difficulty.difficulty` column — see DESIGN.md
for why a single-rater deployment can't do population-level calibration.

`load-gazetteer` needs its two source files downloaded first (US Census
surnames + GeoNames populated places — see `concordance/gazetteer.py`'s
module docstring for the exact URLs); given names come from NLTK's `names`
corpus, already a dependency. Not needed for `ingest` to run — the validity
gate degrades to a no-op proper-noun check without it, same as every other
optional data source here — but see DESIGN.md for why it's the single
highest-leverage piece of the proper-noun-pollution defenses.

### Pronunciation audio

```bash
concordance wordnik-pron       # fetch raw Wordnik transcriptions
concordance ipa                # backfill+validate word.ipa (kaikki -> Wordnik -> local Wiktionary -> OED)
concordance oed-ipa            # backfill word.ipa from OED's verified pronunciations specifically
concordance commons-search     # find real Wikimedia Commons recordings kaikki's dump missed
concordance commons-download   # download the recordings commons-search confirmed
concordance audio              # Commons/MW recording if present, else Azure IPA-guided TTS
concordance audio-guess        # last resort: Azure guesses from spelling alone
```

Real recordings first, IPA-guided synthesis otherwise, spelling-guess only
as a true last resort (`audio-guess` results are tagged `source='azure_guess'`
so the app can flag them unverified). `wordnik-pron`/`ipa` are part of
`maintain`; run `ipa` before `audio` since synthesis quality depends on the
transcription it's given. `commons-search`/`commons-download`/`audio` stay
separate commands — Commons rate-limits hard and is meant to run for hours
unattended, which would starve every other step if interleaved.

## Semantic distance

```bash
concordance train-fasttext           # once (or after a big ingest batch): train on archive/
concordance embed                    # definition_vector for words missing one
concordance embed --signal fasttext  # fasttext_vector instead (needs the trained model)
concordance embed --signal both
```

Two independent per-word vectors — a `sentence-transformers` embedding of
each word's dictionary *gloss* (falls back to synonyms, then the book
example sentence), and a from-scratch FastText model trained on this
project's own `archive/` corpus (covers every lemma, even undefined ones).
Neither is a precomputed all-pairs matrix; both are queried on demand via a
pgvector HNSW index. Powers `/api/words/{id}/neighbors` and the
Visualizations page's word-relatedness graph.

## Vocabulary relatedness

```bash
concordance book-similarity      # each book's top-k most vocabulary-related books
concordance book-clustering      # hierarchical clustering + 2D map over the top-N books
concordance author-similarity    # same, one level up, for authors
concordance author-clustering    # hierarchical clustering + 2D map over the top-N authors
concordance relate               # convenience wrapper: runs all four above with default options
```

Lexical (shared-vocabulary) relatedness — a different axis from the semantic
embeddings above. IDF-weighted cosine similarity over each entity's
word-presence vector; only each entity's top-k neighbors are stored (default
`--top-k 20`, `--min-shared-words 3`), not a full matrix. `book-similarity`/
`author-similarity` always recompute everything in scope (IDF weights are
corpus-wide, so an only-missing mode wouldn't be correct) — this is a
memory-bounded but not cost-bounded computation; see DESIGN.md for the real
tradeoff. `author-clustering`/`book-clustering` take `--top-n` (default 200),
`--n-clusters` (default 12), and `--min-fame` (cluster by fame-score
threshold instead of top-N by volume, writing a separate run rather than
overwriting the default one). `relate` skips flags per step
(`--skip-book-similarity`, etc.) and `--limit` (similarity steps only); use
the standalone commands for `--top-k`/`--top-n`/`--n-clusters`/`--min-fame`.

None of these four are part of `maintain` — run standalone on your own
schedule. Reachable from the web app's Visualizations page.

## Cultural & historical importance

```bash
concordance author-fame              # score every real author, 1-10
concordance book-fame                # score every book, one level down
concordance author-fame --stub       # dry run: gather + print evidence, no LLM call, no write
```

An absolute (not corpus-relative) 1–10 score, LLM-judged against a fixed
external rubric — see DESIGN.md for why absolute over percentile. Run
`author-fame` before `book-fame` (weak context only, never a floor/cap).
Genuinely expensive — several network round-trips plus one LLM generation
per entity, ~5–15s each, so a full multi-thousand-entity corpus is on the
order of a day. `--stale-days` rescores entities checked more than N days
ago (default: never-scored only).

## Analogy relations

```bash
concordance backfill-analogies
```

Builds the word-relation graph behind the quiz's "*A is to B as C is to
___*" question type — WordNet relations plus spaCy-parsed definition
patterns, both mandatory-verified by the local LLM against both terms' real
definitions before use (see DESIGN.md — unverified WordNet "hypernym" edges
are often disguised synonyms). Resumable at two grains: already-scanned
terms are skipped on a later run, and each `--chunk-size` (default 200)
batch commits as its own unit. Runs as the last step of `maintain` by
default (`--skip-analogies` to defer it) — don't run standalone
concurrently with `maintain`, both load a local model onto the same GPU.

## OED reference dictionary

A separate, standalone ingestion pipeline over scanned OED volume PDFs, into
its own `oed` Postgres schema — not merged into the vocabulary pipeline
above, with its own admin-only browsing UI.

```bash
concordance oed-ingest                                    # every .pdf in dictionaries/
concordance oed-ingest dictionaries/09.pdf                # one volume
concordance oed-ingest --page-limit 20 dictionaries/09.pdf   # smoke-test
concordance oed-lemma                                      # flag lemma vs. inflected-form entries
concordance oed-concordance-match                          # cross-reference against concordance's own vocabulary
```

Idempotent and resumable by file hash. Pronunciation is read from a cropped
page image by a local vision-LLM, never trusted from OCR text — falls back
to `pronunciation_needs_review = true` with no transcription if no vision
model is configured. `oed-lemma` needs to run before `oed-concordance-match`
(only lemma-flagged entries are cross-referenced). `oed-ipa` (above) is the
one place OED feeds back into the main pipeline. None of the `oed-*` commands
are part of `maintain` — run as volumes are acquired.

## Web app

The review/quiz/visualization app. Deleting a word never hard-deletes it —
`word.active = false`, so every downstream feature just filters on
`active = true`.

```bash
pip install -e ".[web]"
(cd webapp/frontend && npm install)
concordance create-admin yourname   # first admin account
./webapp/dev.sh                     # backend + frontend together, http://localhost:5173
```

- **Accepted tab** — term/POS/definition/difficulty/validity table, one-click prune (soft delete).
- **Rejected tab** — browse `rejected_word`, filterable by book/reason, "Add" rescues a word back in.
- **OED entries tab** (admin) — read-only browsing of the `oed` schema.
- Every book/author detail page shows fame score + reasoning, a domain-distribution chart, a difficulty histogram, and cross-linked definitions.
- **Home page** (`/app`) — corpus headline stats + links into every section.

### User accounts

```bash
concordance create-admin <username>   # first admin account, prompts for a password
```

The curation UI is admin-only, enforced app-side (`require_admin`). Separate
invite-gated accounts exist for browsing/quizzing: an admin's **"+ Generate
Invite Link"** button (Accepted tab) mints a one-time `/register?token=...`
link (7-day expiry); whoever opens it sets a username/password and lands on
`/app` with no curation access. Sessions are an httpOnly cookie backed by a
`sessions` table (not JWTs, so they're server-revocable), passwords hashed
with Argon2. Run `create-admin` before anything else touches a fresh
deployment — `require_admin` fails closed, so no admin row means no one can
reach the curation API at all.

`./webapp/dev.sh` sets `WATCHFILES_FORCE_POLLING=true` — needed because this
repo lives on a Windows drive mounted into WSL, where native filesystem
change notifications don't reliably reach either dev server's watcher.

### Quizzing (`/app/quiz`)

Any logged-in account can take a quiz. Four question types (multiple choice,
true/false, matching, analogy — see "Analogy relations" above), blendable in
one session; direction (word→definition or definition→word); filters on
length/difficulty/POS/domain/genre; POS-matched distractors blended from
orthographic/semantic/domain/random strategies; optional "None of the above"
at a configurable rate. Feedback timing (immediate vs. end-of-quiz) is a
single admin-controlled global setting (Settings tab). Spaced repetition
(re-surfacing missed words sooner) is opt-in per session.

### Visualizations (`/app/visualizations`)

Every relatedness view in one place: book/author ego-graphs
(`react-force-graph-2d`) from the top-k tables above, the word-embedding
graph, a shared-vocabulary comparison panel (the actual overlapping rare
words between any two books/authors, computed on demand), a cluster
map/similarity matrix/dendrogram over the top-N entities from a clustering
run, a USAS category-overlap graph, and a discipline-category map (PCA over
each entity's category-distribution vector — a different axis from every
shared-vocabulary view above).

### Public deployment

Exposed via a **Cloudflare Tunnel** (DNS-only — no Cloudflare Access; see
DESIGN.md for why that was tried and removed). App-level accounts are the
sole access gate. Runs as two `systemd --user` services (survive
reboot/logout via `loginctl enable-linger`):

- `concordance-web.service` — runs the backend against whatever's already
  built in `webapp/frontend/dist`. Deliberately does **not** rebuild the
  frontend itself (this unit's PATH doesn't include nvm's Node).
- `concordance-tunnel.service` — runs `cloudflared tunnel run <name>` (config
  under `~/.cloudflared/`, not in the repo).

To ship a frontend change:

```bash
cd webapp/frontend && npm run build
systemctl --user restart concordance-web.service
```

Useful commands: `systemctl --user status concordance-web concordance-tunnel`,
`journalctl --user -u concordance-web -u concordance-tunnel -f`.

**If `systemctl --user` fails with "Failed to connect to bus"** on WSL:
WSLg mounts its own tmpfs over `/run/user/1000` during/after boot, hiding the
real per-user runtime dir (the D-Bus socket is still alive, just unreachable
by path). A system-level watcher unit,
`fix-run-user-runtime-dir.service`, polls every 15s and unmounts just the
shadow layer. Machine-specific — not in this repo. Check it's running with
`systemctl status fix-run-user-runtime-dir.service`.

**Restarting the web service while a batch job (`ingest`/`maintain`) is
running** can hang for minutes on a schema-apply lock contending with the
live job's own transaction. Check `ps aux | grep concordance` first; if
something's running, either wait for it or stop it (`SIGTERM`, not `-9`) —
never just kill its DB connection out from under it.

## Corpus housekeeping

Occasional/one-time commands, none part of `maintain`:

```bash
concordance book-merge          # compile split-volume Gutenberg books ("Vol. I"/"Vol. II") into one
concordance incoming-merge      # same, run BEFORE ingestion — scans incoming/, deletes the per-part originals
concordance archive-metadata    # backfill word_count/publication_year/era for books in archive/
concordance book-genres         # tag every book with genre labels (concordance/genre.py's GENRE_LIST)
concordance refresh-rejected-index   # refresh the search index behind the Rejected tab (~15-20s, run on a daily cron)
concordance import-defined      # one-time: import legacy vocab.defined terms as book-less words
```

`book-merge`/`incoming-merge` both take `--dry-run` (report only, zero
writes) — `incoming-merge` is destructive otherwise (deletes the original
per-part files once compiled), so dry-run first. Re-run
`book-similarity`/`book-clustering`/`book-fame` for any book touched by a
merge. `archive-metadata`/`book-genres` are both network-heavy, book-level
passes (Gutenberg RDF lookups paced via `--delay`) — expect hours on a fresh
backlog, `--only-missing` for a cheap re-run after.

## Status

Every stage below is real and running end-to-end, not a stub:

- Core pipeline: extract → judge → enrich, cross-book verdict cache, keep-biased LLM judge (live 14B model, not the no-model fallback).
- Web app: review/prune UI, invite-gated user accounts, quizzing (4 question types, spaced repetition), full visualization suite.
- Enrichment: USAS domain tagging, ex-ante + per-user-calibrated difficulty, quiz-safe definitions, pronunciation audio (real recordings first), unified definition-lookup cascade with a human-review flag for foreign/archaic-spelling hits.
- Scoring: absolute 1–10 cultural-fame score for every book/author.
- Relatedness: lexical (shared-vocabulary) and semantic (embedding) similarity, both for books and authors, surfaced as ego-graphs, cluster maps, similarity matrices, and dendrograms.
- Analogy-relation graph powering a fourth quiz question type.
- A separate OED reference-dictionary pipeline (own schema, own browsing UI), feeding back into the main pipeline as a pronunciation source.

Deferred by choice, not forgotten: other languages, Anki export, scanned-PDF
OCR, a curated names/gazetteer list to close the one known proper-noun-
pollution gap (see DESIGN.md), and a genuinely full-corpus relatedness view —
today's cluster map/matrix/dendrogram are honestly labeled "top N," because
both the clustering math and the rendering itself stop being tractable well
before real corpus scale.

CSV-based ingestion (`run` → hand-edit → `finalize` → `sync-db`) still works
but is no longer the primary workflow — see git history if you're on it.

---

See [DESIGN.md](DESIGN.md) for the reasoning behind every decision above.
