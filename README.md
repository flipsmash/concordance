# Concordance

Extract interesting vocabulary from books you read (EPUB, text PDF, `.txt`) using
a **local** LLM — no paid API. Rare words are surfaced, common ones and junk are
filtered, and the result lands straight in Postgres, ready to review in the
[web app](#web-app-webapp).

The design is deliberately **keep-biased**: a genuine rarity should survive to
your review even at the cost of a little noise, never the reverse. See the
requirements & architecture spec for the full rationale.

## Pipeline

```
extract → clean → tokenize → frequency-floor → cross-book verdict cache
        → strip-proper-nouns → validity-gate → LLM-judge → dictionary-lookup
        → Postgres (ingest) → (refill) → (deepen)
```

- **frequency floor** — a stop-word-style cut of common words (never a rarity *ceiling*)
- **cross-book verdict cache** — a lemma already kept/pruned/judge-rejected in an earlier book is pre-marked from `word`/`rejected_word` and never re-judged: the LLM judge's input is purely `(lemma, frequency band)`, so at temp 0 its verdict on a given lemma is always the same. This is what keeps per-book judge time from scaling with corpus size — cost tracks *distinct new rare words*, which saturates fast on a shared-vocabulary corpus.
- **validity gate** — multi-source, keep-biased. A word is a real word if *any* authority vouches for it — the local `vocab.wiktionary` DB dump (~500k terms, checked first because it's free and carries no "Proper noun" POS to get confused by), then the SymSpell 82k wordlist, **WordNet**, or **NLTK's 234k dictionary corpus** (which carries the archaic vocabulary — *destrier, bartizan, cangue* — that trips up single-dictionary checks). A foreign-language-context check runs early too. Only then is misspelling considered, by *relative* near-neighbor frequency (with a recurrence escape hatch — a "misspelling" that keeps showing up is probably a real coinage). NLTK's `wordnet` and `words` data download automatically on first run.
- **LLM judge** — a local model decides what's worth learning (stubbed until you point it at a model). To keep a weak local model honest it emits a *minimal* per-word verdict (`{"w","k"}`, no free-text reason) so it doesn't truncate its output and silently drop words; any word it omits is re-queried for up to three passes before the keep-biased fallback, so junk can't flood the list by omission. A corpus frequency hint (common / uncommon / rare) steadies its rarity sense but is never a hard cut. Frequency alone can't do this job — *tendril* is rarer than *refectory* yet everyone knows it — which is exactly why the judgment is the model's, not the floor's.
- **dictionary lookup** — the local Wiktionary dump first, then free keyless network sources: Free Dictionary API, then Wiktionary online (which actually carries the rare/archaic words). Fills definition, part of speech, IPA, synonyms, and etymology. Bulk lookups retry with exponential backoff and honour `Retry-After`, so a run of a thousand words doesn't get silently emptied by rate limiting. Whatever's still blank after ingest gets a second, slower pass from `refill`/`deepen` (below).
- **review** — prune too-common/easy terms afterward in the [web app](#web-app-webapp) (a soft delete — `word.active = false` — nothing is destroyed)
- nothing is ever silently dropped — every cut is logged to `rejected_word`, one row per (book, lemma)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

Requires a live `DATABASE_URL` (env or a git-ignored `.env`):

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

The validity gate and enrichment both check the local `vocab.wiktionary` dump
(~500k terms) first, before any of the offline/network fallbacks. It's the
cheapest and, since that dump carries no "Proper noun" POS category at all,
the cleanest authority available: unlike SymSpell/WordNet/wordfreq (all
frequency-derived from general web text, so polluted by real names with any
web footprint), membership alone means "this isn't a name."

## Ingest a book (`ingest`)

```bash
concordance ingest "some book.epub"
concordance ingest "some book.epub" --schema concordance
```

Runs the full extract → filter → judge → enrich pipeline and writes straight
into Postgres — no CSV, no hand-edit, no promotion step. Kept words upsert
into `word`/`word_book`; everything the pipeline dropped goes into
**`rejected_word`** — one row per **(book, lemma)**, deliberately *not*
deduped across books the way `word` is, since the same lemma can be rejected
for a different reason (or recurrence count) in a different book. Nothing is
silently lost; you query the DB instead of opening a CSV. Idempotent —
re-running the same book updates both tables in place rather than duplicating
rows, and never clobbers a field (definition, IPA, etymology, ...) that
already has content with a blank value from a re-run.

Review happens afterward in the **[web app](#web-app-webapp)**: its
**Accepted** tab lets you prune too-common/easy terms (a one-click soft
delete — `active = false`); its **Rejected** tab lets you browse what the
pipeline dropped and "Add" one back if it dropped something worth keeping
(flagged `rescued_from_reject` so the rescue stays traceable).

A word already marked pruned (`active = false`) via the web app, or
judge-rejected in an earlier book, is recognized before it ever reaches the
floor/validity gate/judge — see the cross-book verdict cache above — so
review decisions are never silently re-litigated or wasted as repeat LLM
calls.

**Batch mode — process everything in `incoming/`:**

```bash
concordance ingest              # every .epub/.pdf/.txt in incoming/
```

Name files `[Title] -- [Author].epub` (e.g. `Ulysses -- Joyce, James.txt`) and
the title/author populate `book.title`/`book.author` directly — no delimiter
found just uses the whole filename as the title with a blank author, rather
than erroring out. Each file is moved into `archive/` after processing
(`--no-archive` to leave them in place); explicit single-file mode
(`concordance ingest some/book.epub`) still archives next to the source file
rather than to the top-level `archive/`. The judge model, spaCy, and the
validity gate's dictionaries are loaded **once** for the whole batch (not
once per book), so a long batch pays that cost a single time.

Flags: `--min-zipf` (frequency floor; higher = rarer only), `--limit`,
`--no-lookup`, `--model`, `--stub`, `--schema`, `--database-url`,
`--no-archive` (batch mode only).

## Post-ingest maintenance (`maintain`)

Everything below this point — backfilling, enrichment/scoring, pronunciation
prep, and embeddings — runs in one dependency-ordered pass instead of twelve
commands to remember and re-order by hand:

```bash
concordance maintain   # refill -> deepen -> classify -> normalize-pos -> ngram
                        # -> archaic -> difficulty -> quizdef -> quizzable
                        # -> wordnik-pron -> ipa -> embed
```

Every step runs incrementally (only-missing / blank-only / not-yet-embedded),
so a re-run after everything's caught up is fast — it only touches the newest
batch's words. The **first** run against a corpus with a real backlog is not:
`classify` and `quizdef` both load a local LLM and call it per word, so
catching up a few thousand words is likely to take hours. That cost is paid
once. Use `--skip-refill`, `--skip-classify`, `--skip-quizdef`, etc. (one flag
per step) to defer the slow ones to run separately/overnight instead of
blocking on them inline; `--limit` caps words processed per step, useful for
chunking a large backlog into resumable pieces (`compute_quiz_definitions`,
for one, only commits at the very end of a run — an unlimited invocation
against tens of thousands of words risks losing hours of LLM work to a single
interruption; looping `--limit 3000` and relying on only-missing to pick up
where the last chunk left off is the safer shape for a big catch-up).
`load-taxonomy` and `train-fasttext` are deliberately excluded — both are
one-time/occasional setup, not per-batch maintenance — and so are the
Commons/Azure audio steps, since Commons rate-limits hard and is meant to run
for hours unattended on its own (see "Pronunciation audio" below).

**Definitions can change after these have already run** — the same lemma
reappearing in a later book can resolve to a different dictionary sense, and
`sync_book_results`/`sync_master` will overwrite an existing definition with
the new one. Whenever that happens, whatever was computed from the *old* text
and only ever revisited via an only-missing check — `quiz_definition`, USAS
categories, the definition embedding — gets invalidated (cleared) right there
in the same upsert, so the next `maintain` run regenerates it from the
current text instead of silently going stale. `archaic`/`difficulty`/
`quizzable` don't need this: all three always recompute every row
unconditionally, so they self-correct on the next run with no help.

## Backfilling definitions (`refill`, `deepen`)

A word can be *kept* (a real word, worth learning) without ever getting a
definition — `ingest`'s enrichment sources sometimes miss genuinely rare or
archaic vocabulary. Rather than silently sitting blank forever, every such
word is durably marked `word.flagged_undefined` (+ `_at`) the moment it's
accepted with no definition — **sticky by design**: the marker is never
cleared, even once a definition is later found, because the point is a
permanent "this one needed a second look" audit trail for your own manual
validity review, not a live status flag.

```bash
concordance refill              # cheap sources, same ones ingest already tried
concordance deepen              # slower/deeper sources + a validity estimate
concordance deepen --web        # + web-search/LLM last resort (needs a model)
```

- **`refill`** re-tries the local Wiktionary dump and the free online
  dictionaries (Free Dictionary API, Wiktionary) for every word whose
  `definition` is still blank — useful when the miss was transient (a rate
  limit, a network blip) rather than the word genuinely being undefinable.
- **`deepen`** runs after `refill` and reaches further: **Wordnik** (Century
  Dictionary + Webster's, which carry archaic vocabulary — needs a free
  `WORDNIK_API_KEY` in `.env`, falls back to yourdictionary-only without it)
  and **yourdictionary.com**. Whatever *still* can't be defined gets a
  deterministic, explainable **validity estimate** written to `word.validity_label`
  (`likely-valid` / `uncertain` / `likely-artifact`), `validity_score` (0–1),
  `validity_notes`, and `suggested_correction` — signals are Google Books
  Ngram, wordfreq, WordNet/NLTK wordlists, morphology, and a SymSpell
  near-neighbour check, the same scoring used for the CSV-era `<book>.undefined.csv`
  report. **In practice, most currently-flagged words score `likely-artifact`**
  — OCR misreads, archaic-spelling variants no modern dictionary carries as a
  headword, and foreign-language fragments that slipped past the keep-biased
  validity gate on some other authority's say-so. Cross-reference
  `flagged_undefined = true AND validity_label = 'likely-artifact'` for your
  prune review queue.
- Neither command ever overwrites an existing definition — both only touch
  rows where `definition` is still blank.

Both commands accept `--schema`, `--limit`, `--database-url`.

## Enrichment & scoring (`classify`, `archaic`, `ngram`, `difficulty`, `quizdef`, `quizzable`)

A further pass of DB-only commands (no book/model pipeline; each just reads
and updates rows in the schema `ingest` populated), meant to run in this
order after words exist:

```bash
concordance load-taxonomy   # once: load the USAS category tables
concordance classify        # tag every word with 1-3 USAS domain codes
concordance normalize-pos   # fold part_of_speech into one clean vocabulary
concordance ngram           # cache Google Books Ngram rarity/recency per word
concordance archaic         # set current/dated/archaic/obsolete + confidence
concordance difficulty      # 0-100 ex-ante difficulty scalar + factor breakdown
concordance quizdef         # quiz-safe definitions (rewrite ones that leak the word)
concordance quizzable       # flag variant/inferable-derivative words as unquizzable
```

- **`classify`** — assigns each word 1-3 USAS category codes (word + POS +
  definition + sentence), using the WordNet-Domains mapping as a candidate
  hint the model prunes/confirms against context rather than a hard seed.
  `--only-missing` / `--batch` to backfill incrementally.
- **`archaic`** — an ordinal (current < dated < archaic < obsolete) with a
  0-1 confidence: a register label in the definition or the Wiktionary dump is
  high-confidence; a Google-Books recency decline alone is real but noisy
  (can't distinguish "faded" from "always uncommon"), so it's low-confidence
  and queued for later review rather than trusted outright.
  Needs `ngram` to have run first.
- **`ngram`** — fetches + caches peak/recent frequency and recency ratio per
  word from Google Books Ngram; feeds both `archaic` and `difficulty`.
- **`difficulty`** — blends rarity (dominant), archaic confidence, USAS domain
  specificity, and morphological transparency into a single 0-100 scalar,
  storing the factor breakdown alongside it (a principled ex-ante estimate,
  not yet a fitted/IRT model — that comes once quiz response data exists).
- **`quizdef`** — ~37% of definitions leak the target word's root ("audaciously"
  → "in an audacious manner"), making recall quizzing trivial; this builds a
  separate `quiz_definition` per word — passed through as-is if already clean,
  LLM-paraphrased (then machine-verified leak-free) if not, redacted as a last
  resort.
- **`quizzable`** — flags words whose only difference from an already-known
  base form is grammatical (plurals, inflections) or a transparently inferable
  derivative, so quizzing doesn't waste a card on something not actually new.

### Pronunciation audio (`wordnik-pron`, `ipa`, `commons-search`, `commons-download`, `audio`, `audio-guess`)

Real human recordings where they exist, IPA-guided synthesis otherwise —
never a blind spelling-to-speech guess unless nothing else is available:

```bash
concordance wordnik-pron      # fetch raw Wordnik transcriptions (ARPAbet/AHD-5/IPA)
concordance ipa               # backfill+validate word.ipa from kaikki, then Wordnik, then local Wiktionary
concordance commons-search    # find real Commons recordings kaikki's dump missed
concordance commons-download  # download the recordings commons-search confirmed
concordance audio             # Commons recording if present, else Azure IPA-guided TTS
concordance audio-guess       # last resort: Azure guesses from spelling alone
```

`wordnik-pron` and `ipa` are both part of the `maintain` chain above — rerun
`maintain` (or just `ipa`) before `audio`, since synthesis quality depends on
the transcription it's given. `wordnik-pron` is rate-limited (~1 word/several
seconds on the free tier) and `ipa`'s primary source is a 2.7GB dump scan, so
both stay batch passes rather than per-word ingest-time lookups.
`commons-search`/`commons-download`/`audio` are deliberately separate
commands rather than folded into `maintain`: Commons rate-limits hard and is
meant to run for hours unattended, which would starve every other step if
interleaved. `audio-guess` results are tagged `source='azure_guess'` (vs.
`'azure'` for IPA-guided) so the app can flag them as unverified.

## Semantic distance (`train-fasttext`, `embed`)

Two independent per-word vectors, for visualizing word relationships and — a
later feature this doesn't build — generating quiz distractors. Neither is a
precomputed all-pairs distance matrix (already 200M+ pairs at 20k+ words, and
only growing); both are queried on demand via a pgvector HNSW index, so
finding a word's nearest neighbors is O(log N), not O(N²).

```bash
concordance train-fasttext          # once (or after a big ingest batch): train on archive/
concordance embed                   # definition_vector for words missing one
concordance embed --signal fasttext # fasttext_vector instead (needs the trained model)
concordance embed --signal both     # both in one pass
```

- **Definition embedding** (`sentence-transformers`, `BAAI/bge-small-en-v1.5`)
  embeds each word's dictionary *gloss* — modern English text — rather than
  the rare headword itself, falling back to `synonyms` then the book
  example `sentence` when a definition is missing. This is what makes rare/
  archaic vocabulary tractable at all: the word is rare, its definition
  usually isn't. In practice this reaches ~100% coverage (the `sentence`
  fallback catches almost everything a real definition doesn't).
- **FastText subword vectors** are trained from scratch on this project's own
  `archive/` corpus via `train-fasttext` (not a generic pretrained binary),
  so its character n-grams are learned from the same archaic/literary
  English these words come from. Because it composes a vector from
  subwords, it produces one for *every* lemma regardless of definition
  coverage — genuinely 100%, including words no dictionary could define.
- These capture different things on purpose — meaning vs. spelling — and are
  stored/queried independently (`word_embedding.definition_vector` /
  `.fasttext_vector`), not fused into one score. `train-fasttext` is a
  holistic pass (must see the whole corpus at once — rerun periodically as
  the archive grows); `embed` is the familiar incremental maintenance pass.

The review webapp's backend exposes this as `/api/words/search` (word
picker) and `/api/words/{id}/neighbors` (`signal=definition|fasttext`, with
optional POS/quizzable/difficulty-band/USAS-domain filters and synonym
exclusion) — query infrastructure for a future visualization UI and future
distractor generation, not those features themselves.

## Running the local model (RTX 3060, 12 GB)

The judge talks to `llama.cpp` through the `llama-cpp-python` bindings — no
separate server.

**1. Install the bindings with CUDA.** Easiest is a prebuilt CUDA wheel:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

If no wheel matches your setup, build it (needs the CUDA toolkit on your WSL/Linux):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

**2. Get a model.** A Qwen2.5-14B-Instruct GGUF at Q4_K_M is ~9 GB and fits your
12 GB VRAM with context to spare:

```bash
pip install huggingface-hub
huggingface-cli download bartowski/Qwen2.5-14B-Instruct-GGUF \
  Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir models
```

Want it snappier for big books? Swap in `Qwen2.5-7B-Instruct` or
`Llama-3.1-8B-Instruct` at Q5_K_M.

**3. Run it.**

```bash
concordance ingest "some book.epub" --model models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

The judge defaults to the 14B; if that file is missing it falls back to the
stub judge automatically (which keeps every survivor, letting the pipeline
run end-to-end without a model), and `--stub` forces the stub even when the
model is present. `Config.n_gpu_layers = -1` offloads as many layers to the
GPU as the VRAM allows; drop it if you hit out-of-memory.

## Web app (`webapp/`)

The first slice of a browser-based quizzing/viz/user-management app: a review
table for pruning too-common/easy terms out of the vocab bank. Deleting a term
doesn't hard-delete it — it flips `word.active` to false, so every downstream
feature (quizzing, stats) just needs to filter on `active = true`, and history
(audio, ngram data, etc.) stays intact.

- **Accepted tab** — term/POS/definition/difficulty table, filterable by POS,
  one-click delete (no confirm) that sets `active = false`; whole-row hover
  highlight so a delete click can't land on the wrong term.
- **Rejected tab** — browse `rejected_word`, filterable by book and by reason
  (both multi-select), with an "Add" button that rescues a word back in (live
  dictionary lookup, since rejects were never enriched) and flags it
  `rescued_from_reject` for after-the-fact tracking.

```bash
# first time only
pip install -e ".[web]"
(cd webapp/frontend && npm install)

# every time — runs backend + frontend together, http://localhost:5173
./webapp/dev.sh
```

### User accounts (`create-admin`, invites, `/app`)

The curation UI above (Accepted/Rejected/Graph) is admin-only, enforced by
the app itself (see below) rather than an edge gate. Separately, the app has
its own login — independent accounts for browsing/studying the vocab bank,
gated by one-time invite links rather than open signup:

- `concordance create-admin <username>` seeds the first admin-flagged
  account (prompts for a password via `getpass`; run once per deployment —
  needs to exist and be confirmed working before anything else touches
  who's allowed in, so app-layer auth is never the only thing standing
  between a restart and a lockout).
- A logged-in admin gets a **"+ Generate Invite Link"** button in the
  Accepted tab, which mints a one-time `/register?token=...` link
  (`POST /api/admin/invites`, default 7-day expiry). Whoever opens it sets a
  username/password and lands on `/app` — a non-admin browse page (word
  search → full word detail, including its similarity graph) with no access
  to the curation API.
- Sessions are an httpOnly, Secure cookie (`concordance_session`, 30-day
  expiry) backed by a `sessions` table — not JWTs — so a session can be
  revoked server-side (`/api/auth/logout`) rather than just expiring.
  Passwords are hashed with Argon2 (`argon2-cffi`).
- Every route is one of `require_admin` (curation API) or `require_viewer`
  (word search/detail/audio/graph) — this is the sole, load-bearing gate;
  see "Public access" below for why an edge layer (Cloudflare Access) turned
  out to be structurally incompatible with a fetch()-driven SPA and got
  dropped rather than layered on top. `auth.py` still has a
  `verify_cf_access` JWT-verification helper and both dependencies still
  call it, but with no Access application configured it's simply dead code
  — harmless, since it fails closed (returns `None`) when unconfigured.

`dev.sh` also sets `WATCHFILES_FORCE_POLLING=true` for uvicorn (Vite's own
polling is configured in `frontend/vite.config.js`). Both are needed because
this repo lives on `/mnt/c` — a Windows drive mounted into WSL — where native
fs-change notifications don't reliably reach either dev server's watcher, so
edits silently fail to hot-reload without polling.

### Public access — `vocab.brfinnegan.org`

The app is exposed to that domain via a **Cloudflare Tunnel** running on this
WSL machine (no port-forwarding/firewall changes) — DNS-only, no Cloudflare
Access application in front of it. Access was tried and removed: it gates by
redirecting unauthenticated requests to a `cloudflareaccess.com` login page,
which only works for a full page navigation. Every API call this SPA makes
is a background `fetch()`, and a fetch that gets redirected cross-origin
fails the browser's CORS check outright (a bare "Failed to fetch", no status
code) — so Access couldn't gate a single `/api/*` route without breaking the
page that calls it, and it can't gate `/register` at all without defeating
the entire point of invite links. There was nothing left for it to
usefully protect.

[User accounts](#user-accounts-create-admin-invites-app) are the actual,
sole gate now — `require_admin`/`require_viewer` in
`webapp/backend/main.py`, fail-closed, verified end-to-end by
`tests/test_auth.py`'s HTTP round-trip test. Removing Access doesn't expose
anything Access was reliably protecting before this: same test suite covered
the app-layer boundary the whole time, Access was always documented as
"redundant, not load-bearing" on top of it (see `webapp/backend/auth.py`).

Setup order still matters on any fresh deployment: the admin account
(`concordance create-admin`) has to exist and be confirmed working before
anything else changes, since `require_admin` fails closed — no admin row
means no one, including Brian, can reach the curation API at all.

Both pieces run as **systemd --user services** (survive reboot/logout via
`loginctl enable-linger brian`, already enabled):

- `concordance-web.service` — runs the backend directly against whatever's
  already built in `webapp/frontend/dist`. It deliberately does **not**
  rebuild the frontend itself — this unit's PATH doesn't include nvm's Node
  (only an interactive shell profile sets that up), so a build attempted here
  silently uses the system's older Node and breaks.
- `concordance-tunnel.service` — runs `cloudflared tunnel run concordance-vocab`
  (config at `~/.cloudflared/config.yml`, tunnel id in that file, credentials
  JSON alongside it — none of this lives in the repo).

To ship a frontend change to the public site: rebuild, then bounce the
service so it picks up the new `dist/`:

```bash
cd webapp/frontend && npm run build
systemctl --user restart concordance-web.service
```

Useful commands: `systemctl --user status concordance-web concordance-tunnel`,
`journalctl --user -u concordance-web -u concordance-tunnel -f`.

**If `systemctl --user` fails with "Failed to connect to bus":** WSLg mounts
its own tmpfs (for `wayland-0`/`pulse` sharing) directly on top of
`/run/user/1000` at some point during/after boot, hiding the real per-user
runtime dir that already has the D-Bus session socket bound into it — the
socket is still alive at the kernel level, just unreachable by path. A
system-level watcher unit, `fix-run-user-runtime-dir.service` (installed at
`/etc/systemd/system/`, script at `/usr/local/sbin/fix-run-user-runtime-dir.sh`,
enabled for boot), polls every 15s and unmounts just the `/run/user/1000`
shadow layer whenever it appears — `/mnt/wslg/run/user/1000` is left alone so
WSLg itself keeps working. Neither file lives in this repo (machine-specific,
like the tunnel config). Check it's running with `systemctl status
fix-run-user-runtime-dir.service`.

## Status

Every stage is real and runs end-to-end; the LLM judge is live, running the
14B model against every ingest (not the no-model stub — see the fallback
note above). Beyond the base extract → judge → enrich pipeline:
a cross-book verdict cache, a public review webapp, USAS domain tagging, an
ex-ante difficulty scalar, quiz-safe definitions + a quizzable flag, a
pronunciation-audio pipeline (real recordings first, IPA-guided synthesis
otherwise), a two-stage definition backfill (`refill`/`deepen`, the latter
also scoring the genuinely-undefinable tail for validity), and semantic-
distance vectors (definition-embedding + corpus-trained FastText, queried
via a pgvector HNSW index — infrastructure for future visualization and
quiz-distractor generation, not those features themselves yet) are all in
place. Deferred by choice: other languages, Anki export, scanned-PDF OCR, a
curated names/gazetteer list to close the one known gap in proper-noun
filtering (every validity authority is itself somewhat name-polluted).

CSV-based ingestion (`run` → hand-edit → `finalize` → `sync-db`) still works
but is no longer the primary workflow — `ingest` writing straight to Postgres,
reviewed in the web app, is. The CSV commands (`run`, `finalize`, `sync-db`,
`define`, and the standalone `python -m concordance.refill`) remain in the
codebase for anyone with an existing CSV-based project, but aren't documented
here; see git history for their usage if needed.
