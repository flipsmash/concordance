# Concordance

Extract interesting vocabulary from books you read (EPUB, text PDF, `.txt`) using
a **local** LLM — no paid API. Rare words are surfaced, common ones and junk are
filtered, and you get a CSV of the words worth keeping.

The design is deliberately **keep-biased**: a genuine rarity should survive to
your review even at the cost of a little noise, never the reverse. See the
requirements & architecture spec for the full rationale.

## Pipeline

```
extract → clean → tokenize → frequency-floor → strip-proper-nouns
        → validity-gate → LLM-judge → dictionary-lookup → CSV → (define) → (finalize)
```

- **frequency floor** — a stop-word-style cut of common words (never a rarity *ceiling*)
- **validity gate** — multi-source, keep-biased. A word is a real word if *any* authority vouches for it — the SymSpell 82k wordlist, **WordNet**, or **NLTK's 234k dictionary corpus** (which carries the archaic vocabulary — *destrier, bartizan, cangue* — that trips up single-dictionary checks). Only then is misspelling considered, by *relative* near-neighbor frequency. NLTK's `wordnet` and `words` data download automatically on first run.
- **LLM judge** — a local model decides what's worth learning (stubbed until you point it at a model). To keep a weak local model honest it emits a *minimal* per-word verdict (`{"w","k"}`, no free-text reason) so it doesn't truncate its output and silently drop words; any word it omits is re-queried for up to three passes before the keep-biased fallback, so junk can't flood the list by omission. A corpus frequency hint (common / uncommon / rare) steadies its rarity sense but is never a hard cut. Frequency alone can't do this job — *tendril* is rarer than *refectory* yet everyone knows it — which is exactly why the judgment is the model's, not the floor's.
- **dictionary lookup** — free, keyless sources: Free Dictionary API first, then Wiktionary (which actually carries the rare/archaic words). Fills definition, part of speech, IPA, synonyms, and etymology. Bulk lookups retry with exponential backoff and honour `Retry-After`, so a run of a thousand words doesn't get silently emptied by rate limiting.
- **review** — you mark each word known / unknown; only unknowns are saved
- nothing is ever silently dropped — every cut is logged to `*.rejected.csv`

### Backfilling definitions

If a run's definitions came back sparse (an old build, or the lookup host was
throttling), refill them without redoing the expensive judge pass — it only
touches rows whose `definition` is still blank and is safe to rerun:

```bash
python -m concordance.refill "book.vocab.csv"
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

## Use

```bash
# default: judges with the 14B at models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
concordance run "some book.epub"

# cap the shortlist size
concordance run "some book.epub" --limit 200

# point at a different model (e.g. the faster 7B for a big book)
concordance run "some book.epub" --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# no model at all — stub judge keeps everything past the validity gate
concordance run samples/passage.txt --stub
```

The judge defaults to the 14B (see setup below); if that file is missing it
falls back to the stub judge automatically, and `--stub` forces the stub even
when the model is present. On the RTX 3060 the 14B is both sharper and more
*consistent* than the 7B at deciding everyday-vs-obscure, which is what keeps
common words off the list.

Outputs land next to the book: `book.vocab.csv` and `book.rejected.csv`.

Flags: `--min-zipf` (frequency floor; higher = rarer only), `--limit`,
`--no-lookup`, `--model`, `--stub`.

### Resolving undefined words (`define`)

The archaic / nonce tail (e.g. Shakespeare's *ungenitured*, *scrimer*) is often
absent from the free dictionaries. `define` reaches further, touching only rows
still missing a definition:

```bash
concordance define "some book.vocab.csv"
```

1. It looks each undefined word up in **Wordnik** (Century Dictionary + Webster's,
   which carry archaic vocabulary) and **yourdictionary.com**, and writes any
   definition it finds back into the CSV.
2. Whatever still can't be defined gets a **validity estimate** written to a
   sibling `<book>.undefined.csv`: a 0–1 score, a label (`likely-valid` /
   `uncertain` / `likely-artifact`), explanatory notes, and a suggested
   correction — so you can tell a real rare word (*cobloaf*, *overscutched*) from
   an OCR/old-spelling artifact (*bareheade* → bareheaded) or nonsense. Signals
   are deterministic and explainable: Google Books Ngram, wordfreq, WordNet/NLTK
   wordlists, morphology, and a SymSpell near-neighbour check.

Wordnik needs a free API key. Put it in a git-ignored `.env` at the project root:

```
WORDNIK_API_KEY=your_key_here
```

(or export `WORDNIK_API_KEY`). Without it, `define` uses yourdictionary only.

### Review by editing → master list → archive

Review is just editing the CSV. `run` writes the whole candidate list to
`<book>.vocab.csv` (and immediately drops a pristine copy at
`archive/<book>.vocab.original.csv`, so the untouched list is preserved before
you touch it). Open the working copy and **delete the rows** for words you
already know or that are false positives — whatever survives is approved. Then:

```bash
concordance finalize "some book.vocab.csv"   # add -y to skip the confirm
```

It shows the surviving count for a one-line `y/N` confirm, then:

- appends every surviving term to **`master_vocab.csv`** at the project root,
  carrying its definition/POS/IPA/etymology plus **`date_added`** and
  **`source_book`**. The master keeps **one row per word** — if a word you already
  banked turns up in a later book, that book is added to its `source_book` cell
  rather than duplicating the row.
- moves the per-book files (your cleaned `.vocab.csv`, `.rejected.csv`, and the
  source `.epub`/`.pdf`) into **`archive/`** — which now holds both the original
  and your cleaned version — leaving the working directory to just the books
  still in flight.

### Sync the master list to PostgreSQL (`sync-db`)

The CSVs stay the working format, but the cross-book `master_vocab.csv` can be
mirrored into Postgres for a future web app:

```bash
concordance sync-db                      # loads ./master_vocab.csv
concordance sync-db --schema concordance # tables live in their own schema
```

Set the connection in a git-ignored `.env` (or the environment):

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

It creates a small normalised schema and upserts idempotently (re-run any time):

- **`word`** — one row per lemma (definition, POS, IPA, sentence, etymology,
  `synonyms text[]`, `first_added`, source), unique on `lower(lemma)`.
- **`book`** — the source books.
- **`word_book`** — the many-to-many that unpacks the CSV's `source_book` list, so
  a word surfaced by two books is linked to both.

Tables live in a dedicated schema (default `concordance`) so they can share a
database with other projects. `pg_trgm` is enabled when privileges allow, giving a
trigram index on `lemma` for future fuzzy lookups.

### Ingest straight to the database (`ingest`)

`run` → hand-edit → `finalize` → `sync-db` is the CSV-based path above. Now that
the review [webapp](#web-app-webapp) handles pruning after the fact
(marking a term `active = false` instead of deleting a CSV row before
promotion), a book can skip the CSV step entirely:

```bash
concordance ingest "some book.epub"                 # same pipeline as `run`
concordance ingest "some book.epub" --schema concordance
```

Same extract → filter → judge → enrich pipeline as `run` (same `--model`,
`--stub`, `--min-zipf`, `--limit`, `--no-lookup` flags), but the result goes
straight into Postgres instead of a CSV: kept words upsert into
`word`/`word_book` exactly like `sync-db` does, and everything the pipeline
dropped goes into **`rejected_word`** — one row per **(book, lemma)**,
deliberately *not* deduped across books the way `word` is, since the same
lemma can be rejected for a different reason (or recurrence count) in a
different book. Nothing is silently lost; you just query the DB instead of
opening `<book>.rejected.csv`. The source book file is moved into `archive/`
on success (`--no-archive` to leave it in place); `--database-url` overrides
`DATABASE_URL`/`.env` same as `sync-db`. Idempotent — re-running the same
book updates both tables in place rather than duplicating rows.

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
concordance run "some book.epub" --model models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

`Config.n_gpu_layers = -1` offloads as many layers to the GPU as the VRAM
allows; drop it if you hit out-of-memory.

## Web app (`webapp/`)

The first slice of a browser-based quizzing/viz/user-management app: a review
table for pruning too-common/easy terms out of the vocab bank. Deleting a term
doesn't hard-delete it — it flips `word.active` to false, so every downstream
feature (quizzing, stats) just needs to filter on `active = true`, and history
(audio, ngram data, etc.) stays intact.

```bash
# first time only
pip install -e ".[web]"
(cd webapp/frontend && npm install)

# every time — runs backend + frontend together, http://localhost:5173
./webapp/dev.sh
```

`dev.sh` also sets `WATCHFILES_FORCE_POLLING=true` for uvicorn (Vite's own
polling is configured in `frontend/vite.config.js`). Both are needed because
this repo lives on `/mnt/c` — a Windows drive mounted into WSL — where native
fs-change notifications don't reliably reach either dev server's watcher, so
edits silently fail to hot-reload without polling.

### Public access — `vocab.brfinnegan.org`

The app is exposed to that domain via a **Cloudflare Tunnel** running on this
WSL machine (no port-forwarding/firewall changes) and gated by **Cloudflare
Access** (Zero Trust → Access → Applications → "Vocab Review", policy allows
only `brfinnegan@gmail.com`) — the API has no app-level auth of its own yet,
so Access is the only thing standing between the internet and the delete
button until real user accounts exist.

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

## Status

Walking skeleton — every stage is real and runs end-to-end; the LLM judge is
wired but stubbed until you supply a `.gguf`. Deferred by choice: cross-book
memory, other languages, Anki export, scanned-PDF OCR.
