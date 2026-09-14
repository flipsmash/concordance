#!/usr/bin/env bash
# One-off driver for the full post-mw-backfill maintenance chain, run
# unattended over however long it takes. Each step logs to its own
# timestamped file; failures don't abort later steps (they're independent
# enough that one failing shouldn't cost the rest of the run).
cd /mnt/c/Users/Brian/concordance
source .venv/bin/activate
TS="$1"

run_step() {
    name="$1"; shift
    log="logs/${name}_${TS}.log"
    echo "=== [$(date '+%F %T')] START ${name} ===" | tee -a "logs/chain_${TS}_driver.log"
    "$@" > "$log" 2>&1
    rc=$?
    echo "=== [$(date '+%F %T')] END ${name} (exit ${rc}) ===" | tee -a "logs/chain_${TS}_driver.log"
}

run_step relate                concordance relate
run_step book-genres            concordance book-genres
run_step author-fame            concordance author-fame
run_step book-fame              concordance book-fame
run_step oed-ipa                concordance oed-ipa
run_step commons-search         concordance commons-search
run_step commons-download       concordance commons-download
run_step audio                  concordance audio
run_step audio-guess            concordance audio-guess
run_step oed-concordance-match  concordance oed-concordance-match
run_step train-fasttext         concordance train-fasttext
run_step archive-metadata       concordance archive-metadata
run_step author-stats           concordance author-stats
run_step book-stats              concordance book-stats

echo "=== [$(date '+%F %T')] CHAIN COMPLETE ===" | tee -a "logs/chain_${TS}_driver.log"
