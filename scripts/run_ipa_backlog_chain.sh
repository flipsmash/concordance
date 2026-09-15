#!/usr/bin/env bash
# One-off driver for catching up the wordnik-pron/ipa backlog (~13,270 words
# that predate wordnik_checked_at/ipa_checked_at entirely -- see Phase 0 of
# the audio-pipeline rework). Same shape as run_maintenance_chain.sh, which
# notably does NOT include these two steps -- part of why this backlog
# built up unnoticed. Each step logs to its own timestamped file.
cd /mnt/c/Users/Brian/concordance
source .venv/bin/activate
TS="$1"

run_step() {
    name="$1"; shift
    log="logs/${name}_${TS}.log"
    echo "=== [$(date '+%F %T')] START ${name} ===" | tee -a "logs/ipa_backlog_${TS}_driver.log"
    "$@" > "$log" 2>&1
    rc=$?
    echo "=== [$(date '+%F %T')] END ${name} (exit ${rc}) ===" | tee -a "logs/ipa_backlog_${TS}_driver.log"
}

run_step wordnik-pron  concordance wordnik-pron
run_step ipa           concordance ipa

echo "=== [$(date '+%F %T')] IPA BACKLOG CHAIN COMPLETE ===" | tee -a "logs/ipa_backlog_${TS}_driver.log"
