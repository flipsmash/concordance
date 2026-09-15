#!/usr/bin/env bash
# Waits for the still-running OED reconciliation pass (Phase 1b) to finish,
# then pulls its results into word.ipa and sweeps the existing 'none' audio
# backlog with the newly-wired Piper tier (Phase 2's actual gap-closing step).
cd /mnt/c/Users/Brian/concordance
source .venv/bin/activate
TS="$1"
RECONCILE_PID="$2"

log="logs/phase2_rollout_${TS}_driver.log"

echo "=== [$(date '+%F %T')] waiting for reconcile PID ${RECONCILE_PID} to finish ===" | tee -a "$log"
while kill -0 "$RECONCILE_PID" 2>/dev/null; do
    sleep 30
done
echo "=== [$(date '+%F %T')] reconcile process finished ===" | tee -a "$log"

run_step() {
    name="$1"; shift
    step_log="logs/${name}_${TS}.log"
    echo "=== [$(date '+%F %T')] START ${name} ===" | tee -a "$log"
    "$@" > "$step_log" 2>&1
    rc=$?
    echo "=== [$(date '+%F %T')] END ${name} (exit ${rc}) ===" | tee -a "$log"
}

run_step ipa-refetch    concordance ipa --refetch
# audio (default only_missing=True, NOT --refetch): only touches words with
# no word_audio row at all -- never re-attempts an already-resolved
# commons/mw/azure word, so a transient miss this run can't regress it back
# to 'none'. audio-guess is the actual gap-closing step: it specifically
# targets the existing source='none' backlog and gives it real Piper audio.
run_step audio          concordance audio
run_step audio-guess    concordance audio-guess

echo "=== [$(date '+%F %T')] PHASE 2 ROLLOUT COMPLETE ===" | tee -a "$log"
