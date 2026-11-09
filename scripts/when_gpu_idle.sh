#!/usr/bin/env bash
# Run `concordance` commands one after another once nothing else is using the
# GPU: no upload or respelling job running (their steps take the GPU lock one
# at a time, so merely waiting for the lock could slip in between two of
# their steps), no ingest or maintain. Each command then takes the GPU lock
# itself. Usage: scripts/when_gpu_idle.sh "oed-pronounce" "oed-ipa"
cd /mnt/c/Users/Brian/concordance
source .venv/bin/activate
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

busy() {
    systemctl --user list-units 'concordance-process-uploads-*' 'concordance-refresh-words-*' \
        --state=active --no-legend 2>/dev/null | grep -q . && return 0
    pgrep -f "concordance (ingest|maintain|process-uploads|refresh-words)( |$)" >/dev/null
}

while busy; do sleep 300; done
for cmd in "$@"; do
    echo "=== [$(date '+%F %T')] START $cmd ===" | tee -a "logs/gpu_idle_${TS}_driver.log"
    # shellcheck disable=SC2086
    concordance $cmd > "logs/${cmd%% *}_${TS}.log" 2>&1
    echo "=== [$(date '+%F %T')] END $cmd (exit $?) ===" | tee -a "logs/gpu_idle_${TS}_driver.log"
done
