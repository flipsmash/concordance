#!/usr/bin/env bash
# One-off driver for catching up the quiz_definition backlog (~25k words as of
# 2026-07-18). Not part of the CLI -- `concordance quizdef` itself only commits
# once at the very end of a run (all leaker rewrites held in memory until the
# whole batch finishes), so a single unlimited invocation risks losing many
# hours of LLM work to one interruption. This loops it in chunks instead: each
# chunk is its own process, commits at its own end, and only_missing=true
# (quizdef's default) means the next chunk picks up exactly where the last
# one left off. Safe to Ctrl-C and rerun.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

CHUNK=3000
LOG=scripts/quizdef_backlog.log

echo "$(date -Iseconds) starting quizdef backlog run, chunk size $CHUNK" >> "$LOG"

while true; do
  remaining=$(python3 -c "
from concordance import db
conn = db.connect()
cur = conn.cursor()
cur.execute(\"SELECT count(*) FROM concordance.word WHERE quiz_definition IS NULL AND coalesce(definition,'') <> ''\")
print(cur.fetchone()[0])
conn.close()
")
  echo "$(date -Iseconds) remaining=$remaining" >> "$LOG"
  if [ "$remaining" -le 0 ]; then
    echo "$(date -Iseconds) backlog cleared, done" >> "$LOG"
    break
  fi
  concordance quizdef --limit "$CHUNK" >> "$LOG" 2>&1
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "$(date -Iseconds) chunk failed with exit $status, stopping (rerun this script to resume)" >> "$LOG"
    exit "$status"
  fi
done
