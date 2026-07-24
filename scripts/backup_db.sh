#!/usr/bin/env bash
# Daily full-database backup (pg_dump), keeping only the latest 2 -- run via
# cron (see crontab: 0 4 * * *, after refresh-rejected-index's 3 AM slot).
# Plain-SQL dump piped through gzip -1 (fast, not max compression -- at this
# DB's real size, ~36GB -> ~6GB/~5min measured live, and the daily cadence
# means backup #2 is only ever a day old, so squeezing further isn't worth
# the extra CPU/time). Restore: gunzip -c db_backups/concordance_*.sql.gz |
# psql "$DATABASE_URL".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKUP_DIR=db_backups
LOG="$BACKUP_DIR/backup.log"
mkdir -p "$BACKUP_DIR"

# .env isn't sourced by login/cron shells automatically -- same reason the
# CLI's own commands read it via python-dotenv; a plain shell script has to
# load it itself.
if [ -z "${DATABASE_URL:-}" ] && [ -f .env ]; then
  set -a
  source .env
  set +a
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$BACKUP_DIR/concordance_${TIMESTAMP}.sql.gz"

echo "$(date -Iseconds) starting backup -> $OUTFILE" >> "$LOG"

if pg_dump "$DATABASE_URL" | gzip -1 > "$OUTFILE"; then
  echo "$(date -Iseconds) backup complete: $(du -h "$OUTFILE" | cut -f1)" >> "$LOG"
else
  echo "$(date -Iseconds) backup FAILED, removing partial file" >> "$LOG"
  rm -f "$OUTFILE"
  exit 1
fi

# Keep only the latest 2 -- `ls -t` lists newest-first, so everything from
# line 3 onward is the 3rd-newest and older; delete just those.
ls -1t "$BACKUP_DIR"/concordance_*.sql.gz | tail -n +3 | while read -r old; do
  echo "$(date -Iseconds) removing old backup: $old" >> "$LOG"
  rm -f "$old"
done
