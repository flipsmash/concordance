#!/usr/bin/env bash
# Run the review UI's backend + frontend dev servers together.
#
# WATCHFILES_FORCE_POLLING and Vite's usePolling (set in frontend/vite.config.js)
# both exist because this repo lives on /mnt/c (a Windows drive mounted into
# WSL) — native fs-change notifications don't reliably reach either watcher
# there, so edits silently fail to hot-reload without polling.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source .venv/bin/activate

trap 'kill 0' EXIT

WATCHFILES_FORCE_POLLING=true \
  uvicorn webapp.backend.main:app --reload --port 8000 --app-dir . &

(cd webapp/frontend && npm run dev) &

wait
