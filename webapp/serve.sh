#!/usr/bin/env bash
# Build the frontend and run the backend as a single process serving both —
# this is what should sit behind the Cloudflare Tunnel (one port, one origin,
# no dev-server websockets exposed publicly). For local development with hot
# reload, use dev.sh instead.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

(cd webapp/frontend && npm run build)

source .venv/bin/activate
uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 --app-dir .
