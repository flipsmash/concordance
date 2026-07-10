#!/usr/bin/env bash
# Build the frontend and run the backend as a single process serving both —
# for manually testing the "public" single-origin setup locally. For local
# development with hot reload, use dev.sh instead. The actual public deploy
# (concordance-web.service) intentionally does NOT call this — it serves
# whatever's already in frontend/dist without rebuilding (see README's
# "Public access" section for why, and how to ship a frontend change).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

(cd webapp/frontend && npm run build)

source .venv/bin/activate
uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000 --app-dir .
