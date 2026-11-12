#!/usr/bin/env bash
# Gate 1: unit tests, then the live demo checks against Capital.com and Binance.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Installing dependencies (uv) =="
uv sync --extra dev

echo "== Unit tests =="
uv run pytest -q

echo "== Live demo checks =="
# Separate journal so test trades never mix with the engine's own records.
DATABASE_URL="sqlite+aiosqlite:///./data/gate1.db" ./scripts/with-secrets.sh uv run tradeengine gate1 "$@"
