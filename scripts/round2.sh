#!/usr/bin/env bash
# Round 2 (pre-registered 12 Nov 2026): history from 2013, broker check, then the DEVELOPMENT run only.
# The holdout is a separate, deliberate command (see README "Round 2"); it can be run once per strategy.
# Keys are read from Secret Manager once, for the broker check and the download; the backtest needs none.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Installing dependencies (uv) =="
uv sync --extra dev

echo "== Unit tests =="
uv run pytest -q

echo "== Broker check (codes, sizes, live spread vs backtest cost) + history download from 2013 =="
./scripts/with-secrets.sh bash -c 'uv run tradeengine instruments && uv run tradeengine download-all --from "${FROM:-2013-01-01}"'

echo "== Round 2 development run (check 1) =="
uv run tradeengine round2 --dev
