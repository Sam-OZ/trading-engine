#!/usr/bin/env bash
# Round 3 (crypto, pre-registered 12 Nov 2026): the DEVELOPMENT run only, on the Binance history
# already on disk (no keys, no download needed). The holdout is a separate, deliberate command.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Installing dependencies (uv) =="
uv sync --extra dev

echo "== Unit tests =="
uv run pytest -q

echo "== Round 3 development run (check 1) =="
uv run tradeengine round3 --dev
