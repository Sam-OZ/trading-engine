#!/usr/bin/env bash
# Phase 2: ~3 years of hourly history, then the walk-forward backtest and the Gate 2 verdict.
# Keys are read from Secret Manager once, for the download only; the backtest needs none.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Installing dependencies (uv) =="
uv sync --extra dev

echo "== Unit tests =="
uv run pytest -q

echo "== History download (Binance + Capital.com + Dukascopy) =="
./scripts/with-secrets.sh uv run tradeengine download-all --days "${DAYS:-1100}"

echo "== Walk-forward backtest + Gate 2 =="
uv run tradeengine gate2
