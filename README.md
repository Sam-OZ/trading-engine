# Trading engine (demo accounts only)

Autonomous Forex and crypto trading engine. Forex trades on a **Capital.com Demo**
account; spot crypto on **Binance Spot Demo Mode**. Claude reviews signals and trades
a capped budget of its own (Phase 3). The design, limits and phased plan are in the
Phase 0 proposal.

**Safety by construction**
- `TRADING_MODE` accepts only `demo`, and the code contains only demo trading
  endpoints. Real money needs a code change, never a setting.
- Every position gets a stop loss held **at the broker**. If a stop cannot be
  confirmed, the position is closed.
- Duplicate orders are blocked by a signal-derived client order ID with a unique
  database key, written before anything is sent.
- Keys are never stored in this repo. They live in Google Secret Manager and are
  loaded into the process environment by `scripts/with-secrets.sh`.

## Layout

```
src/tradeengine/
  config.py          infrastructure config (secrets from environment, demo-only guard)
  settings_store.py  admin-editable limits, budgets and toggles (stored in the DB)
  models.py          shared types; starting instruments
  db.py              trade journal schema
  state.py           kill switch, start-of-day and peak equity
  ids.py             deterministic client order IDs
  sizing.py          position sizing and currency conversion
  risk.py            risk manager (all limits, incl. Claude's budget)
  execution.py       idempotent submit, protection enforcement, close
  reconciler.py      journal vs broker, every 30 s and at start-up
  marketdata.py      tick freshness, 1-minute bars, resampling
  history.py         Binance / Capital.com / Dukascopy history files
  brokers/capital.py Capital.com demo REST + WebSocket
  brokers/binance.py Binance Spot Demo Mode (orders) + live public data (signals)
  engine.py          wiring and the trade path
  gate1.py           live demo checks for Gate 1
  cli.py             `tradeengine` command
tests/               unit tests (no network; brokers are mocked)
scripts/             with-secrets.sh, gate1.sh
```

## One-time setup on your Mac

1. Install `uv` (it fetches Python 3.12 for this project; your Mac's own python3 is not used):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Make sure `gcloud` is logged in to the account that owns the `televita-trading` project:

   ```bash
   gcloud auth login
   ```

3. From this folder (already a git repo), install and run the unit tests, and switch on
   the secret scanner (it blocks any commit that contains something that looks like a key):

   ```bash
   cd ~/Downloads/Source_Code/Crypto_Forex
   uv sync --extra dev
   uv run pytest -q
   cp scripts/pre-commit-config.yaml .pre-commit-config.yaml
   uv tool install pre-commit
   pre-commit install
   ```

4. In the Capital.com platform, set the demo account balance to 10,000. (Binance Demo
   Mode keeps its own default balances; the engine trades a virtual 10,000 USDT of it.)

## Gate 1: live demo checks

Needs the demo keys in Secret Manager (see the proposal's "Steps to start").

```bash
cd ~/Downloads/Source_Code/Crypto_Forex
./scripts/gate1.sh
```

It opens one small position on each demo account, checks the broker-side stop and
target, blocks a duplicate, tightens the stop, reconciles, closes, and checks the
kill switch. Forex checks are skipped while the FX market is closed (Friday 17:00
to Sunday 17:00 New York time); run it on a weekday for the full set. The report is
saved to `data/reports/`.

## Everyday commands

```bash
./scripts/with-secrets.sh uv run tradeengine status
./scripts/with-secrets.sh uv run tradeengine run
uv run tradeengine kill "reason"            # blocks new orders
./scripts/with-secrets.sh uv run tradeengine kill "reason" --flatten   # also closes all engine positions
uv run tradeengine unkill
uv run tradeengine settings
uv run tradeengine settings --set risk_per_trade_pct=0.4 away_mode=false
uv run tradeengine download --source dukascopy --symbol EUR/USD --days 365
./scripts/with-secrets.sh uv run tradeengine download --source binance --symbol BTC/USDT --tf M1 --days 30
```

The kill switch also engages if the file `data/KILL` exists.

## Docker

`docker compose up -d` runs Postgres and the engine (used on the cloud VM in Phase 3).
Secrets come from the environment of the `docker compose` command, never from a file in the repo.
