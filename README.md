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
  indicators.py      EMA, RSI, ATR, ADX, MACD, Bollinger, Donchian (match TA-Lib)
  strategies.py      round 1: trend pullback, mean reversion, volatility breakout on hourly bars
  trend.py           round 2: daily channel breakout, daily time-series momentum, 4-hour breakout with daily filter
  backtest.py        fill model with costs, metrics, round 1 walk-forward, Gate 2 report
  portfolio.py       portfolio backtest: all instruments together through the live risk limits
  round2.py          research-round machinery (plan, walk-forward, frozen basket, one-look lock) + round 2's plan
  round3.py          round 3's plan (crypto only)
  runner.py          hourly live runner: signals + trail/exits on each strategy's own bars (off until gate2-apply)
  gate1.py           live demo checks for Gate 1
  cli.py             `tradeengine` command
tests/               unit tests (no network; brokers are mocked)
scripts/             with-secrets.sh, gate1.sh, phase2.sh, round2.sh, round3.sh
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

4. In the Capital.com platform, set the demo account balance (11,000 AUD in use since 12 Nov 2026). (Binance Demo
   Mode starts at 5,000 USDT and cannot be changed; the engine trades a virtual balance set by
   `binance_virtual_capital`, default 5,000.)

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

## Phase 2: backtest and Gate 2 (round 1, hourly strategies)

First run 12 Nov 2026: no strategy passed, so all stay off. Kept for reference; round 2 below replaces it.

```bash
cd ~/Downloads/Source_Code/Crypto_Forex
./scripts/phase2.sh
```

Downloads about 3 years of hourly bars (Binance for crypto; Capital.com and Dukascopy for
Forex, keeping whichever covers more), then runs every strategy through a walk-forward
test: parameters are picked on 12 months, judged on the next 3 months they never saw,
rolled forward 3 months at a time. Costs (spread, slippage, commission, overnight funding)
are charged on every trade. Gate 2 judges only those unseen months: profit factor >= 1.2,
max drawdown < 10%, >= 200 trades. The report goes to `data/reports/gate2-<time>/`
(`report.md`, `result.json`, `oos_trades.csv`).

Strategies stay OFF until you switch on the ones that passed:

```bash
uv run tradeengine gate2-apply data/reports/gate2-<time>/result.json --dry-run
uv run tradeengine gate2-apply data/reports/gate2-<time>/result.json
```

From then on `tradeengine run` trades them on the demo accounts once an hour, with the
same code the backtest used (signal at the hour close, entry at the next open, broker-side
stop, breakeven at 1R, ATR trail, time exit). Thresholds, costs and walk-forward lengths
are admin settings (`gate2_*`, `cost_model`, `walk_forward_*`).

## Round 2: slower trend-following, pre-registered (12 Nov 2026)

The plan (hypotheses, grids, instruments, costs, pass criteria) was written and approved before any
round 2 result existed. The constants in `src/tradeengine/round2.py` are that plan: do not change them
after seeing a result.

```bash
cd ~/Downloads/Source_Code/Crypto_Forex
./scripts/round2.sh
```

1. `tradeengine instruments` checks every instrument at the broker (Capital.com code, minimum size,
   price step, and the live spread against the backtest's cost). An instrument Capital.com does not list
   is left out, with candidate codes printed. A spread wider than the backtest cost means raising
   `cost_model` before the holdout (costs may only go up).
2. `download-all --from 2013-01-01` fetches hourly history: Dukascopy for Forex and gold (monthly files,
   checked against the hours already on disk), Binance for crypto, Capital.com's last 30 days.
3. `round2 --dev` is the development run (check 1). Data is cut at 1 Nov 2023 before anything is computed.
   Per instrument, walk-forward picks parameters (36/6 months daily, 24/6 months 4-hour) from trades that
   entered and exited inside training, or sits the window out if the best training profit factor is below 1.0.
   Then every instrument is replayed together through the live limits. Pass: PF >= 1.2, worst-account
   drawdown < 10%, >= 200 trades. The holdout basket is frozen here (>= 30 trades and PF >= 1.0).

The holdout is one deliberate run, after reading the development report:

```bash
uv run tradeengine round2 --holdout data/reports/round2-dev-<time>/result.json
```

It runs only strategies that passed development, only on their frozen basket, 1 Nov 2023 to
31 Oct 2026, and asks you to type HOLDOUT. Each strategy's run is recorded with a fingerprint of the
code and settings before the result is shown; a second run is refused, and so is a development result
made with different code or settings. Pass: PF >= 1.2, drawdown < 10%, >= 100 trades. A pass is switched
on for demo with `gate2-apply` on the holdout's `result.json`.

## Round 3: crypto only, pre-registered (12 Nov 2026)

Round 2 found nothing on the 13-instrument basket (all three strategies hit the 10% drawdown stop).
Round 3 tests two long-only rules on Bitcoin and Ethereum spot (no overnight funding): round 2's
4-hour breakout with daily filter, unchanged, and a fast daily momentum rule (hold while the 14/28/56-day
return is positive). Constants in `src/tradeengine/round3.py`; the machinery is round 2's.

```bash
cd ~/Downloads/Source_Code/Crypto_Forex
./scripts/round3.sh
```

Development 2018 – Oct 2023, walk-forward 24/6 months. Check 1: PF >= 1.2, drawdown < 10%,
>= 60 trades, and a statistical check (the 5th percentile of 10,000 seeded bootstrap averages of trade
R must be above zero). The holdout (Nov 2023 – Oct 2026, >= 40 trades, same checks) is one run per
strategy, ever, with its own lock:

```bash
uv run tradeengine round3 --holdout data/reports/round3-dev-<time>/result.json
```

If nothing passes, strategy research stops: evidence after round 3 comes only from demo forward tests.

## Everyday commands

```bash
./scripts/with-secrets.sh uv run tradeengine status
./scripts/with-secrets.sh uv run tradeengine run
uv run tradeengine kill "reason"            # blocks new orders
./scripts/with-secrets.sh uv run tradeengine kill "reason" --flatten   # also closes all engine positions
uv run tradeengine unkill
uv run tradeengine settings
uv run tradeengine settings --set risk_per_trade_pct=0.4 away_mode=false
./scripts/with-secrets.sh uv run tradeengine instruments
./scripts/with-secrets.sh uv run tradeengine download-all --from 2013-01-01
uv run tradeengine round2 --dev
```

The kill switch also engages if the file `data/KILL` exists.

## Docker

`docker compose up -d` runs Postgres and the engine (used on the cloud VM in Phase 3).
Secrets come from the environment of the `docker compose` command, never from a file in the repo.
