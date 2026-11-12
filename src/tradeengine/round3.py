"""Round 3: crypto only, pre-registered and approved 12 Nov 2026 19:31 Perth (plan: Claude Docs
"Trading engine - Round 3 crypto research plan (pre-registered)"). The constants here ARE the
pre-registration; the machinery is round2.py's, driven by this Plan.

- Bitcoin and Ethereum spot on Binance, long-only, no overnight funding.
- C1 breakout_h4_dfilter: round 2's 4-hour rule, unchanged. C2 tsmom_fast_d1: hold while the
  14/28/56-day return is positive, checked at every daily close.
- Development 1 Jan 2018 - 31 Oct 2023, walk-forward 24/6 months; holdout 1 Nov 2023 - 31 Oct 2026, one look.
- Check 1: PF >= 1.2, drawdown < 10%, >= 60 trades, statistical check. Check 2: same with >= 40 trades.
  Statistical check: the 5th percentile of 10,000 seeded bootstrap averages of trade R must be above zero.
- Coins for the holdout frozen from development: >= 15 trades and PF >= 1.0.
- Round 3 is the last backtest round for these markets; after it, evidence comes from demo forward tests only.
"""

from __future__ import annotations

import pandas as pd

from .round2 import Plan, Rules
from .trend import ROUND3_STRATEGIES

ROUND3_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
ROUND3_DATA_FROM = pd.Timestamp("2018-01-01", tz="UTC")
ROUND3_DEV_END = pd.Timestamp("2023-11-01", tz="UTC")
ROUND3_HOLDOUT_END = pd.Timestamp("2026-11-01", tz="UTC")
ROUND3_RULES = Rules(
    dev_min_pf=1.2, dev_max_dd_pct=10.0, dev_min_trades=60,
    holdout_min_pf=1.2, holdout_max_dd_pct=10.0, holdout_min_trades=40,
    basket_min_trades=15, basket_min_pf=1.0, sitout_min_pf=1.0,
    bootstrap=True, bootstrap_samples=10_000, bootstrap_pct=5.0,
)


def plan() -> Plan:
    """Round 3, read from the module constants at call time."""
    return Plan(3, ROUND3_SYMBOLS, {"crypto": ROUND3_DATA_FROM, "fx": ROUND3_DATA_FROM}, ROUND3_DEV_END, ROUND3_HOLDOUT_END,
                ROUND3_RULES, ROUND3_STRATEGIES, "round3_walk_forward")
