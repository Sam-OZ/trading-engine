"""Admin-editable runtime settings.

Every tunable business value lives here with its default. Values are stored in
the `settings` table so the dashboard (Phase 3) can change them without a code
change or redeploy. Validation stops a typo from loosening a limit to nonsense.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select

from .db import Database, SettingRow


class RuntimeSettings(BaseModel):
    # --- Engine-wide risk ---
    # Binance Demo Mode starts every account at 5,000 USDT and cannot be set to another amount,
    # so the engine trades a virtual balance of this size on it (Capital.com uses its real balance).
    binance_virtual_capital: Decimal = Field(Decimal("5000"), gt=0)
    risk_per_trade_pct: Decimal = Field(Decimal("0.5"), gt=0, le=2)
    daily_loss_limit_pct: Decimal = Field(Decimal("2.0"), gt=0, le=10)
    max_drawdown_pct: Decimal = Field(Decimal("10.0"), gt=0, le=50)
    max_open_positions: int = Field(4, ge=1, le=20)
    max_positions_per_instrument: int = Field(1, ge=1, le=5)
    max_positions_per_currency: int = Field(2, ge=1, le=10)
    max_effective_leverage: Decimal = Field(Decimal("5"), gt=0, le=30)
    stale_data_seconds: int = Field(60, ge=5, le=3600)
    weekend_fx_flatten: bool = True
    fx_no_new_entries_hours_before_close: int = Field(2, ge=0, le=24)
    instruments: list[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
        "XAU/USD", "BTC/USDT", "ETH/USDT",
    ]
    reconcile_interval_seconds: int = Field(30, ge=5, le=600)
    # Telegram alerts (Phase 3): trades, warnings/errors, kill switch, and a daily summary.
    telegram_alerts_enabled: bool = False
    telegram_chat_id: str | None = None  # set by `tradeengine telegram-setup`; not a secret (the bot token is)
    telegram_summary_utc: str = Field("00:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")  # 08:30 Perth, after the 00:00 UTC daily close
    telegram_repeat_hours: float = Field(6.0, ge=0, le=72)  # the same warning text is sent at most once per this many hours
    # A brief broker outage (HTTP 5xx, rate limit, timeout) is retried quietly; Telegram is told only when the
    # same job has kept failing this long (0 = at once), and again when it recovers.
    transient_alert_minutes: float = Field(3.0, ge=0, le=120)

    # --- Strategies (all OFF until they pass Gate 2; set with `tradeengine gate2-apply`) ---
    strategies_enabled: dict[str, list[str]] = {}  # strategy -> symbols it may trade
    strategy_params: dict[str, dict[str, dict]] = {}  # strategy -> symbol -> chosen parameters

    # --- Backtest / Gate 2 ---
    cost_model: dict[str, dict[str, float]] = {}  # per-symbol overrides of backtest.DEFAULT_COSTS
    gate2_min_pf: Decimal = Field(Decimal("1.2"), gt=0)
    gate2_max_dd_pct: Decimal = Field(Decimal("10"), gt=0)
    gate2_min_trades: int = Field(200, ge=1)
    gate2_symbol_min_pf: Decimal = Field(Decimal("1.0"), gt=0)
    gate2_symbol_min_trades: int = Field(30, ge=1)
    walk_forward_train_months: int = Field(12, ge=3, le=36)
    walk_forward_test_months: int = Field(3, ge=1, le=12)
    # Round 2 walk-forward, per strategy: [training months, test months] (pre-registered 12 Nov 2026)
    round2_walk_forward: dict[str, list[int]] = {
        "channel_breakout_d1": [36, 6], "tsmom_d1": [36, 6], "breakout_h4_dfilter": [24, 6],
    }

    # Round 3 (crypto) walk-forward, per strategy (pre-registered 12 Nov 2026)
    round3_walk_forward: dict[str, list[int]] = {"breakout_h4_dfilter": [24, 6], "tsmom_fast_d1": [24, 6]}

    @field_validator("round2_walk_forward", "round3_walk_forward")
    @classmethod
    def _wf(cls, v: dict[str, list[int]]) -> dict[str, list[int]]:
        for name, pair in v.items():
            if len(pair) != 2 or not (6 <= pair[0] <= 60 and 1 <= pair[1] <= 12):
                raise ValueError(f"{name}: needs [training months 6-60, test months 1-12]")
        return v

    # --- Claude reviewer ---
    claude_review_enabled: bool = True
    claude_review_timeout_seconds: int = Field(20, ge=2, le=120)
    claude_fallback_mode: Literal["half_size", "skip"] = "half_size"

    # --- Claude trader (its own budget) ---
    claude_trader_enabled: bool = True
    claude_budget: Decimal = Field(Decimal("2000"), ge=0)
    claude_max_loss_per_trade: Decimal = Field(Decimal("20"), gt=0)
    claude_daily_loss_limit: Decimal = Field(Decimal("60"), gt=0)
    claude_loss_floor: Decimal = Field(Decimal("1700"), ge=0)
    claude_max_open_positions: int = Field(2, ge=0, le=10)
    claude_max_trades_per_day: int = Field(6, ge=0, le=50)
    away_mode: bool = False
    away_schedule_enabled: bool = True
    away_schedule_start: str = "22:00"
    away_schedule_end: str = "07:00"
    away_timezone: str = "Australia/Perth"
    approval_timeout_seconds: int = Field(300, ge=30, le=3600)

    @field_validator("away_schedule_start", "away_schedule_end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        hh, mm = v.split(":")
        if not (0 <= int(hh) < 24 and 0 <= int(mm) < 60):
            raise ValueError("time must be HH:MM")
        return f"{int(hh):02d}:{int(mm):02d}"

    @model_validator(mode="after")
    def _cross_field(self) -> RuntimeSettings:
        if self.claude_loss_floor > self.claude_budget:
            raise ValueError("claude_loss_floor cannot exceed claude_budget")
        if self.claude_max_loss_per_trade > self.claude_daily_loss_limit:
            raise ValueError("claude_max_loss_per_trade cannot exceed claude_daily_loss_limit")
        return self


class SettingsStore:
    """Loads settings (DB values over defaults) and saves validated changes."""

    def __init__(self, db: Database):
        self.db = db

    async def load(self) -> RuntimeSettings:
        async with self.db.sessions() as s:
            rows = (await s.execute(select(SettingRow))).scalars().all()
        overrides = {r.key: r.value["v"] for r in rows if r.key in RuntimeSettings.model_fields}
        return RuntimeSettings(**overrides)

    async def update(self, changes: dict, updated_by: str) -> RuntimeSettings:
        current = (await self.load()).model_dump()
        unknown = set(changes) - set(RuntimeSettings.model_fields)
        if unknown:
            raise KeyError(f"unknown settings: {sorted(unknown)}")
        current.update(changes)
        validated = RuntimeSettings(**current)  # raises before anything is saved
        dumped = validated.model_dump(mode="json")
        async with self.db.sessions() as s, s.begin():
            for key in changes:
                row = await s.get(SettingRow, key)
                if row is None:
                    s.add(SettingRow(key=key, value={"v": dumped[key]}, updated_by=updated_by))
                else:
                    row.value = {"v": dumped[key]}
                    row.updated_by = updated_by
        return validated
