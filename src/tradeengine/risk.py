"""Risk manager: the final word on every order, from strategies and from Claude.

It is deliberately pure (no I/O): the caller gathers a RiskContext snapshot and
gets back a RiskDecision. That makes every rule unit-testable and means a
failure elsewhere (broker, Claude, database) cannot weaken a limit.

Limits are applied per venue in that venue's account currency, because each
venue is its own demo account. Position-count limits are engine-wide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from .models import (
    AccountSnapshot,
    AssetKind,
    Instrument,
    MarketSpec,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from .settings_store import RuntimeSettings
from .sizing import conversion_rate, loss_per_unit, position_size, round_down

NEW_YORK = ZoneInfo("America/New_York")
HUNDRED = Decimal(100)


@dataclass(frozen=True)
class OpenPositionInfo:
    symbol: str
    venue: Venue
    kind: AssetKind
    side: Side
    currencies: tuple[str, str]
    source: Source
    notional: Decimal  # in the venue's account currency


@dataclass
class RiskContext:
    now: datetime  # timezone-aware
    settings: RuntimeSettings
    halted: bool
    halt_reason: str
    accounts: dict[Venue, AccountSnapshot]
    sod_equity: dict[Venue, Decimal]
    peak_equity: dict[Venue, Decimal]
    open_positions: list[OpenPositionInfo]
    tick_age_seconds: dict[str, float]
    rates: dict[str, Decimal]  # mid prices for currency conversion, e.g. {"AUD/USD": 0.65}
    specs: dict[str, MarketSpec]
    # Claude sleeve, all in USD
    claude_realised_total: Decimal = Decimal(0)
    claude_pnl_today: Decimal = Decimal(0)  # realised today + open unrealised
    claude_unrealised: Decimal = Decimal(0)
    claude_trades_today: int = 0
    away_active: bool = False


@dataclass
class RiskDecision:
    approved: bool
    size: Decimal = Decimal(0)
    risk_amount: Decimal = Decimal(0)  # account currency at the stop
    reasons: list[str] = field(default_factory=list)
    halt: bool = False  # caller must engage the kill switch
    disable_claude: bool = False  # caller must switch the Claude trader off
    needs_approval: bool = False  # Claude trade while Away mode is off

    def reject(self, reason: str) -> RiskDecision:
        self.approved = False
        self.reasons.append(reason)
        return self


def fx_market_state(now: datetime, block_hours_before_close: int) -> tuple[bool, str]:
    """(entries_allowed, reason). FX trades Sunday 17:00 to Friday 17:00 New York time."""
    ny = now.astimezone(NEW_YORK)
    wd = ny.weekday()  # Mon=0 .. Sun=6
    five_pm = time(17, 0)
    if wd == 5 or (wd == 6 and ny.time() < five_pm) or (wd == 4 and ny.time() >= five_pm):
        return False, "FX market closed for the weekend"
    if wd == 4:
        close = ny.replace(hour=17, minute=0, second=0, microsecond=0)
        if close - ny <= timedelta(hours=block_hours_before_close):
            return False, f"no new FX entries in the last {block_hours_before_close} h before the weekend close"
    return True, ""


def evaluate(intent: TradeIntent, inst: Instrument, ctx: RiskContext) -> RiskDecision:
    s = ctx.settings
    d = RiskDecision(approved=False)

    if ctx.halted:
        return d.reject(f"kill switch engaged: {ctx.halt_reason}")
    if inst.symbol not in s.instruments:
        return d.reject(f"{inst.symbol} is not on the allowed instrument list")

    # --- stop and target sanity ------------------------------------------
    if intent.stop is None or intent.stop <= 0:
        return d.reject("no stop loss: every trade needs one")
    if intent.side is Side.BUY and not intent.stop < intent.entry:
        return d.reject("stop must be below entry for a buy")
    if intent.side is Side.SELL and not intent.stop > intent.entry:
        return d.reject("stop must be above entry for a sell")
    if intent.target is not None:
        if intent.side is Side.BUY and not intent.target > intent.entry:
            return d.reject("target must be above entry for a buy")
        if intent.side is Side.SELL and not intent.target < intent.entry:
            return d.reject("target must be below entry for a sell")
    if not (Decimal(0) < intent.size_factor <= Decimal(1)):
        return d.reject("size factor must be in (0, 1]")

    if inst.kind is AssetKind.CRYPTO_SPOT and intent.side is Side.SELL:
        return d.reject("spot crypto is long-only: a sell signal means stay flat")

    age = ctx.tick_age_seconds.get(inst.symbol)
    if age is None or age > s.stale_data_seconds:
        return d.reject(f"market data stale for {inst.symbol}")

    if inst.kind in (AssetKind.FX,):
        ok, why = fx_market_state(ctx.now, s.fx_no_new_entries_hours_before_close)
        if not ok:
            return d.reject(why)

    spec = ctx.specs.get(inst.symbol)
    if spec is None:
        return d.reject(f"no dealing rules loaded for {inst.symbol}")
    if not spec.tradeable:
        return d.reject(f"{inst.symbol} is not tradeable right now")

    acct = ctx.accounts.get(inst.venue)
    if acct is None or acct.equity <= 0:
        return d.reject(f"no account data for {inst.venue}")

    # --- account-level loss limits (per venue) ---------------------------
    sod = ctx.sod_equity.get(inst.venue, acct.equity)
    if sod > 0 and (sod - acct.equity) / sod * HUNDRED >= s.daily_loss_limit_pct:
        return d.reject(f"daily loss limit of {s.daily_loss_limit_pct}% reached on {inst.venue}")
    peak = ctx.peak_equity.get(inst.venue, acct.equity)
    if peak > 0 and (peak - acct.equity) / peak * HUNDRED >= s.max_drawdown_pct:
        d.halt = True
        return d.reject(f"maximum drawdown of {s.max_drawdown_pct}% reached on {inst.venue}")

    # --- position counts and exposure ------------------------------------
    if len(ctx.open_positions) >= s.max_open_positions:
        return d.reject(f"already {len(ctx.open_positions)} open positions (max {s.max_open_positions})")
    same = sum(1 for p in ctx.open_positions if p.symbol == inst.symbol)
    if same >= s.max_positions_per_instrument:
        return d.reject(f"already holding {inst.symbol}")
    if inst.kind is AssetKind.FX:
        for ccy in inst.currencies:
            n = sum(1 for p in ctx.open_positions if p.kind is AssetKind.FX and ccy in p.currencies)
            if n >= s.max_positions_per_currency:
                return d.reject(f"already {n} open positions in {ccy} (max {s.max_positions_per_currency})")

    # --- currency conversion ---------------------------------------------
    q2a = conversion_rate(inst.quote, acct.currency, ctx.rates)
    if q2a is None:
        return d.reject(f"no exchange rate from {inst.quote} to {acct.currency}")

    # --- risk amount -------------------------------------------------------
    if intent.source is Source.CLAUDE:
        if not s.claude_trader_enabled:
            return d.reject("Claude trader is switched off")
        sleeve = s.claude_budget + ctx.claude_realised_total + ctx.claude_unrealised
        if sleeve <= s.claude_loss_floor:
            d.disable_claude = True
            return d.reject(f"Claude budget at {sleeve:.2f}, at or below the {s.claude_loss_floor} floor")
        if -ctx.claude_pnl_today >= s.claude_daily_loss_limit:
            return d.reject(f"Claude daily loss limit of {s.claude_daily_loss_limit} reached")
        claude_open = sum(1 for p in ctx.open_positions if p.source is Source.CLAUDE)
        if claude_open >= s.claude_max_open_positions:
            return d.reject(f"Claude already has {claude_open} open positions")
        if ctx.claude_trades_today >= s.claude_max_trades_per_day:
            return d.reject(f"Claude already made {ctx.claude_trades_today} trades today")
        usd_risk = min(
            s.claude_max_loss_per_trade,
            sleeve - s.claude_loss_floor,
            s.claude_daily_loss_limit + ctx.claude_pnl_today,
        )
        usd_to_acct = conversion_rate("USD", acct.currency, ctx.rates)
        if usd_to_acct is None:
            return d.reject(f"no exchange rate from USD to {acct.currency}")
        risk_amount = usd_risk * usd_to_acct * intent.size_factor
        d.needs_approval = not ctx.away_active
    else:
        risk_amount = acct.equity * s.risk_per_trade_pct / HUNDRED * intent.size_factor

    sized = position_size(intent.entry, intent.stop, risk_amount, spec, q2a)
    if sized.size is None:
        return d.reject(sized.reason)
    size = sized.size

    # --- leverage cap: shrink to fit, or reject ---------------------------
    if inst.kind is not AssetKind.CRYPTO_SPOT:
        venue_notional = sum(p.notional for p in ctx.open_positions if p.venue is inst.venue)
        cap = acct.equity * s.max_effective_leverage
        room = cap - venue_notional
        unit_notional = intent.entry * spec.contract_multiplier * q2a
        if size * unit_notional > room:
            size = round_down(room / unit_notional, spec.size_step) if room > 0 else Decimal(0)
            if size < spec.min_size:
                return d.reject(f"leverage cap of {s.max_effective_leverage}:1 reached on {inst.venue}")
            d.reasons.append(f"size reduced to stay within {s.max_effective_leverage}:1 leverage")
    else:
        if size * intent.entry * q2a > acct.available:
            size = round_down(acct.available * Decimal("0.98") / (intent.entry * q2a), spec.size_step)
            if size < spec.min_size or size * intent.entry < spec.min_notional:
                return d.reject("not enough free balance for this spot order")
            d.reasons.append("size reduced to available balance")

    d.approved = True
    d.size = size
    d.risk_amount = size * loss_per_unit(intent.entry, intent.stop, spec, q2a)
    return d
