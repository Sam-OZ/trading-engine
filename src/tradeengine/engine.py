"""The engine: wires config, journal, brokers, market data, risk, execution and reconciler.

Phase 1 runs market data, reconciliation, equity tracking and the trade path.
Strategies plug into `Engine.trade()` in Phase 2; Claude's reviewer and trader in Phase 3.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .brokers.base import BrokerAdapter
from .brokers.binance import BinanceSpotDemoAdapter
from .brokers.capital import CapitalComAdapter
from .config import InfraConfig
from .db import Database, EquityRow, PositionRow
from .execution import ExecutionOutcome, ExecutionService
from .ids import client_order_id
from .marketdata import BarStore, MinuteBarBuilder, TickTracker
from .models import (
    DEFAULT_INSTRUMENTS,
    AccountSnapshot,
    AssetKind,
    Bar,
    Instrument,
    MarketSpec,
    OrderPlan,
    PositionStatus,
    Quote,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from .reconciler import Reconciler
from .risk import OpenPositionInfo, RiskContext, RiskDecision, evaluate
from .runner import StrategyRunner
from .settings_store import RuntimeSettings, SettingsStore
from .sizing import conversion_rate
from .state import EngineState, log_event

log = logging.getLogger(__name__)


def away_active(s: RuntimeSettings, now: datetime) -> bool:
    if s.away_mode:
        return True
    if not s.away_schedule_enabled:
        return False
    local = now.astimezone(ZoneInfo(s.away_timezone)).time()
    start = time.fromisoformat(s.away_schedule_start)
    end = time.fromisoformat(s.away_schedule_end)
    return (start <= local or local < end) if start > end else (start <= local < end)


class Engine:
    def __init__(self, cfg: InfraConfig, adapters: dict[Venue, BrokerAdapter] | None = None, instruments: dict[str, Instrument] | None = None):
        self.cfg = cfg
        self.db = Database(cfg.database_url)
        self.state = EngineState(self.db, cfg.kill_switch_file)
        self.settings_store = SettingsStore(self.db)
        self.instruments = instruments or DEFAULT_INSTRUMENTS
        self.adapters: dict[Venue, BrokerAdapter] = adapters if adapters is not None else self._build_adapters()
        self.execution = ExecutionService(self.db, self.adapters, self.state)
        self.reconciler = Reconciler(self.db, self.adapters, self.execution, self.instruments)
        self.ticks = TickTracker()
        self.bars = BarStore()
        self.bar_builder = MinuteBarBuilder()
        self.specs: dict[str, MarketSpec] = {}
        self.accounts: dict[Venue, AccountSnapshot] = {}
        self._tasks: list[asyncio.Task] = []

    def _build_adapters(self) -> dict[Venue, BrokerAdapter]:
        insts = list(self.instruments.values())
        out: dict[Venue, BrokerAdapter] = {}
        c = self.cfg
        if c.has_capital():
            out[Venue.CAPITAL] = CapitalComAdapter(
                c.capital_api_key.get_secret_value(),
                c.capital_identifier.get_secret_value(),
                c.capital_api_password.get_secret_value(),
                insts,
            )
        if c.has_binance():
            out[Venue.BINANCE] = BinanceSpotDemoAdapter(c.binance_api_key.get_secret_value(), c.binance_api_secret.get_secret_value(), insts)
        return out

    def active_instruments(self) -> list[Instrument]:
        return [i for i in self.instruments.values() if i.venue in self.adapters]

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self.db.create_all()
        for venue, a in self.adapters.items():
            await a.connect()
            log.info("connected to %s", venue)
        for inst in self.active_instruments():
            self.specs[inst.symbol] = await self.adapters[inst.venue].market_spec(inst)
            self.ticks.update(await self.adapters[inst.venue].quote(inst))
        await self.refresh_accounts()
        await self._add_conversion_feeds()

    async def _add_conversion_feeds(self) -> None:
        """An AUD (or other non-USD) demo account needs an AUD/USD price to size USD-quoted trades.

        The pair is streamed for conversion only; it is not tradeable unless it is
        also on the admin instrument list.
        """
        cap = self.adapters.get(Venue.CAPITAL)
        acct = self.accounts.get(Venue.CAPITAL)
        if cap is None or acct is None or acct.currency.upper() in ("USD", "USDT"):
            return
        ccy = acct.currency.upper()
        if conversion_rate("USD", ccy, self.rates()) is not None:
            return
        for base, quote in ((ccy, "USD"), ("USD", ccy)):
            inst = Instrument(f"{base}/{quote}", Venue.CAPITAL, f"{base}{quote}", AssetKind.FX, base, quote)
            try:
                cap.add_instrument(inst)
                self.specs[inst.symbol] = await cap.market_spec(inst)
                self.ticks.update(await cap.quote(inst))
                self.instruments = {**self.instruments, inst.symbol: inst}
                log.info("streaming %s for %s conversion", inst.symbol, ccy)
                return
            except Exception as e:
                log.warning("conversion pair %s unavailable: %s", inst.symbol, e)
        await log_event(self.db, "ERROR", "engine", f"No USD/{ccy} price available: Capital.com trades will be rejected")

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for a in self.adapters.values():
            await a.close()
        await self.db.close()

    async def refresh_accounts(self) -> None:
        for venue, a in self.adapters.items():
            acct = await a.account()
            if venue is Venue.BINANCE:
                acct = await self._virtual_binance_account(acct)
            self.accounts[venue] = acct
            await self.state.update_peak_equity(venue.value, acct.equity)
            await self.state.start_of_day_equity(venue.value, datetime.now(UTC).date().isoformat(), acct.equity)
            async with self.db.sessions() as s, s.begin():
                s.add(EquityRow(venue=venue.value, currency=acct.currency, balance=acct.balance, equity=acct.equity, available=acct.available))

    async def _virtual_binance_account(self, real: AccountSnapshot) -> AccountSnapshot:
        """Binance Demo Mode hands out its own default balances, which we cannot set to $10,000.

        So the engine trades a virtual balance (admin setting binance_virtual_capital) on that account:
        equity = starting capital + realised P&L of engine trades + open unrealised P&L.
        Spending is also capped by the real free USDT.
        """
        settings = await self.settings_store.load()
        async with self.db.sessions() as s:
            rows = (await s.execute(select(PositionRow).where(PositionRow.venue == Venue.BINANCE.value))).scalars().all()
        realised = sum((p.realised_pnl or Decimal(0)) for p in rows if p.status == PositionStatus.CLOSED.value)
        unrealised = Decimal(0)
        committed = Decimal(0)
        for p in rows:
            if p.status != PositionStatus.OPEN.value:
                continue
            committed += p.size * p.entry_price
            q = self.ticks.last.get(p.symbol)
            if q is not None:
                unrealised += (q.bid - p.entry_price) * p.size
        balance = settings.binance_virtual_capital + realised
        equity = balance + unrealised
        available = max(Decimal(0), min(real.available, balance - committed))
        return AccountSnapshot(Venue.BINANCE, real.currency, balance, equity, available)

    async def run_forever(self) -> None:
        settings = await self.settings_store.load()
        for venue, a in self.adapters.items():
            self._tasks.append(asyncio.create_task(self._consume(venue, a), name=f"stream-{venue}"))
        self._tasks.append(asyncio.create_task(self._loop(self.reconciler.run_once, settings.reconcile_interval_seconds), name="reconcile"))
        self._tasks.append(asyncio.create_task(self._loop(self.refresh_accounts, 60), name="equity"))
        self._tasks.append(asyncio.create_task(self._loop(self._heartbeat, 15), name="heartbeat"))
        self._tasks.append(asyncio.create_task(StrategyRunner(self).run_forever(), name="strategies"))
        await log_event(self.db, "INFO", "engine", "Engine started", {"venues": [v.value for v in self.adapters]})
        await asyncio.gather(*self._tasks)

    async def _consume(self, venue: Venue, adapter: BrokerAdapter) -> None:
        insts = [i for i in self.active_instruments() if i.venue is venue]
        async for ev in adapter.stream(insts):
            self.on_event(ev)

    def on_event(self, ev: Quote | Bar) -> None:
        if isinstance(ev, Quote):
            self.ticks.update(ev)
            if self.instruments.get(ev.symbol, None) and self.instruments[ev.symbol].venue is Venue.CAPITAL:
                bar = self.bar_builder.on_quote(ev)
                if bar:
                    self.bars.add(bar)
        else:
            self.bars.add(ev)

    async def _loop(self, fn, every: float) -> None:
        while True:
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("%s failed", getattr(fn, "__name__", fn))
                await log_event(self.db, "ERROR", "engine", f"{getattr(fn, '__name__', 'task')} failed: {e}")
            await asyncio.sleep(every)

    async def _heartbeat(self) -> None:
        (self.cfg.data_dir / "HEARTBEAT").write_text(datetime.now(UTC).isoformat())

    # ------------------------------------------------------------------ trade path
    def rates(self) -> dict[str, Decimal]:
        return self.ticks.mids()

    async def risk_context(self) -> RiskContext:
        settings = await self.settings_store.load()
        halted, why = await self.state.is_halted()
        now = datetime.now(UTC)
        day = now.date().isoformat()
        sod, peak = {}, {}
        for venue, acct in self.accounts.items():
            sod[venue] = await self.state.start_of_day_equity(venue.value, day, acct.equity)
            peak[venue] = await self.state.update_peak_equity(venue.value, acct.equity)
        async with self.db.sessions() as s:
            open_rows = (await s.execute(select(PositionRow).where(PositionRow.status == PositionStatus.OPEN.value))).scalars().all()
            claude_rows = (await s.execute(select(PositionRow).where(PositionRow.source == Source.CLAUDE.value))).scalars().all()
        rates = self.rates()
        open_info = []
        for p in open_rows:
            inst = self.instruments.get(p.symbol)
            if inst is None:
                continue
            acct = self.accounts.get(inst.venue)
            q2a = conversion_rate(inst.quote, acct.currency, rates) if acct else None
            mult = self.specs[p.symbol].contract_multiplier if p.symbol in self.specs else Decimal(1)
            notional = p.size * p.entry_price * mult * (q2a or Decimal(1))
            open_info.append(OpenPositionInfo(p.symbol, inst.venue, inst.kind, Side(p.side), inst.currencies, Source(p.source), notional))
        realised_total, pnl_today, unrealised, trades_today = self._claude_sleeve(claude_rows, rates, now)
        return RiskContext(
            now=now,
            settings=settings,
            halted=halted,
            halt_reason=why,
            accounts=dict(self.accounts),
            sod_equity=sod,
            peak_equity=peak,
            open_positions=open_info,
            tick_age_seconds=self.ticks.age_seconds(),
            rates=rates,
            specs=dict(self.specs),
            claude_realised_total=realised_total,
            claude_pnl_today=pnl_today,
            claude_unrealised=unrealised,
            claude_trades_today=trades_today,
            away_active=away_active(settings, now),
        )

    def _claude_sleeve(self, rows: list[PositionRow], rates: dict[str, Decimal], now: datetime):
        realised_total = pnl_today = unrealised = Decimal(0)
        trades_today = 0
        today = now.date()
        for p in rows:
            inst = self.instruments.get(p.symbol)
            if inst is None:
                continue
            to_usd = conversion_rate(inst.quote, "USD", rates) or Decimal(1)
            mult = self.specs[p.symbol].contract_multiplier if p.symbol in self.specs else Decimal(1)
            opened = p.opened_at if p.opened_at.tzinfo else p.opened_at.replace(tzinfo=UTC)
            if opened.date() == today:
                trades_today += 1
            if p.status == PositionStatus.CLOSED.value:
                r = (p.realised_pnl or Decimal(0)) * mult * to_usd
                realised_total += r
                closed = p.closed_at if (p.closed_at is None or p.closed_at.tzinfo) else p.closed_at.replace(tzinfo=UTC)
                if closed and closed.date() == today:
                    pnl_today += r
            else:
                q = self.ticks.last.get(p.symbol)
                if q is not None:
                    exit_px = q.bid if p.side == Side.BUY.value else q.ask
                    u = (exit_px - p.entry_price) * p.size * Side(p.side).sign * mult * to_usd
                    unrealised += u
                    pnl_today += u
        return realised_total, pnl_today, unrealised, trades_today

    async def trade(self, intent: TradeIntent) -> tuple[RiskDecision, ExecutionOutcome | None]:
        inst = self.instruments[intent.symbol]
        ctx = await self.risk_context()
        decision = evaluate(intent, inst, ctx)
        await log_event(
            self.db,
            "INFO" if decision.approved else "WARNING",
            "risk",
            f"{'Approved' if decision.approved else 'Rejected'} {intent.side.value} {intent.symbol} ({intent.source.value})",
            {"reasons": decision.reasons, "size": decision.size, "risk": decision.risk_amount},
        )
        if decision.halt:
            await self.state.engage_kill_switch("; ".join(decision.reasons), "risk-manager")
        if decision.disable_claude:
            await self.settings_store.update({"claude_trader_enabled": False}, "risk-manager")
        if not decision.approved or decision.needs_approval:
            return decision, None
        cid = client_order_id(intent.source, intent.strategy, intent.symbol, intent.side, intent.signal_time)
        plan = OrderPlan(cid, inst, intent, decision.size, decision.risk_amount)
        return decision, await self.execution.submit(plan)

    async def emergency_flatten(self, reason: str) -> list[str]:
        """Kill switch + close every engine position at every venue."""
        await self.state.engage_kill_switch(reason, "emergency")
        closed = []
        async with self.db.sessions() as s:
            rows = (await s.execute(select(PositionRow).where(PositionRow.status == PositionStatus.OPEN.value))).scalars().all()
        for p in rows:
            try:
                if await self.execution.close(p.id, f"emergency: {reason}"):
                    closed.append(p.symbol)
            except Exception as e:
                await log_event(self.db, "CRITICAL", "emergency", f"Failed to close {p.symbol}: {e}")
        return closed

