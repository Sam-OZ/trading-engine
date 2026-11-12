"""Gate 1: live checks against the two DEMO accounts.

Run from a machine that can reach the brokers (your Mac or the cloud VM):
    ./scripts/with-secrets.sh uv run tradeengine gate1

Every check opens at most one small demo position and closes it again.
A JSON report is written to data/reports/gate1-<timestamp>.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select

from .db import PositionRow
from .engine import Engine
from .ids import client_order_id
from .models import (
    BrokerPosition,
    OrderPlan,
    OrderStatus,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from .risk import fx_market_state
from .sizing import conversion_rate

log = logging.getLogger(__name__)


@dataclass
class Check:
    id: str
    name: str
    status: str = "SKIP"  # PASS | FAIL | SKIP
    detail: str = ""


@dataclass
class GateReport:
    started: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        ran = [c for c in self.checks if c.status != "SKIP"]
        return bool(ran) and all(c.status == "PASS" for c in ran)


class Gate1:
    def __init__(self, engine: Engine):
        self.e = engine
        self.report = GateReport(datetime.now(UTC).isoformat())

    def _add(self, cid: str, name: str) -> Check:
        c = Check(cid, name)
        self.report.checks.append(c)
        return c

    async def _run(self, c: Check, coro) -> object:
        try:
            out = await coro
            if c.status == "SKIP":
                c.status = "PASS"
            return out
        except Exception as ex:
            c.status, c.detail = "FAIL", f"{type(ex).__name__}: {ex}"
            log.exception("check %s failed", c.id)
            return None

    async def _collect_quotes(self, venue: Venue, seconds: float) -> int:
        adapter = self.e.adapters[venue]
        insts = [i for i in self.e.active_instruments() if i.venue is venue]
        n = 0

        async def consume():
            nonlocal n
            async for ev in adapter.stream(insts):
                self.e.on_event(ev)
                n += 1

        task = asyncio.create_task(consume())
        await asyncio.sleep(seconds)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return n

    async def _our_position(self, venue: Venue, key: str) -> BrokerPosition | None:
        for p in await self.e.adapters[venue].list_positions():
            if key in (p.broker_position_id, p.client_order_id):
                return p
        return None

    # ------------------------------------------------------------------ Capital.com
    async def capital(self) -> None:
        e, v = self.e, Venue.CAPITAL
        a = e.adapters[v]
        inst = e.instruments["EUR/USD"]

        c = self._add("C1", "Capital.com login and demo account")
        acct = await self._run(c, a.account())
        if acct:
            c.detail = f"{acct.currency} balance {acct.balance}, equity {acct.equity}"
            if acct.balance <= 0:
                c.status, c.detail = "FAIL", c.detail + " (zero balance: top up the demo account)"

        c = self._add("C2", "EUR/USD dealing rules")
        spec = await self._run(c, a.market_spec(inst))
        if spec:
            e.specs[inst.symbol] = spec
            c.detail = f"min {spec.min_size}, step {spec.size_step}, tick {spec.price_tick}, lot {spec.contract_multiplier}, tradeable {spec.tradeable}"

        c = self._add("C3", "Price history (15-minute bars, last 2 days)")
        bars = await self._run(c, a.history(inst, "M15", datetime.now(UTC) - timedelta(days=2), datetime.now(UTC)))
        if bars is not None:
            c.detail = f"{len(bars)} bars, last {bars[-1].open_time:%Y-%m-%d %H:%M} close {bars[-1].close}" if bars else "no bars"
            if len(bars) < 20:
                c.status = "FAIL"

        fx_open, why = fx_market_state(datetime.now(UTC), 0)
        c = self._add("C4", "Live price stream")
        if fx_open:
            n = await self._run(c, self._collect_quotes(v, 20))
            c.detail = f"{n} quotes in 20 s, reconnects {getattr(a, 'stream_connects', '?')}"
            if not n:
                c.status = "FAIL"
        else:
            c.detail = why

        c = self._add("C5", "Session renewal after token loss")
        before = a.session_renewals
        a.invalidate_session()
        await self._run(c, a.account())
        if c.status == "PASS":
            c.detail = f"re-logged in automatically ({before} -> {a.session_renewals} logins)"

        if not fx_open or not (spec and spec.tradeable):
            for cid, name in [("C6", "Open with broker-side stop and target"), ("C7", "P&L arithmetic"), ("C8", "Duplicate submit blocked"),
                              ("C9", "Tighten stop at broker"), ("C10", "Reconciler"), ("C11", "Close position")]:
                self._add(cid, name).detail = why or "market not tradeable now"
            return

        q = await a.quote(inst)
        e.ticks.update(q)
        intent = TradeIntent(
            symbol=inst.symbol, side=Side.BUY, entry=q.ask, stop=q.bid - Decimal("0.0030"), target=q.ask + Decimal("0.0030"),
            source=Source.STRATEGY, strategy="gate1", signal_time=datetime.now(UTC), reason="Gate 1 check", size_factor=Decimal("0.2"),
        )
        c = self._add("C6", "Open with broker-side stop and target")
        res = await self._run(c, e.trade(intent))
        if not res:
            return
        decision, outcome = res
        if not decision.approved or outcome is None or outcome.status is not OrderStatus.FILLED:
            c.status = "FAIL"
            c.detail = f"risk: {decision.reasons}; execution: {outcome}"
            return
        cid = client_order_id(intent.source, intent.strategy, intent.symbol, intent.side, intent.signal_time)
        async with e.db.sessions() as s:
            row = (await s.execute(select(PositionRow).where(PositionRow.client_order_id == cid))).scalar_one()
        try:
            bp = await self._our_position(v, row.broker_position_id)
            if bp is None or bp.stop is None or bp.target is None:
                c.status = "FAIL"
                c.detail = f"position {row.broker_position_id} stop={getattr(bp, 'stop', None)} target={getattr(bp, 'target', None)}"
            else:
                c.detail = f"{decision.size} @ {bp.entry_price}, stop {bp.stop}, target {bp.target}, risk {decision.risk_amount:.2f} {acct.currency if acct else ''}"

            c = self._add("C7", "P&L arithmetic matches broker")
            await asyncio.sleep(2)
            bp2 = await self._our_position(v, row.broker_position_id)
            q2 = await a.quote(inst)
            if bp2 and bp2.unrealised_pnl is not None and acct:
                rate = conversion_rate(inst.quote, acct.currency, {**e.rates(), "EUR/USD": q2.mid})
                if rate is None:
                    c.status, c.detail = "FAIL", f"no {inst.quote}->{acct.currency} rate"
                else:
                    ours = (q2.bid - bp2.entry_price) * bp2.size * spec.contract_multiplier * rate
                    tol = max(Decimal("0.10"), abs(bp2.unrealised_pnl) * Decimal("0.25"))
                    c.status = "PASS" if abs(ours - bp2.unrealised_pnl) <= tol else "FAIL"
                    c.detail = f"broker {bp2.unrealised_pnl}, engine {ours:.2f} {acct.currency} (tolerance {tol:.2f})"
            else:
                c.detail = "broker did not report unrealised P&L"

            c = self._add("C8", "Duplicate submit blocked")
            plan = OrderPlan(cid, inst, intent, decision.size, decision.risk_amount)
            n_before = len([p for p in await a.list_positions() if p.symbol == inst.symbol])
            dup = await self._run(c, e.execution.submit(plan))
            n_after = len([p for p in await a.list_positions() if p.symbol == inst.symbol])
            if dup is not None:
                c.status = "PASS" if dup.duplicate and n_after == n_before else "FAIL"
                c.detail = f"duplicate={dup.duplicate}, positions {n_before} -> {n_after}"

            c = self._add("C9", "Tighten stop at broker")
            new_stop = (q2.bid - Decimal("0.0020")).quantize(spec.price_tick)
            ok = await self._run(c, a.modify_protection(bp2 or bp, new_stop, None))
            bp3 = await self._our_position(v, row.broker_position_id)
            if ok and bp3 and bp3.stop is not None and abs(bp3.stop - new_stop) <= spec.price_tick:
                c.detail = f"stop now {bp3.stop}"
            else:
                c.status, c.detail = "FAIL", f"modify ok={ok}, broker stop {getattr(bp3, 'stop', None)}"

            c = self._add("C10", "Reconciler agrees with broker")
            rep = await self._run(c, e.reconciler.run_once())
            if rep:
                c.status = "PASS" if rep.matched >= 1 and not rep.force_closed and not rep.errors else "FAIL"
                c.detail = f"matched {rep.matched}, external {len(rep.external)}, errors {rep.errors}"
        finally:
            c = self._add("C11", "Close position")
            ok = await self._run(c, e.execution.close(row.id, "gate1 cleanup"))
            await asyncio.sleep(1)
            gone = await self._our_position(v, row.broker_position_id) is None
            c.status = "PASS" if ok and gone else "FAIL"
            c.detail = f"close ok={ok}, gone at broker={gone}"

    # ------------------------------------------------------------------ Binance
    async def binance(self) -> None:
        e, v = self.e, Venue.BINANCE
        a = e.adapters[v]
        inst = e.instruments["BTC/USDT"]

        c = self._add("B1", "Binance Demo Mode account")
        acct = await self._run(c, a.account())
        if acct:
            c.detail = f"USDT {acct.balance}, equity {acct.equity:.2f}, free {acct.available}"

        c = self._add("B2", "BTC/USDT exchange filters")
        spec = await self._run(c, a.market_spec(inst))
        if spec:
            e.specs[inst.symbol] = spec
            c.detail = f"min qty {spec.min_size}, step {spec.size_step}, tick {spec.price_tick}, min notional {spec.min_notional}"

        c = self._add("B3", "Live public klines (1-minute, last 2 h)")
        bars = await self._run(c, a.history(inst, "M1", datetime.now(UTC) - timedelta(hours=2), datetime.now(UTC)))
        if bars is not None:
            c.detail = f"{len(bars)} bars"
            if len(bars) < 100:
                c.status = "FAIL"

        c = self._add("B4", "Live public price stream")
        n = await self._run(c, self._collect_quotes(v, 15))
        c.detail = f"{n} events in 15 s"
        if not n:
            c.status = "FAIL"

        if not (acct and spec):
            for cid, name in [("B5", "Buy with exchange-side OCO"), ("B6", "Duplicate submit blocked"), ("B7", "Tighten stop (replace OCO)"),
                              ("B8", "Reconciler"), ("B9", "Close position")]:
                self._add(cid, name).detail = "skipped: account or filters unavailable"
            return

        q = await a.quote(inst)
        e.ticks.update(q)
        await e.refresh_accounts()  # virtual $10,000 for Binance
        intent = TradeIntent(
            symbol=inst.symbol, side=Side.BUY, entry=q.ask, stop=(q.bid * Decimal("0.99")), target=(q.ask * Decimal("1.01")),
            source=Source.STRATEGY, strategy="gate1", signal_time=datetime.now(UTC), reason="Gate 1 check", size_factor=Decimal("0.05"),
        )
        c = self._add("B5", "Buy with exchange-side OCO")
        res = await self._run(c, e.trade(intent))
        if not res:
            return
        decision, outcome = res
        if not decision.approved or outcome is None or outcome.status is not OrderStatus.FILLED:
            c.status, c.detail = "FAIL", f"risk: {decision.reasons}; execution: {outcome}"
            return
        cid = client_order_id(intent.source, intent.strategy, intent.symbol, intent.side, intent.signal_time)
        async with e.db.sessions() as s:
            row = (await s.execute(select(PositionRow).where(PositionRow.client_order_id == cid))).scalar_one()
        try:
            bp = await self._our_position(v, cid)
            if bp is None or bp.stop is None or bp.target is None:
                c.status, c.detail = "FAIL", f"protective orders not found for {cid}"
            else:
                c.detail = f"{row.size} BTC @ {row.entry_price:.2f}, stop {bp.stop}, target {bp.target}, OCO {bp.protection_id}"

            c = self._add("B6", "Duplicate submit blocked")
            plan = OrderPlan(cid, inst, intent, decision.size, decision.risk_amount)
            dup = await self._run(c, e.execution.submit(plan))
            if dup is not None:
                c.status = "PASS" if dup.duplicate else "FAIL"
                c.detail = f"duplicate={dup.duplicate}"

            c = self._add("B7", "Tighten stop (replace OCO)")
            new_stop = (q.bid * Decimal("0.993")).quantize(spec.price_tick)
            if bp:
                await self._run(c, a.modify_protection(bp, new_stop, None))
                bp2 = await self._our_position(v, cid)
                if bp2 and bp2.stop is not None and abs(bp2.stop - new_stop) <= spec.price_tick:
                    c.detail = f"stop now {bp2.stop}, protection {bp2.protection_id}"
                else:
                    c.status, c.detail = "FAIL", f"broker stop {getattr(bp2, 'stop', None)}"

            c = self._add("B8", "Reconciler agrees with exchange")
            rep = await self._run(c, e.reconciler.run_once())
            if rep:
                c.status = "PASS" if rep.matched >= 1 and not rep.force_closed and not rep.errors else "FAIL"
                c.detail = f"matched {rep.matched}, errors {rep.errors}"
        finally:
            c = self._add("B9", "Close position")
            ok = await self._run(c, e.execution.close(row.id, "gate1 cleanup"))
            await asyncio.sleep(1)
            gone = await self._our_position(v, cid) is None
            c.status = "PASS" if ok and gone else "FAIL"
            c.detail = f"close ok={ok}, protective orders gone={gone}"

    # ------------------------------------------------------------------ shared
    async def kill_switch(self) -> None:
        c = self._add("K1", "Kill switch blocks new orders")
        e = self.e
        inst = next((i for i in e.active_instruments()), None)
        if inst is None:
            c.detail = "no venue connected"
            return
        await e.state.engage_kill_switch("gate1 test", "gate1")
        try:
            q = e.ticks.last.get(inst.symbol) or await e.adapters[inst.venue].quote(inst)
            intent = TradeIntent(inst.symbol, Side.BUY, q.ask, q.bid * Decimal("0.99"), None, Source.STRATEGY, "gate1-kill", datetime.now(UTC))
            decision, outcome = await e.trade(intent)
            c.status = "PASS" if not decision.approved and outcome is None else "FAIL"
            c.detail = "; ".join(decision.reasons)
        finally:
            await e.state.release_kill_switch("gate1")

    async def run(self, venues: set[Venue]) -> GateReport:
        if Venue.CAPITAL in venues and Venue.CAPITAL in self.e.adapters:
            await self.capital()
        if Venue.BINANCE in venues and Venue.BINANCE in self.e.adapters:
            await self.binance()
        await self.kill_switch()
        return self.report


def print_report(report: GateReport, path: Path) -> None:
    width = max(len(c.name) for c in report.checks) if report.checks else 10
    print()
    for c in report.checks:
        print(f"  {c.status:<5} {c.id:<4} {c.name:<{width}}  {c.detail}")
    print()
    print(f"  GATE 1: {'PASS' if report.passed else 'NOT PASSED'}   report: {path}")
    print()


def save_report(report: GateReport, data_dir: Path) -> Path:
    out = data_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gate1-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"passed": report.passed, **asdict(report)}, indent=2, default=str))
    return path
