"""Gate 1 script logic, run against in-memory brokers (the real run uses the demo accounts)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

from conftest import FakeAdapter

from tradeengine.config import InfraConfig
from tradeengine.engine import Engine
from tradeengine.gate1 import Gate1, save_report
from tradeengine.models import Bar, Quote, Venue
from tradeengine.risk import fx_market_state


class GateFake(FakeAdapter):
    def __init__(self, venue, currency, price):
        super().__init__(venue, currency)
        self.price = price
        self.session_renewals = 1
        self.stream_connects = 0

    def invalidate_session(self):
        pass

    async def account(self):
        self.session_renewals += 1
        return await super().account()

    async def quote(self, inst):
        return Quote(inst.symbol, self.price - self.price / 10000, self.price + self.price / 10000, datetime.now(UTC))

    async def history(self, inst, tf, start, end):
        return [Bar(inst.symbol, tf, start + timedelta(minutes=i), self.price, self.price, self.price, self.price) for i in range(150)]

    async def list_positions(self):
        out = []
        for p in self.positions.values():
            bid = self.price - self.price / 10000
            p.unrealised_pnl = (bid - p.entry_price) * p.size * p.side.sign  # what a real broker reports
            out.append(p)
        return out

    async def stream(self, instruments):
        self.stream_connects += 1
        for _ in range(3):
            yield await self.quote(instruments[0])
        await asyncio.sleep(3600)


async def test_gate1_runs_all_checks(tmp_path):
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'g.db'}", data_dir=tmp_path)
    cap = GateFake(Venue.CAPITAL, "USD", D("1.1"))
    bnb = GateFake(Venue.BINANCE, "USDT", D("60000"))
    e = Engine(cfg, adapters={Venue.CAPITAL: cap, Venue.BINANCE: bnb})
    await e.start()
    g = Gate1(e)
    g._collect_quotes = lambda venue, seconds: _quick(g, venue)  # don't wait 20 s in tests
    try:
        report = await g.run({Venue.CAPITAL, Venue.BINANCE})
    finally:
        await e.stop()
    status = {c.id: c.status for c in report.checks}
    for cid in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "K1", "C1", "C2", "C3", "C5"):
        assert status[cid] == "PASS", (cid, [c for c in report.checks if c.id == cid])
    if fx_market_state(datetime.now(UTC), 0)[0]:
        for cid in ("C4", "C6", "C7", "C8", "C9", "C10", "C11"):
            assert status[cid] == "PASS", (cid, [c for c in report.checks if c.id == cid])
    else:
        assert status["C6"] == "SKIP"
    assert cap.positions == {} and bnb.positions == {}  # everything closed again
    path = save_report(report, tmp_path)
    assert path.exists() and '"passed"' in path.read_text()


async def _quick(g: Gate1, venue: Venue) -> int:
    adapter = g.e.adapters[venue]
    n = 0
    insts = [i for i in g.e.active_instruments() if i.venue is venue]
    async for ev in adapter.stream(insts):
        g.e.on_event(ev)
        n += 1
        if n == 3:
            break
    return n


