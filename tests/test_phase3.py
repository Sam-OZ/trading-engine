"""Phase 3: exit prices on engine closes, one event per close, Telegram alerts and daily summary,
`gate2-apply --only`, `telegram-setup`, and the server scripts' syntax."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from decimal import Decimal as D
from pathlib import Path

import httpx
import pytest
import respx
from conftest import FakeAdapter
from test_execution import plan, rows

from tradeengine.config import InfraConfig
from tradeengine.db import EventRow, PositionRow, StateRow
from tradeengine.engine import Engine
from tradeengine.execution import ExecutionService
from tradeengine.models import ExitInfo, Venue
from tradeengine.notify import (
    BOOKMARK,
    FORWARD_START,
    LAST_SUMMARY,
    TelegramError,
    TelegramNotifier,
    find_chat_id,
    format_event,
    worth_sending,
)
from tradeengine.state import log_event

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ closes record a real exit price, once
async def test_engine_close_records_the_brokers_exit_price_and_one_event(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state)
    out = await ex.submit(plan())
    fake.exit = ExitInfo("closed", D("1.1030"), "sold by engine")
    assert await ex.close(out.position_id, "tsmom_fast_d1 exit signal")
    pos = (await rows(db, PositionRow))[0]
    assert pos.status == "closed" and pos.exit_price == D("1.1030") and pos.realised_pnl > 0
    closes = [e for e in await rows(db, EventRow) if e.message.startswith("Closed")]
    assert len(closes) == 1 and "exit signal" in closes[0].message and "P&L +" in closes[0].message


async def test_close_still_works_when_the_exit_price_lookup_fails(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state)
    out = await ex.submit(plan())

    async def boom(pos):
        raise httpx.ConnectError("down")

    fake.exit_info = boom
    assert await ex.close(out.position_id, "time exit")
    pos = (await rows(db, PositionRow))[0]
    assert pos.status == "closed" and pos.exit_price is None


async def test_reconciler_close_at_broker_also_logs_one_closed_event(db, state, fake):
    from tradeengine.reconciler import Reconciler

    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state)
    await ex.submit(plan())
    fake.positions.clear()  # the broker's stop filled
    await Reconciler(db, {Venue.CAPITAL: fake}, ex).run_once()
    closes = [e for e in await rows(db, EventRow) if e.message.startswith("Closed")]
    assert len(closes) == 1 and "stop loss" in closes[0].message


# ------------------------------------------------------------------ Telegram
def ev(level: str, component: str, message: str) -> EventRow:
    return EventRow(level=level, component=component, message=message)


def test_which_events_are_sent():
    assert worth_sending(ev("INFO", "execution", "Opened BUY 0.01 BTC/USDT"))
    assert worth_sending(ev("INFO", "execution", "Closed BUY BTC/USDT: stop loss; P&L -25.00"))
    assert worth_sending(ev("WARNING", "reconciler", "Reconcile found issues"))
    assert worth_sending(ev("ERROR", "runner", "Hourly run failed: x"))
    assert worth_sending(ev("CRITICAL", "killswitch", "Kill switch engaged: drawdown"))
    assert not worth_sending(ev("INFO", "runner", "Stop tightened on BTC/USDT"))
    assert not worth_sending(ev("WARNING", "runner", "BTC/USDT: 20 days of hourly history, needs 120; not trading it yet"))
    assert not worth_sending(ev("INFO", "risk", "Approved BUY BTC/USDT (strategy)"))
    assert format_event(ev("INFO", "execution", "Opened BUY 0.01 BTC/USDT")).startswith("🟢")
    assert format_event(ev("CRITICAL", "killswitch", "Kill switch engaged")).startswith("🛑 CRITICAL")


@pytest.fixture
async def eng(tmp_path):
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'p3.db'}", data_dir=tmp_path,
                      telegram_bot_token="123:abc")
    e = Engine(cfg, adapters={Venue.BINANCE: FakeAdapter(Venue.BINANCE, "USDT")})
    await e.start()
    await e.settings_store.update({"telegram_alerts_enabled": True, "telegram_chat_id": "42", "telegram_summary_utc": "00:30"}, "t")
    yield e
    await e.stop()


async def state_of(e, key):
    async with e.db.sessions() as s:
        row = await s.get(StateRow, key)
        return row.value if row else None


@respx.mock
async def test_notifier_starts_from_now_sends_new_events_dedupes_and_retries_after_a_failure(eng):
    route = respx.post("https://api.telegram.org/bot123:abc/sendMessage").mock(return_value=httpx.Response(200, json={"ok": True}))
    n = TelegramNotifier(eng, "123:abc")
    morning = datetime(2026, 11, 13, 0, 10, tzinfo=UTC)  # before the 00:30 summary
    async with httpx.AsyncClient() as client:
        await log_event(eng.db, "CRITICAL", "killswitch", "old event before the notifier started")
        assert await n.run_once(client, morning) == 0 and route.call_count == 0  # history is not replayed
        await log_event(eng.db, "INFO", "execution", "Opened BUY 0.01 BTC/USDT")
        await log_event(eng.db, "WARNING", "reconciler", "Reconcile found issues")
        await log_event(eng.db, "WARNING", "reconciler", "Reconcile found issues")  # same text again: once only
        await log_event(eng.db, "INFO", "runner", "Stop tightened on BTC/USDT")  # not sent
        assert await n.run_once(client, morning) == 2
        texts = [json.loads(c.request.content)["text"] for c in route.calls]
        assert texts[0].startswith("🟢 Opened") and "Reconcile found issues" in texts[1]
        assert all(json.loads(c.request.content)["chat_id"] == "42" for c in route.calls)
        # Telegram down: nothing lost, the bookmark stays and the event goes out on the next pass
        route.mock(return_value=httpx.Response(502))
        await log_event(eng.db, "ERROR", "runner", "Hourly run failed: boom")
        with pytest.raises(TelegramError):
            await n.run_once(client, morning)
        route.mock(return_value=httpx.Response(200, json={"ok": True}))
        assert await n.run_once(client, morning) == 1
        assert "Hourly run failed" in json.loads(route.calls[-1].request.content)["text"]


@respx.mock
async def test_daily_summary_once_a_day_with_the_forward_test_tally(eng):
    route = respx.post("https://api.telegram.org/bot123:abc/sendMessage").mock(return_value=httpx.Response(200, json={"ok": True}))
    async with eng.db.sessions() as s, s.begin():
        s.add(StateRow(key=BOOKMARK, value={"id": 10**9}))
        s.add(StateRow(key=FORWARD_START, value={"at": "2026-11-12T12:00:00+00:00", "strategies": ["tsmom_fast_d1"]}))
        for pnl in (D("40"), D("-20")):
            s.add(PositionRow(client_order_id=f"c{pnl}", venue="binance", symbol="BTC/USDT", side="BUY", size=D("0.01"),
                              entry_price=D("60000"), broker_position_id="x", source="strategy", strategy="tsmom_fast_d1",
                              status="closed", closed_at=datetime(2026, 11, 13, 0, 0, tzinfo=UTC), realised_pnl=pnl))
    n = TelegramNotifier(eng, "123:abc")
    async with httpx.AsyncClient() as client:
        assert await n.run_once(client, datetime(2026, 11, 13, 0, 29, tzinfo=UTC)) == 0  # not yet
        assert await n.run_once(client, datetime(2026, 11, 13, 0, 31, tzinfo=UTC)) == 1
        assert await n.run_once(client, datetime(2026, 11, 13, 9, 0, tzinfo=UTC)) == 0  # once per day
        assert await n.run_once(client, datetime(2026, 11, 14, 0, 31, tzinfo=UTC)) == 1
    text = json.loads(route.calls[0].request.content)["text"]
    assert text.startswith("📊 Daily summary") and "binance: equity" in text and "Kill switch: off" in text
    assert "Forward test since 2026-11-12 (tsmom_fast_d1): 2 closed trades, P&L +20" in text and "profit factor 2.00" in text
    assert (await state_of(eng, LAST_SUMMARY)) == {"date": "2026-11-14"}


async def test_notifier_is_silent_until_set_up(eng):
    await eng.settings_store.update({"telegram_alerts_enabled": False}, "t")
    async with httpx.AsyncClient() as client:
        assert await TelegramNotifier(eng, "123:abc").run_once(client, datetime.now(UTC)) == 0


@respx.mock
async def test_find_chat_id_uses_the_latest_message():
    respx.get("https://api.telegram.org/botT/getUpdates").mock(return_value=httpx.Response(200, json={"ok": True, "result": [
        {"update_id": 1, "message": {"chat": {"id": 1, "first_name": "Old"}}},
        {"update_id": 2, "message": {"chat": {"id": 777, "first_name": "Sam"}}},
    ]}))
    async with httpx.AsyncClient() as client:
        assert await find_chat_id(client, "T") == ("777", "Sam")


@respx.mock
def test_telegram_setup_command(tmp_path, monkeypatch, capsys):
    from tradeengine.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    respx.get("https://api.telegram.org/botT/getUpdates").mock(return_value=httpx.Response(200, json={"ok": True, "result": [
        {"update_id": 2, "message": {"chat": {"id": 777, "first_name": "Sam"}}}]}))
    sent = respx.post("https://api.telegram.org/botT/sendMessage").mock(return_value=httpx.Response(200, json={"ok": True}))
    cli_main(["telegram-setup"])
    assert "Telegram alerts on: chat 777 (Sam)" in capsys.readouterr().out and sent.call_count == 1
    cli_main(["settings"])
    s = json.loads(capsys.readouterr().out)
    assert s["telegram_chat_id"] == "777" and s["telegram_alerts_enabled"] is True


# ------------------------------------------------------------------ gate2-apply --only
def test_apply_only_one_strategy_and_start_the_forward_test_clock(tmp_path, monkeypatch, capsys):
    from tradeengine.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    result = {"verdicts": [
        {"strategy": "breakout_h4_dfilter", "passed": True, "enabled_symbols": ["BTC/USDT", "ETH/USDT"],
         "latest_params": {"BTC/USDT": "dc_n=20,stop_atr=3.0,trail_atr=3.0", "ETH/USDT": "dc_n=55,stop_atr=2.0,trail_atr=3.0"}},
        {"strategy": "tsmom_fast_d1", "passed": True, "enabled_symbols": ["BTC/USDT", "ETH/USDT"],
         "latest_params": {"BTC/USDT": "lookback=14,stop_atr=3.0,trail_atr=4.0", "ETH/USDT": "lookback=56,stop_atr=3.0,trail_atr=4.0"}},
    ]}
    (tmp_path / "r.json").write_text(json.dumps(result))
    with pytest.raises(SystemExit) as e:
        cli_main(["gate2-apply", "r.json", "--only", "nonsense"])
    assert "not in this result" in str(e.value)
    cli_main(["gate2-apply", "r.json", "--only", "tsmom_fast_d1"])
    assert "tsmom_fast_d1: BTC/USDT, ETH/USDT" in capsys.readouterr().out
    cli_main(["settings"])
    s = json.loads(capsys.readouterr().out)
    assert s["strategies_enabled"] == {"tsmom_fast_d1": ["BTC/USDT", "ETH/USDT"]}
    assert s["strategy_params"]["tsmom_fast_d1"]["ETH/USDT"] == {"lookback": 56, "stop_atr": 3.0, "trail_atr": 4.0}

    import asyncio

    from tradeengine.db import Database

    async def started():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
        async with db.sessions() as sess:
            row = await sess.get(StateRow, FORWARD_START)
        await db.close()
        return row.value

    v = asyncio.run(started())
    assert v["strategies"] == ["tsmom_fast_d1"] and datetime.fromisoformat(v["at"]).tzinfo is not None


# ------------------------------------------------------------------ server scripts
@pytest.mark.parametrize("script", ["scripts/vm.sh", "deploy/setup.sh", "deploy/te", "deploy/backup.sh"])
def test_server_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(ROOT / script)], capture_output=True).returncode == 0


def test_server_never_reads_secret_manager_and_unit_runs_as_trader():
    unit = (ROOT / "deploy/tradeengine.service").read_text()
    assert "User=trader" in unit and "EnvironmentFile=/etc/tradeengine/env" in unit and "tradeengine run" in unit
    for f in ("deploy/setup.sh", "deploy/te", "deploy/backup.sh", "deploy/tradeengine.service"):
        assert "secrets versions access" not in (ROOT / f).read_text()
    vm = (ROOT / "scripts/vm.sh").read_text()
    assert 'SERVER_SECRETS="BINANCE_API_KEY BINANCE_API_SECRET TELEGRAM_BOT_TOKEN"' in vm  # no Capital.com or Anthropic keys
    assert "--tunnel-through-iap" in vm and "e2-micro" in vm and "pd-standard" in vm


# ------------------------------------------------------------------ secrets stay out of logs
TOKEN = "9174070385:AAFabcdefghijklmnopqrstuvwxyz012345"


def test_bot_token_is_redacted_from_log_lines_and_tracebacks(caplog):
    import logging

    from tradeengine.logsafe import RedactFilter, redact

    assert redact(f'GET https://api.telegram.org/bot{TOKEN}/getUpdates "HTTP/1.1 200 OK"') == \
        'GET https://api.telegram.org/bot<redacted>/getUpdates "HTTP/1.1 200 OK"'
    handler = logging.StreamHandler()
    handler.addFilter(RedactFilter())
    rec = logging.LogRecord("httpx", logging.INFO, __file__, 1, "HTTP Request: %s %s", ("POST", f"https://api.telegram.org/bot{TOKEN}/sendMessage"), None)
    handler.filter(rec)
    assert TOKEN not in rec.getMessage() and "bot<redacted>" in rec.getMessage()
    try:
        raise RuntimeError(f"boom at https://api.telegram.org/bot{TOKEN}/x")
    except RuntimeError:
        import sys

        rec = logging.LogRecord("x", logging.WARNING, __file__, 1, "failed", None, sys.exc_info())
    handler.filter(rec)
    assert rec.exc_info is None and TOKEN not in rec.exc_text and "bot<redacted>" in rec.exc_text


@respx.mock
async def test_telegram_errors_never_carry_the_token():
    from tradeengine.notify import send

    respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(return_value=httpx.Response(401))
    async with httpx.AsyncClient() as client:
        with pytest.raises(TelegramError) as e:
            await send(client, TOKEN, "1", "hi")
    assert str(e.value) == "Telegram sendMessage failed: HTTP 401" and TOKEN not in str(e.value)


def test_the_cli_installs_the_redaction_on_start(tmp_path, monkeypatch, capsys):
    import logging

    from tradeengine.cli import main as cli_main
    from tradeengine.logsafe import RedactFilter

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    cli_main(["settings"])
    capsys.readouterr()
    assert any(isinstance(f, RedactFilter) for h in logging.getLogger().handlers for f in h.filters)
