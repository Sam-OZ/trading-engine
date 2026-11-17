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


# ------------------------------------------------------------------ telegram-test
@respx.mock
def test_telegram_test_sends_one_message_to_the_saved_chat(tmp_path, monkeypatch, capsys):
    from tradeengine.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
    with pytest.raises(SystemExit) as e:  # nothing linked yet: a clear instruction, no Telegram call
        cli_main(["telegram-test"])
    assert "telegram-setup" in str(e.value)
    cli_main(["settings", "--set", "telegram_chat_id=\"777\"", "telegram_alerts_enabled=true"])
    capsys.readouterr()
    sent = respx.post("https://api.telegram.org/botT/sendMessage").mock(return_value=httpx.Response(200, json={"ok": True}))
    cli_main(["telegram-test"])
    assert "Test message sent to chat 777" in capsys.readouterr().out and sent.call_count == 1
    body = json.loads(sent.calls[0].request.content)
    assert body["chat_id"] == "777" and "Alerts are ON" in body["text"]


@respx.mock
def test_telegram_test_explains_a_revoked_token_without_showing_it(tmp_path, monkeypatch, capsys):
    from tradeengine.cli import main as cli_main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    cli_main(["settings", "--set", "telegram_chat_id=\"777\""])
    capsys.readouterr()
    respx.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage").mock(return_value=httpx.Response(401))
    with pytest.raises(SystemExit) as e:
        cli_main(["telegram-test"])
    assert "401" in str(e.value) and "revoked" in str(e.value) and TOKEN not in str(e.value)


# ------------------------------------------------------------------ Binance location block
@respx.mock
async def test_binance_location_block_gives_a_plain_reason():
    from tradeengine.brokers.base import BrokerError
    from tradeengine.brokers.binance import DEMO_TRADE_REST, BinanceSpotDemoAdapter
    from tradeengine.models import DEFAULT_INSTRUMENTS

    respx.get(f"{DEMO_TRADE_REST}/api/v3/time").mock(return_value=httpx.Response(451, json={
        "code": 0, "msg": "Service unavailable from a restricted location"}))
    a = BinanceSpotDemoAdapter("K", "S", list(DEFAULT_INSTRUMENTS.values()))
    with pytest.raises(BrokerError) as e:
        await a.connect()
    await a.close()
    assert e.value.status == 451 and "restricted location" in str(e.value) and "United States" in str(e.value)


def test_server_vm_is_outside_the_us_and_backups_stay_free():
    s = (Path(__file__).parent.parent / "scripts" / "vm.sh").read_text()
    assert 'ZONE="${ZONE:-australia-southeast2-a}"' in s and 'BUCKET_REGION="us-central1"' in s
    assert "--boot-disk-size=10GB" in s and "us-central1-a" not in s


# ------------------------------------------------------------------ brief outages: quiet retries, one alert, one all-clear
async def last_event_id(e):
    evs = await events_since(e)
    return evs[-1].id if evs else 0


async def events_since(e, after_id=0):
    from sqlalchemy import select

    async with e.db.sessions() as s:
        return (await s.execute(select(EventRow).where(EventRow.id > after_id).order_by(EventRow.id))).scalars().all()


def test_what_counts_as_a_brief_outage():
    from tradeengine.alerting import is_transient
    from tradeengine.brokers.base import BrokerError
    from tradeengine.brokers.binance import _check

    req = httpx.Request("GET", "https://demo-api.binance.com/api/v3/account")
    for code in (502, 504):
        with pytest.raises(BrokerError) as e:
            _check(httpx.Response(code, request=req), "/api/v3/account")
        assert is_transient(e.value)
    with pytest.raises(BrokerError) as e:  # Binance refusing the server's country is NOT brief
        _check(httpx.Response(451, request=req), "/api/v3/time")
    assert not is_transient(e.value)
    assert is_transient(httpx.ConnectError("x")) and is_transient(httpx.ReadTimeout("x")) and is_transient(TimeoutError())
    assert not is_transient(BrokerError("insufficient balance")) and not is_transient(ValueError("bug"))


async def test_outage_is_quiet_for_3_minutes_then_one_alert_then_one_all_clear(eng):
    from tradeengine.alerting import FailureAlerts
    from tradeengine.brokers.base import BrokerError

    t = [datetime(2026, 11, 17, 7, 48, tzinfo=UTC)]
    a = FailureAlerts(eng.db, eng.settings_store, clock=lambda: t[0])
    start = await last_event_id(eng)
    err = BrokerError("Binance /api/v3/account server error 502", status=502, retryable=True)

    await a.failed("account refresh", err)  # 07:48
    await a.succeeded("account refresh")  # 07:49 back: nothing at all reaches Telegram
    assert await events_since(eng, start) == []

    for minute in range(5):  # 07:50 .. 07:54, failing every minute
        t[0] = datetime(2026, 11, 17, 7, 50 + minute, tzinfo=UTC)
        await a.failed("account refresh", err)
    evs = await events_since(eng, start)
    assert [(e.level, e.message) for e in evs] == [
        ("ERROR", "account refresh failing for 3 min (4 tries): Binance /api/v3/account server error 502")]  # once
    t[0] = datetime(2026, 11, 17, 7, 57, tzinfo=UTC)
    await a.succeeded("account refresh")
    evs = await events_since(eng, start)
    assert evs[-1].level == "INFO" and evs[-1].message == "Recovered: account refresh working again after 7 min"
    assert worth_sending(evs[-1]) and format_event(evs[-1]) == "✅ engine: account refresh working again after 7 min"
    assert format_event(evs[0]) == "❗ ERROR engine: account refresh failing for 3 min (4 tries): Binance /api/v3/account server error 502"

    await a.failed("account refresh", ValueError("a real bug"))  # not an outage: alerts at once
    assert (await events_since(eng, start))[-1].message == "account refresh failed: a real bug"


async def test_alert_delay_is_an_admin_setting(eng):
    from tradeengine.alerting import FailureAlerts
    from tradeengine.brokers.base import BrokerError

    await eng.settings_store.update({"transient_alert_minutes": 0}, "t")
    a = FailureAlerts(eng.db, eng.settings_store)
    start = await last_event_id(eng)
    await a.failed("reconcile", BrokerError("binance: Binance /api/v3/account server error 504", retryable=True))
    assert (await events_since(eng, start))[-1].message.startswith("reconcile failing for under a minute (1 tries)")


async def test_reconcile_outage_is_retried_quietly_but_real_findings_alert_with_details(eng):
    from tradeengine.brokers.base import BrokerError
    from tradeengine.reconciler import ReconcileReport

    adapter = eng.adapters[Venue.BINANCE]
    orig = adapter.list_positions

    async def down():
        raise BrokerError("Binance /api/v3/openOrders server error 502", status=502, retryable=True)

    adapter.list_positions = down
    start = await last_event_id(eng)
    with pytest.raises(BrokerError) as e:  # handed to the quiet-retry rule, no reconciler warning
        await eng._reconcile()
    assert e.value.retryable and "binance: Binance /api/v3/openOrders server error 502" in str(e.value)
    assert await events_since(eng, start) == []
    adapter.list_positions = orig
    await eng._reconcile()  # reachable again: clean pass, nothing logged
    assert await events_since(eng, start) == []

    rep = ReconcileReport(reprotected=["BTC/USDT te-abc"], external=["ETH/USDT 123"], errors=["binance: bad request"])
    assert rep.describe() == ("stop re-placed: BTC/USDT te-abc; position the engine did not open: ETH/USDT 123; "
                              "errors: binance: bad request")


async def test_hourly_run_retries_a_brief_outage_inside_the_hour(eng, monkeypatch):
    from tradeengine import runner as runner_mod
    from tradeengine.brokers.base import BrokerError

    monkeypatch.setattr(runner_mod, "HOURLY_RETRY_SECONDS", 0)
    r = runner_mod.StrategyRunner(eng)
    calls = []

    async def flaky(now):
        calls.append(now)
        if len(calls) < 3:
            raise BrokerError("Binance /api/v3/klines server error 502", retryable=True)

    r.on_hour = flaky
    start = await last_event_id(eng)
    await r.run_hour()
    assert len(calls) == 3 and await events_since(eng, start) == []  # recovered on the 3rd try, quietly

    calls.clear()

    async def broken(now):
        calls.append(now)
        raise ValueError("bug")

    r.on_hour = broken
    await r.run_hour()
    assert len(calls) == 1  # a real error is not retried...
    assert (await events_since(eng, start))[-1].message == "hourly run failed: bug"  # ...and alerts at once


async def test_start_up_waits_out_a_broker_outage_with_telegram_running(tmp_path):
    from tradeengine.brokers.base import BrokerError

    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'e.db'}", data_dir=tmp_path,
                      kill_switch_file=tmp_path / "KILL", telegram_bot_token="123:abc")
    fake = FakeAdapter(Venue.BINANCE, "USDT")
    tries = []
    real_connect = fake.connect

    async def connect():
        tries.append(1)
        if len(tries) <= 3:
            raise BrokerError("Binance /api/v3/time server error 502", status=502, retryable=True)
        await real_connect()

    fake.connect = connect
    e = Engine(cfg, adapters={Venue.BINANCE: fake})
    await e.db.create_all()
    await e.settings_store.update({"transient_alert_minutes": 0}, "t")
    await e.start_with_retry(every=0)
    assert len(tries) == 4 and e._notifier_started  # alerts were running throughout the wait
    msgs = [ev.message for ev in await events_since(e)]
    assert msgs[-2].startswith("engine start failing for under a minute (1 tries): Binance /api/v3/time server error 502")
    assert msgs[-1] == "Recovered: engine start working again after under a minute"
    tasks = len(e._tasks)
    e.start_notifier()  # run_forever calls it again: still one notifier
    assert len(e._tasks) == tasks
    await e.stop()


def test_status_says_plainly_when_the_broker_is_down(tmp_path, monkeypatch, capsys):
    from tradeengine import cli
    from tradeengine.brokers.base import BrokerError

    async def down(self):
        raise BrokerError("Binance /api/v3/time server error 502", status=502, retryable=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    monkeypatch.setattr(cli, "_engine", lambda **kw: Engine(InfraConfig(_env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'e.db'}", data_dir=tmp_path, kill_switch_file=tmp_path / "KILL"),
        adapters={Venue.BINANCE: FakeAdapter(Venue.BINANCE, "USDT")}))
    monkeypatch.setattr(FakeAdapter, "connect", down)
    with pytest.raises(SystemExit) as ex:
        cli.main(["status"])
    assert "not reachable right now" in str(ex.value) and "502" in str(ex.value)
