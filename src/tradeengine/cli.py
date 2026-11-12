"""Command line: `tradeengine <command>`.

  status                  accounts, open positions, kill switch
  gate1 [--venue ...]     live demo checks (opens and closes small demo positions)
  run                     market data + reconciler + equity tracking (strategies arrive in Phase 2)
  reconcile               one reconcile pass
  kill "reason" [--flatten]   engage the kill switch (and close all engine positions)
  unkill                  release the kill switch
  settings [--set k=v]    show or change admin settings
  download ...            historical data for one symbol
  download-all            Phase 2 history for every allowed instrument (H1, ~3 years)
  gate2                   walk-forward backtest of every strategy, Gate 2 verdict + report
  gate2-apply <json>      switch on the strategies that passed (demo trading starts on the next hour)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from pydantic import ValidationError
from sqlalchemy import select

from .backtest import parse_params_id, render_report, run_gate2
from .backtest import save_report as save_gate2_report
from .config import load_config
from .db import PositionRow
from .engine import Engine
from .gate1 import Gate1, print_report, save_report
from .history import (
    bars_to_frame,
    download_dukascopy,
    frame_to_bars,
    history_path,
    resample_frame,
    write_bars,
)
from .models import DEFAULT_INSTRUMENTS, PositionStatus, Venue


def _engine(need_brokers: bool = True) -> Engine:
    cfg = load_config()
    eng = Engine(cfg)
    if need_brokers and not eng.adapters:
        sys.exit(
            "No broker keys found. Run through scripts/with-secrets.sh so CAPITAL_* and/or "
            "BINANCE_* are set (see README)."
        )
    return eng


async def cmd_status(_args) -> None:
    eng = _engine()
    try:
        await eng.start()
        halted, why = await eng.state.is_halted()
        print(f"Kill switch: {'ENGAGED - ' + why if halted else 'off'}")
        for v, a in eng.accounts.items():
            print(f"{v.value:<8} {a.currency} balance {a.balance:,.2f}  equity {a.equity:,.2f}  available {a.available:,.2f}")
        for v, ad in eng.adapters.items():
            for p in await ad.list_positions():
                print(f"  {v.value:<8} {p.symbol:<9} {p.side.value:<4} {p.size} @ {p.entry_price} stop {p.stop} target {p.target}")
        async with eng.db.sessions() as s:
            rows = (await s.execute(select(PositionRow).where(PositionRow.status == PositionStatus.OPEN.value))).scalars().all()
        print(f"Journal open positions: {len(rows)}")
    finally:
        await eng.stop()


async def cmd_gate1(args) -> None:
    eng = _engine()
    venues = {Venue(v) for v in args.venue} if args.venue else set(Venue)
    try:
        await eng.start()
        halted, why = await eng.state.is_halted()
        if halted:
            sys.exit(f"Kill switch is engaged ({why}). Run `tradeengine unkill` first.")
        report = await Gate1(eng).run(venues)
        path = save_report(report, eng.cfg.data_dir)
        print_report(report, path)
    finally:
        await eng.stop()


async def cmd_run(_args) -> None:
    eng = _engine()
    try:
        await eng.start()
        await eng.run_forever()
    finally:
        await eng.stop()


async def cmd_reconcile(_args) -> None:
    eng = _engine()
    try:
        await eng.start()
        rep = await eng.reconciler.run_once()
        print(json.dumps(rep.__dict__, indent=2, default=str))
    finally:
        await eng.stop()


async def cmd_kill(args) -> None:
    eng = _engine(need_brokers=args.flatten)
    try:
        await eng.db.create_all()
        if args.flatten:
            await eng.start()
            closed = await eng.emergency_flatten(args.reason)
            print(f"Kill switch engaged; closed: {closed or 'nothing open'}")
        else:
            await eng.state.engage_kill_switch(args.reason, "cli")
            print("Kill switch engaged. New orders are blocked; open positions keep their broker-side stops.")
    finally:
        await eng.stop()


async def cmd_unkill(_args) -> None:
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        await eng.state.release_kill_switch("cli")
        print("Kill switch released.")
    finally:
        await eng.stop()


async def cmd_settings(args) -> None:
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        if args.set:
            changes = {}
            for kv in args.set:
                k, _, v = kv.partition("=")
                try:
                    changes[k] = json.loads(v)
                except json.JSONDecodeError:
                    changes[k] = v
            try:
                s = await eng.settings_store.update(changes, "cli")
            except (ValidationError, KeyError) as e:
                sys.exit(f"Not saved: {e}")
        else:
            s = await eng.settings_store.load()
        print(json.dumps(s.model_dump(mode="json"), indent=2))
    finally:
        await eng.stop()


async def cmd_download(args) -> None:
    cfg = load_config()
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    inst = DEFAULT_INSTRUMENTS.get(args.symbol)
    if args.source == "dukascopy":
        await _download_dukascopy(cfg, args.symbol, start, end)
        return
    if inst is None:
        sys.exit(f"unknown symbol {args.symbol}; known: {', '.join(DEFAULT_INSTRUMENTS)}")
    eng = _engine()
    try:
        await eng.start()
        adapter = eng.adapters.get(Venue.CAPITAL if args.source == "capital" else Venue.BINANCE)
        if adapter is None:
            sys.exit(f"no keys for {args.source}")
        await _download_broker(cfg, adapter, args.source, inst, args.tf, start, end)
    finally:
        await eng.stop()


async def _download_broker(cfg, adapter, source: str, inst, tf: str, start, end) -> None:
    bars = await adapter.history(inst, tf, start, end)
    path = history_path(cfg.data_dir, source, inst.symbol, tf)
    n = write_bars(path, bars)
    gaps = getattr(adapter, "history_gaps", [])
    first = bars[0].open_time.date() if bars else None
    print(f"  {source:<9} {inst.symbol:<9} {tf}: {len(bars):,} bars from {first}; {n:,} rows in file"
          + (f"; {len(gaps)} window(s) unavailable" if gaps else ""))
    if gaps:
        gaps.clear()


async def _download_dukascopy(cfg, symbol: str, start, end) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        bars = await download_dukascopy(client, symbol, start.date(), end.date() + timedelta(days=1))
    m1 = history_path(cfg.data_dir, "dukascopy", symbol, "M1")
    write_bars(m1, bars)
    h1 = resample_frame(bars_to_frame(bars), "1h") if bars else None
    n = write_bars(history_path(cfg.data_dir, "dukascopy", symbol, "H1"), frame_to_bars(h1, symbol, "H1")) if h1 is not None else 0
    first = bars[0].open_time.date() if bars else None
    print(f"  dukascopy {symbol:<9} M1: {len(bars):,} bars from {first}; H1 file {n:,} rows")


async def cmd_download_all(args) -> None:
    """History for every allowed instrument: Binance H1 (crypto); Capital.com H1 and Dukascopy (FX)."""
    logging.getLogger("httpx").setLevel(logging.WARNING)  # thousands of requests: print a summary line per file instead
    cfg = load_config()
    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    eng = _engine()
    try:
        await eng.start()
        settings = await eng.settings_store.load()
        for sym in settings.instruments:
            inst = DEFAULT_INSTRUMENTS.get(sym)
            if inst is None:
                continue
            adapter = eng.adapters.get(inst.venue)
            if adapter is not None:
                try:
                    await _download_broker(cfg, adapter, inst.venue.value, inst, "H1", start, end)
                except Exception as e:
                    print(f"  {inst.venue.value:<9} {sym:<9} failed: {e}")
            if inst.kind.value == "fx" and not args.no_dukascopy:
                try:
                    await _download_dukascopy(cfg, sym, start, end)
                except Exception as e:
                    print(f"  dukascopy {sym:<9} failed: {e}")
    finally:
        await eng.stop()


async def cmd_gate2(args) -> None:
    cfg = load_config()
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        settings = await eng.settings_store.load()
    finally:
        await eng.stop()
    print("Walk-forward backtest (this takes a few minutes):")
    verdicts, results, rules, notes = run_gate2(cfg.data_dir, settings, DEFAULT_INSTRUMENTS)
    md = render_report(verdicts, results, rules, notes)
    out = save_gate2_report(cfg.data_dir / "reports" / f"gate2-{datetime.now().strftime('%Y%m%d-%H%M%S')}", verdicts, results, md)
    print()
    print(md)
    print()
    print(f"Report folder: {out}")
    passed = [v.strategy for v in verdicts if v.passed]
    print(f"GATE 2: {', '.join(passed) + ' PASSED' if passed else 'no strategy passed'}")


async def cmd_gate2_apply(args) -> None:
    """Enable the strategies that passed Gate 2, on their eligible instruments, with the latest window's parameters."""
    data = json.loads(Path(args.result).read_text())
    enabled, params = {}, {}
    for v in data["verdicts"]:
        if v["passed"] and v["enabled_symbols"]:
            enabled[v["strategy"]] = v["enabled_symbols"]
            params[v["strategy"]] = {s: parse_params_id(v["latest_params"][s]) for s in v["enabled_symbols"] if s in v["latest_params"]}
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        if args.dry_run:
            print(json.dumps({"strategies_enabled": enabled, "strategy_params": params}, indent=2))
            return
        await eng.settings_store.update({"strategies_enabled": enabled, "strategy_params": params}, "gate2-apply")
        print("Enabled:" if enabled else "Nothing passed Gate 2: all strategies stay off.")
        for s, syms in enabled.items():
            print(f"  {s}: {', '.join(syms)}")
    finally:
        await eng.stop()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="tradeengine", description="Demo-only trading engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    g = sub.add_parser("gate1")
    g.add_argument("--venue", action="append", choices=[v.value for v in Venue])
    sub.add_parser("run")
    sub.add_parser("reconcile")
    k = sub.add_parser("kill")
    k.add_argument("reason")
    k.add_argument("--flatten", action="store_true", help="also close every engine position")
    sub.add_parser("unkill")
    st = sub.add_parser("settings")
    st.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    d = sub.add_parser("download")
    d.add_argument("--source", choices=["binance", "capital", "dukascopy"], required=True)
    d.add_argument("--symbol", required=True, help='e.g. "BTC/USDT" or "EUR/USD"')
    d.add_argument("--tf", default="M1", choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
    d.add_argument("--days", type=int, default=30)
    da = sub.add_parser("download-all")
    da.add_argument("--days", type=int, default=1100)
    da.add_argument("--no-dukascopy", action="store_true")
    sub.add_parser("gate2")
    ga = sub.add_parser("gate2-apply")
    ga.add_argument("result", help="data/reports/gate2-.../result.json")
    ga.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    handler = {
        "status": cmd_status,
        "gate1": cmd_gate1,
        "run": cmd_run,
        "reconcile": cmd_reconcile,
        "kill": cmd_kill,
        "unkill": cmd_unkill,
        "settings": cmd_settings,
        "download": cmd_download,
        "download-all": cmd_download_all,
        "gate2": cmd_gate2,
        "gate2-apply": cmd_gate2_apply,
    }[args.cmd]
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
