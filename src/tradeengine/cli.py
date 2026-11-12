"""Command line: `tradeengine <command>`.

  status                  accounts, open positions, kill switch
  gate1 [--venue ...]     live demo checks (opens and closes small demo positions)
  run                     market data + reconciler + equity tracking (strategies arrive in Phase 2)
  reconcile               one reconcile pass
  kill "reason" [--flatten]   engage the kill switch (and close all engine positions)
  unkill                  release the kill switch
  settings [--set k=v]    show or change admin settings
  download ...            historical data for backtesting
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import ValidationError
from sqlalchemy import select

from .config import load_config
from .db import PositionRow
from .engine import Engine
from .gate1 import Gate1, print_report, save_report
from .history import download_dukascopy, history_path, write_bars
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
        async with httpx.AsyncClient(timeout=30) as client:
            bars = await download_dukascopy(client, args.symbol, start.date(), end.date() + timedelta(days=1))
        tf = "M1"
    else:
        if inst is None:
            sys.exit(f"unknown symbol {args.symbol}; known: {', '.join(DEFAULT_INSTRUMENTS)}")
        eng = _engine()
        try:
            await eng.start()
            adapter = eng.adapters.get(Venue.CAPITAL if args.source == "capital" else Venue.BINANCE)
            if adapter is None:
                sys.exit(f"no keys for {args.source}")
            bars = await adapter.history(inst, args.tf, start, end)
        finally:
            await eng.stop()
        tf = args.tf
    path = history_path(cfg.data_dir, args.source, args.symbol, tf)
    n = write_bars(path, bars)
    first = bars[0].open_time if bars else None
    print(f"{len(bars)} bars downloaded (from {first}); {n} rows in {path}")


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
    }[args.cmd]
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
