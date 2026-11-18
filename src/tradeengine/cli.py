"""Command line: `tradeengine <command>`.

  status                  accounts, open positions, kill switch
  gate1 [--venue ...]     live demo checks (opens and closes small demo positions)
  run                     market data + reconciler + equity tracking + hourly strategy runner
  reconcile               one reconcile pass
  kill "reason" [--flatten]   engage the kill switch (and close all engine positions)
  unkill                  release the kill switch
  settings [--set k=v]    show or change admin settings
  download ...            historical data for one symbol
  download-all [--from 2013-01-01]   hourly history for every allowed instrument
  instruments             check each instrument at the broker: code, sizes, live spread vs backtest cost
  gate2                   round 1: walk-forward backtest of the hourly strategies, Gate 2 verdict + report
  round2 --dev            round 2 development run (check 1) + frozen basket
  round2 --holdout <dev result.json>   round 2 holdout (check 2): ONE run per strategy, ever
  round3 --dev / --holdout <json>      round 3 (crypto only), same rules of play
  gate2-apply <json> [--only S]   switch on the strategies that passed (Gate 2 or a round holdout result)
  telegram-setup          link the alerts bot to your Telegram chat (send the bot a message first)
  telegram-test           send a test message to the linked chat (checks the token and the chat id)
  round4-kronos --forecast [--limit N]   round 4 Track B: build the Kronos forecasts (resumable; needs --extra kronos)
  round4-kronos --score                  round 4 Track B: the one look, Kronos vs C2 over Aug 2025 - Oct 2026
  round4-follow --select                 round 4 Track A: pick the traders to follow (desktop; writes a JSON file)
  round4-follow --freeze <selection.json>  store that selection on the server, once; the shadow record starts
  round4-follow --status | --score       Track A shadow tally; the one-look score (after 90 days and 40 trades)
  round5 --download                      round 5: Binance BTC/ETH perpetual candles + funding history (no keys)
  round5 --dev                           round 5: development look, C2's short side (Mar 2020 - Oct 2023)
  round5 --holdout                       round 5: the one unseen look (Nov 2023 - Oct 2026), after a development PASS
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
    download_dukascopy_h1,
    history_path,
    read_frame,
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
    from .alerting import is_transient

    eng = _engine()
    try:
        try:
            await eng.start()
        except Exception as e:
            if not is_transient(e):
                raise
            sys.exit(f"Broker not reachable right now ({e}). The engine keeps retrying on its own; "
                     "try `status` again later.")
        halted, why = await eng.state.is_halted()
        print(f"Kill switch: {'ENGAGED - ' + why if halted else 'off'}")
        for sym, why_not in eng.unavailable.items():
            print(f"Left out (broker does not serve it): {sym}: {why_not}")
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
        await eng.start_with_retry()  # a broker outage at start-up is waited out in-process, with Telegram alerts
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
        await _download_dukascopy(cfg, args.symbol, start, end)  # always hourly
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


def cross_check(existing, new, spread: float) -> tuple[int, float, float]:
    """Overlapping hours, max close difference, share of hours differing by more than one spread."""
    if existing is None or new is None or existing.empty or new.empty:
        return 0, 0.0, 0.0
    both = existing[["close"]].rename(columns={"close": "close_old"}).join(new[["close"]].rename(columns={"close": "close_new"}), how="inner")
    if both.empty:
        return 0, 0.0, 0.0
    diff = (both["close_old"] - both["close_new"]).abs()
    return len(both), float(diff.max()), float((diff > spread).mean())


async def _download_dukascopy(cfg, symbol: str, start, end) -> bool:
    """Hourly Dukascopy history, resumable: months already complete on disk are skipped (except the
    latest, re-fetched as a check). New hours are compared with the hours already on disk before they
    are merged in. Months that arrived are saved even if others failed; returns False if any failed."""
    from .backtest import cost_for
    from .history import complete_months

    path = history_path(cfg.data_dir, "dukascopy", symbol, "H1")
    existing = read_frame(path)
    have = complete_months(existing)
    print(f"  dukascopy {symbol:<9} downloading ({len(have)} month(s) already on disk) ...", flush=True)
    async with httpx.AsyncClient(timeout=30) as client:
        bars, stats = await download_dukascopy_h1(
            client, symbol, start.date(), end.date() + timedelta(days=1), have=have,
            progress=lambda msg: print(f"  dukascopy {symbol:<9} {msg}", flush=True),
        )
    first = bars[0].open_time.date() if bars else None
    where = (f"{stats['monthly']} monthly files, {stats['from_minutes']} month(s) from minute files, "
             f"{stats.get('skipped', 0)} already on disk")
    n_overlap, max_diff, share = cross_check(existing, bars_to_frame(bars) if bars else None, cost_for(symbol).spread or 0.0)
    check = ""
    if n_overlap:
        check = f"; matches the {n_overlap:,} hours already on disk (max close difference {max_diff:.5g})"
        if share > 0.01:
            print(f"  dukascopy {symbol:<9} NOT SAVED: {share:.1%} of the {n_overlap:,} overlapping hours differ from the file "
                  f"on disk by more than one spread (max {max_diff:.5g}). Send this line to Claude.")
            return False
    n = write_bars(path, bars) if bars else (len(existing) if existing is not None else 0)
    print(f"  dukascopy {symbol:<9} H1: {len(bars):,} new bars from {first} ({where}){check}; file {n:,} rows")
    if stats.get("failed"):
        shown = "; ".join(stats["failed"][:3]) + (f"; ... {len(stats['failed']) - 3} more" if len(stats["failed"]) > 3 else "")
        print(f"  dukascopy {symbol:<9} {len(stats['failed'])} month(s) still missing ({shown}). "
              "What arrived is saved: run the same command again to fetch the rest.")
        return False
    return True


async def cmd_download_all(args) -> None:
    """Hourly history for every allowed instrument: Binance (crypto); Dukascopy + Capital.com (Forex, gold)."""
    logging.getLogger("httpx").setLevel(logging.WARNING)  # thousands of requests: print a summary line per file instead
    cfg = load_config()
    end = datetime.now(UTC)
    start = datetime.fromisoformat(args.from_date).replace(tzinfo=UTC) if args.from_date else end - timedelta(days=args.days)
    eng = _engine()
    try:
        await eng.start()
        settings = await eng.settings_store.load()
        from .round2 import ROUND2_SYMBOLS

        failed: list[str] = []
        for sym in dict.fromkeys([*settings.instruments, *ROUND2_SYMBOLS]):  # the round 2 basket even if not traded
            inst = DEFAULT_INSTRUMENTS.get(sym)
            if inst is None:
                continue
            adapter = eng.adapters.get(inst.venue)
            if adapter is not None and sym not in eng.unavailable:
                # Capital.com's demo API keeps only ~12 days of hourly prices: ask for 30, not years of empty windows.
                b_start = max(start, end - timedelta(days=30)) if inst.venue is Venue.CAPITAL else start
                try:
                    await _download_broker(cfg, adapter, inst.venue.value, inst, "H1", b_start, end)
                except Exception as e:
                    print(f"  {inst.venue.value:<9} {sym:<9} failed: {e}")
                    if inst.venue is Venue.BINANCE:
                        failed.append(f"binance {sym}")  # Binance is crypto's only history source
            if inst.kind.value in ("fx", "metal") and not args.no_dukascopy:
                try:
                    if not await _download_dukascopy(cfg, sym, start, end):
                        failed.append(f"dukascopy {sym}")
                except Exception as e:
                    print(f"  dukascopy {sym:<9} failed: {e}")
                    failed.append(f"dukascopy {sym}")
    finally:
        await eng.stop()
    if failed:
        # Non-zero exit, so scripts/round2.sh stops here instead of running a backtest on missing data.
        sys.exit(f"Download incomplete ({len(failed)}): {', '.join(failed)}. Send this output to Claude.")


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


async def cmd_instruments(_args) -> None:
    """Each instrument at its broker: code, dealing rules, live spread against the backtest's cost."""
    from .backtest import cost_for

    eng = _engine()
    try:
        await eng.start()
        settings = await eng.settings_store.load()
        from .round2 import ROUND2_SYMBOLS

        for sym in dict.fromkeys([*settings.instruments, *ROUND2_SYMBOLS]):
            inst = eng.instruments.get(sym) or DEFAULT_INSTRUMENTS.get(sym)
            if inst is None:
                print(f"{sym:<9} unknown symbol")
                continue
            if inst.venue not in eng.adapters:
                print(f"{sym:<9} {inst.venue.value}: no keys loaded")
                continue
            if sym in eng.unavailable:
                print(f"{sym:<9} {inst.venue.value} code {inst.broker_symbol}: NOT FOUND ({eng.unavailable[sym]})")
                search = getattr(eng.adapters[inst.venue], "search_markets", None)
                if search is not None:
                    term = "gold" if inst.base == "XAU" else f"{inst.base}/{inst.quote}"
                    for m in (await search(term))[:8]:
                        print(f"            candidate: {m['epic']:<12} {m['instrumentName']} ({m['instrumentType']}, {m['marketStatus']})")
                continue
            spec, q = eng.specs[sym], eng.ticks.last.get(sym)
            line = (f"{sym:<9} {inst.venue.value} code {inst.broker_symbol:<8} min size {spec.min_size} step {spec.size_step} "
                    f"tick {spec.price_tick} lot {spec.contract_multiplier} {'tradeable' if spec.tradeable else 'NOT tradeable now'}")
            if q is not None:
                live = float(q.ask - q.bid)
                c = cost_for(sym, settings.cost_model)
                model = c.spread + float(q.mid) * c.spread_bps / 10000
                verdict = "ok" if live <= model else "WIDER than the backtest cost: raise cost_model before the holdout"
                line += f"; spread now {live:.6g} vs backtest {model:.6g} ({verdict})"
            print(line)
        print("\nOne spread reading is a snapshot; check at a busy hour (London/New York overlap), not at the daily roll-over.")
    finally:
        await eng.stop()


async def cmd_round2(args) -> None:
    await _cmd_round(args, 2)


async def cmd_round3(args) -> None:
    await _cmd_round(args, 3)


async def cmd_round5(args) -> None:
    """Round 5: C2's short side on Binance USDⓈ-M perpetuals (pre-registered in round5.py)."""
    from . import round5 as r5

    cfg = load_config()
    if args.download:
        await r5.download(cfg.data_dir)
        return
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        settings = await eng.settings_store.load()
        if args.dev:
            ok, out = await r5.dev(cfg.data_dir, settings, eng.db)
        else:
            print("UNSEEN LOOK: one look, ever, at Nov 2023 - Oct 2026. The result is locked once shown.")
            if not args.yes and input("Type HOLDOUT to run it now: ").strip() != "HOLDOUT":
                sys.exit("Not run.")
            ok, out = await r5.holdout(cfg.data_dir, settings, eng.db)
        print((out / "report.md").read_text())
        print(f"Report folder: {out}")
    finally:
        await eng.stop()


async def cmd_round4_follow(args) -> None:
    """Round 4 Track A: select (desktop), freeze (server, once), status, score (once)."""
    from . import round4_follow as r4f

    cfg = load_config()
    if args.select:
        async with httpx.AsyncClient(timeout=60, headers={"User-Agent": "tradeengine-follow"}) as client:
            sel = await r4f.select_traders(r4f.Hyperliquid(client), r4f.FuturesMarket(client))
        out = Path(args.out or cfg.data_dir / "reports" / f"round4-follow-selection-{datetime.now():%Y%m%d-%H%M%S}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sel, indent=2))
        print(f"Scanned {sel['scanned']}: {sel['qualified']} qualify, {len(sel['traders'])} chosen; not chosen: {sel['rejected']}")
        if sel["unread"]:
            print(f"Could not read {sel['unread']} accounts even after a retry: {sel['unread_errors']}")
        if sel["unread"] > r4f.MAX_UNREAD_SHARE * sel["scanned"]:
            print(f"INCOMPLETE: more than {r4f.MAX_UNREAD_SHARE:.0%} unread, so this list cannot be frozen. Run --select again later.")
        for t in sel["traders"]:
            print(f"  {t['address']} {(t.get('name') or '')[:20]:<20} realised ${t['realised_pnl']:>14,.0f} | "
                  f"{t['fills_per_day']} fills/day | {t['span_days']} days kept | {t['coverage']:.0%} on Binance futures")
        print(f"Saved {out}." + ("" if sel["unread"] > r4f.MAX_UNREAD_SHARE * sel["scanned"]
                                 else f" Next: ./scripts/vm.sh follow-freeze {out}"))
        return
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        if args.freeze:
            try:
                v = await r4f.freeze(eng.db, json.loads(Path(args.freeze).read_text()))
            except (ValueError, RuntimeError) as e:
                sys.exit(f"Not frozen: {e}")
            print(f"Frozen {len(v['traders'])} traders at {v['frozen_at']}. The hourly shadow record starts with the engine's next hour.")
            return
        st = await r4f.stats(eng.db)
        if st is None:
            sys.exit("Track A has no frozen selection yet (round4-follow --select on your Mac, then vm.sh follow-freeze).")
        if args.status:
            print(r4f.summary_line(st))
            print(f"Recorded for {st.days:.1f} days; the score needs {r4f.MIN_RECORD_DAYS} days and {r4f.RULES.holdout_min_trades} closed trades.")
            return
        try:
            ok, why, st = await r4f.score(eng.db)
        except RuntimeError as e:
            sys.exit(str(e))
        print(r4f.summary_line(st))
        print("TRACK A: PASS" if ok else "TRACK A: FAIL - " + "; ".join(why))
    finally:
        await eng.stop()


async def cmd_round4_kronos(args) -> None:
    """Round 4 Track B. --forecast builds and caches forecasts (stop and rerun at will); --score is the one look."""
    from . import round4_kronos as r4k

    cfg = load_config()
    if args.forecast:
        data, notes = r4k.load_data(cfg.data_dir, DEFAULT_INSTRUMENTS)
        for n in notes:
            print(n)
        daily = {s: r4k.daily_frame(data[s][0]) for s in r4k.SYMBOLS}
        print(f"Loading {r4k.MODEL_ID} (downloaded from Hugging Face the first time) ...")
        fc = r4k.Forecaster(device=args.device)
        print(f"Running on {fc.device}.")
        r4k.make_forecasts(daily, fc, r4k.cache_path(cfg.data_dir), limit=args.limit)
        left = {s: n for s, n in r4k.missing_forecasts(daily, r4k.load_cache(r4k.cache_path(cfg.data_dir))).items() if n}
        print("Every forecast is ready: run `uv run tradeengine round4-kronos --score` (the one look)." if not left
              else f"Still to do: {left}. Run the same command again to continue.")
        return
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        settings = await eng.settings_store.load()
        print("SCORE: one look, ever, at Kronos over Aug 2025 - Oct 2026. The result is locked once shown.")
        if not args.yes and input("Type SCORE to run it now: ").strip() != "SCORE":
            sys.exit("Not run.")
        v, out = await r4k.score(cfg.data_dir, settings, DEFAULT_INSTRUMENTS, eng.db)
        print((out / "report.md").read_text())
        print(f"Report folder: {out}")
    finally:
        await eng.stop()


async def _cmd_round(args, number: int) -> None:
    """A pre-registered research round: `--dev` (check 1) or `--holdout <dev result.json>` (check 2, once)."""
    from . import round2 as r2
    from . import round3 as r3

    pl = r2.plan() if number == 2 else r3.plan()
    rname = pl.name
    cfg = load_config()
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        settings = await eng.settings_store.load()
        fp = r2.fingerprint(settings, pl)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.dev:
            print(f"Round {number} development run (data to {pl.dev_end:%Y-%m-%d}; this takes a few minutes):")
            try:
                runs, verdicts, notes = r2.run_dev(cfg.data_dir, settings, DEFAULT_INSTRUMENTS, pl=pl)
            except r2.IncompleteData as e:
                sys.exit(f"Not run: {e}")
            n = await r2.count_dev_run(eng.db, pl)
            extra = [f"Development run number {n} (every run is counted)."]
            md = r2.render("dev", runs, verdicts, notes, fp, extra, r2.Limits.from_settings(settings), pl)
            out = r2.save(cfg.data_dir / "reports" / f"{rname}-dev-{stamp}", md, r2.dev_payload(runs, verdicts, fp, n, pl), runs)
            print()
            print(md)
            print(f"\nReport folder: {out}")
            passed = [r.strategy for r in runs if verdicts[r.strategy]["passed"]]
            print(f"CHECK 1: {', '.join(passed) + ' passed - eligible for the holdout' if passed else 'no strategy passed; all stay off'}")
            return
        dev_path = Path(args.holdout)
        dev = json.loads(dev_path.read_text())
        if dev.get("kind") != f"{rname}-dev":
            sys.exit(f"{dev_path} is not a round {number} development result.json")
        if dev.get("fingerprint") != fp:
            sys.exit(f"Code or settings changed since that development run (fingerprint {dev.get('fingerprint')} then, {fp} now). "
                     f"Run `tradeengine {rname} --dev` again first.")
        eligible = [name for name, d in dev["strategies"].items() if d["passed"]]
        if not eligible:
            sys.exit("No strategy passed development, so there is nothing to run on the holdout. All strategies stay off.")
        done = await r2.all_holdout_records(eng.db, pl)
        for name in eligible:
            if name in done:
                print(f"{name}: the holdout was already run on {done[name].get('run_at', '?')[:16]} "
                      f"(report {done[name].get('report', '?')}). A second look is not allowed.")
        todo = [n for n in eligible if n not in done]
        if not todo:
            sys.exit("Nothing left to run.")
        print(f"HOLDOUT: one run, ever, for: {', '.join(todo)}. Baskets: "
              + "; ".join(f"{n}: {', '.join(dev['strategies'][n]['basket']) or 'none'}" for n in todo))
        if not args.yes and input("Type HOLDOUT to run it now: ").strip() != "HOLDOUT":
            sys.exit("Not run.")
        dev_todo = {**dev, "strategies": {n: d for n, d in dev["strategies"].items() if n in todo}}
        runs, verdicts, notes = r2.run_holdout(cfg.data_dir, settings, DEFAULT_INSTRUMENTS, dev_todo, pl=pl)
        out_dir = cfg.data_dir / "reports" / f"{rname}-holdout-{stamp}"
        for run in runs:  # recorded BEFORE anything is shown, so the result cannot be looked at and re-rolled
            v = verdicts[run.strategy]
            await r2.record_holdout(eng.db, run.strategy, {
                "fingerprint": fp, "run_at": datetime.now(UTC).isoformat(), "passed": v["passed"], "reasons": v["reasons"],
                "basket": v["basket"], "trades": run.metrics.trades, "profit_factor": run.metrics.profit_factor,
                "max_drawdown_pct": run.metrics.max_drawdown_pct, "report": str(out_dir), "dev_result": str(dev_path),
            }, pl)
        md = r2.render("holdout", runs, verdicts, notes, fp, [f"Development result: `{dev_path}`"], r2.Limits.from_settings(settings), pl)
        out = r2.save(out_dir, md, r2.holdout_payload(runs, verdicts, fp, str(dev_path), pl), runs)
        print()
        print(md)
        print(f"\nReport folder: {out}")
        passed = [r.strategy for r in runs if verdicts[r.strategy]["passed"]]
        print(f"CHECK 2: {', '.join(passed) + ' PASSED' if passed else 'no strategy passed; all stay off'}")
        if passed:
            print(f"To switch on for demo trading: uv run tradeengine gate2-apply {out / 'result.json'} --dry-run")
    finally:
        await eng.stop()


async def cmd_gate2_apply(args) -> None:
    """Enable the strategies that passed Gate 2, on their eligible instruments, with the latest window's parameters."""
    data = json.loads(Path(args.result).read_text())
    enabled, params = {}, {}
    known = {v["strategy"] for v in data["verdicts"]}
    if args.only and not set(args.only) <= known:
        sys.exit(f"--only {', '.join(sorted(set(args.only) - known))}: not in this result (it has {', '.join(sorted(known))})")
    for v in data["verdicts"]:
        if args.only and v["strategy"] not in args.only:
            continue
        if v["passed"] and v["enabled_symbols"]:
            enabled[v["strategy"]] = v["enabled_symbols"]
            params[v["strategy"]] = {s: parse_params_id(v["latest_params"][s]) for s in v["enabled_symbols"] if s in v["latest_params"]}
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        if args.dry_run:
            print(json.dumps({"strategies_enabled": enabled, "strategy_params": params}, indent=2))
            return
        settings = await eng.settings_store.update({"strategies_enabled": enabled, "strategy_params": params}, "gate2-apply")
        if enabled:  # the forward test's clock starts now (the daily Telegram summary counts its trades from here)
            from .notify import FORWARD_START, _set_state

            await _set_state(eng.db, FORWARD_START, {"at": datetime.now(UTC).isoformat(), "strategies": sorted(enabled),
                                                     "result": str(args.result)})
        print("Enabled:" if enabled else "Nothing passed: all strategies stay off.")
        for s, syms in enabled.items():
            print(f"  {s}: {', '.join(syms)}")
        missing = sorted({sym for syms in enabled.values() for sym in syms} - set(settings.instruments))
        if missing:
            print(f"WARNING: not on the allowed instrument list, so the risk manager will refuse them: {', '.join(missing)}. "
                  "Add them with `tradeengine settings --set instruments='[...]'`.")
    finally:
        await eng.stop()


async def cmd_telegram_setup(_args) -> None:
    """Find the chat that messaged the bot, save its id, switch alerts on and send a test message."""
    from .notify import find_chat_id, send

    cfg = load_config()
    if cfg.telegram_bot_token is None:
        sys.exit("TELEGRAM_BOT_TOKEN is not set (run through scripts/with-secrets.sh, or on the server).")
    token = cfg.telegram_bot_token.get_secret_value()
    async with httpx.AsyncClient(timeout=15) as client:
        found = await find_chat_id(client, token)
        if found is None:
            sys.exit("No message found. Open the bot in Telegram, send it any message (e.g. /start), then run this again.")
        chat_id, who = found
        await send(client, token, chat_id, "✅ Trading engine connected. You will get trade alerts, problems and a daily summary here.")
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        await eng.settings_store.update({"telegram_chat_id": chat_id, "telegram_alerts_enabled": True}, "telegram-setup")
    finally:
        await eng.stop()
    print(f"Telegram alerts on: chat {chat_id} ({who}). A test message was sent.")


async def cmd_telegram_test(_args) -> None:
    """Send one test message with the saved chat id and the current token. Changes nothing."""
    import socket

    from .notify import TelegramError, send

    cfg = load_config()
    if cfg.telegram_bot_token is None:
        sys.exit("TELEGRAM_BOT_TOKEN is not set (run through scripts/with-secrets.sh, or on the server).")
    eng = _engine(need_brokers=False)
    try:
        await eng.db.create_all()
        settings = await eng.settings_store.load()
    finally:
        await eng.stop()
    if not settings.telegram_chat_id:
        sys.exit("No chat linked yet. Send the bot any message in Telegram, then run `tradeengine telegram-setup`.")
    text = (f"🔔 Test from the trading engine on {socket.gethostname()} "
            f"({datetime.now(UTC):%Y-%m-%d %H:%M} UTC). Alerts are "
            f"{'ON' if settings.telegram_alerts_enabled else 'OFF'}.")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await send(client, cfg.telegram_bot_token.get_secret_value(), settings.telegram_chat_id, text)
    except TelegramError as e:
        sys.exit(f"Telegram refused the test message: {e}. "
                 "A 401 means the token is wrong or revoked; a 400/403 means the chat id is wrong or the bot was blocked.")
    print(f"Test message sent to chat {settings.telegram_chat_id}. Check Telegram.")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .logsafe import install as _redact_logs

    _redact_logs()  # bot tokens never reach the terminal or the server's journal
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
    da.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD", help="start date (overrides --days)")
    da.add_argument("--no-dukascopy", action="store_true")
    sub.add_parser("instruments")
    sub.add_parser("gate2")
    r2p = sub.add_parser("round2")
    mode = r2p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true", help="development run (check 1)")
    mode.add_argument("--holdout", metavar="DEV_RESULT_JSON", help="the one holdout run (check 2)")
    r2p.add_argument("--yes", action="store_true", help="skip the HOLDOUT confirmation")
    r3p = sub.add_parser("round3")
    mode3 = r3p.add_mutually_exclusive_group(required=True)
    mode3.add_argument("--dev", action="store_true", help="crypto development run (check 1)")
    mode3.add_argument("--holdout", metavar="DEV_RESULT_JSON", help="the one crypto holdout run (check 2)")
    r3p.add_argument("--yes", action="store_true", help="skip the HOLDOUT confirmation")
    r5 = sub.add_parser("round5")
    mode5 = r5.add_mutually_exclusive_group(required=True)
    mode5.add_argument("--download", action="store_true", help="futures candles and funding history (resumable)")
    mode5.add_argument("--dev", action="store_true", help="the development look")
    mode5.add_argument("--holdout", action="store_true", help="the one unseen look")
    r5.add_argument("--yes", action="store_true", help="skip the HOLDOUT confirmation")
    r4f = sub.add_parser("round4-follow")
    mode4f = r4f.add_mutually_exclusive_group(required=True)
    mode4f.add_argument("--select", action="store_true", help="pick the traders (desktop; ~1 hour, read-only)")
    mode4f.add_argument("--freeze", metavar="SELECTION_JSON", help="store the selection, once")
    mode4f.add_argument("--status", action="store_true", help="the shadow tally so far")
    mode4f.add_argument("--score", action="store_true", help="the one-look score")
    r4f.add_argument("--out", default=None, help="where --select writes its JSON")
    r4k = sub.add_parser("round4-kronos")
    mode4 = r4k.add_mutually_exclusive_group(required=True)
    mode4.add_argument("--forecast", action="store_true", help="build the Kronos forecasts (resumable)")
    mode4.add_argument("--score", action="store_true", help="the one look: Kronos vs C2 over the scored window")
    r4k.add_argument("--limit", type=int, default=None, help="forecast at most this many days this run")
    r4k.add_argument("--device", default=None, help="cpu / mps / cuda:0 (default: picked automatically)")
    r4k.add_argument("--yes", action="store_true", help="skip the SCORE confirmation")
    ga = sub.add_parser("gate2-apply")
    ga.add_argument("result", help="data/reports/gate2-.../result.json")
    ga.add_argument("--dry-run", action="store_true")
    ga.add_argument("--only", action="append", metavar="STRATEGY", help="switch on only this strategy (repeatable)")
    sub.add_parser("telegram-setup")
    sub.add_parser("telegram-test")
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
        "instruments": cmd_instruments,
        "gate2": cmd_gate2,
        "round2": cmd_round2,
        "round3": cmd_round3,
        "round4-kronos": cmd_round4_kronos,
        "round4-follow": cmd_round4_follow,
        "round5": cmd_round5,
        "telegram-setup": cmd_telegram_setup,
        "telegram-test": cmd_telegram_test,
        "gate2-apply": cmd_gate2_apply,
    }[args.cmd]
    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
