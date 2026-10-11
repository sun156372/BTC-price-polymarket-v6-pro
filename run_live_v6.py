"""V6 live paper-trading loop.

    python run_live_v6.py probe                  # verify the feed and depth first
    python run_live_v6.py run --stake 10         # paper trade, log everything
    python run_live_v6.py report                 # win rate, profit factor, drawdown

PAPER ONLY, STRUCTURALLY. This process never signs or sends an order. It reads public
market data, evaluates the strategy's gates, records what it WOULD have done, and
scores the result against the venue's own resolution. Cash is a float in a log file.

What changed from V4, and which problem each change addresses
------------------------------------------------------------
1. **Frequency gap.** V4 polled once per second in a loop that also did three network
   calls, so its effective cadence was ~2s, and it recorded the window-open mid only if
   a pass happened to land in the first 25 seconds. Every gate is a difference between
   an open value and a decision value, so a late open shrinks all of them and all three
   gates over-reject. V6:
     * samples spot continuously on a separate thread (``SpotTracker``)
     * records the Up-token mid as close to the true window open as it can, and reports
       the lag it achieved
     * hits the decision second with a short sleep, not a coarse poll
     * logs every gate's MEASURED VALUE for every window, so the live distribution can
       be compared against the backtest's rather than inferred from refusal counts

2. **Depth.** The imbalance threshold was fitted to the dataset's top-10 book, so the
   live book must be measured the same way. The recorder stores imbalance at several
   depth caps on every decision, and ``probe`` prints the live depth against the
   dataset's. If the live book is shallower, the threshold does not transfer and the
   probe says so before a session starts.

3. **Reporting.** Sessions end with win rate, profit factor, expectancy, max drawdown
   and an equity curve, computed by ``engine_v6.metrics`` from the settled trades.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "tools_v6"))

from dataclasses import replace

from engine_v6.feed import (  # noqa: E402
    DATASET_DEPTH, FeedError, PolymarketFeed, SLUG_TEMPLATE, SpotFeed,
    WINDOW_SECONDS,
)
from engine_v6.metrics import TradeRecord, format_report, from_trades  # noqa: E402
from engine_v6.recorder import SessionRecorder, SpotTracker  # noqa: E402
from engine_v6.signals import net_pnl, taker_fee_per_share  # noqa: E402
from engine_v6.strategy import (  # noqa: E402
    READY, V6Params, StrategyV6,
)

DEFAULT_LOG_DIR = os.path.join(PKG, "paper_v6")

#: How early to start the decision sequence, relative to the target second.
WAKE_SLACK_S = 2.5
#: Time reserved for the book fetch that must complete BY the target second.
#: A public REST round trip measured 0.6-1.2s in the live session, so the wake has to
#: lead the target by at least this much or the trade is taken too early.
BOOK_FETCH_BUDGET_S = 1.5

DECISION_FIELDS = [
    "decided_at_utc", "slug", "secs_to_close", "action", "reason", "side",
    "up_mid_open", "up_mid_now", "move", "open_lag_s",
    "book_imbalance_10", "imbalance_5", "imbalance_20", "imbalance_full",
    "bid_notional_10", "ask_notional_10", "bid_depth", "ask_depth",
    "spot_open", "spot_now", "spot_move_bps", "spot_open_lag_s", "spot_source",
    "ask", "spread", "ask_1s", "ask_2s", "breaks_even_at", "shares",
]
SETTLE_FIELDS = [
    "slug", "day", "side", "resolved", "won", "entry", "shares",
    "pnl", "per_share", "pnl_at_fill_2s", "fee_per_share", "settled_at_utc",
]
DROP_FIELDS = [
    "decided_at_utc", "slug", "reason", "up_mid_open", "up_mid_now", "move",
    "book_imbalance_10", "spot_move_bps",
]


def utc(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def current_window_start(now: float | None = None) -> int:
    return int((now or time.time()) // WINDOW_SECONDS) * WINDOW_SECONDS


def log_path(log_dir: str, name: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, name)


def append_row(path: str, fields: list[str], row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------- probe


def cmd_probe(params: V6Params, log_dir: str) -> int:
    print("=" * 92)
    print("V6 FEED AND DEPTH PROBE - run this before any session")
    print("=" * 92)
    print("  The live endpoint's order book depth is unknown and unverifiable from")
    print("  here, and the strategy's core gate depends on it. This probe measures")
    print("  what the endpoint actually returns and compares it to the dataset.")
    print()
    problems = 0

    print("1. SPOT")
    spot_feed = SpotFeed()
    try:
        quote = spot_feed.quote()
    except FeedError as exc:
        quote = None
        print(f"   [FAIL] {exc}")
    if quote is None:
        print("   [FAIL] no spot quote from coinbase or binance")
        problems += 1
    else:
        print(f"   [ok] BTC/USD {quote.price:,.2f} from {quote.source} "
              f"(age {quote.age():.2f}s)")
        print("        NOTE: the research used a 100ms feed. The spot gate needs a")
        print("        4.5bps move, which accrues in ~6s, so polling cadence matters")
        print("        and is measured in the session log, not assumed.")
    print()

    feed = PolymarketFeed()
    epoch = current_window_start()
    print(f"2. MARKET for the window opening {utc(epoch)}")
    market = None
    try:
        market = feed.market_for_epoch(epoch)
    except FeedError as exc:
        print(f"   [FAIL] {exc}")
        problems += 1
    if market is None:
        print("   [FAIL] market not found - is the 5m series live and is the slug")
        print("          format still btc-updown-5m-<epoch>?")
        problems += 1
    else:
        print(f"   [ok] {market.slug}")
        print(f"        up_token   {market.up_token[:28]}...")
        print(f"        down_token {market.down_token[:28]}...")
        print(f"        outcomes   {market.outcomes}")
        print(f"        window     {utc(market.start_ts)} -> {utc(market.end_ts)}")
    print()

    print("3. ORDER BOOK DEPTH  (the question that decides whether 0.30 transfers)")
    if market is not None:
        for label, token in (("UP", market.up_token), ("DOWN", market.down_token)):
            try:
                book = feed.book(token)
            except FeedError as exc:
                print(f"   [FAIL] {label}: {exc}")
                problems += 1
                continue
            span = book.depth_span()
            print(f"   {label}: {book.bid_depth} bid / {book.ask_depth} ask levels"
                  f"   best {book.best_bid} / {book.best_ask}")
            if span:
                print(f"        levels reach {span[0]:.4f} below / {span[1]:.4f} "
                      f"above the touch")
            print(f"        notional top-{DATASET_DEPTH}   bid "
                  f"{book.notional('bids', DATASET_DEPTH):>10,.2f}   ask "
                  f"{book.notional('asks', DATASET_DEPTH):>10,.2f}")
            print(f"        imbalance top-{DATASET_DEPTH} {book.imbalance(DATASET_DEPTH)}"
                  f"   whole book {book.imbalance()}")
        print()
        print("   DATASET REFERENCE, so the live numbers can be judged:")
        print(f"        levels per side            : 10 (82% of snapshots saturate)")
        print(f"        bid notional median        : ~3,620")
        print(f"        ask notional median        : ~4,585")
        print(f"        |imbalance| median         : 0.154")
        print(f"        |imbalance| >= 0.30 passes : 29.0% of windows")
        print()
        print("   VERDICT RULE: if the live level counts are ~10 per side and the")
        print("   top-10 notional is in the same thousands, the 0.30 threshold")
        print("   transfers. If the live book is markedly shallower, or the notional")
        print("   is an order of magnitude smaller, the threshold does NOT transfer")
        print("   and must be refitted on live data before the session means anything.")
    print()

    print("4. DOES THE ENDPOINT HONOUR A LARGER DEPTH REQUEST?")
    if market is not None:
        try:
            deep = feed.book(market.up_token, depth=100)
            print(f"   requested depth=100 -> {deep.bid_depth} bid / "
                  f"{deep.ask_depth} ask levels")
            if max(deep.bid_depth, deep.ask_depth) <= DATASET_DEPTH:
                print("   -> the parameter is IGNORED or the book is capped: the")
                print("      endpoint returns a top-10 view. That matches the dataset,")
                print("      so the gate is comparable. Full depth is NOT obtainable")
                print("      from this endpoint.")
            else:
                print("   -> the parameter is honoured; deeper books are available.")
                print("      The dataset's 10-level cap means the gate was fitted on a")
                print("      SHALLOWER view than the live feed now offers, so the live")
                print("      imbalance will differ in scale and needs refitting.")
        except FeedError as exc:
            print(f"   [FAIL] {exc}")
    print()
    print("=" * 92)
    if problems:
        print(f"PROBE FAILED with {problems} problem(s). Do not start a session.")
        return 1
    print("PROBE PASSED.")
    return 0


# ----------------------------------------------------------------------- run


def cmd_run(params: V6Params, stake: float, log_dir: str, seconds: float,
            spot_interval: float, report_every: int) -> int:
    decisions_path = log_path(log_dir, "decisions.csv")
    settled_path = log_path(log_dir, "settled.csv")
    drops_path = log_path(log_dir, "refusals.csv")

    strategy = StrategyV6(params)
    feed = PolymarketFeed(depth=params.imbalance_depth)
    spot_feed = SpotFeed()
    tracker = SpotTracker(spot_feed, interval=spot_interval)
    recorder = SessionRecorder(WINDOW_SECONDS)

    print("=" * 96)
    print("V6 LIVE PAPER SESSION")
    print("=" * 96)
    for line in strategy.describe():
        print(f"  {line}")
    print(f"  stake {stake:.2f} USDC/trade   NO orders are ever sent")
    print(f"  spot sampling every {spot_interval:.2f}s (separate thread)")
    print(f"  logs: {log_dir}")
    print()
    print("  Ctrl-C to stop. The loop wakes shortly before each decision second.")
    print()

    tracker.start()
    deadline = time.time() + seconds if seconds > 0 else None
    decided: set[str] = set()
    market_cache: dict[int, object] = {}
    last_report = time.time()
    entries = 0
    try:
        while True:
            now = time.time()
            if deadline is not None and now >= deadline:
                print("  session time reached")
                break
            epoch = current_window_start(now)
            slug = SLUG_TEMPLATE.format(epoch=epoch)
            secs_to_close = epoch + WINDOW_SECONDS - now

            market = market_cache.get(epoch)
            if market is None:
                try:
                    market = feed.market_for_epoch(epoch)
                except FeedError as exc:
                    print(f"  [{utc(now)}] market lookup failed: {exc}")
                    time.sleep(2.0)
                    continue
                if market is None:
                    time.sleep(2.0)
                    continue
                market_cache = {epoch: market}
            for stale_epoch in [e for e in market_cache if e < epoch - WINDOW_SECONDS]:
                market_cache.pop(stale_epoch, None)

            # --- record the window open as early as possible -----------------
            if secs_to_close > WINDOW_SECONDS - 20 and slug not in decided:
                try:
                    book = feed.book(market.up_token)
                    recorder.observe(slug, epoch, book.mid, book)
                except FeedError:
                    pass
                time.sleep(spot_interval)
                continue

            # --- approach the decision instant, then decide once -------------
            # The decision needs a book fetch BEFORE the target second, and that
            # fetch costs several hundred milliseconds. The old code allowed only
            # 0.6s of slack, so the observed decision landed 1.2s EARLY - buying a
            # higher ask than the checkpoint the strategy was calibrated on. Since
            # the entry price is the entire edge, that is not a rounding detail.
            # Wake early enough for the fetch to finish at the target.
            target_close = params.entry_t_minus
            if slug not in decided and secs_to_close <= target_close + WAKE_SLACK_S:
                wake_in = secs_to_close - target_close - BOOK_FETCH_BUDGET_S
                if wake_in > 0.02:
                    time.sleep(min(wake_in, 0.5))
                    continue
                decided.add(slug)
                try:
                    _decide_and_log(strategy, feed, tracker, recorder, market, slug,
                                    epoch, params, stake, decisions_path,
                                    drops_path)
                    entries += 1
                except FeedError as exc:
                    print(f"  [{utc()}] {slug}: feed error - {exc}")
                except Exception as exc:                     # noqa: BLE001
                    print(f"  [{utc()}] {slug}: {type(exc).__name__}: {exc}")
                continue

            if report_every and time.time() - last_report > report_every:
                last_report = time.time()
                _progress(strategy, tracker, recorder, entries)
            time.sleep(spot_interval)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        tracker.stop()

    print()
    _session_summary(strategy, tracker, recorder)
    _settle_open_positions(feed, decisions_path, settled_path)
    _report(settled_path, stake)
    return 0


def _decide_and_log(strategy: StrategyV6, feed: PolymarketFeed, tracker: SpotTracker,
                    recorder: SessionRecorder, market, slug: str, epoch: int,
                    params: V6Params, stake: float, decisions_path: str,
                    drops_path: str) -> None:
    """Fetch what the decision needs, evaluate, and log the full measurement."""
    cap = recorder.windows.get(slug)
    decided_at = time.time()
    book = feed.book(market.up_token)
    ask = book.best_ask
    recorder.mark_decision(slug, book.mid, book, ask, book.spread)

    # fill quality: the ask a second and two seconds later
    ask_1s = ask_2s = None
    try:
        time.sleep(1.0)
        ask_1s = feed.book(market.up_token).best_ask
        time.sleep(1.0)
        ask_2s = feed.book(market.up_token).best_ask
    except FeedError:
        pass

    open_point = tracker.at(market.start_ts) if market.start_ts else None
    spot_open = open_point.price if open_point else None
    now_point = tracker.at(decided_at, tolerance=max(2.0, tracker.interval * 4))
    spot_now = now_point.price if now_point else None
    # say so when the spot pair is incomplete, because the spot gate is SKIPPED
    # silently in that case and the window would otherwise look gate-compliant
    spot_incomplete = open_point is None or now_point is None
    if spot_incomplete:
        print(f"  [{utc(decided_at)}] {slug}: SPOT PAIR INCOMPLETE "
              f"(open={'ok' if open_point else 'MISSING'} "
              f"now={'ok' if now_point else 'MISSING'}) - spot gate not applied")

    cap = recorder.windows.get(slug)
    up_mid_open = cap.open_up_mid if cap else None
    up_mid_now = book.mid if ask is not None else None
    # RAW imbalance, not oriented. The strategy orients it itself, using the same
    # direction call that produces the trade. Orienting here as well flipped the sign
    # a SECOND time for every DOWN trade, which inverted the value the gate tested:
    # a genuinely lopsided book (+0.5) read as -0.5 and was refused, while a balanced
    # book could read as lopsided and be accepted. Every DOWN trade was affected.
    raw_imbalance = None
    if cap and ask is not None:
        raw_imbalance = cap.imbalances.get(str(params.imbalance_depth))

    shares = stake / ask if ask else 0.0
    depth_cost = None
    if ask:
        cost = book.cost_to_buy(shares)
        depth_cost = cost if cost is not None else None

    decision = strategy.evaluate(
        slug=slug, now=decided_at,
        secs_to_close=market.end_ts - decided_at,
        up_mid_open=up_mid_open, up_mid_now=up_mid_now,
        book_imbalance=raw_imbalance,
        bid_notional=cap.notionals.get(f"bid_{params.imbalance_depth}") if cap else None,
        ask_notional=cap.notionals.get(f"ask_{params.imbalance_depth}") if cap else None,
        bid_depth=book.bid_depth, ask_depth=book.ask_depth,
        spot_open=spot_open, spot_now=spot_now, ask=ask, spread=book.spread,
        quote_ts=book.fetched_at, shares_wanted=shares, depth_cost=depth_cost,
    )

    row = {
        "decided_at_utc": utc(decided_at), "slug": slug,
        "secs_to_close": round(market.end_ts - decided_at, 2),
        "action": decision.action, "reason": decision.reason,
        "side": decision.side or (cap.side if cap else ""),
        "up_mid_open": up_mid_open, "up_mid_now": up_mid_now,
        "move": None if (up_mid_open is None or up_mid_now is None)
                else round(up_mid_now - up_mid_open, 5),
        "open_lag_s": None if not cap else (round(cap.open_lag_s, 2)
                                            if cap.open_lag_s is not None else None),
        "book_imbalance_10": None if not cap else cap.imbalances.get(
            str(params.imbalance_depth)),
        "imbalance_5": None if not cap else cap.imbalances.get("5"),
        "imbalance_20": None if not cap else cap.imbalances.get("20"),
        "imbalance_full": None if not cap else cap.imbalances.get("full"),
        "bid_notional_10": None if not cap else cap.notionals.get(
            f"bid_{params.imbalance_depth}"),
        "ask_notional_10": None if not cap else cap.notionals.get(
            f"ask_{params.imbalance_depth}"),
        "bid_depth": book.bid_depth, "ask_depth": book.ask_depth,
        "spot_open": spot_open, "spot_now": spot_now,
        "spot_move_bps": None if (not spot_open or not spot_now)
                         else round((spot_now / spot_open - 1.0) * 1e4, 3),
        "spot_open_lag_s": None if (open_point is None or not market.start_ts)
                           else round(open_point.ts - market.start_ts, 2),
        "spot_source": now_point.source if now_point else "",
        "ask": ask, "spread": book.spread, "ask_1s": ask_1s, "ask_2s": ask_2s,
        "breaks_even_at": decision.break_even or None,
        "shares": round(shares, 4),
    }
    append_row(decisions_path, DECISION_FIELDS, row)
    if decision.action != "enter":
        append_row(drops_path, DROP_FIELDS, {
            "decided_at_utc": row["decided_at_utc"], "slug": slug,
            "reason": decision.reason, "up_mid_open": up_mid_open,
            "up_mid_now": up_mid_now, "move": row["move"],
            "book_imbalance_10": row["book_imbalance_10"],
            "spot_move_bps": row["spot_move_bps"],
        })

    marker = "ENTER" if decision.entering else "hold "
    print(f"  [{utc(decided_at)}] {marker} {slug}  {decision.side or '-':<4} "
          f"move={row['move']}  imb={row['book_imbalance_10']}  "
          f"spot={row['spot_move_bps']}bps  open_lag={row['open_lag_s']}s  "
          f"-> {decision.reason}")


def _progress(strategy: StrategyV6, tracker: SpotTracker,
              recorder: SessionRecorder, entries: int) -> None:
    stats = tracker.stats()
    print(f"  -- {utc()} windows={len(strategy.ledger.readings)} "
          f"entries~{entries} spot_samples={stats['samples']} "
          f"spot_gap_med={stats['gap_median_s']}")


def _session_summary(strategy: StrategyV6, tracker: SpotTracker,
                     recorder: SessionRecorder) -> None:
    print()
    print("=" * 96)
    print("SESSION MEASUREMENT SUMMARY")
    print("=" * 96)
    print(f"  windows evaluated : {len(strategy.ledger.readings)}")
    print()
    print("  GATE FUNNEL (reached -> passed):")
    for gate, reached, passed in strategy.ledger.funnel():
        share = (passed / reached * 100) if reached else 0.0
        print(f"    {reached:>5}  {gate:<34} -> {passed:>5} kept  ({share:5.1f}%)")
    print()
    print("  LIVENESS OF THE MEASUREMENT (the V4 gap was here):")
    lag = recorder.open_lag_summary()
    if lag:
        print(f"    window-open mid captured after: median {lag['median_s']:.2f}s  "
              f"p90 {lag['p90_s']:.2f}s  max {lag['max_s']:.2f}s  (n={lag['n']})")
        print("      In the backtest this lag is 0: the open is a real tick. A lag of")
        print("      a few seconds shrinks every measured move and over-rejects.")
    else:
        print("    window-open mid: NEVER captured - all moves would read as zero")
    ss = tracker.stats()
    print(f"    spot samples {ss['samples']}  median gap {ss['gap_median_s']}s  "
          f"p90 gap {ss['gap_p90_s']}s  errors {ss['errors']}")
    print()
    print("  MEASURED DISTRIBUTIONS (compare to the backtest reference):")
    ref = {"move": ("|move|", "median 0.25, >=0.20 in ~73%"),
           "book_imbalance": ("|imbalance|", "median 0.154, >=0.30 in 29.0%"),
           "spot_move": ("|spot move|", ">=4.5bps in ~90%")}
    for field_name, (label, reference) in ref.items():
        dist = strategy.ledger.distribution(field_name)
        if not dist:
            print(f"    {label:<12} no data")
            continue
        print(f"    {label:<12} p10 {dist['p10']:>9.4f}  median {dist['median']:>9.4f}"
              f"  p90 {dist['p90']:>9.4f}    backtest: {reference}")
    print()
    print("  If a live distribution sits well below its backtest reference, that gate")
    print("  is over-rejecting and the frequency gap is a measurement problem, not a")
    print("  market change. That is the hypothesis this log exists to test.")
    print("=" * 96)


def _settle_open_positions(feed: PolymarketFeed, decisions_path: str,
                           settled_path: str) -> None:
    if not os.path.exists(decisions_path):
        return
    with open(decisions_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    settled: set[str] = set()
    if os.path.exists(settled_path):
        with open(settled_path, encoding="utf-8") as handle:
            settled = {r["slug"] for r in csv.DictReader(handle)}
    for row in rows:
        if row["action"] != "enter" or row["slug"] in settled:
            continue
        try:
            epoch = int(row["slug"].rsplit("-", 1)[-1])
            market = feed.market_for_epoch(epoch)
            if market is None or time.time() < market.end_ts + 15:
                continue
            outcome = feed.resolved_outcome(market)
        except (FeedError, ValueError):
            continue
        if outcome is None:
            continue
        side = row["side"]
        won = side == outcome
        entry = float(row["ask"])
        shares = float(row["shares"])
        day = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
        pnl = net_pnl(entry, won, shares)
        ask2 = float(row["ask_2s"]) if row.get("ask_2s") else entry
        append_row(settled_path, SETTLE_FIELDS, {
            "slug": row["slug"], "day": day, "side": side, "resolved": outcome,
            "won": won, "entry": round(entry, 5), "shares": round(shares, 4),
            "pnl": round(pnl, 6), "per_share": round(pnl / shares, 6) if shares else 0,
            "pnl_at_fill_2s": round(net_pnl(ask2, won, shares), 6),
            "fee_per_share": round(taker_fee_per_share(entry), 6),
            "settled_at_utc": utc(),
        })
        print(f"  settled {row['slug']} {side} -> {outcome} "
              f"{'WIN' if won else 'loss'}  pnl {pnl:+.4f} USDC")


def _load_settled(settled_path: str, basis: str) -> list[TradeRecord]:
    """Load settled trades, scored at either the decision ask or the 2s-later ask.

    The settled log stores total P&L per trade at the decision price and, separately,
    total P&L at the 2-second price. The entry price for the latter is derived from
    that P&L and the share count, because the fee depends on the price and cannot be
    backed out in closed form.
    """
    if not os.path.exists(settled_path):
        return []
    out = []
    with open(settled_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                shares = float(row["shares"])
                entry = float(row["entry"])
                won = row["won"].strip() == "True"
                if basis == "fill_2s":
                    pnl = float(row["pnl_at_fill_2s"])
                    entry = _implied_entry(pnl, won, shares)
                    if entry is None:
                        continue
                else:
                    pnl = float(row["pnl"])
            except (KeyError, TypeError, ValueError):
                continue
            if shares <= 0:
                continue
            out.append(TradeRecord(
                slug=row["slug"], day=row.get("day", ""), side=row["side"],
                entry=entry, shares=shares, won=won, pnl=pnl, basis=basis))
    return out


def _implied_entry(pnl: float, won: bool, shares: float) -> float | None:
    """Recover the entry price from a settled net P&L, given the outcome.

    net = shares*(1 - p) - shares*fee(p)  when won
    net = -shares*p     - shares*fee(p)  when lost, fee(p) = c*p*(1-p)

    Both are cubic in p, so bisect on the bracketed range instead of inverting.
    """
    if shares <= 0:
        return None

    def net_at(p: float) -> float:
        return net_pnl(p, won, shares)

    low, high = 0.001, 0.999
    f_low, f_high = net_at(low), net_at(high)
    if (pnl - f_low) * (pnl - f_high) > 0:
        return None                       # not bracketed: P&L is inconsistent
    for _ in range(80):
        mid = (low + high) / 2.0
        if (net_at(mid) - pnl) * (f_low - pnl) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def _report(settled_path: str, stake: float, basis: str = "decision") -> None:
    trades = _load_settled(settled_path, basis)
    print()
    if not trades:
        print("  no settled trades yet - win rate, profit factor and drawdown need")
        print("  at least one resolved window. Re-run `report` after some settle.")
        return
    print(format_report(from_trades(trades),
                        f"PAPER SESSION PERFORMANCE (basis: {basis})", stake=stake))
    if basis == "decision":
        alt = _load_settled(settled_path, "fill_2s")
        if alt:
            alt_m = from_trades(alt)
            print()
            print("  SAME TRADES AT THE 2s-LATER PRICE (the latency cost):")
            print(f"    expectancy {alt_m.expectancy:+.4f} USDC/trade   "
                  f"edge/share {(alt_m.edge_per_share or 0)*100:+.3f}%   "
                  f"win rate {(alt_m.win_rate or 0)*100:.2f}%")
            print(f"    the difference from the decision-basis run is what latency costs")


def cmd_report(log_dir: str, stake: float, basis: str) -> int:
    decisions_path = log_path(log_dir, "decisions.csv")
    settled_path = log_path(log_dir, "settled.csv")
    print("=" * 96)
    print("V6 SESSION REPORT")
    print("=" * 96)
    if not os.path.exists(decisions_path):
        print(f"  no decisions log at {decisions_path}")
        return 1
    with open(decisions_path, encoding="utf-8") as handle:
        decisions = list(csv.DictReader(handle))
    print(f"  windows evaluated : {len(decisions)}")
    reasons: dict[str, int] = {}
    for row in decisions:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    print("  refusals by reason:")
    total = len(decisions) or 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>6}  {count/total*100:>5.1f}%  {reason}")
    entries = [r for r in decisions if r["action"] == "enter"]
    print(f"  would have entered: {len(entries)}  "
          f"({len(entries)/total*100:.1f}% of windows)")
    if decisions:
        intervals = [float(r["secs_to_close"]) for r in decisions
                     if r.get("secs_to_close")]
        if intervals:
            print(f"  decision timing   : median T-{statistics.median(intervals):.1f}s "
                  f"(target T-180s)")
        lags = [float(r["open_lag_s"]) for r in decisions
                if r.get("open_lag_s") not in (None, "")]
        if lags:
            lags.sort()
            print(f"  window-open lag   : median {statistics.median(lags):.2f}s  "
                  f"p90 {lags[9*len(lags)//10]:.2f}s")
        drift = []
        for row in entries:
            if row.get("ask") and row.get("ask_2s"):
                drift.append(float(row["ask_2s"]) - float(row["ask"]))
        if drift:
            drift.sort()
            print(f"  latency drift 2s  : median {statistics.median(drift)*100:+.3f}c  "
                  f"p90 {drift[9*len(drift)//10]*100:+.3f}c   "
                  f"(the edge tolerates ~4c)")
    _report(settled_path, stake, basis)
    return 0


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="V6 live paper trading (read-only, never sends orders)")
    parser.add_argument("command", choices=("probe", "run", "report"))
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after N seconds (0 = until Ctrl-C)")
    parser.add_argument("--entry-t-minus", type=float, default=None)
    parser.add_argument("--spot-interval", type=float, default=0.25,
                        help="seconds between spot samples")
    parser.add_argument("--report-every", type=float, default=120.0,
                        help="seconds between progress lines (0 = never)")
    parser.add_argument("--basis", default="decision",
                        choices=("decision", "fill_2s"),
                        help="which entry price the report scores")
    args = parser.parse_args(argv)

    params = V6Params()
    if args.entry_t_minus is not None:
        params = replace(params, entry_t_minus=args.entry_t_minus)

    if args.command == "probe":
        return cmd_probe(params, args.log_dir)
    if args.command == "run":
        return cmd_run(params, args.stake, args.log_dir, args.seconds,
                       args.spot_interval, args.report_every)
    return cmd_report(args.log_dir, args.stake, args.basis)


if __name__ == "__main__":
    raise SystemExit(main())
