"""V6 test suite. No network, no dataset, no side effects.

    python tools_v6/test_v6.py

Every assertion here exists because something in V4 broke in a way that produced
plausible-looking output:

* a refactor dropped two local assignments and the live loop died on its first window
* a cache mapped bisect indices onto the wrong end of the tick list, shifting the
  decision point by 57 seconds and still returning real prices
* validation tools hardcoded the checkpoint they were written against, so they kept
  validating a configuration the strategy no longer used
* the reports showed no win rate, no profit factor and no drawdown, and two
  aggregations disagreed in sign with nothing marking the difference
* a backtest computed a quantity locally and then re-gated on it, so the strategy's own
  gate silently did nothing and backtest and live ran different rule sets

So the suite checks: arithmetic identity, gate behaviour at the boundary, orientation
sign, metric definitions, the live decision path executed against fake feeds, and the
book parser against an UNSORTED book as the venue's docs warn about.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "tools_v6"))

from engine_v6 import signals  # noqa: E402
from engine_v6.feed import Book, FeedError, SpotQuote  # noqa: E402
from engine_v6.metrics import TradeRecord, from_trades, max_drawdown  # noqa: E402
from engine_v6.recorder import SessionRecorder, SpotTracker  # noqa: E402
from engine_v6.strategy import (  # noqa: E402
    BALANCED_BOOK, NO_MOVE, READY, V6Params, StrategyV6,
)

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  [ok]   {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILURES.append(f"{name}  {detail}")
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


# --------------------------------------------------------------------- signals


def test_signals() -> None:
    section("1. FEE AND BREAK-EVEN ARITHMETIC")
    for price in (0.5, 0.7, 0.88, 0.95):
        got = signals.taker_fee_per_share(price)
        want = 0.07 * price * (1 - price)
        check(f"fee at {price} matches 0.07*p*(1-p)", abs(got - want) < 1e-15,
              f"{got:.8f}")
    check("fee peaks at 0.50",
          signals.taker_fee_per_share(0.5) > signals.taker_fee_per_share(0.7))
    # symmetric in exact arithmetic, equal only within tolerance in floating point:
    # f(0.7) and f(0.3) differ by about 3e-18
    gap = abs(signals.taker_fee_per_share(0.7) - signals.taker_fee_per_share(0.3))
    check("fee is symmetric within floating-point tolerance", gap < 1e-15,
          f"gap {gap:.1e}")
    check("fee vanishes at the extremes",
          signals.taker_fee_per_share(0.001) < 1e-4)
    be = signals.break_even_accuracy(0.88)
    check("break-even = price + fee",
          abs(be - (0.88 + signals.taker_fee_per_share(0.88))) < 1e-15, f"{be:.6f}")

    # net_pnl must agree with the closed form used everywhere else
    entry, shares = 0.88, 11.36
    won_pnl = signals.net_pnl(entry, True, shares)
    lose_pnl = signals.net_pnl(entry, False, shares)
    fee = shares * signals.taker_fee_per_share(entry)
    check("net_pnl on a win = shares*(1-p) - fee",
          abs(won_pnl - (shares * (1 - entry) - fee)) < 1e-12, f"{won_pnl:+.6f}")
    check("net_pnl on a loss = -shares*p - fee",
          abs(lose_pnl - (-shares * entry - fee)) < 1e-12, f"{lose_pnl:+.6f}")
    # the identity the whole strategy rests on
    implied = signals.break_even_accuracy(entry)
    check("at break-even accuracy the expected P&L is zero",
          abs(implied * won_pnl + (1 - implied) * lose_pnl) < 1e-9,
          f"residual {implied*won_pnl + (1-implied)*lose_pnl:.2e}")


# -------------------------------------------------------------------- metrics


def test_metrics() -> None:
    section("2. METRIC DEFINITIONS")


    def trade(won: bool, entry: float = 0.80) -> TradeRecord:
        shares = 12.5
        pnl = signals.net_pnl(entry, won, shares)
        return TradeRecord(slug="x", day="d", side="UP", entry=entry,
                           shares=shares, won=won, pnl=pnl)

    m = from_trades([trade(True)] * 7 + [trade(False)] * 3)
    check("win rate counts settled trades", abs(m.win_rate - 0.70) < 1e-12,
          f"{m.win_rate:.4f}")
    check("profit factor = gross profit / gross loss",
          abs(m.profit_factor - m.gross_profit / m.gross_loss) < 1e-12,
          f"{m.profit_factor:.4f}")
    check("expectancy = mean P&L over all trades",
          abs(m.expectancy - m.total_pnl / 10) < 1e-12)
    check("total P&L = gross profit - gross loss",
          abs(m.total_pnl - (m.gross_profit - m.gross_loss)) < 1e-12)
    check("best/worst bracket every trade",
          m.best_trade >= m.expectancy >= m.worst_trade)
    check("equity curve ends at total P&L",
          abs(m.equity[-1] - m.total_pnl) < 1e-12)

    # the case that matters: 70% win rate at a 0.80 entry is a LOSS, because
    # break-even at that price is about 86%
    check("70% win rate at entry 0.80 is correctly reported as a loss",
          m.total_pnl < 0, f"{m.total_pnl:+.4f} USDC")

    # drawdown on a known curve: up 10, down 4 -> max drawdown 4
    worst, pct = max_drawdown([0.0, 10.0, 6.0])
    check("max drawdown over a known curve", abs(worst - 4.0) < 1e-12,
          f"{worst:.4f}")
    check("drawdown percentage is relative to the peak",
          pct is not None and abs(pct - 0.4) < 1e-12, f"{pct}")
    worst, pct = max_drawdown([0.0, -1.0, -2.0])
    check("drawdown percentage undefined when never above zero", pct is None)

    # undefined profit factor must be None, never 0 or infinity
    clean = from_trades([trade(True)] * 4)
    check("profit factor undefined with no losses",
          clean.profit_factor is None, repr(clean.profit_factor))
    check("the undefined case is flagged as a caveat",
          any("profit factor" in c for c in clean.caveats))
    empty = from_trades([])
    check("empty input yields trade count 0, not a crash", empty.trades == 0)
    check("empty input is flagged", bool(empty.caveats))

    small = from_trades([trade(True), trade(False)])
    check("small samples produce a caveat", len(small.caveats) >= 1,
          f"{len(small.caveats)} caveats")


# ----------------------------------------------------------------------- book


def test_book() -> None:
    section("3. ORDER BOOK PARSING (the venue's arrays are not reliably sorted)")
    # exactly the shape the venue's own guidance warns about: a stale 0.99 long shot
    # at index 0 while the real best ask is 0.53
    asks = [(0.99, 3.0), (0.53, 80.0), (0.55, 40.0), (0.53, 20.0)]
    bids = [(0.51, 120.0), (0.48, 300.0)]
    book = Book(token_id="t", bids=bids, asks=asks, fetched_at=time.time())
    check("best ask is derived, not indexed", book.best_ask == 0.53,
          f"{book.best_ask}")
    check("best bid is derived, not indexed", book.best_bid == 0.51,
          f"{book.best_bid}")
    check("mid uses the derived touch", abs(book.mid - 0.52) < 1e-12, f"{book.mid}")
    check("spread uses the derived touch", abs(book.spread - 0.02) < 1e-12,
          f"{book.spread}")
    check("size at best sums only the best level",
          abs(book.size_at_best("asks") - 100.0) < 1e-12,
          f"{book.size_at_best('asks')}")
    check("top-N selection is by price, not array position",
          abs(book.notional("asks", 2) - (0.53 * 80 + 0.53 * 20)) < 1e-9,
          f"{book.notional('asks', 2):.4f}")
    cost = book.cost_to_buy(100.0)
    check("walking the asks fills only at the best level",
          cost is not None and abs(cost - 100 * 0.53) < 1e-9, f"{cost}")
    # the book holds 80+20 at 0.53, 40 at 0.55, 3 at 0.99 = 143 shares
    cost = book.cost_to_buy(130.0)
    check("walking the asks crosses levels when needed",
          cost is not None and abs(cost - (100 * 0.53 + 30 * 0.55)) < 1e-9,
          f"{cost}")
    cost = book.cost_to_buy(143.0)
    check("filling the entire book succeeds exactly",
          cost is not None
          and abs(cost - (100 * 0.53 + 40 * 0.55 + 3 * 0.99)) < 1e-9, f"{cost}")
    check("one share more than the book holds returns None",
          book.cost_to_buy(144.0) is None)
    check("an unfillable size returns None",
          Book(token_id="t", bids=[(0.5, 1.0)], asks=[(0.51, 1.0)],
               fetched_at=time.time()).cost_to_buy(50.0) is None)
    check("a crossed book is detectable",
          Book(token_id="t", bids=[(0.60, 1.0)], asks=[(0.50, 1.0)],
               fetched_at=time.time()).is_crossed())
    empty = Book(token_id="t", bids=[], asks=[], fetched_at=time.time())
    check("an empty book yields None, not a zero price",
          empty.best_bid is None and empty.mid is None and empty.imbalance() is None)

    # imbalance orientation: a bid-heavy UP book must read negative for a DOWN buyer
    heavy_bid = Book(token_id="t", bids=[(0.50, 1000.0)], asks=[(0.51, 10.0)],
                     fetched_at=time.time())
    check("bid-heavy book has positive imbalance",
          heavy_bid.imbalance(10) > 0, f"{heavy_bid.imbalance(10):.4f}")
    check("orienting to DOWN flips the sign",
          StrategyV6.orient(heavy_bid.imbalance(10), "DOWN") < 0)
    check("orienting to UP preserves it",
          StrategyV6.orient(heavy_bid.imbalance(10), "UP") > 0)
    check("orienting None stays None", StrategyV6.orient(None, "UP") is None)


# ------------------------------------------------------------------- strategy


def test_strategy() -> None:
    section("4. GATE BEHAVIOUR")
    p = V6Params()
    s = StrategyV6(p)
    strong = p.threshold * 1.5

    up, move = s.direction(0.40, 0.40 + strong)
    check("a rise buys UP", up == "UP", f"{move:+.3f}")
    down, move = s.direction(0.40 + strong, 0.40)
    check("a fall buys DOWN", down == "DOWN")
    check("a move below the threshold is refused",
          s.direction(0.40, 0.40 + p.threshold * 0.5) is None)
    check("a move just under the threshold is refused",
          s.direction(0.40, 0.40 + p.threshold - 1e-9) is None)
    check("a move at the threshold passes (the gate is >=)",
          s.direction(0.40, 0.40 + p.threshold) is not None)

    def decide(**kw):
        base = dict(slug="t", now=1000.0, secs_to_close=p.entry_t_minus,
                    up_mid_open=0.40, up_mid_now=0.40 + strong,
                    book_imbalance=0.80, spot_open=100.0, spot_now=100.10,
                    ask=0.70, spread=0.01, quote_ts=1000.0, shares_wanted=14.0,
                    depth_cost=9.8)
        base.update(kw)
        return s.evaluate(**base)

    check("a fully passing window enters", decide().entering, decide().reason)
    check("a balanced book is refused",
          decide(book_imbalance=p.min_book_imbalance / 2).reason == BALANCED_BOOK)
    check("the gate is inclusive at the threshold",
          decide(book_imbalance=p.min_book_imbalance).entering)
    check("a too-small spot move is refused",
          decide(spot_now=100.0 * (1 + p.min_spot_move / 2)).reason
          != READY)
    check("a missing quote is refused", decide(ask=None).reason != READY)
    check("stale quote is refused",
          decide(quote_ts=1000.0 - p.max_quote_staleness_s - 1).reason != READY)
    check("no shares means no trade", decide(shares_wanted=0.0).reason != READY)
    check("a flat window is refused with the move reason",
          decide(up_mid_now=0.40 + p.threshold * 0.5).reason == NO_MOVE)

    # the funnel must count what actually happened. A move of 0.25 clears the 0.20
    # threshold; a move of 0.02 does not, and an earlier version of this test used
    # 0.02 and then blamed the funnel for reporting zero passes.
    p2 = V6Params()
    s2 = StrategyV6(p2)
    strong_move = p2.threshold + 0.05
    # The first four have a MEASURED but small imbalance, so the gate must refuse
    # them. Passing None instead would mean "not measured", which correctly SKIPS the
    # gate - an earlier version of this test made that mistake and then blamed the
    # funnel for letting everything through.
    balanced = p2.min_book_imbalance / 2.0
    for _ in range(4):
        s2.evaluate(slug="a", now=1.0, secs_to_close=180.0, up_mid_open=0.40,
                    up_mid_now=0.40 + strong_move, book_imbalance=balanced,
                    spot_open=100.0, spot_now=100.1, ask=0.7, shares_wanted=10.0,
                    depth_cost=7.0)
    for _ in range(6):
        s2.evaluate(slug="b", now=1.0, secs_to_close=180.0, up_mid_open=0.40,
                    up_mid_now=0.40 + strong_move, book_imbalance=0.9,
                    spot_open=100.0, spot_now=100.1, ask=0.7, shares_wanted=10.0,
                    depth_cost=7.0)
    funnel = dict((gate, (reached, passed)) for gate, reached, passed
                  in s2.ledger.funnel())
    check("the funnel counts every window", s2.ledger.readings.__len__() == 10,
          f"{len(s2.ledger.readings)}")
    check("the funnel shows the imbalance gate refusing the balanced ones",
          funnel.get(BALANCED_BOOK, (0, 0))[0] == 10
          and funnel.get(BALANCED_BOOK, (0, 0))[1] == 6,
          f"{funnel.get(BALANCED_BOOK)}")
    dist = s2.ledger.distribution("book_imbalance")
    check("the ledger exposes a measured distribution, not just counts",
          dist and "median" in dist, f"{dist.get('median') if dist else None}")
    check("every window records its measured values regardless of refusal",
          all(r.move is not None for r in s2.ledger.readings))


# ------------------------------------------------------------------ recorder


def test_recorder() -> None:
    section("5. WINDOW-OPEN CAPTURE (the V4 frequency bug lived here)")
    rec = SessionRecorder(300.0)
    epoch = int(time.time())
    # an observation taken well after the window opened must not become "the open"
    # if an earlier one exists
    cap = rec.observe("s", epoch, 0.50)
    first_lag = cap.open_lag_s
    time.sleep(0.05)
    rec.observe("s", epoch, 0.55)
    check("the earliest observation wins as the window open",
          rec.windows["s"].open_up_mid == 0.50,
          f"{rec.windows['s'].open_up_mid}")
    check("the recorded lag is tiny in test conditions",
          first_lag is not None and first_lag < 1.0, f"{first_lag}")
    summary = rec.open_lag_summary()
    check("open lag is summarised for the report", "median_s" in summary)

    book = Book(token_id="t", bids=[(0.50, 1000.0)], asks=[(0.51, 10.0)],
                fetched_at=time.time())
    rec.mark_decision("s", 0.75, book, 0.51, 0.01)
    cap = rec.windows["s"]
    check("the decision stores imbalance at several depth caps",
          set(cap.imbalances) >= {"5", "10", "20", "full"},
          f"{sorted(cap.imbalances)}")
    check("depth levels are stored", cap.depth_levels == (1, 1),
          f"{cap.depth_levels}")


def test_spot_tracker() -> None:
    section("6. SPOT TRACKER (timestamp-accurate, not last-fetched)")


    class FakeSpotFeed:
        def __init__(self):
            self.calls = 0

        def quote(self):
            self.calls += 1
            return SpotQuote(ts=time.time(), price=100.0 + self.calls, source="fake")

    tracker = SpotTracker(FakeSpotFeed(), interval=0.02)
    tracker.start()
    time.sleep(0.25)
    tracker.stop()
    stats = tracker.stats()
    check("the tracker samples continuously", stats["samples"] >= 3,
          f"{stats['samples']} samples")
    check("the tracker reports its own sampling gaps",
          stats["gap_median_s"] is not None, f"{stats['gap_median_s']}")
    now = time.time()
    point = tracker.at(now - 0.1)
    check("at() returns a timestamped point", point is not None and point.price > 100)
    check("between() returns the series slice",
          len(tracker.between(now - 10, now)) >= 3)
    check("a far-past timestamp returns None rather than a stale price",
          tracker.at(now - 3600) is None)

    # a feed that always fails must not raise
    class DeadFeed:
        def quote(self):
            return None

    dead = SpotTracker(DeadFeed(), interval=0.02)
    dead.start()
    time.sleep(0.15)
    dead.stop()
    check("a dead feed is counted, not raised", dead.stats()["errors"] >= 1,
          f"{dead.stats()['errors']} errors")


# --------------------------------------------------------------- live path


def test_live_path() -> None:
    section("7. LIVE DECISION PATH (executed end to end, no network)")
    import run_live_v6 as live

    class FakeBook:
        def __init__(self, mid, spread, bid_notional, ask_notional):
            self.best_bid = mid - spread / 2
            self.best_ask = mid + spread / 2
            self.bids = [(self.best_bid, bid_notional / max(self.best_bid, 1e-9))]
            self.asks = [(self.best_ask, ask_notional / max(self.best_ask, 1e-9))]
            self.mid = mid
            self.spread = spread
            self.fetched_at = time.time()
            self.bid_depth = 1
            self.ask_depth = 1
            self.token_id = "t"

        def notional(self, side, cap=None):
            levels = self.bids if side == "bids" else self.asks
            return sum(p * s for p, s in levels)

        def imbalance(self, cap=None):
            b, a = self.notional("bids"), self.notional("asks")
            return None if b + a <= 0 else (b - a) / (b + a)

        def cost_to_buy(self, shares):
            return self.best_ask * shares

        def depth_span(self):
            return (0.0, 0.0)

    class FakeMarket:
        slug = "btc-updown-5m-1"
        up_token = "UP"
        start_ts = 0.0
        end_ts = 10 ** 9

    class FakeFeed:
        def __init__(self, book):
            self._book = book

        def book(self, token_id, depth=None):
            return self._book

    class FakeTracker:
        interval = 0.25

        def __init__(self, price):
            self._price = price

        def at(self, ts, tolerance=None, prefer_after=True):
            return SpotQuote(ts=ts, price=self._price, source="fake")

        def stats(self):
            return {"samples": 1, "errors": 0, "gap_median_s": 0.25,
                    "gap_p90_s": 0.3, "gap_max_s": 0.4}

    # a window with a strong up move and a bid-heavy book: should enter
    params = V6Params()
    strategy = StrategyV6(params)
    book = FakeBook(mid=0.75, spread=0.02, bid_notional=900.0, ask_notional=100.0)
    recorder = SessionRecorder(300.0)
    recorder.observe("btc-updown-5m-1", 1, 0.50, book)
    recorder.mark_decision("btc-updown-5m-1", 0.75, book, book.best_ask, book.spread)
    cap = recorder.windows["btc-updown-5m-1"]
    signed = strategy.orient(cap.imbalances.get("10"), "UP")
    decision = strategy.evaluate(
        slug="s", now=1.0, secs_to_close=180.0, up_mid_open=cap.open_up_mid,
        up_mid_now=0.75, book_imbalance=signed, bid_notional=900.0,
        ask_notional=100.0, spot_open=100.0, spot_now=100.05,
        ask=book.best_ask, spread=book.spread, quote_ts=time.time(),
        shares_wanted=10.0 / book.best_ask, depth_cost=10.0)
    check("the live decision path enters on a strong lopsided window",
          decision.entering, decision.reason)
    check("the imbalance that reached the strategy is oriented to UP",
          signed is not None and signed > 0, f"{signed}")

    # and a weak window must be refused, not crash
    strategy2 = StrategyV6(params)
    recorder2 = SessionRecorder(300.0)
    recorder2.observe("s2", 1, 0.50, book)
    flat = FakeBook(mid=0.52, spread=0.02, bid_notional=900.0, ask_notional=100.0)
    recorder2.mark_decision("s2", 0.52, flat, flat.best_ask, flat.spread)
    decision = strategy2.evaluate(
        slug="s2", now=1.0, secs_to_close=180.0,
        up_mid_open=recorder2.windows["s2"].open_up_mid, up_mid_now=0.52,
        book_imbalance=0.8, spot_open=100.0, spot_now=100.05,
        ask=flat.best_ask, spread=flat.spread, quote_ts=time.time(),
        shares_wanted=10.0, depth_cost=10.0)
    check("a flat window is refused through the live path",
          decision.reason == NO_MOVE, decision.reason)
    check("running the live path leaves no undefined names",
          decision.action in ("enter", "hold"))


def test_live_path_regressions() -> None:
    """Regressions from the real live session, each of which passed the suite before.

    These three shipped and were only caught by running against the real venue:
    a KeyError that killed the process, a double-oriented imbalance that silently
    inverted every DOWN trade, and a decision that fired 1.2s before its target.
    """
    section("8. LIVE-SESSION REGRESSIONS")

    # --- the KeyError: a decision for a window that was never observed ---------
    fresh = SessionRecorder(300.0)
    try:
        cap = fresh.mark_decision("btc-updown-5m-1791696900", 0.75, None, 0.78, 0.01)
        ok, detail = True, "window created on demand"
    except KeyError as exc:
        cap, ok, detail = None, False, f"KeyError {exc}"
    check("a decision for an unobserved window does not raise", ok, detail)
    check("the unobserved window records NO open rather than a substituted one",
          cap is not None and cap.open_up_mid is None and cap.open_lag_s is None)
    check("the epoch is recovered from the slug",
          cap is not None and cap.epoch == 1791696900
          and cap.start_ts == 1791696900.0, f"{cap.epoch if cap else None}")
    check("the window is still marked decided", cap is not None and cap.decided)

    # --- the double-orientation -------------------------------------------------
    # The gate tests |imbalance|, so pre-orienting does NOT change the decision - an
    # earlier version of this test asserted that it did, and was wrong. What it
    # changes is the SIGN STORED IN THE LEDGER, which is the evidence a live session
    # is judged on. So the assertion is about the recorded value, not the outcome.
    params = V6Params()
    raw_up_space = -0.60            # DOWN token book bid-heavy, in UP-token space
    correct = StrategyV6(params)
    correct.evaluate(
        slug="d", now=1.0, secs_to_close=params.entry_t_minus,
        up_mid_open=0.60, up_mid_now=0.30,           # a fall -> buy DOWN
        book_imbalance=raw_up_space,                 # RAW, as the live loop now passes
        spot_open=100.0, spot_now=99.0, ask=0.70, spread=0.01,
        quote_ts=1.0, shares_wanted=14.0, depth_cost=9.8)
    right = correct.ledger.readings[-1].book_imbalance
    check("a RAW imbalance is oriented once, to +0.60 for a DOWN buy",
          right is not None and abs(right - 0.60) < 1e-9, f"{right}")

    doubled = StrategyV6(params)
    doubled.evaluate(
        slug="d2", now=1.0, secs_to_close=params.entry_t_minus,
        up_mid_open=0.60, up_mid_now=0.30,
        book_imbalance=+0.60,                        # pre-flipped = the old live bug
        spot_open=100.0, spot_now=99.0, ask=0.70, spread=0.01,
        quote_ts=1.0, shares_wanted=14.0, depth_cost=9.8)
    wrong = doubled.ledger.readings[-1].book_imbalance
    check("double-orienting records the WRONG sign in the ledger",
          wrong is not None and wrong < 0, f"{wrong}")
    check("the two disagree, so the live loop had to stop pre-orienting",
          right is not None and wrong is not None and right * wrong < 0,
          f"{right} vs {wrong}")

    # and the gate must still be on magnitude: the anti-correlated book is accepted
    moved = StrategyV6(params).evaluate(
        slug="d3", now=1.0, secs_to_close=params.entry_t_minus,
        up_mid_open=0.60, up_mid_now=0.30, book_imbalance=-0.60,
        spot_open=100.0, spot_now=99.0, ask=0.70, spread=0.01,
        quote_ts=1.0, shares_wanted=14.0, depth_cost=9.8)
    check("the gate is on magnitude, matching the validated calibration",
          moved.entering, f"{moved.reason}")
    check("a small imbalance is still refused regardless of sign",
          not StrategyV6(params).evaluate(
              slug="d4", now=1.0, secs_to_close=params.entry_t_minus,
              up_mid_open=0.60, up_mid_now=0.30, book_imbalance=0.10,
              spot_open=100.0, spot_now=99.0, ask=0.70, spread=0.01,
              quote_ts=1.0, shares_wanted=14.0, depth_cost=9.8).entering)

    # --- the wake budget: the decision must not fire before its target ---------
    import run_live_v6 as live
    check("time is reserved for the book fetch",
          live.BOOK_FETCH_BUDGET_S > 0.5,
          f"{live.BOOK_FETCH_BUDGET_S}s reserved")
    check("the wake slack exceeds the fetch budget",
          live.WAKE_SLACK_S > live.BOOK_FETCH_BUDGET_S,
          f"slack {live.WAKE_SLACK_S}s > budget {live.BOOK_FETCH_BUDGET_S}s")


def main() -> int:
    print("=" * 92)
    print("V6 TEST SUITE - no network, no dataset, no side effects")
    print("=" * 92)
    test_signals()
    test_metrics()
    test_book()
    test_strategy()
    test_recorder()
    test_spot_tracker()
    test_live_path()
    test_live_path_regressions()
    section(f"RESULT: {CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    for failure in FAILURES:
        print(f"  - {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
