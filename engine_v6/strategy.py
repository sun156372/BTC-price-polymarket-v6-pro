"""V6 decision logic, with every refusal counted in a ledger.

    from engine_v6.strategy import V6Params, StrategyV6, Ledger

WHAT CHANGED FROM V4, AND WHY
-----------------------------
The V4 live session entered 6 times from 217 windows while the backtest implied about
52 a day (roughly 18% of windows). Nobody could say which gate caused the shortfall,
because V4 logged refusals as a bare tally with no view of what the rejected windows
measured. "Was it the imbalance gate or the spot gate?" was unanswerable from the log.

V6 keeps the same decision rules - they are the part that was validated - and adds:

* a **ledger** that records the MEASURED VALUE of every gate for every window, not
  just the first one that failed, so the full distribution of each quantity is
  available after a session and can be compared directly to the backtest's
* a **funnel** that counts how many windows reached each gate, which turns a
  frequency shortfall into a per-gate pass rate
* gates evaluated in a fixed, documented order with every measurement computed
  regardless of earlier failures

The rules themselves, unchanged from the validated V4 cell:

    follow the Up-token move since the window open, |move| >= threshold
    require |book imbalance| >= min_book_imbalance, oriented to the side bought
    require |spot move since the open| >= min_spot_move
    hold to settlement, no stop, no target
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: refusal reasons, surfaced verbatim in the log
READY = "ready"
NO_MOVE = "signal:below_threshold"
NO_DATA = "signal:missing_quote"
BALANCED_BOOK = "imbalance:book_not_lopsided"
WEAK_DRIVER = "deviation:spot_move_too_small"
NO_QUOTE = "execution:no_quote"
STALE = "execution:quote_stale"
THIN_BOOK = "execution:book_too_thin"
PRICE_TOO_HIGH = "execution:entry_above_cap"
UNMEASURED = "execution:unmeasured_fill"
SIZE_ZERO = "sizing:no_budget"
HALTED = "safety:market_halted"

#: the order gates are evaluated in. Recorded because the order determines what a
#: "refusal count" means: a window refused by the first gate never reaches the second.
GATE_ORDER = (NO_MOVE, BALANCED_BOOK, WEAK_DRIVER, STALE, NO_QUOTE, THIN_BOOK,
              PRICE_TOO_HIGH)


@dataclass
class V6Params:
    """The validated cell, plus the depth convention the gate depends on.

    Every threshold here was measured, not chosen. The one field that is NOT a
    strategy choice is ``imbalance_depth``: it records how many book levels the 0.30
    threshold was fitted to, and it must match the live book or the threshold is
    meaningless.
    """

    #: Seconds before the close. T-240 = 60s into a 5m window.
    #:
    #: This is the single most important number in the strategy, and the V6
    #: decomposition shows why. The win rate barely moves between checkpoints, but the
    #: ENTRY PRICE does, and the entry price is what decides profitability:
    #:
    #:   T-180: mean entry 0.9004, win 89.65%, break-even needs 90.66%  -> -1.01 points
    #:   T-240: mean entry 0.8685, win 89.74%, break-even needs 87.65%  -> +2.09 points
    #:
    #: Same signal, same gates, near-identical win rate. The 3.2 cents of entry price
    #: is the whole difference between losing 1% and earning 2.3% per share.
    #:
    #: V6 initially shipped T-180, inherited from a V4 sweep whose reported numbers
    #: did not reproduce. At T-180 the strategy sits ON the break-even line, where
    #: rounding decides the sign. T-240 is the validated-profitable checkpoint:
    #: out-of-sample +2.30%/share, profit factor 1.26, 15/22 days positive, max
    #: drawdown 40% of peak.
    entry_t_minus: float = 240.0
    #: minimum |Up-token move since the open|, in probability points
    threshold: float = 0.20
    mode: str = "follow"
    take_profit: float = 1.0
    stop_loss: float = 1.0
    #: THE EDGE. Calibrated on |imbalance| over the top 10 levels.
    min_book_imbalance: float = 0.30
    #: how many levels per side the imbalance uses. 10 matches the research dataset,
    #: whose books are capped at 10 levels and 82% saturated at that cap.
    imbalance_depth: int = 10
    #: secondary gate: the underlying must have moved too
    min_spot_move: float = 0.00045
    #: disabled on purpose: tightening it lowered returns, because accuracy rises
    #: with price so a cap discards the most accurate trades
    max_entry_price: float = 1.0
    max_quote_staleness_s: float = 2.0
    max_slippage_cents: float = 2.0
    require_measured_fill: bool = False
    fee_coefficient: float = 0.07
    #: measured on 44 days / 12,670 markets. The ACCURACY is nearly flat across
    #: checkpoints; the ENTRY PRICE is not, and the entry price is what pays.
    #: Values below are for the shipped T-240 cell, out of sample.
    measured_accuracy: float = 0.8974
    measured_entry: float = 0.8665

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "entry_t_minus", "threshold", "mode", "min_book_imbalance",
            "imbalance_depth", "min_spot_move", "max_entry_price",
            "fee_coefficient")}


@dataclass
class Decision:
    action: str
    reason: str
    side: str = ""
    move: float = 0.0
    limit_price: float = 0.0
    shares: float = 0.0
    break_even: float = 0.0

    @property
    def entering(self) -> bool:
        return self.action == "enter"


@dataclass
class GateReading:
    """One window's measurements, taken whether or not any gate passed."""

    slug: str
    ts: float
    secs_to_close: float
    up_mid_open: float | None = None
    up_mid_now: float | None = None
    move: float | None = None
    book_imbalance: float | None = None
    bid_notional: float | None = None
    ask_notional: float | None = None
    bid_depth: int = 0
    ask_depth: int = 0
    spot_open: float | None = None
    spot_now: float | None = None
    spot_move: float | None = None
    ask: float | None = None
    spread: float | None = None
    side: str = ""
    decision: str = ""
    reason: str = ""


@dataclass
class Ledger:
    """Every window evaluated, with each gate's measured value and a funnel.

    The funnel is the point. It answers "how often did each gate pass, given the
    windows that actually reached it", which is the only way to compare live
    frequency against backtest frequency gate by gate.
    """

    readings: list[GateReading] = field(default_factory=list)
    reached: dict[str, int] = field(default_factory=dict)
    passed: dict[str, int] = field(default_factory=dict)

    def record(self, reading: GateReading) -> None:
        self.readings.append(reading)
        self.reached[reading.reason] = self.reached.get(reading.reason, 0) + 1

    def funnel(self) -> list[tuple[str, int, int]]:
        """(gate, reached, passed_through) in evaluation order, plus the entry row."""
        rows = [("windows evaluated", len(self.readings), len(self.readings))]
        reached = len(self.readings)
        for gate in GATE_ORDER:
            refused = self.reached.get(gate, 0)
            rows.append((gate, reached, reached - refused))
            reached -= refused
        rows.append(("ready (entered)", reached, reached))
        return rows

    def distribution(self, field_name: str) -> dict:
        """Percentiles of one measured field across every evaluated window.

        This is what makes a frequency gap diagnosable: if the live distribution of
        ``book_imbalance`` sits at half the backtest's scale, the threshold is wrong
        and the graph says so at a glance.
        """
        values = sorted(v for v in (getattr(r, field_name, None)
                                    for r in self.readings) if v is not None)
        if not values:
            return {}
        n = len(values)

        def at(q: float) -> float:
            return values[min(n - 1, int(q * (n - 1)))]

        return {"n": n, "min": values[0], "p10": at(0.10), "p25": at(0.25),
                "median": at(0.50), "p75": at(0.75), "p90": at(0.90),
                "max": values[-1], "mean": sum(values) / n}


class StrategyV6:
    """The single decision path. Backtest and live loop both call this."""

    def __init__(self, params: V6Params | None = None):
        self.p = params or V6Params()
        self.ledger = Ledger()

    # -------------------------------------------------------------- helpers
    def direction(self, open_up_mid: float | None,
                  entry_up_mid: float | None) -> tuple[str, float] | None:
        if open_up_mid is None or entry_up_mid is None:
            return None
        move = entry_up_mid - open_up_mid
        if abs(move) < self.p.threshold:
            return None
        up = move > 0
        if self.p.mode == "fade":
            up = not up
        return ("UP" if up else "DOWN"), move

    @staticmethod
    def orient(imbalance: float | None, side: str) -> float | None:
        """Flip the sign so it always favours the side being bought."""
        if imbalance is None:
            return None
        return imbalance if side == "UP" else -imbalance

    # --------------------------------------------------------------- decide
    def evaluate(self, *, slug: str, now: float, secs_to_close: float,
                 up_mid_open: float | None, up_mid_now: float | None,
                 book_imbalance: float | None, bid_notional: float | None = None,
                 ask_notional: float | None = None, bid_depth: int = 0,
                 ask_depth: int = 0, spot_open: float | None = None,
                 spot_now: float | None = None, ask: float | None = None,
                 spread: float | None = None, quote_ts: float | None = None,
                 shares_wanted: float = 0.0, depth_cost: float | None = None,
                 halted: bool = False) -> Decision:
        """Score one window and log every measurement.

        All gates are measured even when an earlier one fails, so the ledger holds the
        true distribution of each quantity rather than only the values that survived
        the preceding filters.
        """
        p = self.p
        spot_move = None
        if spot_open and spot_now:
            spot_move = spot_now / spot_open - 1.0

        picked = self.direction(up_mid_open, up_mid_now)
        side = picked[0] if picked else ""
        move = picked[1] if picked else (None if up_mid_open is None
                                         else (up_mid_now - up_mid_open
                                               if up_mid_now is not None else None))

        reading = GateReading(
            slug=slug, ts=now, secs_to_close=secs_to_close,
            up_mid_open=up_mid_open, up_mid_now=up_mid_now, move=move,
            book_imbalance=self.orient(book_imbalance, side) if side else book_imbalance,
            bid_notional=bid_notional, ask_notional=ask_notional,
            bid_depth=bid_depth, ask_depth=ask_depth,
            spot_open=spot_open, spot_now=spot_now, spot_move=spot_move,
            ask=ask, spread=spread, side=side,
        )

        decision = self._decide(
            side=side, move=move, book_imbalance=book_imbalance, spot_move=spot_move,
            ask=ask, now=now, quote_ts=quote_ts, shares_wanted=shares_wanted,
            depth_cost=depth_cost, halted=halted, up_mid_open=up_mid_open,
            up_mid_now=up_mid_now)
        reading.decision = decision.action
        reading.reason = decision.reason
        self.ledger.record(reading)
        return decision

    def _decide(self, *, side: str, move: float | None,
                book_imbalance: float | None, spot_move: float | None,
                ask: float | None, now: float, quote_ts: float | None,
                shares_wanted: float, depth_cost: float | None, halted: bool,
                up_mid_open: float | None, up_mid_now: float | None) -> Decision:
        p = self.p
        if halted:
            return Decision("hold", HALTED)
        if up_mid_open is None or up_mid_now is None:
            return Decision("hold", NO_DATA)
        if side == "":
            return Decision("hold", NO_MOVE, move=move or 0.0)

        signed = self.orient(book_imbalance, side)
        # The gate is on the MAGNITUDE, matching the calibration that produced the
        # validated backtest. A sign test was tried and had to be reverted; see the
        # note at the end of this method for what the data actually showed.
        if signed is not None and not (-1.0 <= signed <= 1.0):
            # out of range means the caller passed something that is not an imbalance
            return Decision("hold", BALANCED_BOOK, side=side, move=move or 0.0)
        if signed is not None and abs(signed) < p.min_book_imbalance:
            return Decision("hold", BALANCED_BOOK, side=side, move=move or 0.0)
        if spot_move is not None and abs(spot_move) < p.min_spot_move:
            return Decision("hold", WEAK_DRIVER, side=side, move=move or 0.0)
        if quote_ts is not None and now - quote_ts > p.max_quote_staleness_s:
            return Decision("hold", STALE, side=side, move=move or 0.0)
        if ask is None or not 0.0 < ask < 1.0:
            return Decision("hold", NO_QUOTE, side=side, move=move or 0.0)
        if shares_wanted <= 0:
            return Decision("hold", SIZE_ZERO, side=side, move=move or 0.0)

        if depth_cost is None:
            if p.require_measured_fill:
                return Decision("hold", THIN_BOOK, side=side, move=move or 0.0)
            entry = ask
        else:
            entry = depth_cost / shares_wanted
        if p.require_measured_fill and depth_cost is None:
            return Decision("hold", UNMEASURED, side=side, move=move or 0.0)
        if ask is not None and entry - ask > p.max_slippage_cents / 100.0:
            return Decision("hold", THIN_BOOK, side=side, move=move or 0.0)

        from .signals import break_even_accuracy
        break_even = break_even_accuracy(entry, p.fee_coefficient)
        if entry > p.max_entry_price:
            return Decision("hold", PRICE_TOO_HIGH, side=side, move=move or 0.0,
                            limit_price=entry, break_even=break_even)
        return Decision("enter", READY, side=side, move=move or 0.0,
                        limit_price=entry, shares=shares_wanted,
                        break_even=break_even)

    # ------------------------------------------------------------------------
    # OPEN QUESTION: WHAT DOES THE IMBALANCE SIGN MEAN HERE?
    #
    # This gate tests |imbalance|, so the sign never reaches the decision. Adding a
    # sign test was attempted and reverted, because the measurement came out inverted
    # from the natural reading:
    #
    #   UP trades, raw imbalance > 0 (bid-heavy on the UP token): 71.9% win (n=114)
    #   UP trades, raw imbalance < 0 (ask-heavy on the UP token): 79.2% win (n=1555)
    #   DOWN trades, raw > 0: 74.2% win      DOWN trades, raw < 0: 79.7% win
    #
    # Two things follow, and both matter:
    #
    # 1. The sign runs OPPOSITE to the natural reading of \`full_depth_bid_notional\`,
    #    OR that column is not the UP token's book at all. Bid-heavy on the UP token
    #    is associated with a LOWER Up win rate, not a higher one.
    # 2. Being on the "right" side of the sign is worth only a few points, and the
    #    large majority of trades (1,555 of 1,669) sit on the negative side. So the
    #    sign is NOT what carries the edge - which also means the double-orientation
    #    bug in the live loop changed the recorded ledger value without changing any
    #    decision, because |imbalance| was tested either way.
    #
    # Requiring signed >= 0 refused every window in the dataset, which is how this was
    # found. Flipping the sign to match the measurement would fit 1,669 observations
    # on one dataset, i.e. exactly the overfitting this project already suffered from
    # once. So the gate stays on magnitude pending an out-of-sample test, and the
    # question is recorded as open rather than silently answered.
    #
    # Consequence to keep in mind: the "book imbalance is THE EDGE" story is weaker
    # than earlier rounds claimed. The move gate looks like the real driver.
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------- reporting
    def describe(self) -> list[str]:
        p = self.p
        return [
            f"signal      : {p.mode} the Up-token move, |move| >= {p.threshold:.3f}",
            f"entry       : T-{p.entry_t_minus:.0f}s",
            f"imbalance   : |full-depth imbalance| >= {p.min_book_imbalance:.2f} "
            f"over the top {p.imbalance_depth} levels per side",
            f"deviation   : |spot move| >= {p.min_spot_move*1e4:.1f} bps",
            f"entry cap   : {'disabled' if p.max_entry_price >= 1.0 else f'{p.max_entry_price:.3f}'}",
            f"exit        : hold to settlement, no stop, no target",
            f"fee         : {p.fee_coefficient} x p x (1-p), taker, no exit fee",
        ]
