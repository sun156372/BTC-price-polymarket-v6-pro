"""Session recording: capture the window OPEN properly, then measure at the decision.

    from engine_v6.recorder import SessionRecorder, SpotTracker

THE MEASUREMENT BUG THIS FIXES
------------------------------
The V4 live loop polled once per second in a loop that also fetched the market, the
spot quote and two order books per iteration. Each iteration cost roughly 0.6-1.2s of
network time, so the effective cadence was closer to 2 seconds, and the loop only
recorded the window-open mid on a pass that happened to fall inside the first 25
seconds of the window.

Every gate is a DIFFERENCE between an open value and a decision value:

    direction gate   up_mid_now - up_mid_open
    spot gate        spot_now / spot_open - 1
    imbalance gate   signed imbalance at the decision instant

A late or sparse ``open`` therefore shrinks every measured difference, and all three
gates reject more often. The live session showed exactly that: 48% of windows refused
for too small a move against roughly 27% in backtest, and 87% refused by the spot gate
against roughly 10% in backtest.

The fix is to separate the two jobs:

* **SpotTracker** runs in its own thread and samples the spot price continuously, so
  ``spot_open`` is the first quote at or after the window start and ``spot_now`` is the
  quote nearest the decision instant. It keeps the whole series, so the decision can
  use a timestamp-accurate pair instead of "whatever was last fetched".
* **SessionRecorder** records the Up-token mid as soon as it sees a window, and
  re-checks at a fixed cadence until the decision second, keeping the value closest to
  the true window open.

The recorder also stores the imbalance computed at SEVERAL depth caps (5, 10, 20,
whole book) on every decision, so the live distribution can be compared against the
backtest's top-10 distribution without re-running the session. That comparison is what
decides whether the 0.30 threshold transfers.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

#: depth caps recorded on every decision, to compare live against the dataset
DEPTH_PROBES = (5, 10, 20, None)


def _epoch_from_slug(slug: str) -> int:
    """Recover the window epoch from `btc-updown-5m-<epoch>`; 0 if unparseable."""
    try:
        return int(str(slug).rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return 0


@dataclass
class SpotPoint:
    ts: float
    price: float
    source: str


class SpotTracker:
    """Continuous spot sampling on its own thread.

    A 4.5bps move accrues in roughly six seconds at the sampled volatility, so a
    multi-second gap between samples loses a large part of what the gate is testing.
    Sampling every ``interval`` seconds keeps that loss small and, importantly,
    bounds it: the tracker reports its own observed gaps so the bound is measurable
    rather than assumed.
    """

    def __init__(self, spot_feed, interval: float = 0.25):
        self.spot_feed = spot_feed
        self.interval = interval
        self.points: list[SpotPoint] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0
        self.gaps: list[float] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="spot-tracker",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        last_ts = None
        while not self._stop.is_set():
            try:
                quote = self.spot_feed.quote()
            except Exception:                              # noqa: BLE001
                quote = None
            if quote is not None:
                with self._lock:
                    self.points.append(SpotPoint(quote.ts, quote.price,
                                                 quote.source))
                    if last_ts is not None:
                        self.gaps.append(quote.ts - last_ts)
                    last_ts = quote.ts
            else:
                self.errors += 1
            self._stop.wait(self.interval)

    def at(self, ts: float, tolerance: float | None = None,
           prefer_after: bool = True) -> SpotPoint | None:
        """The sample nearest ``ts``.

        With ``prefer_after`` the first sample at or after ``ts`` wins when one exists
        within tolerance, because the window open is a lower bound: a quote from
        before the window started is not the open price.
        """
        with self._lock:
            points = list(self.points)
        if not points:
            return None
        tol = tolerance if tolerance is not None else max(2.0, self.interval * 8)
        after = [p for p in points if p.ts >= ts and p.ts - ts <= tol]
        if prefer_after and after:
            return min(after, key=lambda p: p.ts - ts)
        near = [p for p in points if abs(p.ts - ts) <= tol]
        if near:
            return min(near, key=lambda p: abs(p.ts - ts))
        return None

    def between(self, ts_from: float, ts_to: float) -> list[SpotPoint]:
        with self._lock:
            return [p for p in self.points if ts_from <= p.ts <= ts_to]

    def stats(self) -> dict:
        with self._lock:
            n = len(self.points)
            gaps = sorted(self.gaps)
        return {
            "samples": n, "errors": self.errors,
            "gap_median_s": (gaps[len(gaps) // 2] if gaps else None),
            "gap_p90_s": (gaps[9 * len(gaps) // 10] if gaps else None),
            "gap_max_s": (gaps[-1] if gaps else None),
        }


@dataclass
class WindowCapture:
    """What was observed for one 5-minute window."""

    slug: str
    epoch: int
    start_ts: float
    #: (timestamp, up_mid) nearest the true window open, plus how late it was
    open_ts: float | None = None
    open_up_mid: float | None = None
    open_lag_s: float | None = None
    #: (timestamp, book) at the decision instant
    decision_ts: float | None = None
    decision_up_mid: float | None = None
    decision_book: object | None = None
    decision_ask: float | None = None
    decision_spread: float | None = None
    #: imbalance measured at several depth caps: cap -> value
    imbalances: dict = field(default_factory=dict)
    notionals: dict = field(default_factory=dict)
    depth_levels: tuple[int, int] = (0, 0)
    #: fill quality: the ask one and two seconds after the decision
    ask_1s: float | None = None
    ask_2s: float | None = None
    decided: bool = False


class SessionRecorder:
    """Tracks window opens and builds a capture per window."""

    def __init__(self, window_seconds: float = 300.0):
        self.window_seconds = window_seconds
        self.windows: dict[str, WindowCapture] = {}

    def observe(self, slug: str, epoch: int, up_mid: float | None,
                book: object | None = None) -> WindowCapture:
        """Record an observation, keeping the one closest to the window open."""
        capture = self.windows.setdefault(
            slug, WindowCapture(slug=slug, epoch=epoch, start_ts=float(epoch)))
        if up_mid is None:
            return capture
        lag = time.time() - capture.start_ts
        # keep the earliest observation, but only if it is not from BEFORE the start
        if lag >= -0.5 and (capture.open_lag_s is None or lag < capture.open_lag_s):
            capture.open_lag_s = lag
            capture.open_ts = time.time()
            capture.open_up_mid = up_mid
        return capture

    def mark_decision(self, slug: str, up_mid: float | None,
                      book: object | None, ask: float | None,
                      spread: float | None,
                      epoch: int | None = None) -> WindowCapture:
        """Record the decision for a window.

        Uses ``setdefault`` rather than ``self.windows[slug]``, which crashed the live
        loop with ``KeyError: 'btc-updown-5m-...'`` whenever the decision fired for a
        window that had never been observed. That happens routinely, not rarely:

        * the loop starts mid-window and the decision second arrives before any
          observation of that window completed
        * a slow market lookup or book fetch consumes the whole observation phase
        * a transient feed error skips the observation branch for that window

        A KeyError here takes down the entire session, so the capture is created on
        demand and the missing open is recorded as such. ``open_lag_s`` stays None,
        which the session summary already reports as "NEVER captured" - the window is
        then refused for a missing open rather than silently traded on a substituted
        value.
        """
        if epoch is None:
            epoch = _epoch_from_slug(slug)
        capture = self.windows.setdefault(
            slug, WindowCapture(slug=slug, epoch=epoch, start_ts=float(epoch)))
        capture.decision_ts = time.time()
        capture.decision_up_mid = up_mid
        capture.decision_book = book
        capture.decision_ask = ask
        capture.decision_spread = spread
        capture.decided = True
        if book is not None:
            capture.depth_levels = (book.bid_depth, book.ask_depth)
            for cap in DEPTH_PROBES:
                key = "full" if cap is None else str(cap)
                try:
                    capture.imbalances[key] = book.imbalance(cap)
                    capture.notionals[f"bid_{key}"] = book.notional("bids", cap)
                    capture.notionals[f"ask_{key}"] = book.notional("asks", cap)
                except (AttributeError, TypeError):
                    continue
        return capture

    def open_lag_summary(self) -> dict:
        lags = sorted(w.open_lag_s for w in self.windows.values()
                      if w.open_lag_s is not None)
        if not lags:
            return {}
        n = len(lags)
        return {"n": n, "median_s": lags[n // 2], "p90_s": lags[9 * n // 10],
                "max_s": lags[-1]}
