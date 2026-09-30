"""Temporal split (P2 task 0). Reads configs/p2_split.yaml; never hard-codes a boundary.

Runtime-safe: no ground truth is read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPLIT_CONFIG = ROOT / "configs" / "p2_split.yaml"
P1_DECISIONS = ROOT / "configs" / "p1_data_decisions.yaml"

FITTED_PERIODS = ("TRAIN", "VALIDATION", "CALIBRATION")
PRE_TEST_PERIODS = FITTED_PERIODS
ORDER = ("TRAIN", "EMBARGO_1", "VALIDATION", "EMBARGO_2", "CALIBRATION", "EMBARGO_3", "TEST")


class SplitError(ValueError):
    """The split config violates a frozen constraint."""


def _ts(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    if d.tzinfo is None or d.utcoffset() != timedelta(0):
        raise SplitError(f"timestamp {s!r} must be timezone-aware UTC")
    return d


@dataclass(frozen=True)
class Split:
    l_max: timedelta
    window: timedelta
    burn_in: tuple[datetime, datetime]
    periods: dict[str, tuple[datetime, datetime]]
    span: tuple[datetime, datetime]
    status: str
    raw: dict

    # ------------------------------------------------------------ queries
    def period_of(self, window_start: datetime) -> str | None:
        """Period containing a window start; None for burn-in / outside the span."""
        _ts(window_start.isoformat())
        for name, (a, b) in self.periods.items():
            if a <= window_start < b:
                return name
        return None

    def window_starts(self, period: str | None = None) -> list[datetime]:
        """All daily window starts (in a period, or in every non-embargo period)."""
        names = [period] if period else [p for p in ORDER if not p.startswith("EMBARGO")]
        out: list[datetime] = []
        for n in names:
            a, b = self.periods[n]
            t = a
            while t < b:
                out.append(t)
                t += self.window
        return sorted(out)

    def as_of(self, window_start: datetime) -> datetime:
        return window_start + self.window

    def window_end(self, window_start: datetime) -> datetime:
        return self.as_of(window_start) - timedelta(seconds=1)

    def lookback_start(self, as_of: datetime) -> datetime:
        """Earliest visible timestamp for a rule/feature with this as_of (inclusive)."""
        return as_of - self.l_max


def load_split(path: Path = SPLIT_CONFIG, p1_path: Path = P1_DECISIONS) -> Split:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    p1 = yaml.safe_load(p1_path.read_text(encoding="utf-8"))
    if set(cfg["periods"]) != set(ORDER):
        raise SplitError(f"periods must be exactly {ORDER}, got {sorted(cfg['periods'])}")
    periods = {k: (_ts(cfg["periods"][k]["start"]), _ts(cfg["periods"][k]["end"])) for k in ORDER}
    s = Split(
        l_max=timedelta(hours=cfg["l_max_hours"]),
        window=timedelta(hours=cfg["window"]["length_hours"]),
        burn_in=(_ts(cfg["burn_in"]["start"]), _ts(cfg["burn_in"]["end"])),
        periods=periods,
        span=(_ts(p1["span"]["start"]), _ts(p1["span"]["end_exclusive"])),
        status=cfg["status"],
        raw=cfg,
    )
    validate(s, p1)
    return s


def validate(s: Split, p1: dict) -> None:
    """Every structural constraint of the temporal split. Raises SplitError."""
    if tuple(s.periods) != ORDER:
        raise SplitError(f"periods must be exactly {ORDER} in order, got {tuple(s.periods)}")
    if s.window != timedelta(hours=24):
        raise SplitError("window must be 24 h (daily batch)")
    if s.l_max > s.window:
        raise SplitError("rules/features may not look back further than L_max; L_max > window")
    # contiguous cover: burn-in then periods, no gaps/overlaps, ending at the span end
    if s.burn_in[0] != s.span[0]:
        raise SplitError("burn-in must start at the span start")
    prev_end = s.burn_in[1]
    for name in ORDER:
        a, b = s.periods[name]
        if a != prev_end:
            raise SplitError(f"{name} starts {a}, expected {prev_end} (gap or overlap)")
        if not a < b:
            raise SplitError(f"{name} is empty")
        for t in (a, b):
            if t.hour or t.minute or t.second or t.microsecond:
                raise SplitError(f"{name} boundary {t} is not a day boundary")
        if name.startswith("EMBARGO") and (b - a) < s.l_max:
            raise SplitError(f"{name} is {b - a}, shorter than L_max {s.l_max}")
        prev_end = b
    if prev_end != s.span[1]:
        raise SplitError(f"TEST must end at the span end {s.span[1]} (D1), got {prev_end}")
    test_floor = _ts(p1["provisional_holdout"]["test_start_not_before"])
    if s.periods["TEST"][0] < test_floor:
        raise SplitError(f"TEST starts before {test_floor} (D4)")
    if s.status not in ("PROVISIONAL", "FROZEN"):
        raise SplitError(f"unknown status {s.status}")
    if s.status == "FROZEN" and not s.raw.get("frozen_on"):
        raise SplitError("a FROZEN split needs frozen_on")
    frac = s.raw["agent_split"]["fraction_agent_test"]
    if not 0 < frac < 1:
        raise SplitError("fraction_agent_test must be in (0, 1)")
