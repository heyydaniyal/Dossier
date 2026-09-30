"""P2 task 0: the temporal split obeys every frozen constraint, and violations are caught."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from src.data.periods import ORDER, SPLIT_CONFIG, SplitError, load_split

D = lambda day: datetime(2022, 9, day, tzinfo=UTC)  # noqa: E731


def test_split_loads_and_matches_confirmed_calendar():
    s = load_split()
    assert s.l_max == timedelta(hours=24)
    assert s.burn_in == (D(1), D(2))
    assert s.periods["TRAIN"] == (D(2), D(6))
    assert s.periods["VALIDATION"] == (D(7), D(8))
    assert s.periods["CALIBRATION"] == (D(9), D(12))
    assert s.periods["TEST"] == (D(13), D(17))
    assert [len(s.window_starts(p)) for p in ("TRAIN", "VALIDATION", "CALIBRATION", "TEST")] == [
        4,
        1,
        3,
        4,
    ]


def test_embargo_at_least_lmax_and_ordering():
    s = load_split()
    assert tuple(s.periods) == ORDER
    for name, (a, b) in s.periods.items():
        if name.startswith("EMBARGO"):
            assert b - a >= s.l_max
    # a window of the next period cannot see anything that an earlier period's window saw, and a
    # further L_max of data separates them (the embargo)
    fitted = ["TRAIN", "VALIDATION", "CALIBRATION", "TEST"]
    for x, y in zip(fitted, fitted[1:], strict=False):
        last_as_of = s.as_of(s.window_starts(x)[-1])
        first_lookback = s.lookback_start(s.as_of(s.window_starts(y)[0]))
        assert first_lookback - last_as_of >= s.l_max


def test_period_of_and_window_times():
    s = load_split()
    assert s.period_of(D(1)) is None  # burn-in
    assert s.period_of(D(2)) == "TRAIN"
    assert s.period_of(D(6)) == "EMBARGO_1"
    assert s.period_of(D(12)) == "EMBARGO_3"
    assert s.period_of(D(16)) == "TEST"
    assert s.period_of(D(17)) is None  # after span (D1)
    ws = D(5)
    assert s.as_of(ws) == D(6)
    assert s.window_end(ws) == D(6) - timedelta(seconds=1)
    assert s.lookback_start(s.as_of(ws)) == ws
    assert all(s.period_of(w) and not s.period_of(w).startswith("EMB") for w in s.window_starts())


def _write(tmp: Path, mut) -> Path:
    cfg = yaml.safe_load(SPLIT_CONFIG.read_text(encoding="utf-8"))
    mut(cfg)
    p = tmp / "split.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "mut,msg",
    [
        (lambda c: c.update(l_max_hours=48), "shorter than L_max|L_max > window"),
        (
            lambda c: c["periods"]["TEST"].update(start="2022-09-12T00:00:00+00:00"),
            "starts",
        ),
        (lambda c: c["periods"]["TRAIN"].update(end="2022-09-05T00:00:00+00:00"), "gap"),
        (lambda c: c["periods"]["TEST"].update(end="2022-09-18T00:00:00+00:00"), "span end"),
        (lambda c: c["periods"]["TRAIN"].update(start="2022-09-02T00:00:00"), "timezone"),
        (lambda c: c.update(status="FROZEN", frozen_on=None), "frozen_on"),
    ],
)
def test_violations_are_rejected(tmp_path: Path, mut, msg):
    with pytest.raises(SplitError, match=msg):
        load_split(_write(tmp_path, mut))


def test_test_start_never_before_d4(tmp_path: Path):
    def earlier(c):
        c["periods"]["EMBARGO_3"].update(end="2022-09-12T00:00:00+00:00")
        c["periods"]["TEST"].update(start="2022-09-12T00:00:00+00:00")
        c["periods"]["CALIBRATION"].update(end="2022-09-11T00:00:00+00:00")
        c["periods"]["EMBARGO_3"].update(start="2022-09-11T00:00:00+00:00")

    with pytest.raises(SplitError, match="D4"):
        load_split(_write(tmp_path, earlier))


def test_committed_split_is_frozen():
    from src.data.periods import load_split

    s = load_split()
    assert s.status == "FROZEN" and s.raw["frozen_on"] == "2026-09-30"
