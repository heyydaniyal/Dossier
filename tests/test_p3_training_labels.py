"""src/eval/training_labels.py: the only label loader for model fitting (open risk 26)."""

from __future__ import annotations

import polars as pl
import pytest

from src.eval.training_labels import FIT_PERIODS, HoldoutAccessError, load_fit_labels


@pytest.mark.parametrize(
    "periods", [["TEST"], ["TRAIN", "TEST"], [], ["train"], ["EMBARGO_1"], ["AGENT-TEST"]]
)
def test_refuses_anything_but_fit_periods_before_opening_files(tmp_path, periods):
    # tmp_path holds no store at all: refusal must come before any file access
    with pytest.raises(HoldoutAccessError):
        load_fit_labels(periods, "FIX", tmp_path)


def test_returns_exactly_the_requested_periods(p2_run):
    root = p2_run["base"].parent
    alerts = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    truth = pl.read_parquet(p2_run["base"] / "eval" / "alert_labels.parquet")
    for periods in (["TRAIN"], ["TRAIN", "VALIDATION"], list(FIT_PERIODS)):
        got = load_fit_labels(periods, "FIX", root)
        want = alerts.filter(pl.col("period").is_in(periods)).select("alert_id", "period")
        assert got.columns == ["alert_id", "period", "y"]
        assert sorted(got["alert_id"].to_list()) == sorted(want["alert_id"].to_list())
        j = got.join(truth, on="alert_id")
        assert (j["y"] == j["is_true_positive"].cast(pl.Int8)).all()
        assert not got.filter(pl.col("period") == "TEST").height
    assert load_fit_labels(["TRAIN"], "FIX", root)["y"].sum() > 0  # the fixture has TRAIN positives


def test_missing_label_rows_fail_loudly(p2_run, tmp_path):
    base = tmp_path / "FIX"
    (base / "runtime").mkdir(parents=True)
    (base / "eval").mkdir()
    a = pl.read_parquet(p2_run["base"] / "runtime" / "alerts.parquet")
    a.write_parquet(base / "runtime" / "alerts.parquet")
    lab = pl.read_parquet(p2_run["base"] / "eval" / "alert_labels.parquet")
    drop = a.filter(pl.col("period") == "TRAIN")["alert_id"][0]
    lab.filter(pl.col("alert_id") != drop).write_parquet(base / "eval" / "alert_labels.parquet")
    with pytest.raises(RuntimeError, match="no label row"):
        load_fit_labels(["TRAIN"], "FIX", tmp_path)
