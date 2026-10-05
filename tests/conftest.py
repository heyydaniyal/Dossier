"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run_fixture_pipeline(root: Path) -> dict:
    from scripts.p2 import run_p2
    from tests import p2_fixture

    info = p2_fixture.make(root)
    cfg = yaml.safe_load((ROOT / "configs" / "p2_rules.yaml").read_text(encoding="utf-8"))
    # The fixture is tiny: widen the TRAIN precision band and disable the strength checks so
    # calibration always completes. The real bands are tested on the real results JSON.
    cfg["calibration"]["precision_band"] = [0.0, 1.0]
    cfg["calibration"]["too_strong"] = {"precision_above": 1.01, "recall_above": 1.01}
    cfg["calibration"]["max_layer_recall"] = 1.01
    cfg["calibration"]["min_peer_rows"] = 50
    cfg["calibration"]["precision_target"] = 0.0
    cfg["calibration"]["per_rule_min_true_alerts"] = 0
    cfg["calibration"]["per_rule_min_precision"] = 0.0
    cfg["calibration"]["min_active_rules"] = 1
    rules_path = root / "rules.yaml"
    rules_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    # The fixture's TEST period has a handful of positives, so the AGENT split gates (>= 150
    # positives per group, largest component <= 20%) cannot pass: relax them for the fixture only,
    # so split v2 selects the most conservative variant. The selection rule itself is unit-tested.
    scfg = yaml.safe_load((ROOT / "configs" / "p2_split.yaml").read_text(encoding="utf-8"))
    scfg["feasibility"]["min_positive_alerts_per_agent_group"] = 0
    scfg["feasibility"]["max_share_of_test_positives_in_one_component"] = 1.0
    split_path = root / "split.yaml"
    split_path.write_text(yaml.safe_dump(scfg, sort_keys=False), encoding="utf-8")
    cp = run_p2.ConfigPaths(rules=rules_path, sources=info["sources"], split=split_path)
    cdir = root / "configs"
    cdir.mkdir()
    doc = run_p2.run(
        "all",
        "FIX",
        info["interim"],
        root / "out",
        cdir,
        root / "results.json",
        20260930,
        cp=cp,
        primary="FIX",
    )
    base = root / "out" / "FIX"
    return {"doc": doc, "root": root, "base": base, "configs": cdir, "cp": cp, "info": info}


@pytest.fixture(scope="session")
def p2_run(tmp_path_factory) -> dict:
    return _run_fixture_pipeline(tmp_path_factory.mktemp("p2run"))


@pytest.fixture(scope="session")
def p2_run_again(tmp_path_factory) -> dict:
    return _run_fixture_pipeline(tmp_path_factory.mktemp("p2run_again"))
