"""P2 task 4: rule thresholds calibrated on TRAIN only. Offline; reads TRAIN alert labels.

Procedure (declared in configs/p2_rules.yaml before any measurement):
  1. For every tau in the grid: threshold_r = max(floor_r, TRAIN quantile_tau(stat_r)), per peer
     group for R02. Measure combined TRAIN precision/recall of the whole layer.
  2. Pick the tau whose precision is closest to the target (log space) inside the band.
  3. Any rule that is 'too strong' gets its own tau loosened one grid step at a time; if the grid
     runs out, the rule is dropped. Every action is recorded.
  4. Re-check the band and the layer-recall ceiling. Anything outside -> CalibrationError (no
     silent fix: the result is printed and a human decides).

Implemented on numpy arrays (not per-evaluation DataFrame copies) so ~6M TRAIN account-days fit
in laptop memory. Firing semantics are identical to src.alerts.rules.fire (tested).
Quantile 'higher' = the sorted value at index ceil(tau * (n - 1)): always an observed value.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.alerts.rules import rule_ids


class CalibrationError(RuntimeError):
    pass


def _sig(x: float) -> float:
    return float(f"{float(x):.10g}")


class Arrays:
    """Column arrays of the TRAIN account-days needed for calibration, sorted once per rule."""

    def __init__(self, stats: pl.DataFrame, cfg: dict):
        self.n = stats.height
        self.pos = stats["is_pos"].to_numpy().astype(bool)
        self.peer = stats["peer_group"].to_numpy().astype(object)
        self.groups = sorted(set(self.peer.tolist()))
        self.peer_mask = {g: self.peer == g for g in self.groups}
        self.col: dict[str, np.ndarray] = {}
        self.sorted: dict[str, np.ndarray] = {}
        self.sorted_peer: dict[tuple[str, str], np.ndarray] = {}
        for r in rule_ids(cfg):
            spec = cfg["rules"][r]
            c = spec["stat"]
            v = stats[c].cast(pl.Float64).to_numpy()
            self.col[c] = v
            self.sorted[c] = np.sort(v)
            if spec.get("peer_group"):
                for g in self.groups:
                    self.sorted_peer[(c, g)] = np.sort(v[self.peer_mask[g]])


def _q(sorted_v: np.ndarray, tau: float) -> float:
    if sorted_v.size == 0:
        return math.inf
    return float(sorted_v[math.ceil(tau * (sorted_v.size - 1))])


def thresholds_at(a: Arrays, cfg: dict, taus: dict[str, float], active: dict[str, bool]) -> dict:
    cal = cfg["calibration"]
    out: dict = {}
    for r in rule_ids(cfg):
        spec = cfg["rules"][r]
        col, floor, tau = spec["stat"], float(spec["floor"]), taus[r]
        entry: dict = {"active": active[r], "tau": tau, "stat": col, "floor": floor}
        if spec.get("peer_group"):
            by_peer = {
                g: _sig(max(floor, _q(a.sorted_peer[(col, g)], tau)))
                for g in a.groups
                if a.sorted_peer[(col, g)].size >= cal["min_peer_rows"]
            }
            entry |= {"by_peer": by_peer, "default": _sig(max(floor, _q(a.sorted[col], tau)))}
        else:
            entry["threshold"] = _sig(max(floor, _q(a.sorted[col], tau)))
        out[r] = entry
    return {"rules": out}


def fired(a: Arrays, thr: dict, cfg: dict) -> dict[str, np.ndarray]:
    """Same semantics as src.alerts.rules.fire: stat > 0 and stat >= threshold."""
    out = {}
    for r in rule_ids(cfg):
        spec = thr["rules"][r]
        if not spec["active"]:
            continue
        v = a.col[cfg["rules"][r]["stat"]]
        if "by_peer" in spec:
            t = np.full(a.n, float(spec["default"]))
            for g, x in spec["by_peer"].items():
                if g in a.peer_mask:
                    t[a.peer_mask[g]] = float(x)
        else:
            t = float(spec["threshold"])
        out[r] = (v > 0) & (v >= t)
    return out


def evaluate_arrays(a: Arrays, thr: dict, cfg: dict, n_pos_total: int, detail: bool = True) -> dict:
    f = fired(a, thr, cfg)
    active = list(f)
    n_rules = np.zeros(a.n, dtype=np.int64)
    for v in f.values():
        n_rules += v
    alerted = n_rules > 0
    n_al, n_tp = int(alerted.sum()), int((alerted & a.pos).sum())
    per_rule = {}
    for r in active:
        m = f[r]
        na, tp = int(m.sum()), int((m & a.pos).sum())
        per_rule[r] = {
            "n_alerts": na,
            "n_true": tp,
            "precision": _sig(tp / na) if na else None,
            "recall": _sig(tp / n_pos_total) if n_pos_total else None,
            "n_alerts_only_this_rule": int((m & (n_rules == 1)).sum()),
        }
    res = {
        "n_alerts": n_al,
        "n_true_alerts": n_tp,
        "precision": _sig(n_tp / n_al) if n_al else None,
        "recall": _sig(n_tp / n_pos_total) if n_pos_total else None,
        "n_positive_account_days": n_pos_total,
        "per_rule": per_rule,
    }
    if detail:
        res["overlap_p_b_given_a"] = {
            x: {
                y: (_sig(int((f[x] & f[y]).sum()) / int(f[x].sum())) if f[x].any() else None)
                for y in active
                if y != x
            }
            for x in active
        }
        vals, cnt = np.unique(n_rules[alerted], return_counts=True)
        res["n_rules_triggered_hist"] = {
            str(int(v)): int(c) for v, c in zip(vals, cnt, strict=True)
        }
    return res


def evaluate(stats: pl.DataFrame, thr: dict, cfg: dict, n_pos_total: int) -> dict:
    """stats must carry peer_group and is_pos."""
    return evaluate_arrays(Arrays(stats, cfg), thr, cfg, n_pos_total)


def _too_strong(m: dict, cfg: dict) -> bool:
    ts = cfg["calibration"]["too_strong"]
    p, r = m["precision"] or 0.0, m["recall"] or 0.0
    return p > ts["precision_above"] and r > ts["recall_above"]


def calibrate(stats: pl.DataFrame, cfg: dict, n_pos_total: int) -> dict:
    """stats: TRAIN account-days with peer_group and is_pos. Returns the thresholds document."""
    cal = cfg["calibration"]
    if cal.get("quantile_interpolation") != "higher":
        raise CalibrationError("only quantile_interpolation 'higher' is implemented")
    grid = list(cal["tau_grid"])
    if grid != sorted(grid, reverse=True):
        raise CalibrationError("tau_grid must run from strict to loose")
    rules = rule_ids(cfg)
    lo, hi = cal["precision_band"]
    target = cal["precision_target"]
    a = Arrays(stats, cfg)

    curve = []
    for tau in grid:
        thr = thresholds_at(a, cfg, dict.fromkeys(rules, tau), dict.fromkeys(rules, True))
        m = evaluate_arrays(a, thr, cfg, n_pos_total, detail=False)
        curve.append(
            {
                "tau": tau,
                "n_alerts": m["n_alerts"],
                "precision": m["precision"],
                "recall": m["recall"],
            }
        )
    inside = [c for c in curve if c["precision"] is not None and lo <= c["precision"] <= hi]
    if not inside:
        raise CalibrationError(f"no tau gives TRAIN precision in {lo}-{hi}; curve: {curve}")
    best = min(inside, key=lambda c: (abs(math.log(c["precision"] / target)), -c["recall"]))
    tau_star = best["tau"]

    taus = dict.fromkeys(rules, tau_star)
    active = dict.fromkeys(rules, True)
    actions: list[dict] = []
    for r in rules:  # config order
        while True:
            thr = thresholds_at(a, cfg, taus, active)
            m = evaluate_arrays(a, thr, cfg, n_pos_total, detail=False)["per_rule"][r]
            if not _too_strong(m, cfg):
                break
            i = grid.index(taus[r])
            step = {"rule": r, "precision": m["precision"], "recall": m["recall"]}
            if i + 1 < len(grid):
                actions.append(
                    step | {"action": "loosen", "from_tau": taus[r], "to_tau": grid[i + 1]}
                )
                taus[r] = grid[i + 1]
            else:
                actions.append(step | {"action": "drop"})
                active[r] = False
                break

    thr = thresholds_at(a, cfg, taus, active)
    final = evaluate_arrays(a, thr, cfg, n_pos_total)
    problems = []
    if final["precision"] is None or not lo <= final["precision"] <= hi:
        problems.append(f"final precision {final['precision']} outside {lo}-{hi}")
    if final["recall"] is not None and final["recall"] >= cal["max_layer_recall"]:
        problems.append(f"layer recall {final['recall']} >= {cal['max_layer_recall']}")
    if sum(active.values()) < 6:
        problems.append("fewer than 6 active rules")
    for r, m in final["per_rule"].items():
        if _too_strong(m, cfg):
            problems.append(f"{r} still too strong: {m}")
    if problems:
        raise CalibrationError("; ".join(problems) + f" | curve: {curve} | actions: {actions}")
    return thr | {
        "tau_global": tau_star,
        "tau_curve": curve,
        "actions": actions,
        "train_metrics": final,
    }
