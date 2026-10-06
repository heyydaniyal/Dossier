"""P3 point-in-time features for alerts. Runtime-safe: no label, typology or eval store is read.

Every feature of an alert with as_of T is computed from transactions with
    T - lookback <= ts < T          (frozen P0 tie rule; lookback = L_max = 24 h)
through src.data.transactions.visible, the one point-in-time filter. Nothing else is an input:
  - the label-free transaction loader (src.data.transactions.scan_transactions),
  - the TRAIN-fitted FX table (configs/p2_fx_usd_per_unit.yaml, D6),
  - the frozen rule definitions and TRAIN-fitted thresholds (P2), used only to recompute the rule
    statistics and fired flags every alert already carries,
  - the alert's own peer_group, used only to recompute the frozen R02 flag (P2 rule; the group
    itself is not a feature: KYC fields are excluded from P3-P5 model features).

Two implementations:
  batch_features()   vectorised, one pass per window (polars + scipy.sparse);
  account_features() pure f(data, account_key, as_of): an independent, slow reference written
                     with plain Python over the same visible rows. Tests require both to agree.
Cross-sectional statistics (peer ranks, graph) are computed over ALL accounts active in the same
visible window, never over other days and never over the whole dataset.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import yaml
from scipy.sparse.csgraph import connected_components

from src.alerts import rules
from src.data.periods import ROOT, Split, load_split
from src.data.transactions import visible

FEATURES_CONFIG = ROOT / "configs" / "p3_features.yaml"
FX_FILE = ROOT / "configs" / "p2_fx_usd_per_unit.yaml"
ALERT_KEYS = ["alert_id", "account_key", "window_start", "as_of"]


class FeatureError(RuntimeError):
    """A feature input violates a frozen constraint. Never swallowed."""


@dataclass(frozen=True)
class FeatureContext:
    lookback: timedelta
    rules_cfg: dict
    thresholds: dict
    formats: tuple[str, ...]
    peer_stats: tuple[str, ...]

    @property
    def active_rules(self) -> list[str]:
        t = self.thresholds["rules"]
        return [r for r in rules.rule_ids(self.rules_cfg) if r in t and t[r]["active"]]


def load_features_config(path: Path = FEATURES_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_context(
    rules_cfg: dict, thresholds: dict, features_cfg: dict, split: Split | None = None
) -> FeatureContext:
    split = split or load_split()
    lookback = timedelta(hours=features_cfg["lookback_hours"])
    if lookback > split.l_max:
        raise FeatureError(f"feature lookback {lookback} exceeds the frozen L_max {split.l_max}")
    if lookback <= timedelta(0):
        raise FeatureError("feature lookback must be positive")
    return FeatureContext(
        lookback=lookback,
        rules_cfg=rules_cfg,
        thresholds=thresholds,
        formats=tuple(features_cfg["payment_formats"]),
        peer_stats=tuple(features_cfg["peer_stats"]),
    )


def default_context() -> FeatureContext:
    return make_context(rules.load_rules(), rules.load_thresholds(), load_features_config())


def slug(x: str) -> str:
    return x.lower().replace(" ", "_")


# ------------------------------------------------------------------ feature names (by group)


def rule_names(ctx: FeatureContext) -> list[str]:
    return [*rules.ALL_STATS, "n_rules_triggered", *[f"fired_{r}" for r in ctx.active_rules]]


BEHAVIOUR = [
    "n_in",
    "n_out",
    "amt_mean_usd",
    "amt_median_usd",
    "amt_min_usd",
    "amt_std_usd",
    "amt_cv",
    "amt_log10_std",
    "share_near_threshold",
    "share_round",
    "share_cross_currency",
    "n_currencies",
    "n_counterparties",
    "cp_hhi",
    "max_txn_one_cp",
    "repeat_share",
    "net_flow_ratio",
    "n_self_transfers",
]
DYNAMICS = [
    "active_hours",
    "span_hours",
    "max_txn_per_hour",
    "median_interarrival_min",
    "in_to_out_lag_min",
]
GRAPH = [
    "reciprocal_cp",
    "cycles3",
    "reach2_out",
    "reach2_in",
    "senders_mean_outdeg",
    "receivers_mean_indeg",
    "wcc_log10_size",
]


def format_names(ctx: FeatureContext) -> list[str]:
    return [f"share_fmt_{slug(f)}" for f in ctx.formats]


def peer_names(ctx: FeatureContext) -> list[str]:
    return [f"pct_{s}" for s in ctx.peer_stats]


def group_names(ctx: FeatureContext) -> dict[str, list[str]]:
    return {
        "rule": rule_names(ctx),
        "behaviour": list(BEHAVIOUR),
        "format_mix": format_names(ctx),
        "dynamics": list(DYNAMICS),
        "peer": peer_names(ctx),
        "graph": list(GRAPH),
    }


def feature_names(ctx: FeatureContext) -> list[str]:
    return [n for names in group_names(ctx).values() for n in names]


# ------------------------------------------------------------------ the visible window


def window_frame(trans_usd: pl.LazyFrame, as_of: datetime, ctx: FeatureContext) -> pl.DataFrame:
    """Every transaction an alert with this as_of may see (lookback <= ts < as_of)."""
    return visible(trans_usd, as_of, as_of - ctx.lookback).collect()


def _legs(win: pl.DataFrame) -> pl.DataFrame:
    """One row per (non-self transaction, role), with the leg's own currency and format."""
    base = win.filter(pl.col("from_key") != pl.col("to_key"))
    common = [pl.col("timestamp"), pl.col("payment_format").alias("fmt")]
    out = base.select(
        pl.col("from_key").alias("account_key"),
        pl.col("to_key").alias("counterparty"),
        pl.lit("out").alias("direction"),
        pl.col("usd_paid").alias("usd"),
        pl.col("payment_currency").alias("currency"),
        *common,
    )
    inn = base.select(
        pl.col("to_key").alias("account_key"),
        pl.col("from_key").alias("counterparty"),
        pl.lit("in").alias("direction"),
        pl.col("usd_received").alias("usd"),
        pl.col("receiving_currency").alias("currency"),
        *common,
    )
    return pl.concat([out, inn])


def _check_formats(win: pl.DataFrame, ctx: FeatureContext) -> None:
    unknown = set(win["payment_format"].unique().to_list()) - set(ctx.formats)
    if unknown:
        raise FeatureError(f"payment formats not in configs/p3_features.yaml: {sorted(unknown)}")


# ------------------------------------------------------------------ batch path


def _rule_block(win: pl.DataFrame, ctx: FeatureContext) -> pl.DataFrame:
    """Frozen P2 statistics for EVERY account active in the window (needed for peer ranks)."""
    st = rules.account_stats(rules.legs(win.lazy(), ctx.rules_cfg), ctx.rules_cfg).collect()
    return st.with_columns([pl.col(c).fill_null(0) for c in rules.ALL_STATS])


def _behaviour_block(legs: pl.DataFrame, win: pl.DataFrame, st: pl.DataFrame, ctx) -> pl.DataFrame:
    cp = (
        legs.group_by("account_key", "counterparty")
        .agg(pl.col("usd").sum().alias("cp_usd"), pl.len().alias("cp_n"))
        .with_columns((pl.col("cp_usd") / pl.col("cp_usd").sum().over("account_key")).alias("w"))
        .group_by("account_key")
        .agg(
            pl.len().cast(pl.Float64).alias("n_counterparties"),
            (pl.col("w") ** 2).sum().alias("cp_hhi"),
            pl.col("cp_n").max().cast(pl.Float64).alias("max_txn_one_cp"),
        )
    )
    amt = legs.group_by("account_key").agg(
        (pl.col("direction") == "in").sum().cast(pl.Float64).alias("n_in"),
        (pl.col("direction") == "out").sum().cast(pl.Float64).alias("n_out"),
        pl.col("usd").mean().alias("amt_mean_usd"),
        pl.col("usd").median().alias("amt_median_usd"),
        pl.col("usd").min().alias("amt_min_usd"),
        pl.col("usd").std(ddof=0).alias("amt_std_usd"),
        pl.col("usd").log10().std(ddof=0).alias("amt_log10_std"),
        pl.col("currency").n_unique().cast(pl.Float64).alias("n_currencies"),
        *[(pl.col("fmt") == f).mean().alias(f"share_fmt_{slug(f)}") for f in ctx.formats],
    )
    selfs = (
        win.filter(pl.col("from_key") == pl.col("to_key"))
        .group_by(pl.col("from_key").alias("account_key"))
        .agg(pl.len().cast(pl.Float64).alias("n_self_transfers"))
    )
    n = pl.col("n_txn").cast(pl.Float64)
    shares = st.select(
        "account_key",
        (pl.col("n_near_threshold") / n).alias("share_near_threshold"),
        (pl.col("n_round_amounts") / n).alias("share_round"),
        (pl.col("n_cross_currency") / n).alias("share_cross_currency"),
        ((pl.col("in_usd") - pl.col("out_usd")) / pl.col("volume_usd")).alias("net_flow_ratio"),
        n.alias("_n"),
    )
    return (
        amt.join(cp, on="account_key")
        .join(shares, on="account_key")
        .join(selfs, on="account_key", how="left")
        .with_columns(
            (pl.col("amt_std_usd") / pl.col("amt_mean_usd")).alias("amt_cv"),
            (1.0 - pl.col("n_counterparties") / pl.col("_n")).alias("repeat_share"),
            pl.col("n_self_transfers").fill_null(0.0),
        )
        .drop("_n")
    )


def _dynamics_block(legs: pl.DataFrame) -> pl.DataFrame:
    hourly = (
        legs.group_by("account_key", pl.col("timestamp").dt.truncate("1h").alias("h"))
        .agg(pl.len().alias("n"))
        .group_by("account_key")
        .agg(
            pl.len().cast(pl.Float64).alias("active_hours"),
            pl.col("n").max().cast(pl.Float64).alias("max_txn_per_hour"),
        )
    )
    minutes = lambda e: e.dt.total_seconds().cast(pl.Float64) / 60.0  # noqa: E731
    gaps = (
        legs.select("account_key", "timestamp")
        .sort("account_key", "timestamp")
        .with_columns(minutes(pl.col("timestamp").diff().over("account_key")).alias("gap"))
        .group_by("account_key")
        .agg(
            (minutes(pl.col("timestamp").max() - pl.col("timestamp").min()) / 60.0).alias(
                "span_hours"
            ),
            pl.col("gap").drop_nulls().median().alias("median_interarrival_min"),
        )
    )
    first_in = (
        legs.filter(pl.col("direction") == "in")
        .group_by("account_key")
        .agg(pl.col("timestamp").min().alias("_fi"))
    )
    lag = (
        legs.filter(pl.col("direction") == "out")
        .join(first_in, on="account_key")
        .filter(pl.col("timestamp") >= pl.col("_fi"))
        .group_by("account_key")
        .agg(minutes(pl.col("timestamp").min() - pl.col("_fi").first()).alias("in_to_out_lag_min"))
    )
    return hourly.join(gaps, on="account_key").join(lag, on="account_key", how="left")


def _peer_block(st: pl.DataFrame, ctx: FeatureContext) -> pl.DataFrame:
    n = st.height
    return st.select(
        "account_key",
        *[
            (pl.col(s).rank("average").cast(pl.Float64) / n).alias(f"pct_{s}")
            for s in ctx.peer_stats
        ],
    )


def window_edges(win: pl.DataFrame) -> pl.DataFrame:
    """Distinct directed counterparty pairs of the window (self-transfers excluded), with the
    first/last timestamp of each pair: the ONLY edges graph features use (graph test)."""
    return (
        win.filter(pl.col("from_key") != pl.col("to_key"))
        .group_by("from_key", "to_key")
        .agg(
            pl.col("timestamp").min().alias("first_ts"), pl.col("timestamp").max().alias("last_ts")
        )
    )


def _graph_block(win: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    edges = window_edges(win)
    nodes = (
        pl.concat([edges["from_key"], edges["to_key"]]).unique().sort()  # index only, not signal
    )
    idx = pl.DataFrame({"k": nodes, "i": np.arange(len(nodes), dtype=np.int64)})
    e = edges.join(idx.rename({"k": "from_key", "i": "s"}), on="from_key").join(
        idx.rename({"k": "to_key", "i": "d"}), on="to_key"
    )
    n = len(nodes)
    a = sp.csr_matrix(
        (np.ones(e.height), (e["s"].to_numpy(), e["d"].to_numpy())), shape=(n, n), dtype=np.float64
    )
    at = a.T.tocsr()
    outdeg = np.diff(a.indptr).astype(np.float64)
    indeg = np.diff(at.indptr).astype(np.float64)
    _, lab = connected_components(a, directed=True, connection="weak")
    comp = np.bincount(lab)
    pos = dict(zip(idx["k"].to_list(), idx["i"].to_list(), strict=True))
    missing = [k for k in targets if k not in pos]
    if missing:
        raise FeatureError(f"{len(missing)} alerted accounts have no edge in their window")
    t = np.array([pos[k] for k in targets], dtype=np.int64)
    rows = np.arange(len(t))
    asub, atsub = a[t], at[t]
    two_out, two_in = asub @ a, atsub @ at
    self_out = np.asarray(two_out[rows, t]).ravel() != 0
    self_in = np.asarray(two_in[rows, t]).ravel() != 0
    ov, iv = outdeg[t], indeg[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(iv > 0, (atsub @ outdeg) / iv, np.nan)
        rm = np.where(ov > 0, (asub @ indeg) / ov, np.nan)
    return pl.DataFrame(
        {
            "account_key": targets,
            "reciprocal_cp": np.asarray(asub.multiply(atsub).sum(axis=1)).ravel(),
            "cycles3": np.asarray(two_out.multiply(atsub).sum(axis=1)).ravel(),
            "reach2_out": np.diff(two_out.indptr) - self_out,
            "reach2_in": np.diff(two_in.indptr) - self_in,
            "senders_mean_outdeg": sm,
            "receivers_mean_indeg": rm,
            "wcc_log10_size": np.log10(comp[lab[t]]),
        }
    ).with_columns(pl.exclude("account_key").cast(pl.Float64))


def features_for_window(
    trans_usd: pl.LazyFrame, as_of: datetime, alerts: pl.DataFrame, ctx: FeatureContext
) -> pl.DataFrame:
    """Features for the alerts of ONE as_of. alerts: alert_id, account_key, window_start, as_of,
    peer_group (one row per account)."""
    if alerts["as_of"].n_unique() != 1 or alerts["as_of"][0] != as_of:
        raise FeatureError("features_for_window needs the alerts of exactly this as_of")
    if alerts["account_key"].n_unique() != alerts.height:
        raise FeatureError("one alert per account per window")
    win = window_frame(trans_usd, as_of, ctx)
    _check_formats(win, ctx)
    targets = alerts["account_key"].to_list()
    st = _rule_block(win, ctx)
    fired = rules.fire(
        st.join(alerts.select("account_key", "peer_group"), on="account_key"),
        ctx.thresholds,
        ctx.rules_cfg,
    )
    rule_df = fired.select(
        "account_key",
        *[pl.col(c).cast(pl.Float64) for c in rules.ALL_STATS],
        pl.col("n_rules_triggered").cast(pl.Float64),
        *[pl.col(f"fired_{r}").cast(pl.Float64) for r in ctx.active_rules],
        pl.col("triggered_rules").alias("_triggered"),
    )
    if rule_df.height != len(targets):
        raise FeatureError("an alerted account has no visible non-self activity in its window")
    legs = _legs(win).filter(pl.col("account_key").is_in(targets))
    out = (
        alerts.select(*ALERT_KEYS)
        .join(rule_df, on="account_key")
        .join(_behaviour_block(legs, win, st, ctx), on="account_key")
        .join(_dynamics_block(legs), on="account_key")
        .join(_peer_block(st, ctx), on="account_key")
        .join(_graph_block(win, targets), on="account_key")
    )
    names = feature_names(ctx)
    # one missing-value convention: NaN (undefined, e.g. no outflow after an inflow), never null
    return out.select(
        *ALERT_KEYS,
        *[pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in names],
        "_triggered",
    )


def batch_features(
    trans_usd: pl.LazyFrame, alerts: pl.DataFrame, ctx: FeatureContext
) -> pl.DataFrame:
    """Features for many alerts: one visible window per distinct as_of. Output sorted by the
    opaque alert_id (never by account ID, D5). '_triggered' is the recomputed rule list, kept
    for the integrity check against the alert store; it is not a feature."""
    parts = []
    for as_of in sorted(alerts["as_of"].unique().to_list()):
        parts.append(
            features_for_window(trans_usd, as_of, alerts.filter(pl.col("as_of") == as_of), ctx)
        )
    if not parts:
        raise FeatureError("no alerts")
    return pl.concat(parts).sort("alert_id")


# ------------------------------------------------------------------ pure reference path


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _pstd(xs: list[float]) -> float:
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def _avg_rank_pct(values: list[float], v: float) -> float:
    less = sum(1 for x in values if x < v)
    equal = sum(1 for x in values if x == v)
    return (less + (equal + 1) / 2.0) / len(values)


def account_features(
    trans_usd: pl.LazyFrame,
    account_key: str,
    as_of: datetime,
    peer_group: str | None,
    ctx: FeatureContext,
) -> dict[str, float]:
    """Pure reference: f(data, account_key, as_of). Plain Python over the visible rows."""
    win = window_frame(trans_usd, as_of, ctx)
    _check_formats(win, ctx)
    # the account's own rows as dicts (small); the whole window only as two key lists (graph)
    rows = (
        win.filter((pl.col("from_key") == account_key) | (pl.col("to_key") == account_key))
        .select(
            "timestamp", "from_key", "to_key", "usd_paid", "usd_received", "payment_currency",
            "receiving_currency", "payment_format",
        )
        .rows(named=True)
    )  # fmt: skip
    all_from, all_to = win["from_key"].to_list(), win["to_key"].to_list()
    # frozen P2 statistics (shared definition), all active accounts for the peer ranks
    st = _rule_block(win, ctx)
    me = st.filter(pl.col("account_key") == account_key)
    if me.height != 1:
        raise FeatureError(f"{account_key} has no visible non-self activity before {as_of}")
    fired = rules.fire(
        me.with_columns(pl.lit(peer_group).alias("peer_group")), ctx.thresholds, ctx.rules_cfg
    ).row(0, named=True)
    f: dict[str, float] = {c: float(fired[c]) for c in rules.ALL_STATS}
    f["n_rules_triggered"] = float(fired["n_rules_triggered"])
    for r in ctx.active_rules:
        f[f"fired_{r}"] = float(fired[f"fired_{r}"])

    # legs of this account (non-self), written out by hand
    legs = []
    for r in rows:
        if r["from_key"] == r["to_key"]:
            continue
        if r["from_key"] == account_key:
            legs.append(("out", r["to_key"], r["usd_paid"], r["payment_currency"], r))
        if r["to_key"] == account_key:
            legs.append(("in", r["from_key"], r["usd_received"], r["receiving_currency"], r))
    usd = [x[2] for x in legs]
    n = len(legs)
    f["n_in"] = float(sum(1 for x in legs if x[0] == "in"))
    f["n_out"] = float(sum(1 for x in legs if x[0] == "out"))
    f["amt_mean_usd"] = sum(usd) / n
    f["amt_median_usd"] = _median(usd)
    f["amt_min_usd"] = min(usd)
    f["amt_std_usd"] = _pstd(usd)
    f["amt_cv"] = f["amt_std_usd"] / f["amt_mean_usd"]
    f["amt_log10_std"] = _pstd([math.log10(u) for u in usd])
    f["share_near_threshold"] = fired["n_near_threshold"] / n
    f["share_round"] = fired["n_round_amounts"] / n
    f["share_cross_currency"] = fired["n_cross_currency"] / n
    f["n_currencies"] = float(len({x[3] for x in legs}))
    by_cp_usd: dict[str, float] = defaultdict(float)
    by_cp_n: dict[str, int] = defaultdict(int)
    for x in legs:
        by_cp_usd[x[1]] += x[2]
        by_cp_n[x[1]] += 1
    tot = sum(usd)
    f["n_counterparties"] = float(len(by_cp_usd))
    f["cp_hhi"] = sum((v / tot) ** 2 for v in by_cp_usd.values())
    f["max_txn_one_cp"] = float(max(by_cp_n.values()))
    f["repeat_share"] = 1.0 - len(by_cp_usd) / n
    f["net_flow_ratio"] = (fired["in_usd"] - fired["out_usd"]) / fired["volume_usd"]
    f["n_self_transfers"] = float(
        sum(1 for r in rows if r["from_key"] == account_key and r["to_key"] == account_key)
    )
    for fmt in ctx.formats:
        f[f"share_fmt_{slug(fmt)}"] = sum(1 for x in legs if x[4]["payment_format"] == fmt) / n

    # dynamics
    ts = sorted(x[4]["timestamp"] for x in legs)
    hours: dict[datetime, int] = defaultdict(int)
    for t in ts:
        hours[t.replace(minute=0, second=0, microsecond=0)] += 1
    f["active_hours"] = float(len(hours))
    f["max_txn_per_hour"] = float(max(hours.values()))
    f["span_hours"] = (ts[-1] - ts[0]).total_seconds() / 3600.0
    gaps = [(b - a).total_seconds() / 60.0 for a, b in zip(ts, ts[1:], strict=False)]
    f["median_interarrival_min"] = _median(gaps) if gaps else math.nan
    ins = [x[4]["timestamp"] for x in legs if x[0] == "in"]
    outs = [x[4]["timestamp"] for x in legs if x[0] == "out"]
    lag = math.nan
    if ins:
        later = [t for t in outs if t >= min(ins)]
        if later:
            lag = (min(later) - min(ins)).total_seconds() / 60.0
    f["in_to_out_lag_min"] = lag

    # peer ranks over all active accounts of the window
    for s in ctx.peer_stats:
        f[f"pct_{s}"] = _avg_rank_pct([float(v) for v in st[s].to_list()], float(fired[s]))

    # graph of distinct non-self pairs, by hand
    out_adj: dict[str, set[str]] = defaultdict(set)
    in_adj: dict[str, set[str]] = defaultdict(set)
    for a_, b_ in zip(all_from, all_to, strict=True):
        if a_ != b_:
            out_adj[a_].add(b_)
            in_adj[b_].add(a_)
    v = account_key
    f["reciprocal_cp"] = float(len(out_adj[v] & in_adj[v]))
    f["cycles3"] = float(sum(1 for u in out_adj[v] for w in out_adj[u] if v in out_adj[w]))
    f["reach2_out"] = float(len({w for u in out_adj[v] for w in out_adj[u]} - {v}))
    f["reach2_in"] = float(len({w for u in in_adj[v] for w in in_adj[u]} - {v}))
    snd, rcv = in_adj[v], out_adj[v]
    f["senders_mean_outdeg"] = sum(len(out_adj[u]) for u in snd) / len(snd) if snd else math.nan
    f["receivers_mean_indeg"] = sum(len(in_adj[u]) for u in rcv) / len(rcv) if rcv else math.nan
    seen, queue = {v}, deque([v])
    while queue:
        x = queue.popleft()
        for y in out_adj[x] | in_adj[x]:
            if y not in seen:
                seen.add(y)
                queue.append(y)
    f["wcc_log10_size"] = math.log10(len(seen))
    return {k: float(f[k]) for k in feature_names(ctx)}
