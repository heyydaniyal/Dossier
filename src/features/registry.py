"""P3 feature registry: name, group, definition, lookback, inputs, as_of-safety evidence, owner.

The registry is the contract between the feature pipeline (src/features/compute.py), the model
(P4/P5) and the agents' tools (P7+). tests/test_p3_features.py requires:
  - registry names == the columns the pipeline produces, in the same order;
  - every lookback <= the frozen L_max;
  - every input in ALLOWED_INPUTS (label-free runtime inputs only: never labels, the patterns
    file, an evaluation store, KYC fields or identifier values);
  - every model feature covered by the as_of tests named in `as_of_tests`.
Runtime-safe: no ground truth is read here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from src.features import compute as fc

OWNER = "Daniyal Ahmad (Modelling seam)"
# Label-free inputs a feature may depend on.
ALLOWED_INPUTS = {
    "transactions",  # src.data.transactions.scan_transactions (label column never selected)
    "fx_train",  # configs/p2_fx_usd_per_unit.yaml, fitted on TRAIN (D6)
    "rule_config",  # configs/p2_rules.yaml (frozen P2 definitions)
    "rule_thresholds_train",  # configs/p2_rule_thresholds.yaml, fitted on TRAIN
    "alert_peer_group",  # the alert's own peer_group, ONLY to recompute the frozen R02 flag
    "dispositions_visible",  # src.data.stores.dispositions_visible(as_of); network tool only
}
AS_OF_TESTS = (
    "test_deletion_recompute_batch",
    "test_outside_window_mutation_changes_nothing",
    "test_pure_path_equals_batch_path",
)
GENERATOR_SIGNATURE = (
    "Generator signature (DATA_CARD §7): real in this data, may not transfer to a bank; "
    "P4 runs a sensitivity check without it."
)


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    definition: str
    lookback_hours: int
    inputs: tuple[str, ...]
    model: bool
    owner: str = OWNER
    as_of_tests: tuple[str, ...] = AS_OF_TESTS
    note: str = ""


_TX = ("transactions", "fx_train")
_RULE = ("transactions", "fx_train", "rule_config")
_FIRED = (*_RULE, "rule_thresholds_train", "alert_peer_group")

_RULE_DEFS = {
    "max_txn_usd": "largest single leg (USD), either role (R01 statistic)",
    "volume_usd": "total USD of all legs in + out (R02 statistic)",
    "n_near_threshold": "legs in [9,000, 10,000) USD (R03 statistic)",
    "n_distinct_senders": "distinct counterparties sending to the account (R04 statistic)",
    "n_distinct_receivers": "distinct counterparties the account sends to (R05 statistic)",
    "pass_through_usd": "min(in, out) USD if out/in in [0.8, 1.25] and an outflow follows the "
    "first inflow, else 0 (R06 statistic; rule inactive)",
    "high_risk_channel_usd": "USD via Cash or Bitcoin (R07 statistic)",
    "n_cross_currency": "legs whose paid and received currency differ (R08; rule inactive)",
    "n_round_amounts": "legs that are exact multiples of 1,000 in their own currency (R09; "
    "rule inactive)",
    "n_txn": "number of legs (non-self transactions, both roles)",
    "in_usd": "USD received",
    "out_usd": "USD sent",
}
_DEFS = {
    "n_in": "legs received",
    "n_out": "legs sent",
    "amt_mean_usd": "mean leg amount (USD)",
    "amt_median_usd": "median leg amount (USD)",
    "amt_min_usd": "smallest leg amount (USD)",
    "amt_std_usd": "population std of leg amounts (USD)",
    "amt_cv": "amt_std_usd / amt_mean_usd",
    "amt_log10_std": "population std of log10(leg USD)",
    "share_near_threshold": "n_near_threshold / n_txn",
    "share_round": "n_round_amounts / n_txn",
    "share_cross_currency": "n_cross_currency / n_txn",
    "n_currencies": "distinct currencies of the account's legs (its own side of each leg)",
    "n_counterparties": "distinct counterparties, either role",
    "cp_hhi": "Herfindahl index of USD across counterparties (1 = one counterparty)",
    "max_txn_one_cp": "most legs with a single counterparty",
    "repeat_share": "1 - n_counterparties / n_txn",
    "net_flow_ratio": "(in_usd - out_usd) / volume_usd",
    "n_self_transfers": "transactions from the account to itself (not legs)",
    "active_hours": "distinct clock-hour buckets with >= 1 leg (a count, not the hour)",
    "span_hours": "hours between the first and last leg",
    "max_txn_per_hour": "most legs in one clock-hour bucket",
    "median_interarrival_min": "median minutes between consecutive legs (NaN if 1 leg)",
    "in_to_out_lag_min": "minutes from the first inflow to the first outflow at or after it "
    "(NaN if none)",
    "reciprocal_cp": "counterparties that both send to and receive from the account",
    "cycles3": "directed 3-cycles v->u->w->v through the account (distinct pairs graph)",
    "reach2_out": "distinct accounts reachable in exactly 2 outgoing hops (excluding itself)",
    "reach2_in": "distinct accounts reaching it in exactly 2 hops (excluding itself)",
    "senders_mean_outdeg": "mean number of distinct receivers of the account's senders (NaN if "
    "none)",
    "receivers_mean_indeg": "mean number of distinct senders of the account's receivers (NaN if "
    "none)",
    "wcc_log10_size": "log10 of the size of the account's weakly connected component in the "
    "window's graph (community feature)",
}
_GROUP_INPUTS = {
    "behaviour": _RULE,
    "format_mix": _TX,
    "dynamics": _TX,
    "peer": _RULE,
    "graph": ("transactions",),
}


def build_registry(ctx: fc.FeatureContext) -> list[Feature]:
    lb = int(ctx.lookback.total_seconds() // 3600)
    groups = fc.group_names(ctx)
    out: list[Feature] = []
    for name in groups["rule"]:
        if name.startswith("fired_"):
            d, inp = f"1 if frozen rule {name[6:]} fires on the window (P2 thresholds)", _FIRED
        elif name == "n_rules_triggered":
            d, inp = "number of active rules that fire on the window", _FIRED
        else:
            d, inp = _RULE_DEFS[name], _RULE
        out.append(Feature(name, "rule", d, lb, inp, True))
    for g in ("behaviour", "dynamics", "graph"):
        for name in groups[g]:
            out.append(Feature(name, g, _DEFS[name], lb, _GROUP_INPUTS[g], True))
    for name, fmt in zip(groups["format_mix"], ctx.formats, strict=True):
        out.append(
            Feature(
                name,
                "format_mix",
                f"share of legs paid by {fmt}",
                lb,
                _TX,
                True,
                note=GENERATOR_SIGNATURE,
            )  # fmt: skip
        )
    for name, s in zip(groups["peer"], ctx.peer_stats, strict=True):
        out.append(
            Feature(
                name,
                "peer",
                f"average-rank percentile of {s} among all accounts with >= 1 non-self leg in "
                "the same visible window (no KYC grouping)",
                lb,
                _GROUP_INPUTS["peer"],
                True,
            )
        )
    order = {g: i for i, g in enumerate(groups)}
    out.sort(key=lambda f: (order[f.group], fc.feature_names(ctx).index(f.name)))
    return out


def registry_records(ctx: fc.FeatureContext) -> list[dict]:
    return [asdict(f) for f in build_registry(ctx)]


def registry_hash(ctx: fc.FeatureContext) -> str:
    """sha256 of the canonical registry JSON: the RunManifest feature_registry_hash."""
    blob = json.dumps(registry_records(ctx), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def model_features(ctx: fc.FeatureContext, groups: list[str] | None = None) -> list[str]:
    reg = build_registry(ctx)
    return [f.name for f in reg if f.model and (groups is None or f.group in groups)]


# Not model features (MDP): provided to agents through a tool only (P10 Network agent).
TOOL_ONLY = {
    "flagged_counterparties": Feature(
        "flagged_counterparties",
        "network_tool",
        "counterparties of the account in the window whose own earlier alert was CONFIRMED and "
        "CLOSED (simulated disposition, pre-TEST alerts) strictly before as_of",
        24,
        ("transactions", "dispositions_visible"),
        False,
        as_of_tests=("test_flagged_neighbours_point_in_time",),
        note="Noisy by design (confirmed_suspicious precision 56.5%); never a label.",
    )
}
