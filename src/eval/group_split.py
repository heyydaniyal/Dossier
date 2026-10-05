"""AGENT-DEV / AGENT-TEST group split of TEST-period alerts. EVALUATION ONLY (reads patterns).

v1 (P2, below first) is kept for the original-vs-corrected record; v2 (further below) is used from
P3 task 0 on (review finding M-6, approved by Dani 2026-10-05).

Method (frozen in configs/p2_split.yaml): build a graph whose nodes are the accounts with a TEST
alert; join two accounts if they take part in the same laundering pattern instance (attempt,
any date) or are the two sides of an unattributed laundering transaction dated in TEST. Every
connected component goes wholly to one group, chosen by a keyed hash of the component (the
smallest keyed hash of its member account_keys) -> AGENT-DEV and AGENT-TEST share no account and
no pattern instance. No ID value or ID order decides anything (D5).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.hashing import keyed_u64, uniforms
from src.eval.alert_labels import KEY_COLS

AGENT_DEV = "AGENT-DEV"
AGENT_TEST = "AGENT-TEST"


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)  # deterministic; roots are never exposed


def split_test_alerts(
    test_alerts: pl.DataFrame,
    legs: pl.DataFrame,
    test_start,
    test_end,
    hash_key: str,
    fraction_agent_test: float,
) -> pl.DataFrame:
    """test_alerts: alert_id, account_key. legs: output of alert_labels.laundering_legs."""
    accounts = set(test_alerts["account_key"].to_list())
    uf = _UF()
    for a in accounts:
        uf.find("acct:" + a)
    # (1) pattern instances, any date
    att = legs.filter(
        pl.col("attempt_id").is_not_null() & pl.col("account_key").is_in(list(accounts))
    )
    for aid, acct in att.select("attempt_id", "account_key").unique().iter_rows():
        uf.union(f"att:{aid}", "acct:" + acct)
    # (2) unattributed laundering transactions dated in TEST
    un = legs.filter(
        pl.col("attempt_id").is_null()
        & (pl.col("timestamp") >= test_start)
        & (pl.col("timestamp") < test_end)
        & pl.col("account_key").is_in(list(accounts))
    )
    for rid, acct in un.select("row_id", "account_key").unique().iter_rows():
        uf.union(f"txn:{rid}", "acct:" + acct)

    acc_list = sorted(accounts)
    h = keyed_u64(acc_list, hash_key, 0) if acc_list else []
    hmap = dict(zip(acc_list, (int(x) for x in h), strict=True))
    comp_min: dict[str, int] = {}
    for a in acc_list:
        r = uf.find("acct:" + a)
        comp_min[r] = min(comp_min.get(r, hmap[a]), hmap[a])
    roots = sorted(comp_min)
    u = (
        uniforms(np.array([comp_min[r] for r in roots], dtype=np.uint64), hash_key + "|group")
        if roots
        else []
    )
    group_of_root = {
        r: (AGENT_TEST if ui < fraction_agent_test else AGENT_DEV)
        for r, ui in zip(roots, u, strict=True)
    }
    comp_of_acct = {a: f"{comp_min[uf.find('acct:' + a)]:016x}" for a in acc_list}
    grp_of_acct = {a: group_of_root[uf.find("acct:" + a)] for a in acc_list}
    return test_alerts.select(
        "alert_id",
        "account_key",
        pl.col("account_key")
        .replace_strict(grp_of_acct, return_dtype=pl.String)
        .alias("agent_group"),
        pl.col("account_key")
        .replace_strict(comp_of_acct, return_dtype=pl.String)
        .alias("component"),
    ).sort("alert_id")


# ---------------------------------------------------------------- v2 (P3 task 0, review M-6)
#
# v1 above builds attempt membership from laundering legs CUT to the usable span and only through
# TEST-alerted accounts. The independent P2 review (2026-10-01, finding M-6) showed three gaps:
#   (a) two TEST-alerted accounts whose attempts meet only at a hub WITHOUT a TEST alert can land
#       in different groups;
#   (b) an attempt member whose legs fall only in the excluded tail (>= 09-17) is not linked, so
#       an attempt can be shared by both groups -- a strict violation the v1 check cannot see,
#       because it uses the same cut legs;
#   (c) unattributed laundering links outside TEST are ignored.
# v2 builds memberships from the WHOLE raw file (patterns file for attempts, every unattributed
# laundering transaction) and offers three graphs, from the frozen method done correctly (A) to the
# most conservative (C). Which one is used is decided by a rule declared before the run
# (configs/p2_agent_split_v2.yaml). Hash key and fraction are unchanged, so a component with the
# same members keeps its v1 group.

VARIANTS = ("A_attempts_full", "B_plus_unattributed_any_date", "C_full_laundering_graph")


def attempt_members(patterns_path) -> pl.DataFrame:
    """(node, account_key) for every pattern attempt, any date: node = 'att:<attempt_id>'."""
    p = pl.scan_parquet(patterns_path)
    out = pl.concat(
        [
            p.select(
                pl.col("attempt_id"),
                pl.concat_str([pl.col("from_bank"), pl.lit("|"), pl.col("from_account")]).alias(
                    "account_key"
                ),
            ),
            p.select(
                pl.col("attempt_id"),
                pl.concat_str([pl.col("to_bank"), pl.lit("|"), pl.col("to_account")]).alias(
                    "account_key"
                ),
            ),
        ]
    )
    return (
        out.select(
            pl.concat_str([pl.lit("att:"), pl.col("attempt_id").cast(pl.String)]).alias("node"),
            "account_key",
        )
        .unique()
        .sort("node", "account_key")
        .collect()
    )


def unattributed_members(trans_path, patterns_path) -> pl.DataFrame:
    """(node, account_key, timestamp) for every laundering transaction in NO pattern, any date
    (burn-in, embargo and the D1 tail included): node = 'txn:<row_id>', both sides."""
    t = (
        pl.scan_parquet(trans_path)
        .select(["row_id", *KEY_COLS, "is_laundering"])
        .filter(pl.col("is_laundering") == 1)
    )
    p = pl.scan_parquet(patterns_path).select(KEY_COLS)
    un = t.join(p, on=KEY_COLS, how="anti").with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC"),
        pl.concat_str([pl.lit("txn:"), pl.col("row_id").cast(pl.String)]).alias("node"),
    )
    sides = [("from_bank", "from_account"), ("to_bank", "to_account")]
    return (
        pl.concat(
            [
                un.select(
                    "node",
                    pl.concat_str([pl.col(b), pl.lit("|"), pl.col(a)]).alias("account_key"),
                    "timestamp",
                )
                for b, a in sides
            ]
        )
        .unique(["node", "account_key"])
        .sort("node", "account_key")
        .collect()
    )


def memberships(
    variant: str,
    att: pl.DataFrame,
    unatt: pl.DataFrame,
    test_accounts: set[str],
    test_start,
    test_end,
) -> pl.DataFrame:
    """(node, account_key) edges of the graph for one variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown split variant {variant}")
    un = unatt
    if variant == VARIANTS[0]:  # frozen v1 scope: unattributed links dated in TEST only
        un = un.filter((pl.col("timestamp") >= test_start) & (pl.col("timestamp") < test_end))
    m = pl.concat([att.select("node", "account_key"), un.select("node", "account_key")])
    if variant != VARIANTS[2]:  # A, B: only TEST-alerted accounts are graph nodes
        m = m.filter(pl.col("account_key").is_in(list(test_accounts)))
    return m.unique().sort("node", "account_key")


def split_v2(
    test_alerts: pl.DataFrame,
    members: pl.DataFrame,
    hash_key: str,
    fraction_agent_test: float,
) -> pl.DataFrame:
    """Components of the bipartite graph {accounts} x {attempt / transaction nodes}, projected onto
    TEST-alerted accounts. Group by keyed hash of the component's smallest member hash (TEST-alerted
    members only, as in v1). Order-free; no ID value or order decides anything (D5)."""
    accounts = set(test_alerts["account_key"].to_list())
    uf = _UF()
    for a in accounts:
        uf.find("acct:" + a)
    for node, acct in members.select("node", "account_key").iter_rows():
        uf.union(node, "acct:" + acct)
    acc_list = sorted(accounts)
    h = keyed_u64(acc_list, hash_key, 0) if acc_list else []
    hmap = dict(zip(acc_list, (int(x) for x in h), strict=True))
    comp_min: dict[str, int] = {}
    for a in acc_list:
        r = uf.find("acct:" + a)
        comp_min[r] = min(comp_min.get(r, hmap[a]), hmap[a])
    roots = sorted(comp_min)
    u = (
        uniforms(np.array([comp_min[r] for r in roots], dtype=np.uint64), hash_key + "|group")
        if roots
        else []
    )
    group_of_root = {
        r: (AGENT_TEST if ui < fraction_agent_test else AGENT_DEV)
        for r, ui in zip(roots, u, strict=True)
    }
    comp_of_acct = {a: f"{comp_min[uf.find('acct:' + a)]:016x}" for a in acc_list}
    grp_of_acct = {a: group_of_root[uf.find("acct:" + a)] for a in acc_list}
    return test_alerts.select(
        "alert_id",
        "account_key",
        pl.col("account_key")
        .replace_strict(grp_of_acct, return_dtype=pl.String)
        .alias("agent_group"),
        pl.col("account_key")
        .replace_strict(comp_of_acct, return_dtype=pl.String)
        .alias("component"),
    ).sort("alert_id")


def check_disjoint_v2(
    assign: pl.DataFrame,
    att: pl.DataFrame,
    unatt: pl.DataFrame,
    variant: str | None = None,
    test_start=None,
    test_end=None,
) -> dict:
    """On the FULL memberships (any date): no account and no attempt has TEST-alerted members in
    both groups; and no unattributed laundering transaction the variant links (A: dated in TEST;
    B, C: any date) does either. Raises on violation; returns counts."""
    acct_groups = assign.group_by("account_key").agg(pl.col("agent_group").n_unique().alias("g"))
    if (acct_groups["g"] > 1).any():
        raise AssertionError("an account is in both AGENT-DEV and AGENT-TEST")
    groups = assign.select("account_key", "agent_group").unique()
    out: dict = {"n_accounts": acct_groups.height}

    def crossing(m: pl.DataFrame) -> tuple[int, int]:
        g = (
            m.select("node", "account_key")
            .unique()
            .join(groups, on="account_key")
            .group_by("node")
            .agg(pl.col("agent_group").n_unique().alias("g"))
        )
        return g.height, int((g["g"] > 1).sum())

    out["n_attempt_nodes_touching_test_alerts"], out["n_attempt_nodes_in_both_groups"] = crossing(
        att
    )
    n_u, x_u = crossing(unatt)
    out["n_unattributed_txn_nodes_touching_test_alerts"] = n_u
    out["n_unattributed_txn_nodes_in_both_groups"] = x_u
    if out["n_attempt_nodes_in_both_groups"]:
        raise AssertionError("a pattern instance is in both AGENT-DEV and AGENT-TEST")
    if variant is not None:
        linked = unatt
        if variant == VARIANTS[0]:
            linked = unatt.filter(
                (pl.col("timestamp") >= test_start) & (pl.col("timestamp") < test_end)
            )
        if crossing(linked)[1]:
            raise AssertionError(
                f"variant {variant}: a linked unattributed laundering transaction is in both groups"
            )
    return out


def check_disjoint(assign: pl.DataFrame, legs: pl.DataFrame) -> dict:
    """No account and no attempt in both groups. Returns counts; raises on violation."""
    acct_groups = assign.group_by("account_key").agg(pl.col("agent_group").n_unique().alias("g"))
    if (acct_groups["g"] > 1).any():
        raise AssertionError("an account is in both AGENT-DEV and AGENT-TEST")
    att = (
        legs.filter(pl.col("attempt_id").is_not_null())
        .select("attempt_id", "account_key")
        .unique()
        .join(assign.select("account_key", "agent_group").unique(), on="account_key")
        .group_by("attempt_id")
        .agg(pl.col("agent_group").n_unique().alias("g"))
    )
    if (att["g"] > 1).any():
        raise AssertionError("a pattern instance is in both AGENT-DEV and AGENT-TEST")
    return {"n_accounts": acct_groups.height, "n_attempts_touching_test_alerts": att.height}
