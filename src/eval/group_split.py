"""AGENT-DEV / AGENT-TEST group split of TEST-period alerts. EVALUATION ONLY (reads patterns).

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
