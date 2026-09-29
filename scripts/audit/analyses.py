"""P1 audit analyses. Every function takes Polars LazyFrames and returns JSON-ready dicts.

HOLDOUT DISCIPLINE (project HOLDOUT RULE):
  - The provisional holdout start is derived from transaction VOLUME per day only (no labels).
  - Anything that could inform a design choice (label rates by category, separability trees,
    currency rates, identifier leakage) uses rows with timestamp < holdout_start ONLY.
  - For the holdout and the post-span tail we report AGGREGATE label counts per period only,
    which is the single permitted exposure (evaluability check).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

KEY_COLS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
]
ALL_COLS = [*KEY_COLS, "is_laundering"]
COMPLETE_DAY_FRACTION = 0.5  # a day is "complete" if its volume >= 50% of the median day
HOLDOUT_FRACTION = 0.75  # provisional holdout starts after 75% of the complete days
SEPARABILITY_FIT_FRACTION = 2 / 3  # inside pre-holdout: first 2/3 of days fit, last 1/3 eval
MIN_SUPPORT = 1000  # minimum group size before a rate/lift is reported as a finding


def _c(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def _rows(df: pl.DataFrame) -> list[dict]:
    out = []
    for r in df.iter_rows(named=True):
        out.append(
            {k: (v.isoformat() if isinstance(v, date | datetime) else v) for k, v in r.items()}
        )
    return out


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


# ---------------------------------------------------------------- time


def time_audit(t: pl.LazyFrame) -> tuple[dict, dict]:
    """Span, per-day volume, boundary completeness, file order, timestamp ties.

    Returns (report, bounds). bounds uses VOLUME ONLY to define the complete-day span and the
    provisional holdout start; laundering counts per day are reported as aggregates.
    """
    daily = _c(
        t.group_by(pl.col("timestamp").dt.date().alias("day"))
        .agg(n=pl.len(), n_laundering=pl.col("is_laundering").cast(pl.Int64).sum())
        .sort("day")
    )
    med = float(daily["n"].median())
    daily = daily.with_columns(complete=pl.col("n") >= COMPLETE_DAY_FRACTION * med)
    comp = daily.filter(pl.col("complete"))["day"].to_list()
    first_c, last_c = comp[0], comp[-1]
    n_complete = len(comp)
    expected = (last_c - first_c).days + 1
    gaps = expected - n_complete  # incomplete days INSIDE the span
    ho_day = first_c + timedelta(days=math.floor(HOLDOUT_FRACTION * n_complete))
    bounds = {
        "span_start": datetime.combine(first_c, datetime.min.time()),
        "span_end_exclusive": datetime.combine(last_c + timedelta(days=1), datetime.min.time()),
        "holdout_start": datetime.combine(ho_day, datetime.min.time()),
    }

    mm = _c(
        t.select(pl.col("timestamp").min().alias("min"), pl.col("timestamp").max().alias("max"))
    )
    ts_min, ts_max = mm["min"][0], mm["max"][0]

    def hourly(day: date) -> list[int]:
        h = _c(
            t.filter(pl.col("timestamp").dt.date() == day)
            .group_by(pl.col("timestamp").dt.hour().alias("h"))
            .agg(pl.len())
        )
        m = dict(zip(h["h"].to_list(), h["len"].to_list(), strict=True))
        return [m.get(i, 0) for i in range(24)]

    days_all = daily["day"].to_list()
    # file order: is the raw file sorted by time? (order-dependent -> in-memory engine)
    ts = t.select("timestamp").collect()["timestamp"]
    n_inversions = int((ts.diff() < timedelta(0)).sum())
    del ts
    ties = _c(t.group_by("timestamp").agg(pl.len()))
    n_rows = int(ties["len"].sum())
    shared = int(ties.filter(pl.col("len") > 1)["len"].sum())

    report = {
        "ts_min": ts_min.isoformat(),
        "ts_max": ts_max.isoformat(),
        "raw_span_days": (ts_max - ts_min).total_seconds() / 86400,
        "timestamp_resolution": "minute (format YYYY/MM/DD HH:MM, no seconds)",
        "median_daily_volume": med,
        "complete_day_rule": (
            f"volume >= {COMPLETE_DAY_FRACTION} x median daily volume (labels not used)"
        ),
        "n_calendar_days_with_data": len(days_all),
        "n_complete_days": n_complete,
        "incomplete_days_inside_span": gaps,
        "complete_span_first_day": first_c.isoformat(),
        "complete_span_last_day": last_c.isoformat(),
        "incomplete_days_outside_span": _rows(daily.filter(~pl.col("complete")).drop("complete")),
        "hourly_volume_first_day": hourly(days_all[0]),
        "hourly_volume_last_complete_day": hourly(last_c),
        "daily": _rows(daily),
        "file_sorted_by_time": n_inversions == 0,
        "n_time_inversions_in_file_order": n_inversions,
        "n_distinct_timestamps": ties.height,
        "max_txns_same_minute": int(ties["len"].max()),
        "share_rows_sharing_timestamp": shared / n_rows,
        "provisional_holdout": {
            "rule": f"span_start + floor({HOLDOUT_FRACTION} x n_complete_days) days, at 00:00",
            "holdout_start": bounds["holdout_start"].isoformat(),
            "span_end_exclusive": bounds["span_end_exclusive"].isoformat(),
            "pre_holdout_days": (ho_day - first_c).days,
            "holdout_days": (last_c - ho_day).days + 1,
        },
    }
    return report, bounds


def period_expr(bounds: dict) -> pl.Expr:
    ts = pl.col("timestamp")
    return (
        pl.when(ts < bounds["span_start"])
        .then(pl.lit("before_span"))
        .when(ts < bounds["holdout_start"])
        .then(pl.lit("pre_holdout"))
        .when(ts < bounds["span_end_exclusive"])
        .then(pl.lit("holdout"))
        .otherwise(pl.lit("after_span"))
        .alias("period")
    )


def pre_holdout(t: pl.LazyFrame, bounds: dict) -> pl.LazyFrame:
    return t.filter(
        (pl.col("timestamp") >= bounds["span_start"])
        & (pl.col("timestamp") < bounds["holdout_start"])
    )


# ---------------------------------------------------------------- schema


def schema_audit(t: pl.LazyFrame) -> dict:
    """Types, nulls, empties, ranges, category counts. No label rates here (label-free)."""
    schema = t.collect_schema()
    str_cols = [c for c, d in schema.items() if d == pl.String]
    exprs = [pl.len().alias("n_rows")]
    exprs += [pl.col(c).null_count().alias(f"null__{c}") for c in schema]
    exprs += [(pl.col(c) == "").sum().alias(f"empty__{c}") for c in str_cols]
    for a in ("amount_paid", "amount_received"):
        exprs += [
            pl.col(a).min().alias(f"{a}__min"),
            pl.col(a).max().alias(f"{a}__max"),
            pl.col(a).median().alias(f"{a}__median"),
            pl.col(a).quantile(0.99).alias(f"{a}__p99"),
            (pl.col(a) == 0).sum().alias(f"{a}__n_zero"),
            (pl.col(a) < 0).sum().alias(f"{a}__n_negative"),
            (pl.col(a) == pl.col(a).round(0)).sum().alias(f"{a}__n_integer"),
        ]
    exprs += [
        (pl.col("payment_currency") != pl.col("receiving_currency"))
        .sum()
        .alias("n_cross_currency"),
        (
            (pl.col("payment_currency") == pl.col("receiving_currency"))
            & (pl.col("amount_paid") != pl.col("amount_received"))
        )
        .sum()
        .alias("n_same_currency_amount_mismatch"),
        (
            (pl.col("from_bank") == pl.col("to_bank"))
            & (pl.col("from_account") == pl.col("to_account"))
        )
        .sum()
        .alias("n_self_transfer"),
        (
            (pl.col("from_bank") != pl.col("to_bank"))
            & (pl.col("from_account") == pl.col("to_account"))
        )
        .sum()
        .alias("n_same_account_number_different_bank"),
        pl.col("is_laundering").is_in([0, 1]).not_().sum().alias("n_label_not_0_1"),
        pl.col("is_laundering").cast(pl.Int64).sum().alias("n_laundering"),
    ]
    row = _c(t.select(exprs)).row(0, named=True)
    out: dict = {"dtypes": {c: str(d) for c, d in schema.items()}}
    out["n_rows"] = row.pop("n_rows")
    out["nulls"] = {k[6:]: v for k, v in row.items() if k.startswith("null__")}
    out["empty_strings"] = {k[7:]: v for k, v in row.items() if k.startswith("empty__")}
    out["amounts"] = {k: v for k, v in row.items() if "__" in k and k.startswith("amount")}
    for k in (
        "n_cross_currency",
        "n_same_currency_amount_mismatch",
        "n_self_transfer",
        "n_same_account_number_different_bank",
        "n_label_not_0_1",
        "n_laundering",
    ):
        out[k] = row[k]
    out["laundering_rate_overall"] = out["n_laundering"] / out["n_rows"]

    def counts(col: str) -> dict:
        df = _c(t.group_by(col).agg(pl.len()).sort(["len", col], descending=[True, False]))
        return dict(zip(df[col].to_list(), df["len"].to_list(), strict=True))

    out["payment_format_counts"] = counts("payment_format")
    out["payment_currency_counts"] = counts("payment_currency")
    out["receiving_currency_counts"] = counts("receiving_currency")
    fmt_self = _c(
        t.filter(
            (pl.col("from_bank") == pl.col("to_bank"))
            & (pl.col("from_account") == pl.col("to_account"))
        )
        .group_by("payment_format")
        .agg(pl.len())
        .sort("payment_format")
    )
    out["self_transfer_by_format"] = dict(
        zip(fmt_self["payment_format"].to_list(), fmt_self["len"].to_list(), strict=True)
    )
    return out


def label_rates_by_category(t: pl.LazyFrame, bounds: dict) -> dict:
    """P(laundering | category) on PRE-HOLDOUT rows only (candidate artefacts)."""
    pre = pre_holdout(t, bounds).with_columns(
        cross_currency=pl.col("payment_currency") != pl.col("receiving_currency"),
        self_transfer=(pl.col("from_bank") == pl.col("to_bank"))
        & (pl.col("from_account") == pl.col("to_account")),
        integer_amount=pl.col("amount_paid") == pl.col("amount_paid").round(0),
        hour=pl.col("timestamp").dt.hour(),
        minute=pl.col("timestamp").dt.minute(),
    )
    tot = _c(pre.select(pl.len(), pl.col("is_laundering").cast(pl.Int64).sum().alias("pos"))).row(0)
    base = tot[1] / tot[0]
    out: dict = {
        "scope": "pre_holdout",
        "n_rows": tot[0],
        "n_laundering": tot[1],
        "base_rate": base,
    }
    for col in (
        "payment_format",
        "payment_currency",
        "receiving_currency",
        "cross_currency",
        "self_transfer",
        "integer_amount",
        "hour",
        "minute",
    ):
        df = _c(
            pre.group_by(col)
            .agg(n=pl.len(), n_laundering=pl.col("is_laundering").cast(pl.Int64).sum())
            .with_columns(rate=pl.col("n_laundering") / pl.col("n"))
            .with_columns(lift=pl.col("rate") / base)
            .sort(col)
        )
        out[col] = _rows(df.with_columns(pl.col(col).cast(pl.String)))
    return out


# ---------------------------------------------------------------- duplicates


def duplicates(t: pl.LazyFrame) -> dict:
    """Exact duplicates (all 11 fields) and label conflicts (same 10 fields, labels differ).

    Hash pre-filter, then EXACT re-grouping of the candidates (hash collisions cannot count).
    """

    def exact_groups(cols: list[str]) -> pl.DataFrame:
        # 1) one u64 fingerprint per row (~256 MB for 32M rows), sorted in numpy: low memory
        hx = pl.struct(cols).hash(seed=0)
        h = _c(t.select(hx.alias("h")))["h"].to_numpy()
        h.sort()
        dup = np.unique(h[1:][h[1:] == h[:-1]])
        del h
        # 2) exact re-grouping of the (few) candidate rows: collisions cannot inflate counts
        cand = t.filter(hx.is_in(dup.tolist()))
        return _c(
            cand.group_by(cols).agg(
                n=pl.len(),
                n_pos=pl.col("is_laundering").cast(pl.Int64).sum(),
                ts=pl.col("timestamp").first(),
            )
        ).filter(pl.col("n") > 1)

    full = exact_groups(ALL_COLS)
    near = exact_groups(KEY_COLS)
    conflict = near.filter((pl.col("n_pos") > 0) & (pl.col("n_pos") < pl.col("n")))
    return {
        "exact_duplicate_groups": full.height,
        "exact_duplicate_extra_rows": int((full["n"] - 1).sum()) if full.height else 0,
        "exact_duplicate_groups_laundering": int((full["n_pos"] > 0).sum()) if full.height else 0,
        "exact_duplicate_max_copies": int(full["n"].max()) if full.height else 1,
        "same_fields_except_label_groups": near.height,
        "label_conflict_groups": conflict.height,
        "label_conflict_rows": int(conflict["n"].sum()) if conflict.height else 0,
    }


# ---------------------------------------------------------------- keys


def _norm_bank(e: pl.Expr) -> pl.Expr:
    s = e.str.strip_chars_start("0")
    return pl.when(s == "").then(pl.lit("0")).otherwise(s)


def key_audit(t: pl.LazyFrame, a: pl.LazyFrame) -> dict:
    """Is an account number unique on its own, or only within a bank? Do bank IDs join?"""
    nodes = pl.concat(
        [
            t.select(pl.col("from_bank").alias("bank"), pl.col("from_account").alias("account")),
            t.select(pl.col("to_bank").alias("bank"), pl.col("to_account").alias("account")),
        ]
    ).unique()
    nodes_df = _c(nodes)
    n_pairs = nodes_df.height
    per_acct = nodes_df.group_by("account").agg(pl.col("bank").n_unique().alias("nb"))
    banks = nodes_df.select(pl.col("bank").unique())
    banks_norm = banks.with_columns(norm=_norm_bank(pl.col("bank")))
    collisions = banks_norm.group_by("norm").agg(pl.len()).filter(pl.col("len") > 1)
    nodes_norm = nodes_df.with_columns(bank_norm=_norm_bank(pl.col("bank")))
    n_pairs_norm = nodes_norm.select("bank_norm", "account").unique().height

    acc = _c(a.select("bank_id", "account_number", "entity_id", "entity_type", "bank_name"))
    acc_pairs = acc.select("bank_id", "account_number").unique().height
    acc_bank_norm_differs = int(
        acc.select((pl.col("bank_id") != _norm_bank(pl.col("bank_id"))).sum()).item()
    )
    acc_keys = acc.select(
        _norm_bank(pl.col("bank_id")).alias("bank_norm"), pl.col("account_number").alias("account")
    ).unique()
    trans_keys = nodes_norm.select("bank_norm", "account").unique()
    in_acc = trans_keys.join(acc_keys, on=["bank_norm", "account"], how="semi").height
    acc_in_trans = acc_keys.join(trans_keys, on=["bank_norm", "account"], how="semi").height
    per_entity = acc.group_by("entity_id").agg(n_acc=pl.len(), n_banks=pl.col("bank_id").n_unique())
    etype = acc.group_by("entity_type").agg(pl.len()).sort("entity_type")
    return {
        "n_distinct_account_numbers": per_acct.height,
        "n_distinct_bank_account_pairs": n_pairs,
        "n_account_numbers_in_multiple_banks": int((per_acct["nb"] > 1).sum()),
        "max_banks_per_account_number": int(per_acct["nb"].max()),
        "account_number_alone_is_unique": n_pairs == per_acct.height,
        "n_distinct_bank_ids_raw": banks.height,
        "n_bank_ids_with_leading_zero": int(
            banks.select(pl.col("bank").str.starts_with("0").sum()).item()
        ),
        "n_bank_id_collisions_after_stripping_zeros": collisions.height,
        "n_pairs_after_bank_normalisation": n_pairs_norm,
        "accounts_file_rows": acc.height,
        "accounts_file_distinct_bank_account_pairs": acc_pairs,
        "accounts_file_key_unique": acc_pairs == acc.height,
        "accounts_file_bank_ids_with_leading_zero": acc_bank_norm_differs,
        "trans_pairs_found_in_accounts_file": in_acc,
        "trans_pairs_missing_from_accounts_file": trans_keys.height - in_acc,
        "accounts_file_pairs_never_transacting": acc_keys.height - acc_in_trans,
        "n_entities": per_entity.height,
        "n_entities_with_multiple_accounts": int((per_entity["n_acc"] > 1).sum()),
        "max_accounts_per_entity": int(per_entity["n_acc"].max()),
        "n_entities_spanning_multiple_banks": int((per_entity["n_banks"] > 1).sum()),
        "entity_type_counts": dict(
            zip(etype["entity_type"].to_list(), etype["len"].to_list(), strict=True)
        ),
        "n_entity_name_unparsed": int(acc["entity_type"].null_count()),
    }


# ---------------------------------------------------------------- labels and patterns


def labels_by_period(t: pl.LazyFrame, bounds: dict) -> dict:
    """AGGREGATE label counts per period: the one permitted holdout exposure."""
    df = _c(
        t.with_columns(period_expr(bounds))
        .group_by("period")
        .agg(n=pl.len(), n_laundering=pl.col("is_laundering").cast(pl.Int64).sum())
        .with_columns(rate=pl.col("n_laundering") / pl.col("n"))
        .sort("period")
    )
    return {r["period"]: {k: v for k, v in r.items() if k != "period"} for r in _rows(df)}


def patterns_vs_trans(t: pl.LazyFrame, p: pl.DataFrame, bounds: dict) -> dict:
    """How the patterns file relates to transaction labels."""
    pk = p.select([*KEY_COLS, "attempt_id", "txn_index"]).lazy()
    m = _c(t.select([*KEY_COLS, "row_id", "is_laundering"]).join(pk, on=KEY_COLS, how="inner"))
    matched_ptx = m.select("attempt_id", "txn_index").unique().height
    per_ptx = m.group_by("attempt_id", "txn_index").agg(pl.len())
    per_row = m.group_by("row_id").agg(pl.col("attempt_id").n_unique().alias("na"))
    lab_rows = m.select("row_id", "is_laundering").unique()
    unmatched = p.join(
        m.select("attempt_id", "txn_index").unique(), on=["attempt_id", "txn_index"], how="anti"
    ).with_columns(period_expr(bounds))
    um_period = unmatched.group_by("period").agg(pl.len()).sort("period")
    attributed = lab_rows.filter(pl.col("is_laundering") == 1).select("row_id")
    pos = t.filter(pl.col("is_laundering") == 1).with_columns(period_expr(bounds))
    unattr = _c(
        pos.join(attributed.lazy(), on="row_id", how="anti")
        .group_by("period")
        .agg(pl.len())
        .sort("period")
    )
    n_pos = _c(pos.select(pl.len())).item()
    return {
        "n_pattern_txns": p.height,
        "n_pattern_txns_matched_in_trans": matched_ptx,
        "n_pattern_txns_not_in_trans": p.height - matched_ptx,
        "pattern_txns_not_in_trans_by_period": dict(
            zip(um_period["period"].to_list(), um_period["len"].to_list(), strict=True)
        ),
        "n_pattern_txns_matching_multiple_trans_rows": int((per_ptx["len"] > 1).sum()),
        "n_trans_rows_matched": lab_rows.height,
        "n_matched_trans_rows_label_0": int((lab_rows["is_laundering"] == 0).sum()),
        "n_trans_rows_in_multiple_attempts": int((per_row["na"] > 1).sum()),
        "n_laundering_rows_total": n_pos,
        "n_laundering_rows_attributed_to_a_pattern": attributed.height,
        "n_laundering_rows_not_in_any_pattern": n_pos - attributed.height,
        "unattributed_laundering_by_period": dict(
            zip(unattr["period"].to_list(), unattr["len"].to_list(), strict=True)
        ),
    }


def typology_audit(p: pl.DataFrame, bounds: dict) -> dict:
    """Attempt-level structure. Period counts are aggregates (attempt assigned by its START)."""
    att = p.group_by("attempt_id").agg(
        typology=pl.col("typology").first(),
        start=pl.col("timestamp").min(),
        end=pl.col("timestamp").max(),
        n_txn=pl.len(),
        n_accounts=pl.concat_str("from_bank", pl.lit("|"), "from_account")
        .append(pl.concat_str("to_bank", pl.lit("|"), "to_account"))
        .n_unique(),
    )
    att = att.with_columns(
        duration_h=(pl.col("end") - pl.col("start")).dt.total_minutes() / 60,
        period=pl.when(pl.col("start") < bounds["holdout_start"])
        .then(pl.lit("pre_holdout"))
        .when(pl.col("start") < bounds["span_end_exclusive"])
        .then(pl.lit("holdout"))
        .otherwise(pl.lit("after_span")),
        crosses_holdout=(pl.col("start") < bounds["holdout_start"])
        & (pl.col("end") >= bounds["holdout_start"]),
        extends_past_span=pl.col("end") >= bounds["span_end_exclusive"],
    )
    by_typ = (
        att.group_by("typology")
        .agg(
            n_attempts=pl.len(),
            n_txns=pl.col("n_txn").sum(),
            median_txns=pl.col("n_txn").median(),
            max_txns=pl.col("n_txn").max(),
            median_accounts=pl.col("n_accounts").median(),
            max_accounts=pl.col("n_accounts").max(),
            median_duration_h=pl.col("duration_h").median(),
            p90_duration_h=pl.col("duration_h").quantile(0.9),
            max_duration_h=pl.col("duration_h").max(),
            n_extends_past_span=pl.col("extends_past_span").sum(),
        )
        .sort("typology")
    )
    per_period = att.group_by("typology", "period").agg(pl.len()).sort("typology", "period")
    pp: dict = {}
    for r in per_period.iter_rows(named=True):
        pp.setdefault(r["typology"], {})[r["period"]] = r["len"]
    return {
        "n_attempts": att.height,
        "by_typology": _rows(by_typ),
        "attempts_by_typology_and_start_period": pp,
        "attempt_duration_h": {
            "median": float(att["duration_h"].median()),
            "p90": float(att["duration_h"].quantile(0.9)),
            "p99": float(att["duration_h"].quantile(0.99)),
            "max": float(att["duration_h"].max()),
        },
        "n_attempts_crossing_holdout_start": int(att["crosses_holdout"].sum()),
        "n_attempts_extending_past_span_end": int(att["extends_past_span"].sum()),
    }


def laundering_accounts(t: pl.LazyFrame, p: pl.DataFrame, bounds: dict) -> dict:
    """Accounts ever involved in laundering (label=1 txns), and roles in patterns."""
    pos = t.filter(pl.col("is_laundering") == 1)
    src = pos.select(
        pl.concat_str("from_bank", pl.lit("|"), "from_account").alias("k"), pl.lit("out").alias("r")
    )
    dst = pos.select(
        pl.concat_str("to_bank", pl.lit("|"), "to_account").alias("k"), pl.lit("in").alias("r")
    )
    roles = _c(pl.concat([src, dst]).group_by("k").agg(pl.col("r").unique().alias("roles")))
    n_both = int(roles.select(pl.col("roles").list.len() == 2).to_series().sum())
    n_out_only = int(
        roles.select((pl.col("roles").list.len() == 1) & (pl.col("roles").list.first() == "out"))
        .to_series()
        .sum()
    )
    all_nodes = _c(
        pl.concat(
            [
                t.select(pl.concat_str("from_bank", pl.lit("|"), "from_account").alias("k")),
                t.select(pl.concat_str("to_bank", pl.lit("|"), "to_account").alias("k")),
            ]
        )
        .unique()
        .select(pl.len())
    ).item()
    pk = pl.concat(
        [
            p.select(
                pl.concat_str("from_bank", pl.lit("|"), "from_account").alias("k"),
                "attempt_id",
                "typology",
            ),
            p.select(
                pl.concat_str("to_bank", pl.lit("|"), "to_account").alias("k"),
                "attempt_id",
                "typology",
            ),
        ]
    )
    per = pk.group_by("k").agg(na=pl.col("attempt_id").n_unique(), nt=pl.col("typology").n_unique())
    return {
        "n_accounts_total": all_nodes,
        "n_accounts_in_laundering_txns": roles.height,
        "share_accounts_in_laundering_txns": roles.height / all_nodes,
        "n_laundering_accounts_sending_and_receiving": n_both,
        "n_laundering_accounts_sending_only": n_out_only,
        "n_laundering_accounts_receiving_only": roles.height - n_both - n_out_only,
        "n_accounts_in_patterns": per.height,
        "n_pattern_accounts_in_multiple_attempts": int((per["na"] > 1).sum()),
        "max_attempts_per_account": int(per["na"].max()),
        "n_pattern_accounts_in_multiple_typologies": int((per["nt"] > 1).sum()),
    }


# ---------------------------------------------------------------- currency


def currency_rates(t: pl.LazyFrame, start: datetime, end: datetime) -> dict:
    """Implied rates amount_received / amount_paid from cross-currency txns in [start, end).

    If each pair has one constant rate, the generator uses fixed FX and a deterministic
    conversion table can be derived from the data itself (no real-world FX imported).
    """
    cc = t.filter(
        (pl.col("timestamp") >= start)
        & (pl.col("timestamp") < end)
        & (pl.col("payment_currency") != pl.col("receiving_currency"))
        & (pl.col("amount_paid") > 0)
        & (pl.col("amount_received") > 0)  # rounding to 0.00 would give a meaningless rate
    ).with_columns(
        rate=pl.col("amount_received") / pl.col("amount_paid"), day=pl.col("timestamp").dt.date()
    )
    pairs = _c(
        cc.group_by("payment_currency", "receiving_currency")
        .agg(
            n=pl.len(),
            median=pl.col("rate").median(),
            q25=pl.col("rate").quantile(0.25),
            q75=pl.col("rate").quantile(0.75),
            min=pl.col("rate").min(),
            max=pl.col("rate").max(),
        )
        .with_columns(
            rel_iqr=pl.when(pl.col("median") > 0).then(
                (pl.col("q75") - pl.col("q25")) / pl.col("median")
            )
        )
        .sort("payment_currency", "receiving_currency")
    )
    daily = _c(
        cc.group_by("payment_currency", "receiving_currency", "day")
        .agg(m=pl.col("rate").median())
        .group_by("payment_currency", "receiving_currency")
        .agg(
            daily_median_min=pl.col("m").min(), daily_median_max=pl.col("m").max(), n_days=pl.len()
        )
    )
    pairs = pairs.join(
        daily, on=["payment_currency", "receiving_currency"], how="left"
    ).with_columns(
        daily_drift=pl.when(pl.col("median") > 0).then(
            (pl.col("daily_median_max") - pl.col("daily_median_min")) / pl.col("median")
        )
    )
    usd = {}
    for r in pairs.iter_rows(named=True):
        if r["payment_currency"] == "US Dollar":  # 1 USD -> median units of receiving ccy
            usd.setdefault(r["receiving_currency"], {})["from_usd"] = r["median"]
        if r["receiving_currency"] == "US Dollar":  # 1 unit of paying ccy -> USD
            usd.setdefault(r["payment_currency"], {})["to_usd"] = r["median"]
    usd_per_unit = {}
    for ccy, d in sorted(usd.items()):
        cands = [d["to_usd"]] if "to_usd" in d else []
        if "from_usd" in d:
            cands.append(1 / d["from_usd"])
        usd_per_unit[ccy] = {
            "usd_per_unit": float(np.median(cands)),
            "direct_vs_inverse_rel_diff": (abs(cands[0] - cands[1]) / cands[0])
            if len(cands) == 2
            else None,
        }
    usd_per_unit["US Dollar"] = {"usd_per_unit": 1.0, "direct_vs_inverse_rel_diff": 0.0}
    return {
        "scope": f"[{start.isoformat()}, {end.isoformat()})",
        "n_cross_currency_txns": int(pairs["n"].sum()) if pairs.height else 0,
        "n_pairs": pairs.height,
        "max_rel_iqr": float(pairs["rel_iqr"].max()) if pairs.height else None,
        "max_daily_drift": float(pairs["daily_drift"].max()) if pairs.height else None,
        "pairs": _rows(pairs),
        "usd_per_unit": usd_per_unit,
    }


# ---------------------------------------------------------------- trivial separability


def _tree_summary(clf, names: list[str]) -> list[dict]:
    tr = clf.tree_
    out = []
    for node in range(tr.node_count):
        v = tr.value[node][0]
        rate = float(v[1] / v.sum()) if v.sum() > 0 else None
        leaf = tr.children_left[node] == -1
        out.append(
            {
                "node": node,
                "depth": None,
                "split": None if leaf else f"{names[tr.feature[node]]} <= {tr.threshold[node]:.6g}",
                "n_samples": int(tr.n_node_samples[node]),
                "positive_rate": rate,
                "left": None if leaf else int(tr.children_left[node]),
                "right": None if leaf else int(tr.children_right[node]),
            }
        )
    # depth by walking from the root
    stack = [(0, 0)]
    while stack:
        n, d = stack.pop()
        out[n]["depth"] = d
        if out[n]["left"] is not None:
            stack += [(out[n]["left"], d + 1), (out[n]["right"], d + 1)]
    return out


def separability(t: pl.LazyFrame, bounds: dict) -> dict:
    """Depth-1 / depth-3 trees on naive transaction features, PRE-HOLDOUT data only.

    Temporal split inside pre-holdout: first 2/3 of days = fit, remaining days = evaluate.
    Currency conversion uses rates derived from the FIT window only.
    """
    from sklearn.metrics import average_precision_score
    from sklearn.tree import DecisionTreeClassifier

    start, ho = bounds["span_start"], bounds["holdout_start"]
    pre_days = (ho - start).days
    fit_end = start + timedelta(days=math.floor(SEPARABILITY_FIT_FRACTION * pre_days))
    rates = currency_rates(t, start, fit_end)["usd_per_unit"]
    rate_df = pl.DataFrame(
        {"payment_currency": list(rates), "usd": [v["usd_per_unit"] for v in rates.values()]}
    )
    cats = _c(pre_holdout(t, bounds).select(pl.col("payment_currency").unique().sort()))[
        "payment_currency"
    ].to_list()
    fmts = _c(pre_holdout(t, bounds).select(pl.col("payment_format").unique().sort()))[
        "payment_format"
    ].to_list()
    base = (
        pre_holdout(t, bounds)
        .join(rate_df.lazy(), on="payment_currency", how="left")
        .select(
            "timestamp",
            "amount_paid",
            amount_usd=pl.col("amount_paid") * pl.col("usd"),
            no_rate=pl.col("usd").is_null(),
            ccy=pl.col("payment_currency").cast(pl.Enum(cats)).to_physical(),
            fmt=pl.col("payment_format").cast(pl.Enum(fmts)).to_physical(),
            hour=pl.col("timestamp").dt.hour(),
            self_transfer=(pl.col("from_bank") == pl.col("to_bank"))
            & (pl.col("from_account") == pl.col("to_account")),
            cross_currency=pl.col("payment_currency") != pl.col("receiving_currency"),
            y=pl.col("is_laundering"),
        )
    )
    groups = {
        "amount_paid_raw": ["amount_paid_raw"],
        "amount_usd": ["log10_amount_usd"],
        "payment_currency": [f"payment_currency={c}" for c in cats],
        "payment_format": [f"payment_format={f}" for f in fmts],
        "hour": ["hour"],
        "self_transfer": ["self_transfer"],
        "cross_currency": ["cross_currency"],
    }
    names = [n for g in groups.values() for n in g]

    def compact(lo: datetime, hi: datetime) -> pl.DataFrame:
        # compact numeric columns only (codes instead of strings): ~22 bytes per row
        return _c(
            base.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi)).drop("timestamp")
        )

    def build_x(df: pl.DataFrame, gnames: list[str]) -> np.ndarray:
        width = sum(len(groups[g]) for g in gnames)
        x = np.empty((df.height, width), dtype=np.float32)
        j = 0
        for g in gnames:
            if g == "amount_paid_raw":
                x[:, j] = df["amount_paid"].to_numpy()
                j += 1
            elif g == "amount_usd":
                u = df["amount_usd"].fill_null(-1.0).to_numpy()
                x[:, j] = np.where(u > 0, np.log10(np.clip(u, 1e-12, None)), -99.0)
                j += 1
            elif g in ("payment_currency", "payment_format"):
                codes = df["ccy" if g == "payment_currency" else "fmt"].to_numpy()
                for k in range(len(groups[g])):
                    x[:, j] = codes == k
                    j += 1
            else:
                x[:, j] = df[g].to_numpy()
                j += 1
        return x

    all_groups = list(groups)
    specs = {"depth_1": (all_groups, 1), "depth_3": (all_groups, 3)}
    specs |= {f"single:{g}": ([g], 3) for g in all_groups}

    fit = compact(start, fit_end)
    y_fit = fit["y"].to_numpy().astype(np.int8)
    nr_fit = int(fit["no_rate"].sum())
    models = {}
    if y_fit.sum() > 0:
        x_full = build_x(fit, all_groups)
        for key in ("depth_1", "depth_3"):
            models[key] = DecisionTreeClassifier(max_depth=specs[key][1], random_state=0).fit(
                x_full, y_fit
            )
        del x_full
        for g in all_groups:
            xg = build_x(fit, [g])
            models[f"single:{g}"] = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
                xg, y_fit
            )
            del xg
    del fit
    ev = compact(fit_end, ho)
    y_ev = ev["y"].to_numpy().astype(np.int8)
    res: dict = {
        "scope": "pre_holdout only",
        "fit_window": [start.isoformat(), fit_end.isoformat()],
        "eval_window": [fit_end.isoformat(), ho.isoformat()],
        "n_fit": int(len(y_fit)),
        "n_fit_positive": int(y_fit.sum()),
        "n_eval": int(len(y_ev)),
        "n_eval_positive": int(y_ev.sum()),
        "eval_prevalence": float(y_ev.mean()) if len(y_ev) else None,
        "rows_without_usd_rate_fit_eval": [nr_fit, int(ev["no_rate"].sum())],
        "features": names,
        "models": {},
        "single_feature_group_depth3": {},
    }
    if y_fit.sum() == 0 or y_ev.sum() == 0:
        res["error"] = "no positives in fit or eval window"
        return res
    prev = float(y_ev.mean())
    x_full = build_x(ev, all_groups)
    for key in ("depth_1", "depth_3"):
        clf = models[key]
        ap = float(average_precision_score(y_ev, clf.predict_proba(x_full)[:, 1]))
        imp = sorted(
            ((names[i], float(v)) for i, v in enumerate(clf.feature_importances_) if v > 0),
            key=lambda kv: -kv[1],
        )
        res["models"][key] = {
            "pr_auc_eval": ap,
            "pr_auc_over_prevalence": ap / prev,
            "feature_importance": imp,
            "tree": _tree_summary(clf, names),
        }
    del x_full
    for g in all_groups:
        ap = float(
            average_precision_score(
                y_ev, models[f"single:{g}"].predict_proba(build_x(ev, [g]))[:, 1]
            )
        )
        res["single_feature_group_depth3"][g] = {
            "pr_auc_eval": ap,
            "pr_auc_over_prevalence": ap / prev,
        }
    return res


# ---------------------------------------------------------------- identifier leakage


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def identifier_leakage(t: pl.LazyFrame, a: pl.LazyFrame, bounds: dict) -> dict:
    """Do identifier values or file order carry the label? PRE-HOLDOUT data only.

    Account-level label = account appears in >= 1 laundering txn before holdout_start.
    """
    pre = pre_holdout(t, bounds)
    nodes = pl.concat(
        [
            pre.select(
                pl.col("from_bank").alias("bank"),
                pl.col("from_account").alias("account"),
                "is_laundering",
            ),
            pre.select(
                pl.col("to_bank").alias("bank"),
                pl.col("to_account").alias("account"),
                "is_laundering",
            ),
        ]
    )
    acc = _c(nodes.group_by("bank", "account").agg(y=pl.col("is_laundering").max().cast(pl.Int8)))
    acc = acc.with_columns(
        id_int=pl.col("account").str.to_integer(base=16, strict=False),
        last_hex=pl.col("account").str.slice(-1),
        prefix2=pl.col("account").str.slice(0, 2),
        id_len=pl.col("account").str.len_chars(),
        bank_norm=_norm_bank(pl.col("bank")),
    )
    y = acc["y"].to_numpy()
    prev = float(y.mean())
    out: dict = {
        "scope": "pre_holdout accounts",
        "n_accounts": acc.height,
        "n_laundering_accounts": int(y.sum()),
        "account_prevalence": prev,
        "n_account_ids_not_hex": int(acc["id_int"].null_count()),
    }
    ok = acc.filter(pl.col("id_int").is_not_null())
    yy = ok["y"].to_numpy()
    ids = ok["id_int"].to_numpy().astype(np.float64)
    out["pr_auc_account_id_as_score"] = _ap(yy, ids)
    out["pr_auc_negated_account_id_as_score"] = _ap(yy, -ids)
    # contiguity: is the next account in ID order also laundering more often than chance?
    srt = ok.sort("id_int", "bank")
    ys = srt["y"].to_numpy()
    nxt = ys[1:][ys[:-1] == 1]
    out["contiguity"] = {
        "p_next_id_laundering_given_laundering": float(nxt.mean()) if len(nxt) else None,
        "baseline_prevalence": prev,
        "lift": (float(nxt.mean()) / prev) if len(nxt) and prev > 0 else None,
    }

    def rate_table(col: str) -> list[dict]:
        return _rows(
            acc.group_by(col)
            .agg(n=pl.len(), n_laundering=pl.col("y").cast(pl.Int64).sum())
            .with_columns(
                rate=pl.col("n_laundering") / pl.col("n"),
                lift=(pl.col("n_laundering") / pl.col("n")) / prev,
            )
            .sort(col)
            .with_columns(pl.col(col).cast(pl.String))
        )

    out["by_last_hex_digit"] = rate_table("last_hex")
    out["by_id_length"] = rate_table("id_len")
    pref = (
        acc.group_by("prefix2")
        .agg(n=pl.len(), k=pl.col("y").cast(pl.Int64).sum())
        .filter(pl.col("n") >= MIN_SUPPORT)
    )
    pref = pref.with_columns(lift=(pl.col("k") / pl.col("n")) / prev).sort("lift", descending=True)
    out["id_prefix2_max_lift_with_support"] = _rows(pref.head(5))
    banks = (
        acc.group_by("bank_norm")
        .agg(n=pl.len(), k=pl.col("y").cast(pl.Int64).sum())
        .filter(pl.col("n") >= MIN_SUPPORT)
    )
    banks = banks.with_columns(lift=(pl.col("k") / pl.col("n")) / prev).sort(
        "lift", "bank_norm", descending=[True, False]
    )
    out["bank_max_lift_with_support"] = _rows(banks.head(5))
    out["n_banks_with_support"] = banks.height

    # KYC attributes from the accounts file
    ak = _c(
        a.select(
            _norm_bank(pl.col("bank_id")).alias("bank_norm"),
            pl.col("account_number").alias("account"),
            "entity_type",
            pl.when(pl.col("bank_name").str.contains(r"^[A-Za-z ]+ Bank #\d+$"))
            .then(pl.col("bank_name").str.extract(r"^([A-Za-z ]+) Bank #\d+$", 1))
            .otherwise(pl.lit("US-named bank"))
            .alias("bank_country_label"),
        )
    )
    j = acc.join(ak, on=["bank_norm", "account"], how="left")
    out["n_accounts_without_kyc_row"] = int(j["entity_type"].null_count())
    for col in ("entity_type", "bank_country_label"):
        tab = (
            j.group_by(col)
            .agg(n=pl.len(), n_laundering=pl.col("y").cast(pl.Int64).sum())
            .with_columns(rate=pl.col("n_laundering") / pl.col("n"))
            .with_columns(lift=pl.col("rate") / prev)
            .sort(col, nulls_last=True)
        )
        out[f"by_{col}"] = _rows(tab)

    # file order vs label (transaction level, pre-holdout rows)
    dec = _c(
        pre.select("row_id", "is_laundering")
        .with_columns(decile=(pl.col("row_id") * 10 // (pl.col("row_id").max() + 1)).cast(pl.Int32))
        .group_by("decile")
        .agg(n=pl.len(), n_laundering=pl.col("is_laundering").cast(pl.Int64).sum())
        .with_columns(rate=pl.col("n_laundering") / pl.col("n"))
        .sort("decile")
    )
    out["txn_rate_by_file_position_decile"] = _rows(dec)
    return out


# ---------------------------------------------------------------- footprint


def footprint(t: pl.LazyFrame, n_rows: int, sample_rows: int = 1_000_000) -> dict:
    """In-memory size of the full transactions table (strings as Categorical), by extrapolation."""
    s = t.head(sample_rows).collect()
    cat = s.with_columns(pl.col(pl.String).cast(pl.Categorical))
    k = n_rows / max(s.height, 1)
    return {
        "est_in_memory_gb_strings": s.estimated_size() * k / 1e9,
        "est_in_memory_gb_categorical": cat.estimated_size() * k / 1e9,
        "sample_rows": s.height,
    }
