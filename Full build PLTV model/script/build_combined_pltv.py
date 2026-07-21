#!/usr/bin/env python3
# ============================================================================
# COMBINED PLTV BUILD — E[orders] x E[AOV] -> per-customer PLTV
# ----------------------------------------------------------------------------
# WHAT THIS DOES
#   Joins the two phases into one dollar score per customer:
#
#       PLTV_c  =  E[future orders]_c   x   E[AOV]_c
#                  \_______________/       \________/
#                   Phase 1 (churn)         Phase 2 (AOV)
#
#   where, for a customer currently on their k-th order with next-order
#   probability p = p_repeat (from the churn model on their LATEST order):
#
#       E[future orders] = p * (1 + T_{k+1})
#
#   T_{k+1} is the population forward tail (expected additional orders that
#   follow order k+1), baked into the churn artifact by the notebook export
#   snippet. This is the exact recursion T_k = r_k * (1 + T_{k+1}).
#
# INPUTS (all produced upstream — see README):
#   1. churn_order_level_artifact.joblib   <- churn notebook (v0.2 export snippet)
#        carries: model, feature_cols, cat_features, forward_tail_T, max_bucket,
#        horizon, risk_tiers.
#   2. pltv_combined_scoring_extract.csv    <- sql/pltv_combined_scoring_extract.sql
#        one row per customer = their latest order's features (+ order_seq = k).
#   3. expected_aov_by_customer.csv         <- AOV notebook §4 output
#        columns: shopify_customer_id, expected_aov (E[AOV_c]).
#
# OUTPUT:
#   pltv_scores.csv  — one row per customer:
#        shopify_customer_id, order_seq, churn_prob, p_repeat, risk_tier,
#        expected_future_orders, expected_aov,
#        remaining_pltv (future $), realized_value, total_pltv.
#
# This is an OUTLINE / reference implementation: paths and a couple of column
# names (flagged CONFIRM) need wiring to your environment. Logic is complete.
# ============================================================================

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Resolve paths relative to THIS file, not the current working directory, so the
# script runs identically whether you launch it from a terminal or hit VS Code's
# "Run Python File" button (which sets cwd to the workspace root, not script/).
PROJECT_DIR = Path(__file__).resolve().parent.parent      # .../Full build PLTV model
INPUTS_DIR  = PROJECT_DIR / "inputs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# ----------------------------------------------------------------------------
# 1) Load the churn artifact and rebuild the feature-prep used at training time
# ----------------------------------------------------------------------------
def load_artifact(path: str) -> dict:
    art = joblib.load(path)
    # sanity: the projection payload must be present (v0.2 export snippet)
    for key in ("forward_multiplier_M", "max_bucket", "horizon", "feature_cols",
                "cat_features", "risk_tiers"):
        if key not in art:
            raise KeyError(f"artifact missing '{key}' — re-export with the v0.2 "
                           f"snippet in notebook_snippets/export_churn_artifact.py")
    return art


def prepare_features(frame: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Mirror the notebook's prepare_features + reindex to the trained columns.
    Booleans -> 0/1, categorical NaNs -> 'missing', reindexed to feature_cols."""
    cat_cols = art["cat_features"]
    f = frame.copy()
    for c in f.columns:
        if f[c].dtype == bool:
            f[c] = f[c].astype(int)
    for c in cat_cols:
        if c in f.columns:
            f[c] = np.where(f[c].isna(), "missing", f[c].astype(str))
    # exact training column order; any missing feature is added as NaN (CatBoost-safe)
    return f.reindex(columns=art["feature_cols"])


# ----------------------------------------------------------------------------
# 2) Phase 1 — score churn and project future orders
# ----------------------------------------------------------------------------
def score_orders(scoring_df: pd.DataFrame, art: dict) -> pd.DataFrame:
    model = art["model"]
    horizon = art["horizon"]
    M = {int(k): float(v) for k, v in art["forward_multiplier_M"].items()}  # M_k = 1 + T_{k+1}
    hi = art["risk_tiers"]["high"]
    med = art["risk_tiers"]["medium"]

    X = prepare_features(scoring_df, art)
    churn_prob = model.predict_proba(X)[:, 1]
    p_repeat = 1.0 - churn_prob

    k = scoring_df["order_seq"].to_numpy(dtype=int)
    # M_k, order stage capped at the horizon (matches the notebook's clip)
    k_cap = np.minimum(k, horizon)
    M_k = np.array([M.get(int(kk), 1.0) for kk in k_cap])

    # E[future orders] = p * M_k = p * (1 + T_{k+1})   (notebook-exact)
    expected_future_orders = p_repeat * M_k

    risk_tier = np.where(churn_prob >= hi, "High",
                 np.where(churn_prob >= med, "Medium", "Low"))

    return pd.DataFrame({
        "shopify_customer_id": scoring_df["shopify_customer_id"].to_numpy(),
        "order_seq": k,
        "churn_prob": churn_prob,
        "p_repeat": p_repeat,
        "risk_tier": risk_tier,
        "expected_future_orders": expected_future_orders,
        # realized value so far (net spend to date) for a total-PLTV view
        "realized_value": scoring_df.get("cum_net_spend", pd.Series(np.nan, index=scoring_df.index)).to_numpy(),
    })

# ----------------------------------------------------------------------------
# 2.5) Give orders with 75% churn risk E[AOV] = 0 (0.75 churn score ~ 86.7% precision)
# ----------------------------------------------------------------------------
def filter_high_churn(scored: pd.DataFrame, cutoff: float = 0.75) -> pd.DataFrame:
    """Customers with churn_prob >= cutoff are considered "lost" and get E[AOV] = 0.
    This is a business decision, not a model limitation. The cutoff can be tuned."""
    out = scored.copy()
    out.loc[out["churn_prob"] >= cutoff, "expected_future_orders"] = 0.0
    return out

# ----------------------------------------------------------------------------
# 3) Phase 2 — join expected AOV and multiply
# ----------------------------------------------------------------------------
def combine(scored: pd.DataFrame, aov: pd.DataFrame,
            fallback_aov: float | None = None) -> pd.DataFrame:
    aov = aov.rename(columns={"expected_aov": "expected_aov"})  # CONFIRM column name
    out = scored.merge(aov[["shopify_customer_id", "expected_aov"]],
                       on="shopify_customer_id", how="left")

    # customers with no AOV estimate (e.g. brand-new) -> population fallback AOV
    if fallback_aov is not None:
        out["expected_aov"] = out["expected_aov"].fillna(fallback_aov)

    out["remaining_pltv"] = out["expected_future_orders"] * out["expected_aov"]
    out["total_pltv"] = out["realized_value"].fillna(0) + out["remaining_pltv"]
    return out.sort_values("total_pltv", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 4) Create data visualization of the PLTV distribution
# ----------------------------------------------------------------------------
def visualize_pltv_distribution(pltv: pd.DataFrame):
    # Check number of customers with non-null remaining PLTV vs number of customers with null remaining PLTV
    num_non_null = pltv["remaining_pltv"].notnull().sum()
    num_null = pltv["remaining_pltv"].isnull().sum()
    print(f"\nNumber of customers with non-null remaining PLTV: {num_non_null}")
    print(f"Number of customers with null remaining PLTV: {num_null}")

    # Check number of customers in expected AOV DataFrame vs number of customers in the scored DataFrame
    num_customers_aov = pltv["expected_aov"].notnull().sum()
    num_customers_scored = pltv.shape[0]
    print(f"\nNumber of customers in expected AOV DataFrame: {num_customers_aov}")
    print(f"Number of customers in scored DataFrame: {num_customers_scored}")

    # Mean remaining PLTV by order_seq (k) — useful for sanity check and business insight
    mean_by_order_seq = pltv.groupby("order_seq")["remaining_pltv"].mean().round(2)
    print("\nMean remaining PLTV by order_seq:")
    print(mean_by_order_seq.to_string())

    # # Optional: visualize with matplotlib (if installed)
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 5))
    #     plt.hist(pltv["remaining_pltv"], bins=50, color='skyblue', edgecolor='black')
    #     plt.title("Distribution of Remaining PLTV (Future $)")
    #     plt.xlabel("Remaining PLTV ($)")
    #     plt.ylabel("Number of Customers")
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.show()
    # except ImportError:
    #     print("matplotlib not installed; skipping visualization.")
# ----------------------------------------------------------------------------
# 5) Wire it together
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build combined PLTV = E[orders] x E[AOV].")
    # Defaults live in inputs/ and outputs/ next to the script; override if needed.
    ap.add_argument("--artifact",    default=str(INPUTS_DIR / "churn_order_level_artifact.joblib"))
    ap.add_argument("--scoring-csv", default=str(INPUTS_DIR / "pltv_combined_scoring_extract.csv"))
    ap.add_argument("--aov-csv",     default=str(INPUTS_DIR / "expected_aov_by_customer.csv"))
    ap.add_argument("--out",         default=str(OUTPUTS_DIR / "pltv_scores.csv"))
    ap.add_argument("--fallback-aov", type=float, default=None,
                    help="population AOV for customers with no per-customer estimate")
    args = ap.parse_args()

    art = load_artifact(args.artifact)
    scoring_df = pd.read_csv(args.scoring_csv)
    aov = pd.read_csv(args.aov_csv)

    scored = score_orders(scoring_df, art)
    cut_scored = filter_high_churn(scored, cutoff=0.75)
    pltv = combine(cut_scored, aov, fallback_aov=args.fallback_aov)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    pltv.to_csv(args.out, index=False)
    print(f"Wrote {len(pltv):,} customers -> {args.out}")
    print(pltv[["shopify_customer_id", "order_seq", "p_repeat",
                "expected_future_orders", "expected_aov",
                "remaining_pltv", "total_pltv"]].head(10).to_string(index=False))
    print("\nDecile check (remaining_pltv):")
    print(pltv["remaining_pltv"].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
    visualize_pltv_distribution(pltv)

if __name__ == "__main__":
    main()
