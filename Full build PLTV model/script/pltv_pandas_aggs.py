#!/usr/bin/env python3
# ============================================================================
# COMBINED PLTV BUILD — aggregation functions for pandas DataFrames
# ============================================================================
#
# WHAT THIS FILE DOES
#   This file is used to aggregate the final PLTV table for reporting and visualization.
#   It draws from the scored DataFrame produced by the build_combined_pltv.py script, located in outputs as 'pltv_scores.csv'.
#   Create visualizations using matplotlib and seaborn to analyze the distribution of remaining PLTV across deciles and groups.
# 
# VALIDATIONS:
#   1) The decile check for remaining PLTV should show a decreasing trend in the mean and median values across deciles.
#   2) The distribution of remaining PLTV should be right-skewed, with most customers having low remaining PLTV and a few having high remaining PLTV.
#   3) The percentage of total PLTV contributed by the top 10% of customers should be significantly higher than that contributed by the bottom 10%.
#   4) There should be <0.5% of customers with null remaining PLTV and expected AOV values.
#   5) Customers with high churn risk (churn_prob >= 0.75) should have their expected future orders set to 0, resulting in a remaining PLTV of 0.
#
# VISUALIZATIONS:
#   1) Histogram of remaining PLTV to visualize the distribution and identify any outliers
#   2) Scatter plot of expected future orders vs. remaining PLTV to analyze the relationship between these two variables
#   3) Value concentration plot to show the contribution of different deciles to the total PLTV
#   4) Histogram of remaining PLTV by risk tier (high, medium, low) to visualize the distribution of remaining PLTV across different churn risk levels

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent      # .../Full build PLTV model
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# ----------------------------------------------------------------------------
# FILTER: Customers with >20 orders (order_seq > 20) are considered "loyal" are excluded from visualizations to avoid skewing the 
# distribution. They are still included in the final PLTV table.
# Customers with 0 remaining PLTV are also excluded from visualizations to avoid skewing the distribution.
# ----------------------------------------------------------------------------
def filter_loyal_customers(pltv: pd.DataFrame, loyal_threshold: int = 20) -> pd.DataFrame:
    """Filter out loyal customers (order_seq > loyal_threshold) and those with 0 remaining PLTV for visualization purposes."""
    filtered = pltv[(pltv["order_seq"] <= loyal_threshold) & (pltv["remaining_pltv"] > 0)].copy()
    print(f"Filtered out {len(pltv) - len(filtered)} loyal customers and those with 0 remaining PLTV.")
    return filtered

# ----------------------------------------------------------------------------
# VALIDATIONS
# ----------------------------------------------------------------------------

def validate_pltv_scores(pltv: pd.DataFrame) -> None:
    """Run a series of checks on the PLTV scores DataFrame to ensure data integrity and expected distributions."""
    
    # Check for null values in key columns
    null_counts = pltv[["expected_aov"]].isnull().sum()
    print(f"Null counts in expected_aov: {null_counts.max()}")
    
    # Check decile distribution of remaining PLTV
    deciles = pltv["remaining_pltv"].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    print("Decile distribution of remaining PLTV:")
    print(deciles)
    
    # Check contribution of top vs bottom deciles
    top_10_pct = pltv.nlargest(int(0.1 * len(pltv)), "remaining_pltv")["remaining_pltv"].sum().round(2)
    bottom_10_pct = pltv.nsmallest(int(0.1 * len(pltv)), "remaining_pltv")["remaining_pltv"].sum().round(2)
    print(f"Top 10% contribution to total PLTV: {top_10_pct}")
    print(f"Bottom 10% contribution to total PLTV: {bottom_10_pct}")

    # Check that high churn risk customers have expected future orders set to 0
    high_churn_customers = pltv[pltv["churn_prob"] >= 0.75]
    high_churn_percent = len(high_churn_customers) / len(pltv) * 100
    print(f"Percentage of high churn risk customers (churn_prob >= 0.75): {high_churn_percent:.2f}%")
    print(f"Number of high churn risk customers (churn_prob >= 0.75): {len(high_churn_customers)}")
    print(f"Expected future orders for high churn risk customers (should be 0): {high_churn_customers['expected_future_orders'].unique()}")

# ----------------------------------------------------------------------------
# VISUALIZATIONS
# ----------------------------------------------------------------------------

def visualize_pltv_distribution(pltv: pd.DataFrame) -> None:
    """Create visualizations to analyze the distribution of remaining PLTV."""

    # Defensive: compute decile if it doesn't already exist
    if "decile" not in pltv.columns:
        pltv["decile"] = pd.qcut(pltv["remaining_pltv"].rank(method="first"), 10, labels=False) + 1

    # Defensive: fall back if customer_segment is missing
    segment_col = "customer_segment" if "customer_segment" in pltv.columns else None

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("PLTV Distribution Visualizations (churn<0.75, order_seq <= 20)", fontsize=12)

    # Histogram of remaining PLTV (log scale needs strictly positive values)
    positive_pltv = pltv["remaining_pltv"][pltv["remaining_pltv"] > 0]
    ax[0,0].hist(positive_pltv, bins=50, color='skyblue', edgecolor='black')
    ax[0,0].set_title("Distribution of Remaining PLTV")
    ax[0,0].set_xlabel("Remaining PLTV ($)")
    ax[0,0].set_ylabel("Frequency")
    ax[0,0].set_xlim(0, 2000)

    # Scatter plot of expected future orders vs. remaining PLTV
    mask = (pltv["expected_future_orders"] > 0) & (pltv["remaining_pltv"] > 0)
    ax[0,1].scatter(pltv.loc[mask, "expected_future_orders"], pltv.loc[mask, "remaining_pltv"], alpha=0.5, color='salmon')
    ax[0,1].set_title("Expected Future Orders vs. Remaining PLTV")
    ax[0,1].set_xlabel("Expected Future Orders")
    ax[0,1].set_ylabel("Remaining PLTV ($)")
    ax[0,1].set_ylim(0, 2000)
    z = np.polyfit(pltv.loc[mask, "expected_future_orders"], pltv.loc[mask, "remaining_pltv"], 1)
    p = np.poly1d(z)
    ax[0,1].plot(pltv.loc[mask, "expected_future_orders"], p(pltv.loc[mask, "expected_future_orders"]), "r--", linewidth=1)

    # Value concentration plot by decile — % of total PLTV
    decile_totals = pltv.groupby("decile")["remaining_pltv"].sum().sort_index()
    decile_pct = decile_totals / decile_totals.sum() * 100

    ax[1,0].bar(decile_pct.index, decile_pct.values, color='lightgreen', edgecolor='black')
    ax[1,0].set_title("% of Total PLTV by Decile")
    ax[1,0].set_xlabel("Decile")
    ax[1,0].set_ylabel("% of Total Remaining PLTV")
    ax[1,0].set_xticks(decile_pct.index)
    # percentage read above each bar
    ax[1,0].bar_label(ax[1,0].containers[0], fmt='%.1f%%', label_type='edge', fontsize=8, padding=2)

    # Bar chart of remaining PLTV by risk tier (high, medium, low)
    sns.barplot(data=pltv, x="risk_tier", y="remaining_pltv", hue="risk_tier", ax=ax[1,1], palette="Set2")
    ax[1,1].set_title("Distribution of Remaining PLTV by Risk Tier")
    ax[1,1].set_xlabel("Risk Tier")
    ax[1,1].set_ylabel("Remaining PLTV")
    plt.tight_layout()
    # plt.show()
    # Save plots
    plt.savefig(OUTPUTS_DIR / "pltv_distribution_plots.png")
# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    # Load the PLTV scores DataFrame
    pltv_scores_path = OUTPUTS_DIR / "pltv_scores.csv"
    pltv = pd.read_csv(pltv_scores_path)

    # Validate the PLTV scores
    validate_pltv_scores(pltv)

    # Filter out loyal customers
    pltv = filter_loyal_customers(pltv)

    # Visualize the distribution of remaining PLTV
    visualize_pltv_distribution(pltv)

if __name__ == "__main__":
    main()
