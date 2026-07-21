# ============================================================================
# CHURN NOTEBOOK — replacement for "## 9. Export model artifact" (cell 26)
# ----------------------------------------------------------------------------
# WHY: the combined PLTV build needs the churn model to be fully self-contained
# so the downstream script can compute E[orders] = p * (1 + T_{k+1}) WITHOUT
# re-deriving the survival curve. This version bakes the population survival
# curve r_k and the forward tail T_k INTO the artifact, and drops the
# google.colab dependency so it runs anywhere.
#
# Run this AFTER the projection / calibration cells so `test_scored` (or `val`)
# and `probs` exist. It reuses the same MAX_BUCKET / HORIZON as the notebook.
# ============================================================================
import numpy as np, sklearn, catboost, joblib

MAX_BUCKET = 8      # order_seq >= 8 share one "8+" steady-state repeat rate
HORIZON    = 15     # cap on look-ahead so loyal customers don't project to infinity

# --- 1) Population survival curve r_k = P(order k -> k+1), from scored TEST rows ---
#     test_scored is built in the projection cell (§8). Fall back to `val` (§ Val-3).
_src = test_scored if "test_scored" in dir() else val
_cap = np.minimum(_src["order_seq"].to_numpy(), MAX_BUCKET)
r_arr = np.array([_src.loc[_cap == k, "p_repeat"].mean() for k in range(1, MAX_BUCKET + 1)])
r_arr = pd.Series(r_arr).ffill().bfill().to_numpy()          # fill any empty high stage
r_at  = lambda m: r_arr[min(m, MAX_BUCKET) - 1]

# --- 2) Forward multiplier M_k = 1 + r_{k+1} * M_{k+1}, horizon-capped ---
#     This is EXACTLY the recursion used in the notebook's projection (§ Val-3):
#         expected_future_orders = p_repeat * M_k
#     M_k = expected orders "in the bank" for a customer at order k IF they make
#     their next order (the immediate next order + everything expected to follow).
#     It maps to the user's formula E[orders] = p * (1 + T_{k+1}) with M_k = 1 + T_{k+1}.
#     Stored (not T) so the downstream score is bit-for-bit consistent with the notebook.
M = {HORIZON + 1: 1.0}
for k in range(HORIZON, 0, -1):
    M[k] = 1.0 + r_at(k + 1) * M[k + 1]

# --- 3) Bundle everything the scoring script needs ---
print("PIN:", "scikit-learn==" + sklearn.__version__, "catboost==" + catboost.__version__)
artifact = {
    "model":         model,
    "feature_cols":  list(X.columns),
    "cat_features":  cat_features,
    "id_cols":       ID_COLS,
    "leakage_cols":  LEAKAGE_COLS,
    "label_col":     LABEL_COL,
    "group_col":     GROUP_COL,
    "churn_window_days": CHURN_WINDOW,
    "train_vocab":   {c: sorted(set(map(str, X[c].unique()))) for c in cat_features},
    # ----- projection payload (NEW) -----
    "max_bucket":    MAX_BUCKET,
    "horizon":       HORIZON,
    "survival_curve_r": r_arr.tolist(),                       # r_arr[k-1] = r_k
    "forward_multiplier_M": {int(k): float(v) for k, v in M.items()},  # M_k = 1 + T_{k+1}
    "risk_tiers":    {"high": 0.7, "medium": 0.5},            # churn_prob thresholds
    "sklearn_version": sklearn.__version__,
    "catboost_version": catboost.__version__,
    "model_version": "churn_order_level_v0.2",               # +projection payload
}
joblib.dump(artifact, "churn_order_level_artifact.joblib")
print("Saved churn_order_level_artifact.joblib")
print("r_k:", [round(x, 3) for x in r_arr])
print("M_1..M_5:", [round(M[k], 3) for k in range(1, 6)])

# Optional Colab download (keep the try/except so the cell is portable):
try:
    from google.colab import files
    files.download("churn_order_level_artifact.joblib")
except Exception:
    pass
