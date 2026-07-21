# Churn Model — 400-Day Repeat-Purchase Hazard (Phase 1)

A discrete-time repeat-purchase **hazard** model. Every **customer-order** (purchase-day) is
one row; given a customer's state *as of* an order, it predicts whether they will order again
within **400 days**. It is the order-count engine for PLTV: each order's repeat probability
chains into an expected number of future orders, which Phase 2 (AOV) turns into dollars.

Phase 1 answers *"how many more orders?"* · Phase 2 (AOV model) answers *"worth how much each?"* ·
PLTV = **projected orders × expected AOV**.

---

## Pipeline (run in this order)

1. **`churn_pltv_features_order_level.sql`** — run in BigQuery, export the result to CSV
   (Google Drive), and copy the share link.
2. **`Train_Churn_OrderLevel_CatBoost.ipynb`** — point `DRIVE_CSV_URL` at that CSV and run top
   to bottom. It splits (grouped by customer), trains CatBoost, evaluates, validates the
   projection, and exports the model artifact.
3. **Dashboards** (`../Dashboards/`) — the shareable explainers: data exploration and the
   plain-English write-up of how a churn score becomes projected future orders.

---

## Files

| Path | What it is |
|------|------------|
| `churn_pltv_features_order_level.sql` | **Feature extract.** One row per `(shopify_customer_id, purchase-day)` from `core.orders`. Builds leakage-safe cumulative / sequence / cadence / breadth / prior-return features and the `churned_400d` label. Emits only orders whose full 400-day window is observed. |
| `Train_Churn_OrderLevel_CatBoost.ipynb` | **The model.** Loads the extract, grouped train/val/test split, trains CatBoost, reports AUC / PR-AUC, feature importance, calibration by `order_seq`, chained-survival projected orders, and exports `churn_order_level_artifact.joblib`. |
| `../Dashboards/PLTV & Churn Model — Data Exploration Dashboard.html` | Data-exploration dashboard (distributions, signal exploration). |
| `../Dashboards/Expected Orders Survival Calculation.html` | Plain-English explainer of the survival curve → projected-orders math. |

---

## The label

`churned_400d = 1` if the customer places **no further order within 400 days** of *this* order;
`0` if they reorder inside the window. It is anchored to **each order's own date**, so every
order gets an equal 400-day forward look — the label is **cohort-age-neutral**. Only orders with
`order_date ≤ today − 400d` are emitted so the full window is observed. Cumulative features still
look back over the customer's entire history (including pre-2023) for accuracy.

## Grain & why order-level

One row per **purchase-day** (distinct `order_created_date`) unifies the old first-order and
two-order models into a single hazard model, with `order_seq` as a feature. A customer with four
purchase-days contributes four rows. Purchase-day (not raw order id) absorbs same-day
split/exchange artifacts. **Note:** the Phase-2 AOV model uses the `order_id` grain so AOV and
order counts multiply consistently — a known, intentional grain difference between the two phases.

## Leakage discipline (the core design constraint)

- **Grouped split by `shopify_customer_id`** — never split by row. A customer's rows are
  correlated; a random split would leak them across train/test. `GroupShuffleSplit` keeps every
  customer entirely on one side; assertions confirm disjoint customer sets.
- **State-as-of features only.** `order_seq`, cumulative spend, cadence, breadth are legitimate
  state (known at scoring time), not leakage.
- **Current order's own returns/exchanges are excluded** (post-order → leakage). Only *prior*
  orders' cumulative returns/exchanges are kept (`w_prior` frame stops at `1 PRECEDING`). A
  per-order return-propensity score is the planned leakage-safe replacement.
- **Label-side columns** (`next_order_date`, `days_to_next_order`) are dropped before training.

## Features (as-of each order)

Sequence & cadence (`order_seq`, `days_since_first_order`, `days_since_prev_order`,
`avg_gap_so_far`, `gap_vs_cadence`), monetary (cumulative net spend, `cum_avg_order_value`,
`spend_velocity`, this-order spend / items / discount), breadth & channel (distinct departments,
cross-gender, footwear + non-footwear, distinct physical stores, ever-ecomm / ever-retail / both),
prior return & exchange quantities, this-order product context, and entry/static attributes
(entry department, first style, acquisition channel/store, customer state).

## Model & evaluation

**CatBoost** classifier (1000 iters, depth 6, lr 0.05, AUC metric), native categorical handling,
early stopping on a grouped validation set.

- **Per-row (ranking):** AUC / PR-AUC on held-out customers.
- **Per-stage leakage check:** AUC *within* each `order_seq` bucket, to separate how much of the
  headline AUC is `order_seq` sorting stages apart vs. genuine within-stage discrimination.
- **Calibration by `order_seq`:** predicted mean repeat rate vs. actual repeat rate at each order
  number — the projection depends on this tracking.
- **Backtest:** projected future orders vs. actual future orders for 2023–2024 customers.

## From churn score to projected orders (the PLTV bridge)

For a customer's **latest** order, `p_repeat = 1 − churn_prob`. Chain it with the population
survival curve `r_k` (repeat rate at order `k`) into expected additional orders:

```
E[future orders] = p_repeat × (1 + T_{k+1})     where   T_k = r_k × (1 + T_{k+1})
```

`T_k` is the expected additional orders for a typical customer at order `k`, built by walking
forward and capped at a horizon `H` so very loyal customers can't project a near-infinite total.
See `Expected Orders Survival Calculation.html` for the full walkthrough. Phase 2 multiplies these
projected orders by the customer's expected AOV to get dollar PLTV.

## Deployment

Score each customer's **latest** order row → `churn_prob` → `p_repeat` → risk tier
(High ≥ 0.7, Medium ≥ 0.5, Low otherwise) → projected future orders. The artifact
(`churn_order_level_artifact.joblib`) carries the model, feature list, categorical vocab, and
metadata needed to score a fresh feature panel.

---

## Confirm before production

- **Column names in `core.orders`:** `net_sales_value`, `gross_sales_amount`, `discounts`,
  `gross_product_quantity`, `exchanged_quantity`, `return_quantity` are flagged `CONFIRM`.
- **`LIMIT 100000`** in the extract is a development guard — remove it for the full population run.
- **Colab dependency:** the export cell uses `google.colab.files.download`. Use the portable
  export in `../Full build PLTV model/` when running outside Colab.
