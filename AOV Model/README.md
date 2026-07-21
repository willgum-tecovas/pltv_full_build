# AOV Model — Expected Average Order Value (Phase 2)

An empirical-Bayes hierarchical (credibility) model that estimates each customer's
**expected average order value (AOV)**. It feeds dollar CLV as `AOV × projected orders`
(projected orders come from the churn / Phase-1 model).

Two ideas do the work: department and exotic leather are priced **at the order level**
(so a customer's boot orders are priced like boots, accessories like accessories), and
each customer's baseline is **shrunk toward a population prior** by how many orders
we've actually seen — thin histories lean on the population, loyal customers on
themselves.

---

## Pipeline (run in this order)

1. **`sql/pltv_order_level_extract.sql`** — run in BigQuery, export the result to CSV.
2. **`notebook/eb_aov_order_level.ipynb`** — point `DRIVE_CSV_URL` at the exported CSV;
   run top to bottom. It fits the model and writes one expected AOV per customer.
3. **`sql/pltv_holdout_validation_extract.sql`** + notebook **Section 8** — temporal
   holdout: fit on pre-cutoff orders, score against realized post-cutoff spend.
4. **`dashboard/AOV Model Validation & Defensibility.html`** — the shareable
   explainer + validation write-up (open in any browser).

---

## Files

| Path | What it is |
|------|------------|
| `sql/pltv_order_level_extract.sql` | **Production extract.** One row per `(customer, order_department, order_is_exotic)` cell, at the **order (`order_id`) grain**. Emits log-scale sufficient statistics for the fit. |
| `sql/pltv_holdout_validation_extract.sql` | **Holdout extract.** Same grain; fits on orders before a cutoff and carries each customer's post-cutoff actuals for out-of-sample scoring. |
| `notebook/eb_aov_order_level.ipynb` | **The model.** Loads the extract (via Google Drive), fits per-cell department/exotic premiums, within-cell σ, credibility shrinkage, mix-shrinkage, and expected AOV. Sections: 1 premiums · 1b exotic diagnostic · 2 σ + baseline · 3 prior/τ/K · 4 expected AOV · 5 checks · 6 worked example · 7 vs. actual · 8 holdout validation · 9–11 distributions & spend. |
| `dashboard/AOV Model Validation & Defensibility.html` | **Self-contained dashboard.** Why this approach, the value ladder, variance decomposition, the math, an interactive AOV playground, and holdout results. No dependencies — just open it. |
| `working/build_notebook.py` | Source that **generates** the notebook. Edit here and re-run to rebuild the `.ipynb` (don't hand-edit the notebook). |
| `working/make_synth.py`, `working/make_holdout.py` | Synthetic-data generators used **only** to test the pipeline offline — not part of the production run. |

---

## Current version / design decisions (as of 2026-07-20)

- **Grain: order (`order_id`)** — changed from purchase-day so AOV and Phase-1 order
  counts multiply consistently. Same-day back-to-back orders stay separate.
- **Department label:** primary department by **line count**, revenue breaks ties
  (kept count-first to avoid value-circularity in the label).
- **Exotic:** `order_is_exotic` = any exotic line in the order; premium is estimated
  **per cell (saturated)**, so it varies by department (≈2× accessories, 1.8× men's
  boots, 1.35× women's boots) rather than one shared multiplier.
- **Baseline prior:** **featureless** — shrinks toward the population mean `μ`, grand-mean
  centered so `γ` reads as a premium/discount vs the average order. Two candidate baseline
  features were tested and dropped as inert: `buys_exotic` (~$1) and first-order department
  (~0.5% of baseline variance). Department and exotic are order-level premiums only.
- **Predictor:** `mean` (Duan smearing retransform); **mix-shrinkage `M = 5`**.
- **Channel:** removed everywhere (not a feature).

---

## Confirm before production

- **Column names in `core.orders`:** `order_id` and `is_exotic` are flagged `CONFIRM`
  in the SQL — verify the real names.
- **`gross_sales_less_discount`:** confirm this is the correct (returns-adjusted?)
  value field. It drives low-value orders and the model-free validation.
- **Notebook `DRIVE_CSV_URL`:** update to your own exported CSV's Drive link.
- **`merchandise_department` values:** confirm the department set matches expectations
  (e.g. whether `(Not Provided)` should be excluded or merged).
