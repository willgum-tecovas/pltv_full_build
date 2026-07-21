# PLTV & Churn Model

Predicted lifetime value for Tecovas customers, built in two phases that multiply
into one dollar score:

**PLTV = projected future orders × expected order value**

- **Phase 1 — Churn** answers *"how many more orders?"* (a 400-day repeat-purchase
  hazard model)
- **Phase 2 — AOV** answers *"worth how much each?"* (an empirical-Bayes hierarchical
  order-value model)
- **Full build** joins them into a per-customer PLTV score.

## Repository layout

| Folder | What's inside |
|--------|---------------|
| `Churn Model/` | Phase 1. Feature SQL + CatBoost training notebook. Outputs the churn model artifact. See its README. |
| `AOV Model/` | Phase 2. Order-level extract SQL + empirical-Bayes notebook + validation dashboard. Outputs expected AOV per customer. See its README. |
| `Full build PLTV model/` | Combines Phase 1 × Phase 2. SQL scoring extract, build script, artifact-export snippet, `inputs/`, `outputs/`, and `VALIDATION.md`. See its README. |
| `Dashboards/` | Self-contained HTML explainers (data exploration, the survival → projected-orders math). Just open in a browser. |
| `Documents/` | Project write-ups and reference material. |

## How it fits together

- Churn Model ─► churn_order_level_artifact.joblib (output = model artifact)
- AOV Model ─► expected_aov_by_customer.csv (output = expected AOV of customer base)
- Full build PLTV model ─► pltv_scores.csv

## Getting started

1. Start with the folder README for the phase you're working in — each one lists its
   own SQL → notebook → dashboard pipeline and the columns to confirm before production.
2. To produce final PLTV scores, follow `Full build PLTV model/README.md`: re-export the
   churn artifact, generate the AOV output, run the scoring SQL, then run
   `script/build_combined_pltv.py`.
3. Validation for the combined score lives in `Full build PLTV model/VALIDATION.md`.

## Notes

- All SQL runs against `tecovas-prod-edw` in BigQuery; notebooks load their extracts via CSV.
- Some column names are flagged `CONFIRM` in the SQL — verify against the live schema
  before a production run.
- The two phases use slightly different order grains (churn = purchase-day, AOV =
  `order_id`) on purpose; see the Full build README for how they reconcile.
