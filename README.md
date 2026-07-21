# Full Build PLTV Model — Combining Phase 1 (Churn) × Phase 2 (AOV)

The final assembly: one **dollar PLTV score per customer**, built by multiplying projected
future orders (churn model) by expected order value (AOV model).

```
PLTV_c  =  E[future orders]_c   ×   E[AOV]_c

E[future orders] = p × (1 + T_{k+1})
    p       = p_repeat on the customer's LATEST order   (Phase-1 churn model)
    k       = the order stage the customer is currently on
    T_{k+1} = expected additional orders that follow order k+1  (population survival tail)

E[AOV]_c   = the customer's expected order value           (Phase-2 empirical-Bayes AOV model)
```

`E[orders] = p(1 + T_{k+1})` is the churn model's own next-order probability chained onto the
population forward tail `T_k = r_k(1 + T_{k+1})` — the recursion documented in
`../Dashboards/Expected Orders Survival Calculation.html`.

---

## Do this first — re-export the churn artifact

The build script needs the churn model to be **self-contained** — it must carry the population
survival curve `r_k` and forward tail `T_k` so PLTV can be computed without re-deriving them.
The notebook's current export cell (§9) does **not** store these and depends on
`google.colab.files`.

**Replace the churn notebook's "## 9. Export model artifact" cell (cell 26) with
`notebook_snippets/export_churn_artifact.py`, then re-run it.** It produces
`churn_order_level_artifact.joblib` (v0.2) with the projection payload baked in and no Colab
dependency. Run it after the projection/calibration cells so `test_scored` and `probs` exist.

---

## Pipeline (run in this order)

1. **Re-export the churn artifact** — swap in `notebook_snippets/export_churn_artifact.py` and
   run it → `churn_order_level_artifact.joblib`.
2. **AOV per-customer output** — run the AOV model (`../AOV Model/`) top to bottom; export its
   §4 expected-AOV table to `expected_aov_by_customer.csv`
   (columns: `shopify_customer_id`, `expected_aov`).
3. **Scoring extract** — run `sql/pltv_combined_scoring_extract.sql` in BigQuery, export to
   `pltv_combined_scoring_extract.csv` (one row per customer = their latest order's features).
4. **Combine** — run the build script:
   ```bash
   python script/build_combined_pltv.py \
       --artifact    churn_order_level_artifact.joblib \
       --scoring-csv pltv_combined_scoring_extract.csv \
       --aov-csv     expected_aov_by_customer.csv \
       --fallback-aov 310.23 \
       --out         pltv_scores.csv
   ```
5. **Validate** — follow `VALIDATION.md`.

---

## Files

| Path | What it is |
|------|------------|
| `sql/pltv_combined_scoring_extract.sql` | **Scoring extract.** Same feature logic as the churn training extract, but with the maturity cutoff and label columns removed, reduced to each customer's **latest** order (`QUALIFY ROW_NUMBER`). Produces the live feature panel the churn model scores. |
| `notebook_snippets/export_churn_artifact.py` | **Drop-in replacement** for the churn notebook's export cell. Bakes `r_k` / `T_k` / risk tiers into the artifact and removes the Colab dependency. |
| `script/build_combined_pltv.py` | **The combiner.** Loads the artifact, scores latest orders → `p_repeat`, projects `E[orders] = p(1+T_{k+1})`, joins Phase-2 `E[AOV]`, writes `pltv_scores.csv`. |
| `VALIDATION.md` | The validation process for the combined score. |

---

## Output columns (`pltv_scores.csv`)

`shopify_customer_id`, `order_seq` (current stage k), `churn_prob`, `p_repeat`, `risk_tier`
(High ≥ 0.7 / Medium ≥ 0.5 / Low), `expected_future_orders`, `expected_aov`,
`remaining_pltv` (= future orders × AOV, the forward-looking $), `realized_value` (net spend to
date), `total_pltv` (realized + remaining).

**`remaining_pltv` is the decision variable** — it's the go-forward value the business can still
influence. `total_pltv` is for ranking / segmentation.

---

## Design notes & assumptions

- **Grain mismatch is intentional.** Phase 1 is purchase-day; Phase 2 (AOV) is `order_id`. The
  AOV README documents the move to `order_id` so counts and AOV multiply consistently. `E[orders]`
  from Phase 1 counts purchase-days; if a materially different number of same-day back-to-back
  orders exists, reconcile the two grains before trusting absolute dollar levels (see VALIDATION).
- **Independence assumption.** `PLTV = E[orders] × E[AOV]` treats order count and AOV as
  independent. If high-frequency customers systematically spend more (or less) per order, the
  product is biased; the validation step tests this with a `E[orders] × AOV` interaction check.
- **AOV is already retransformed.** The AOV model returns `E[AOV]` on the dollar scale (Duan
  smearing), so no further exp/bias correction here.
- **Fallback AOV.** New customers with no per-customer AOV estimate fall back to the population
  AOV (`--fallback-aov`); the placeholder `310.23` matches the churn notebook.

## Confirm before production

- **`expected_aov` column name** in the AOV export (flagged CONFIRM in the script).
- **`cum_net_spend`** present in the scoring extract (used for `realized_value` / `total_pltv`).
- **Artifact version** is `churn_order_level_v0.2` (has the projection payload). v0.1 will raise.
- **`LIMIT`** removed from the scoring extract for a full-population run.
