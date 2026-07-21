# Validation Process — Combined PLTV Score

The combined score has three failure surfaces: the **order projection** (Phase 1), the **AOV
estimate** (Phase 2), and the **way they're multiplied** (the combination itself). Each phase is
already validated in its own notebook; this document covers what must be checked *for the product*
`PLTV = E[orders] × E[AOV]`, end to end. Validate each factor, then the product, then the dollars
against reality.

## Principle

Validate on a **temporal holdout**, out of sample, **grouped by customer**. Fit both phases on
orders up to a cutoff date; score each customer's state as of the cutoff; then compare projections
to what those same customers *actually* did in the observed window after it. Never let a customer
appear on both sides of any split — the churn extract already enforces this with a customer-grouped
split; the AOV holdout extract (`../AOV Model/sql/pltv_holdout_validation_extract.sql`) mirrors it.

---

## 1. Component validation (confirm each factor before multiplying)

**1a. Order projection (Phase 1).** Reuse the churn notebook's checks:

- **Calibration by `order_seq`** — predicted mean repeat rate vs. actual at each stage. The
  survival curve `r_k` must track observed rates within a few points (it does: ≤ ~3 pts). This is
  the ingredient the whole projection rests on.
- **Within-stage AUC** — AUC inside each `order_seq` bucket, to confirm real discrimination beyond
  `order_seq` merely sorting stages apart.
- **Projected vs. actual future orders** — the notebook's backtest on 2023–2024 customers:
  `E[orders]` at their 2nd–5th order vs. the orders they actually went on to make. Bias should be
  small and non-systematic across stages.

**1b. AOV estimate (Phase 2).** Reuse the AOV notebook's §8 temporal holdout: fit on pre-cutoff
orders, score `E[AOV]` against realized post-cutoff spend per customer. Check the model-free
arithmetic-mean baseline and calibration of predicted vs. realized AOV across deciles.

Gate: **do not combine until both factors pass their own calibration.** A biased factor makes the
product biased no matter how good the other is.

---

## 2. Combination validation (the multiply is where new error enters)

**2a. Independence / interaction check.** `E[orders] × E[AOV]` assumes order frequency and AOV are
independent. Test it: bucket holdout customers by projected order frequency, and within each bucket
compare predicted AOV to realized AOV. If AOV drifts systematically with frequency (e.g. loyal
customers spend less per order), the product is biased and needs an interaction term or a joint
model. Quantify the bias; decide if it's material.

**2b. Grain reconciliation.** Phase 1 counts **purchase-days**; Phase 2 prices **`order_id`s**.
Count, per holdout customer, orders-per-purchase-day. If it's ≈1 for the vast majority, the product
is safe; if a meaningful tail splits same-day orders, `E[orders]` (purchase-days) and `E[AOV]`
(per order_id) are on different units and absolute dollars will be off. Report the distribution and
a correction factor if needed.

---

## 3. End-to-end dollar validation (the number the business uses)

On the **temporal holdout**, for each customer compute predicted `remaining_pltv` as of the cutoff,
and the **realized** forward value = actual net spend in the observed post-cutoff window
(annualized/normalized to the same horizon). Then:

- **Calibration by decile.** Rank customers by predicted `remaining_pltv`, form deciles, and plot
  predicted vs. realized mean dollars per decile. The line should track the diagonal; a monotone
  predicted→realized relationship is the minimum bar for using the score to rank.
- **Aggregate bias.** Total predicted vs. total realized dollars over the holdout population —
  within a stated tolerance (e.g. ±10%). Flag systematic over/under-projection.
- **Rank quality.** Spearman correlation and a gains/lift chart (does the top predicted decile
  actually capture a disproportionate share of realized dollars?). For most PLTV uses, **ranking
  correctness matters more than absolute-dollar accuracy** — state which the downstream use needs.
- **Segment stability.** Repeat calibration within key cuts (`order_seq` 1 vs. repeat, acquisition
  channel, footwear vs. non-footwear entry, high vs. low AOV state) to confirm the score isn't only
  calibrated on average while badly wrong for an important segment.

---

## 4. Robustness & operational checks

- **Horizon sensitivity.** Re-run with the survival-tail horizon `H` at, say, 10 / 15 / 20 and
  confirm rankings (and total dollars) are stable — the plateau in `T_k` should make them so.
- **Fallback-AOV share.** Report the fraction of customers on the population fallback AOV (new
  customers). If large, that segment's PLTV is essentially a constant × `E[orders]`; note it.
- **Extreme values / sanity.** Cap or inspect `p_repeat` near 1 (projection blows up as `p→1`);
  confirm no negative or absurd PLTV; check the top-100 by `total_pltv` are plausible real customers.
- **Stability over time.** Re-score the same customers a month apart; scores should move smoothly
  with new orders, not lurch — a proxy for production drift.

---

## Pass criteria (suggested)

1. Both components pass their own calibration (§1).
2. No material frequency×AOV interaction bias, or it's corrected (§2a).
3. Grain reconciled or correction applied (§2b).
4. Decile calibration monotone and near-diagonal; aggregate dollar bias within tolerance;
   top-decile lift materially > 1 (§3).
5. Calibration holds within key segments (§3) and is stable to horizon and over time (§4).

Document each result alongside the AOV dashboard style
(`../AOV Model/dashboard/AOV Model Validation & Defensibility.html`) so the combined score has a
matching, shareable defensibility write-up.
