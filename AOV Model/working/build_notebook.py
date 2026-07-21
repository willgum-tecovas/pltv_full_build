import json, uuid

cells = []

def md(src):
    cells.append({"cell_type": "markdown",
                  "metadata": {},
                  "id": uuid.uuid4().hex[:8],
                  "source": src})

def code(src):
    cells.append({"cell_type": "code",
                  "metadata": {},
                  "id": uuid.uuid4().hex[:8],
                  "execution_count": None,
                  "outputs": [],
                  "source": src})

# ---------------------------------------------------------------- title
md(r"""# Phase 2 — Order-level EB Hierarchical AOV (with mix-shrinkage)
### Department & exotic enter at the ORDER level, so `sigma` stays clean, `K` is small, and one blended AOV comes out per customer

Reads the `pltv_order_level_extract.sql` output — one row per **(customer, order_department, order_is_exotic)** cell with log-scale sufficient stats. Department and exotic are modeled as order-level effects, so a customer's boot orders are priced like boots and their accessory orders like accessories; the leftover `sigma_resid` is the true within-category noise (~0.41), which lets a customer with only a few orders carry real weight. The per-cell pieces collapse into a single expected AOV that multiplies Phase-1 orders.""")

# ---------------------------------------------------------------- model & formulas
md(r"""## Model & formulas

**Order value** (order $i$ of customer $c$, in department $d$, exotic flag $e$):

$$\log(\text{value}) = \theta_c + \gamma_{d} + \delta\,e + \varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma_{\text{resid}}^2)$$

$$\theta_c \sim \mathcal N(\mu,\ \tau^2)\quad(\text{no customer features — the baseline shrinks to the population mean})$$

- $\gamma_d,\ \delta$ — **order-level** department / exotic effects (population/marginal).
- $\sigma_{\text{resid}}$ — within-(customer, department, exotic) noise → the clean ~0.41.
- $\theta_c$ — customer's **department/exotic-adjusted** baseline.

**Credibility** for the baseline, with $K=\sigma_{\text{resid}}^2/\tau^2$ (now small):

$$\hat\theta_c = w_c\,\theta_c^{\text{raw}} + (1-w_c)\,(\mu+X_c\beta),\qquad w_c=\frac{n_c}{n_c+K}$$

**Mix-shrinkage** (Dirichlet–multinomial). A customer's department×exotic mix is shrunk toward the population mix with $M$ pseudo-orders, so a one-order customer isn't assumed to buy their one cell forever:

$$\hat p_c(\text{cell}) = \frac{n_c(\text{cell}) + M\,p_{\text{pop}}(\text{cell})}{n_c + M}$$

**Expected AOV** — value-per-order in each cell, weighted by the shrunk mix. It simplifies so we never expand to all cells:

$$\mathbb E[\text{AOV}_c]=\sum_{\text{cell}}\hat p_c(\text{cell})\,e^{\hat\theta_c+\gamma_{\text{cell}}+\sigma^2/2}
=\frac{e^{\hat\theta_c}}{n_c+M}\Big(\underbrace{\textstyle\sum_{\text{obs cells}} n_c(\text{cell})\,A_{\text{cell}}}_{\text{their orders}} + M\,\underbrace{\textstyle\sum_{\text{cell}} p_{\text{pop}}\,A_{\text{cell}}}_{\bar A_{\text{pop}}\ (\text{global})}\Big),\quad A_{\text{cell}}=e^{\gamma_{\text{cell}}+\sigma^2/2}$$

That single number multiplies Phase-1's projected orders for dollar CLV.""")

# ---------------------------------------------------------------- imports
code(r'''import numpy as np, pandas as pd, patsy, json
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.formula.api as smf
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
TCV = ["#3A4B38", "#A94619", "#69757C", "#E9A877", "#B6B28B", "#F6957D"]

MIX_SHRINK_M = 2.0     # pseudo-orders for department/exotic-mix shrinkage (small => trust the observed mix)
PREDICTOR    = "mean"  # "mean"   = expected value (Duan smearing retransform); use when AOV is summed into $ CLV
                       # "median" = typical order (no variance lift); lower, minimises median % error''')

# ---------------------------------------------------------------- load & prep
code(r'''import re, gdown

DRIVE_CSV_URL = "https://drive.google.com/file/d/13TsAhP8_ymOUDavBFMFir0aWpFnHDXmx/view"   # <-- point at your linked/exported extract

def load_drive_csv(url, out="orders_panel.csv"):
    m = re.search(r"/d/([A-Za-z0-9_-]+)", url) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if not m: raise ValueError("Could not find a file id in DRIVE_CSV_URL.")
    gdown.download(id=m.group(1), output=out, quiet=False)
    return out

df = pd.read_csv(load_drive_csv(DRIVE_CSV_URL)).dropna(how="all").dropna(axis=1, how="all")

# --- normalise types -------------------------------------------------------
# booleans may arrive as "True"/"true"/1/"t"; coerce to plain 0/1 integers
for b in ["order_is_exotic", "buys_exotic"]:
    df[b] = df[b].astype(str).str.lower().isin(["true", "1", "t", "yes"]).astype(int)
df["order_department"] = df["order_department"].fillna("Unassigned")

# --- per-cell log-scale summaries ------------------------------------------
# A "cell" = one (customer, order_department, order_is_exotic) row.
# cell_mean_log = mean of log(order_value) over the cell's orders = (sum log x) / n
df["cell_mean_log"] = df["sum_log_value"] / df["n_orders"]

# cell_ss = within-cell sum of squared deviations of the log values:
#           sum (log x - mean)^2  =  sum (log x)^2  -  n * mean^2
#           (this is the numerator of a variance; pooled into sigma_resid below)
df["cell_ss"] = df["sum_log_value_sq"] - df["n_orders"] * df["cell_mean_log"] ** 2

n_cust = df["shopify_customer_id"].nunique()
print(f"{len(df):,} cells across {n_cust:,} customers")
df.head()''')

# ---------------------------------------------------------------- sec 1 md
md(r"""## 1 · Department & exotic premiums

`gamma_cell` is the **population value of an order** in each (department × exotic) cell, relative to the **average order** — estimated **saturated**, i.e. each observed cell gets its own order-weighted mean, anchored at the order-weighted grand mean so `gamma = 0` is a typical order (negative below, positive above). This lets the exotic premium **differ by department**, which it does in the data (~2.0× accessories, ~1.8× men's boots, only ~1.35× women's boots). An additive single-`delta` model (`dept + exotic`) would force one identical exotic premium everywhere and over-price women's-footwear-exotic. Only cells that actually occur are estimated, so impossible cells (exotic apparel, exotic kids) never appear. The anchor is arbitrary for prediction (the model is reference-invariant — `theta_c` absorbs any shift); the grand-mean anchor just makes both the baseline and `gamma` readable. `exp(gamma)` = multiplier vs the average order.""")

# ---------------------------------------------------------------- sec 1 code
code(r'''# ---------------------------------------------------------------------------
# gamma_cell = the value of an order in each OBSERVED (department, exotic) cell,
# relative to a reference cell. Estimated SATURATED -- each cell gets its own
# order-weighted mean log value, so the exotic premium is FREE TO DIFFER BY
# DEPARTMENT. It does: ~2.0x on accessories, ~1.8x on men's boots, only ~1.35x
# on women's boots. A single additive delta (cell_mean_log ~ dept + exotic) would
# force one identical exotic premium on every department, which the data rejects
# (it over-prices women's-footwear-exotic). Only cells that OCCUR are estimated,
# so impossible cells (exotic apparel, exotic kids) simply never appear.
# ---------------------------------------------------------------------------
cellagg = (df.groupby(["order_department", "order_is_exotic"])
             .agg(sum_log=("sum_log_value", "sum"), n=("n_orders", "sum")).reset_index())
cellagg["cell_logmean"] = cellagg["sum_log"] / cellagg["n"]        # order-weighted mean log value
cellagg = cellagg.sort_values(["order_department", "order_is_exotic"]).reset_index(drop=True)
# Anchor at the ORDER-WEIGHTED GRAND MEAN, not a cell. Then gamma = 0 is the
# "average order," negative below it, positive above -- so the baseline exp(theta)
# reads as a typical order (not the cheapest cell) and gamma is an interpretable
# premium/discount. The choice of anchor is arbitrary for prediction (the model
# is reference-invariant: theta absorbs any shift); this one just reads better.
ref_val = cellagg["sum_log"].sum() / cellagg["n"].sum()            # grand mean log order value
cellagg["gamma_cell"] = cellagg["cell_logmean"] - ref_val

combos = cellagg[["order_department", "order_is_exotic", "gamma_cell"]].copy()
df = df.merge(combos, on=["order_department", "order_is_exotic"], how="left")

# exotic premium is now PER DEPARTMENT (not one delta): exotic cell / normal cell
wide = (combos.assign(mult=np.exp(combos["gamma_cell"]))
              .pivot_table(index="order_department", columns="order_is_exotic", values="mult"))
print("per-cell multiplier vs the AVERAGE order (exp gamma; 1.00 = average):")
print(combos.assign(x_premium=np.exp(combos.gamma_cell)).sort_values("x_premium").to_string(index=False))
if 0 in wide.columns and 1 in wide.columns:
    wide["exotic_premium"] = wide[1] / wide[0]
    print("\nexotic premium BY DEPARTMENT (exotic $ / normal $) -- varies, so a single delta is wrong:")
    print(wide["exotic_premium"].dropna().round(3).to_string())''')

# ---------------------------------------------------------------- sec 1b md
md(r"""### 1b · Diagnostic — is the order-level exotic premium real, and is it estimated cleanly?

Three questions decide whether to keep `delta` at the order level, fix how it's estimated, or drop it in favor of `buys_exotic` alone. This cell only *measures* — it does not change the model.

1. **Mix rate.** Do customers actually mix exotic and non-exotic orders? If almost none do, exotic is really a customer *type* and the order-level term is redundant.
2. **Marginal vs within-customer `delta`.** The Section-1 `delta` is a *pooled* estimate: it compares exotic vs non-exotic orders across different people, so it absorbs the fact that exotic buyers are higher spenders (selection). The **within-customer** estimate demeans each order by its own customer first — removing `theta_c` — so `delta` is identified only from customers who mix, i.e. the pure per-order premium. If within ≪ marginal, selection was inflating it.
3. **`sigma_resid` with vs without the term.** Collapsing exotic into the residual is only safe if it doesn't re-inflate `sigma` (and therefore `K`).""")

# ---------------------------------------------------------------- sec 1b code
code(r'''# ===========================================================================
# DIAGNOSTIC ONLY — does not modify the model. Decides how to treat exotic.
# ===========================================================================

# --- (1) exotic mix rate ---------------------------------------------------
# per customer: exotic vs non-exotic order counts
ex = (df.assign(ex_n = df.n_orders * df.order_is_exotic,
                nex_n = df.n_orders * (1 - df.order_is_exotic))
        .groupby("shopify_customer_id")
        .agg(n_c=("n_orders", "sum"), ex=("ex_n", "sum"), nex=("nex_n", "sum")))
share_exotic = ex.ex.sum() / ex.n_c.sum()
mc = ex[ex.n_c >= 2]                                   # need >=2 orders to be able to mix
mix_rate = ((mc.ex > 0) & (mc.nex > 0)).mean()
print("(1) EXOTIC MIX")
print(f"    share of all orders that are exotic ............. {share_exotic*100:5.1f}%")
print(f"    of customers with >=2 orders, share who have BOTH")
print(f"    an exotic AND a non-exotic order (true mixers) .. {mix_rate*100:5.1f}%")

# --- (2) exotic premium: PER DEPARTMENT, plus marginal vs within averages --
# per-department exotic premium from Section 1's saturated cells (log gap)
w_ = combos.pivot_table(index="order_department", columns="order_is_exotic", values="gamma_cell")
dept_logprem = ((w_[1] - w_[0]).dropna() if (0 in w_.columns and 1 in w_.columns)
                else pd.Series(dtype=float))
print("\n(2) EXOTIC PREMIUM BY DEPARTMENT (exotic $ / normal $) -- not one constant:")
print(np.exp(dept_logprem).round(3).to_string() if len(dept_logprem) else "  (no exotic cells)")

# volume-weighted marginal average (single number, for the selection check below)
exo_vol = df[df.order_is_exotic == 1].groupby("order_department")["n_orders"].sum()
delta_marg = (float(np.average(dept_logprem, weights=exo_vol.reindex(dept_logprem.index).fillna(0)))
              if len(dept_logprem) else 0.0)

# Build the design (exotic + full department dummies, no intercept), then DEMEAN
# every column and the target by each customer's order-weighted mean. Demeaning
# subtracts theta_c, so what's left identifies delta within-customer only.
#
# VECTORISED: the customer-weighted mean of a column v is sum(v*w)/sum(w) per
# customer. groupby(...).transform("sum") does both sums in Cython (fast even
# with hundreds of thousands of customers) -- no per-group Python loop.
w    = df["n_orders"].to_numpy(float)
grp  = df["shopify_customer_id"].to_numpy()
wsum = df.groupby("shopify_customer_id")["n_orders"].transform("sum").to_numpy(float)  # sum of weights per customer

def _demean(v):
    v = np.asarray(v, float)
    vw_sum = pd.Series(v * w).groupby(grp).transform("sum").to_numpy(float)             # sum(v*w) per customer
    return v - vw_sum / wsum                                                            # v - weighted customer mean

Xd = patsy.dmatrix("0 + order_is_exotic + C(order_department)", df, return_type="dataframe")
yq = _demean(df["cell_mean_log"].to_numpy(float))
Xq = np.column_stack([_demean(Xd[c].to_numpy(float)) for c in Xd.columns])
sw = np.sqrt(w)                                        # weighted least squares via sqrt-weights
coef, *_ = np.linalg.lstsq(Xq * sw[:, None], yq * sw, rcond=None)
delta_within = coef[list(Xd.columns).index("order_is_exotic")]

# is there any within-customer exotic variation to identify delta from?
within_var = np.average(_demean(Xd["order_is_exotic"].to_numpy(float)) ** 2, weights=w)

print(f"\n    volume-weighted marginal avg  exp(delta) = {np.exp(delta_marg):.3f}   (premium + selection)")
print(f"    within-customer avg           exp(delta) = {np.exp(delta_within):.3f}   (same-customer; pure premium)")
if within_var < 1e-8:
    print("    NOTE: essentially no within-customer exotic variation -> exotic behaves as a TYPE")

# --- (3) sigma_resid WITH vs WITHOUT the order-level exotic term ------------
# WITH: cells are (customer, dept, exotic) -> exotic variation already removed
m1 = df.n_orders >= 2
sig_with = np.sqrt(df.loc[m1, "cell_ss"].sum() / (df.loc[m1, "n_orders"] - 1).sum())

# WITHOUT: collapse exotic so cells are (customer, dept); re-pool sufficient
# stats  n' = sum n,  S1 = sum(sum_log),  S2 = sum(sum_log_sq),  SS' = S2 - S1^2/n'
coll = (df.groupby(["shopify_customer_id", "order_department"])
          .agg(n=("n_orders", "sum"), S1=("sum_log_value", "sum"), S2=("sum_log_value_sq", "sum"))
          .reset_index())
coll["ss"] = coll.S2 - coll.S1 ** 2 / coll.n
m2 = coll.n >= 2
sig_without = np.sqrt(coll.loc[m2, "ss"].sum() / (coll.loc[m2, "n"] - 1).sum())

print("\n(3) sigma_resid WITH vs WITHOUT the order-level exotic term")
print(f"    with order-level exotic (current) ... {sig_with:.3f}")
print(f"    without (exotic gap in residual) .... {sig_without:.3f}   (+{(sig_without/sig_with-1)*100:.0f}%)")
print(f"    K scales with sigma^2, so dropping the term inflates K by ~{(sig_without**2/sig_with**2-1)*100:.0f}%")

print("\nREAD:")
print("  * within exp(delta) ~ 1 and low mix rate -> exotic is a customer TYPE;")
print("    dropping the order-level term is safe (buys_exotic already carries it).")
print("  * within exp(delta) clearly > 1 but < marginal -> real per-order premium")
print("    that the pooled fit overstated; KEEP the term, use the within estimate.")
print("  * large sigma inflation WITHOUT the term -> you need it to keep K small.")''')

# ---------------------------------------------------------------- sec 2 md
md(r"""## 2 · Clean `sigma_resid` and the department-adjusted baseline `theta_c`

`sigma_resid` is the within-cell spread (same customer, same department, same exotic status) — the residual order-to-order noise after department/exotic are removed. `theta_raw` is each customer's mean log value with their department/exotic mix stripped out (their baseline "level").""")

# ---------------------------------------------------------------- sec 2 code
code(r'''# ---------------------------------------------------------------------------
# sigma_resid = order-to-order noise WITHIN a cell (same customer, department,
# exotic status). Pool the within-cell sums of squares over cells with >=2
# orders (a 1-order cell has no internal spread):
#     sigma^2 = sum(cell_ss) / sum(n_orders - 1)      (pooled, unbiased variance)
# ---------------------------------------------------------------------------
multi  = df["n_orders"] >= 2
sigma2 = df.loc[multi, "cell_ss"].sum() / (df.loc[multi, "n_orders"] - 1).sum()
sigma  = np.sqrt(sigma2)

# ---------------------------------------------------------------------------
# theta_raw = each customer's department/exotic-ADJUSTED baseline (log scale):
# their mean log order value with the cell effects (gamma) subtracted out.
#     theta_raw_c = mean_i [ log(value_i) - gamma_cell(i) ]
#                 = (sum log value) / n_c   -   (sum n_cell * gamma_cell) / n_c
# ---------------------------------------------------------------------------
df["n_gamma"] = df["n_orders"] * df["gamma_cell"]      # per-cell piece of the mix-weighted gamma
adj = df.groupby("shopify_customer_id").agg(
    n_c=("n_orders", "sum"),                # total orders for the customer
    sum_log=("sum_log_value", "sum"),       # sum log(value) over all their orders
    mix_gamma_num=("n_gamma", "sum"),       # sum n_cell * gamma_cell
).reset_index()
adj["cust_mean_log"] = adj["sum_log"] / adj["n_c"]                        # mean log value
adj["theta_raw"]     = adj["cust_mean_log"] - adj["mix_gamma_num"] / adj["n_c"]

# customer-level buys_exotic flag — kept only for the descriptive spend section
# (Section 11); it is NOT a prior feature (the baseline has no features).
feat = df.groupby("shopify_customer_id").agg(
    buys_exotic=("buys_exotic", "max")).reset_index()
cust_df = adj.merge(feat, on="shopify_customer_id")

print(f"sigma_resid (within cell): {sigma:0.3f}   (vs the ~0.81 basket-inflated pooled sigma)")
cust_df[["n_c", "theta_raw", "buys_exotic"]].describe()''')

# ---------------------------------------------------------------- sec 3 md
md(r"""## 3 · Prior, `tau`, and `K`

`theta_raw` is shrunk toward the **population baseline** `mu` — no customer features. Both candidates were tested and proved inert (`buys_exotic` ~$1, first-order department ~0.5% of baseline variance), so department and exotic live only as order-level premiums, not baseline types. `tau` is fit by maximum likelihood (bounded >= 0, singleton-robust); `K = sigma_resid^2 / tau^2`.""")

# ---------------------------------------------------------------- sec 3 code
code(r'''def fit_random_effects(formula, data, sigma2, iters=60):
    """Fit theta_raw_c ~ N(X_c beta, tau^2), where each theta_raw_c is itself
    observed with sampling variance sigma^2 / n_c. So the marginal model is

        theta_raw_c ~ N(X_c beta, tau^2 + sigma^2 / n_c)

    Alternate until tau^2 converges:
      (1) GLS for beta given tau^2  -- weight each customer by 1 / (tau^2 + s2n)
          so precisely estimated (high-n) customers count more.
      (2) 1-D ML for tau^2 given beta -- minimise the Gaussian negative
          log-likelihood 0.5 * sum[ log(tau^2 + s2n) + resid^2 / (tau^2 + s2n) ].
    tau^2 is bounded >= 0 (a variance) and the fit tolerates singleton (n=1)
    customers.
    """
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    di = X.design_info
    y  = y.values.ravel()
    Xv = X.values
    s2n = sigma2 / data["n_c"].values.astype(float)      # sampling variance of each theta_raw

    tau2 = max(y.var() - s2n.mean(), 1e-3)               # moment-based starting value
    for _ in range(iters):
        w    = 1.0 / (tau2 + s2n)                        # GLS weights
        XtW  = Xv.T * w
        beta = np.linalg.solve(XtW @ Xv, XtW @ y)        # weighted normal equations
        r    = y - Xv @ beta                             # residuals
        tau2_new = minimize_scalar(
            lambda t2: 0.5 * np.sum(np.log(t2 + s2n) + r**2 / (t2 + s2n)),
            bounds=(0.0, 5.0), method="bounded").x
        if abs(tau2_new - tau2) < 1e-9:
            tau2 = tau2_new
            break
        tau2 = tau2_new
    return beta, tau2, di

# prior has NO customer features (intercept only). Both candidate baseline features
# were tested and proved inert (~0% of baseline variance): buys_exotic (~$1) and
# first_order_department (~0.5%). Department and exotic act purely as ORDER-level
# premiums, not baseline types, so theta shrinks toward the single population mu.
beta, tau2, di = fit_random_effects("theta_raw ~ 1", cust_df, sigma2)

# K = prior strength in orders: how many real orders it takes to match the prior
K = sigma2 / tau2
print(f"sigma_resid={sigma:0.3f}   tau={np.sqrt(tau2):0.3f}   K={K:0.2f}")
print(f"credibility weight:  w(2)={2/(2+K):.2f}  w(5)={5/(5+K):.2f}  w(10)={10/(10+K):.2f}")

# prior mean per customer:  mu + X_c beta   (design matrix from the fitted spec)
prior_mean = np.asarray(patsy.build_design_matrices([di], cust_df)[0]) @ beta

# credibility shrinkage of the baseline:
#   w_c = n_c / (n_c + K);  theta_hat = w_c * theta_raw + (1 - w_c) * prior_mean
w_c = cust_df["n_c"] / (cust_df["n_c"] + K)
cust_df["prior_mean"] = prior_mean
cust_df["theta_hat"]  = w_c * cust_df["theta_raw"] + (1 - w_c) * prior_mean
cust_df["w"]          = w_c''')

# ---------------------------------------------------------------- sec 4 md
md(r"""## 4 · Mix-shrinkage → single expected AOV per customer

`A_cell = exp(gamma_cell) * RT` is the dollar multiplier for each department×exotic cell. `RT` is the **retransformation factor** set by `PREDICTOR`: `mean` uses Duan's smearing factor `S = mean(exp(residual))` — estimated empirically as arithmetic-mean / geometric-mean over multi-order cells, so it's robust when residuals aren't exactly lognormal (unlike the parametric `exp(sigma^2/2)`) — giving a true expected value for summed dollar CLV; `median` uses `RT = 1` for the typical order (lower, minimises median % error). Expected AOV then uses the closed form from the model section — the customer's own cells plus `M` pseudo-orders at the population-average cell — so it stays one pass over the data.""")

# ---------------------------------------------------------------- sec 4 code
code(r'''# ---------------------------------------------------------------------------
# Retransformation factor RT turns a log-scale level into a dollar level.
#   mean  : RT = Duan's smearing factor S = mean(exp(residual)). Estimated from
#           multi-order cells as (arithmetic mean / geometric mean), because for
#           a cell  arith_mean = exp(cell_mean_log) * mean(exp(e_i)), so
#           mean(exp(e_i)) = (sum_value / n) / exp(cell_mean_log). Nonparametric,
#           so it self-corrects when residuals are not exactly lognormal.
#   median: RT = 1  -> predicts the typical (median) order, no variance lift.
# Per-cell multiplier:  A_cell = exp(gamma_cell) * RT
# ---------------------------------------------------------------------------
c2 = df["n_orders"] >= 2                                       # cells with within-cell residuals
smear_num  = (df.loc[c2, "sum_value"] / np.exp(df.loc[c2, "cell_mean_log"])).sum()   # sum exp(e_i)
S          = smear_num / df.loc[c2, "n_orders"].sum()          # Duan smearing factor
parametric = np.exp(sigma2 / 2)                                # what a perfect lognormal implies
RT = {"mean": S, "median": 1.0}[PREDICTOR]

df["A_cell"] = np.exp(df["gamma_cell"]) * RT
print(f"retransform: PREDICTOR={PREDICTOR!r}  smearing S={S:.3f}  "
      f"parametric exp(sigma^2/2)={parametric:.3f}  ->  RT={RT:.3f}")

# Population-average cell multiplier, each cell weighted by its share of all
# orders:   Abar_pop = sum_cell p_pop(cell) * A_cell = sum(n_cell A_cell) / sum(n_cell)
pop = df.groupby(["order_department", "order_is_exotic"]).agg(
    n=("n_orders", "sum"), A=("A_cell", "first")).reset_index()
Abar_pop = np.average(pop["A"], weights=pop["n"])

# value ladder for the artifact: typical $ per order in each OBSERVED cell,
# with its share of all orders. usd = exp(mean theta_hat) * A_cell.
import json
base_theta = float(np.exp(cust_df["theta_hat"].mean()))
pop_ladder = pop.assign(usd=base_theta * pop["A"], order_share=pop["n"] / pop["n"].sum())
pop_ladder = pop_ladder.sort_values("usd")
print("VIZ_LADDER_JSON=" + json.dumps([
    {"department": r.order_department, "exotic": int(r.order_is_exotic),
     "usd": round(float(r.usd)), "order_share": round(float(r.order_share), 4)}
    for _, r in pop_ladder.iterrows()]))

# Each customer's own-cell sum:  sum_cell n_cell * A_cell
obs = (df.assign(nA=df["n_orders"] * df["A_cell"])
         .groupby("shopify_customer_id")["nA"].sum())
cust_df = cust_df.merge(obs.rename("obs_sum"), on="shopify_customer_id")

# Expected AOV, closed form (mix shrunk toward the population with M pseudo-
# orders, so a thin customer is not assumed to buy only their one cell):
#
#                        sum n_cell A_cell  +  M * Abar_pop
#   E[AOV_c] = exp(theta_hat) * ---------------------------------
#                                          n_c + M
M = MIX_SHRINK_M
cust_df["expected_aov"] = (np.exp(cust_df["theta_hat"]) *
                           (cust_df["obs_sum"] + M * Abar_pop) / (cust_df["n_c"] + M))

print(f"Abar_pop (population avg cell multiplier) = {Abar_pop:0.3f}")
print(f"expected AOV: min={cust_df.expected_aov.min():,.2f}  "
      f"median={cust_df.expected_aov.median():,.2f}  max={cust_df.expected_aov.max():,.2f}")''')

# ---------------------------------------------------------------- sec 5 md
md(r"""## 5 · Checks & tier summary""")

# ---------------------------------------------------------------- sec 5 code
code(r'''# sanity checks on the fit
checks = {}
checks["tau^2 > 0"]            = tau2 > 0
checks["expected AOV all > 0"] = bool((cust_df.expected_aov > 0).all())
checks[f"K small enough (loyalists earn weight): w(5)={5/(5+K):.2f}"] = (5/(5+K)) > 0.4
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)

# how weight and expected AOV move with order count (the shrinkage story)
cust_df["n_tier"] = pd.cut(cust_df.n_c, [0, 1, 4, 10, 10**9], labels=["1", "2-4", "5-10", "11+"])
print("\n" + "-"*60)
print(cust_df.groupby("n_tier", observed=True).agg(
    customers=("shopify_customer_id", "size"),
    pct_of_base=("shopify_customer_id", lambda s: 100*len(s)/len(cust_df)),
    median_w=("w", "median"),
    median_expected_aov=("expected_aov", "median")).round(3))''')

# ---------------------------------------------------------------- sec 6 md
md(r"""## 6 · One customer, fully decomposed

A random 3-order, cross-department customer, showing every element that feeds their expected AOV and how much weight each carries: the baseline (their data vs the feature prior, split by the credibility weight `w`), the department mix (with the pseudo-order shrinkage), and each cell's dollar contribution to the final number. The explicit per-cell sum should reproduce the vectorised closed form exactly — that equality is the core validation of Section 4.""")

# ---------------------------------------------------------------- sec 6 decomposition
code(r'''# Pick a random 3-order, cross-department customer so every moving part shows.
ndept = df.groupby("shopify_customer_id")["order_department"].nunique().rename("n_dept")
cand  = cust_df.merge(ndept, on="shopify_customer_id")
cand  = cand[(cand.n_c == 3) & (cand.n_dept >= 2)]
cid   = cand.sample(1, random_state=3)["shopify_customer_id"].iloc[0]

row    = cust_df[cust_df.shopify_customer_id == cid].iloc[0]
ccells = df[df.shopify_customer_id == cid].copy()
n_c, w, th_raw, th_hat, pm = row.n_c, row.w, row.theta_raw, row.theta_hat, row.prior_mean

# prior mean — just the population intercept (no features)
xrow    = np.asarray(patsy.build_design_matrices([di], cust_df[cust_df.shopify_customer_id == cid])[0])[0]
contrib = pd.Series(xrow * beta, index=di.column_names)

# Rebuild expected AOV the LONG way (explicit per-cell sum) and confirm it
# matches the vectorised closed form stored in cust_df.
Mv = MIX_SHRINK_M
ccells["weight"]       = ccells["n_orders"] / (n_c + Mv)                       # own-order share of the mix
ccells["cell_value"]   = np.exp(th_hat + ccells["gamma_cell"]) * RT            # $ per order in the cell
ccells["contribution"] = ccells["weight"] * ccells["cell_value"]
pop_weight       = Mv / (n_c + Mv)                                             # pseudo-order share of the mix
pop_value        = Abar_pop * np.exp(th_hat)                                   # $ per order at the population cell
pop_contribution = pop_weight * pop_value
expected_aov     = ccells["contribution"].sum() + pop_contribution

print(f"customer {cid}: {int(n_c)} orders across {ccells.order_department.nunique()} departments\n")
print("STEP 1  baseline (log scale)")
print(f"  their adjusted level  theta_raw = {th_raw:.3f}")
print(f"  feature prior mean    mu+Xb     = {pm:.3f}   parts: "
      + ", ".join(f"{k.split('[')[0]}={v:+.3f}" for k, v in contrib.items()))
print(f"  credibility weight    w=n/(n+K) = {w:.2f}  ->  their data {w*100:.0f}% / prior {(1-w)*100:.0f}%  (K={K:.2f})")
print(f"  shrunk  theta_hat = {w:.2f}*{th_raw:.3f} + {1-w:.2f}*{pm:.3f} = {th_hat:.3f}\n")
print(f"STEP 2  cell mix  (M={Mv:g} pseudo-orders)")
for _, r in ccells.iterrows():
    print(f"  {r.order_department:18s} exotic={int(r.order_is_exotic)}  orders={int(r.n_orders)}  "
          f"weight={r.weight:.2f}  cell_value=${r.cell_value:,.0f}  ->  ${r.contribution:,.0f}")
print(f"  {'population (unbought)':26s} weight={pop_weight:.2f}  cell_value=${pop_value:,.0f}  ->  ${pop_contribution:,.0f}")
print(f"\nEXPECTED AOV (explicit) = ${expected_aov:,.2f}   (closed form: ${row.expected_aov:,.2f})")''')

# ---------------------------------------------------------------- sec 6 plot
code(r'''tau = np.sqrt(tau2)
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (a) baseline on the $ scale: prior distribution + the three points
grid = np.linspace(pm - 3*tau, pm + 3*tau, 200)
ax[0,0].plot(np.exp(grid), stats.norm.pdf(grid, pm, tau), color=TCV[2])
for val, lab, col in [(pm, "prior mean", TCV[2]), (th_raw, "their data", TCV[1]), (th_hat, "shrunk", TCV[0])]:
    ax[0,0].axvline(np.exp(val), color=col, lw=2, label=f"{lab} = ${np.exp(val):,.0f}")
ax[0,0].set_title(f"(a) baseline: prior vs their data (w={w:.2f})")
ax[0,0].set_xlabel("$ baseline order value"); ax[0,0].set_yticks([]); ax[0,0].legend(fontsize=8)

# (b) theta_hat composition (log scale)
ax[0,1].barh(["theta_hat"], [w*th_raw], color=TCV[1], label=f"w*theta_raw ({w*100:.0f}%)")
ax[0,1].barh(["theta_hat"], [(1-w)*pm], left=[w*th_raw], color=TCV[2], label=f"(1-w)*prior ({(1-w)*100:.0f}%)")
ax[0,1].set_title("(b) where theta_hat comes from"); ax[0,1].legend(fontsize=8); ax[0,1].set_xlabel("log $")

labels = [f"{d[:14]}\nexotic={int(e)}" for d, e in zip(ccells.order_department, ccells.order_is_exotic)] + ["population"]
cols   = [TCV[i % len(TCV)] for i in range(len(labels))]
# (c) mix weights
ax[1,0].bar(range(len(labels)), list(ccells["weight"]) + [pop_weight], color=cols)
ax[1,0].set_xticks(range(len(labels))); ax[1,0].set_xticklabels(labels, fontsize=8)
ax[1,0].set_title("(c) mix weight per cell (sums to 1)"); ax[1,0].set_ylabel("weight")
# (d) $ contributions
ax[1,1].bar(range(len(labels)), list(ccells["contribution"]) + [pop_contribution], color=cols)
ax[1,1].set_xticks(range(len(labels))); ax[1,1].set_xticklabels(labels, fontsize=8)
ax[1,1].set_title(f"(d) $ contribution -> expected AOV = ${expected_aov:,.0f}"); ax[1,1].set_ylabel("$")
plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- sec 7 md
md(r"""## 7 · Expected AOV vs actual arithmetic mean (1–5 order customers)

For 20 sampled customers with 1–5 orders, the model's expected AOV against their **actual** arithmetic mean order value so far (model-free, from raw dollars in `sum_value`). They should track — but expected AOV is deliberately **shrunk** toward the department/feature prior, most at `n=1` and least at `n=5`, so low-order customers pull off the `y=x` line toward their group. That's the shrinkage doing its job, not error.""")

# ---------------------------------------------------------------- sec 7 code
code(r'''# Model-free actual mean per customer, straight from raw dollars:
#   actual_mean = sum(order value) / sum(orders)
if "sum_value" not in df.columns:
    print("sum_value not in the extract — re-run the SQL (it emits SUM(order_value)) to enable this check.")
else:
    g_act = df.groupby("shopify_customer_id").agg(sv=("sum_value", "sum"), no=("n_orders", "sum"))
    act = (g_act.sv / g_act.no).rename("actual_mean")     # vectorised customer arithmetic mean
    comp = cust_df.merge(act, on="shopify_customer_id")

    # 20-row spot check: expected vs actual, plus their ratio
    samp = comp[comp.n_c.between(1, 5)].sample(20, random_state=5).copy()
    samp["ratio"] = samp["expected_aov"] / samp["actual_mean"]
    print(samp[["shopify_customer_id", "n_c", "actual_mean", "expected_aov", "ratio"]]
          .round(2).to_string(index=False))

    # full scatter for 1-5 order customers, coloured by order count
    sub = comp[comp.n_c.between(1, 5)]
    plt.figure(figsize=(6.5, 6.5))
    sc = plt.scatter(sub["actual_mean"], sub["expected_aov"], c=sub["n_c"],
                     cmap="viridis", s=10, alpha=0.4)
    lim = [0, np.nanpercentile(sub["actual_mean"], 99)]
    plt.plot(lim, lim, "k--", lw=1, label="y = x")
    plt.colorbar(sc, label="orders (n)"); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("actual arithmetic mean order value ($)")
    plt.ylabel("expected AOV ($)")
    plt.title("Expected AOV vs observed mean (1-5 orders)")
    plt.legend(); plt.tight_layout(); plt.show()
    print(f"\nmedian expected/actual ratio: {samp['ratio'].median():.2f}  "
          f"(low-n rows shrink toward the group; n=5 rows sit near y=x)")''')

# ---------------------------------------------------------------- sec 8 md
md(r"""## 8 · Temporal-holdout validation against REAL actuals

The plots above only compare predictions to the *same* orders the model was fit on, which is circular — a shrunk estimate is *supposed* to disagree with its own noisy in-sample mean, so being off `y=x` proves nothing. This section does the honest test.

**How to run it.** Run `pltv_holdout_validation_extract.sql` in BigQuery and export the result, then set `CSV_PATH` (top of the notebook) to that file and **re-run the whole notebook**. That extract fits the model on each customer's orders *before* a cutoff (leakage-safe: `buys_exotic` and every statistic are pre-cutoff) and carries each customer's *post-cutoff* actuals on the side. This cell scores the model's `expected_aov` — a pure out-of-sample prediction — against the customer's realized mean order value *after* the cutoff.

**What "defensible" means here.** Three things, in order of importance:
1. **Beats the baselines.** For each train-order tier, median error must be lower than (a) just predicting the population mean, and (b) the customer's own raw (unshrunk) train mean. If shrinkage doesn't beat the raw mean out-of-sample, it isn't earning its keep.
2. **Calibrated in the large.** Mean prediction ≈ mean realized actual over the scorable set (ratio ≈ 1.00), i.e. no systematic over/under-prediction.
3. **Calibrated across the range.** Binning customers by prediction, each bin's mean prediction ≈ its mean realized actual (monotonic, on the diagonal).

**Honest limitation.** Only customers with ≥1 post-cutoff order are scorable, so this validates AOV for *returning* customers; one-and-done customers can't be scored this way.""")

# ---------------------------------------------------------------- sec 8 code
code(r'''if "test_n_orders" not in df.columns:
    print("This looks like the PRODUCTION extract (no test_* columns).\n"
          "To validate: run pltv_holdout_validation_extract.sql, set CSV_PATH to\n"
          "its export, re-run the whole notebook, then run this cell.")
else:
    # --- out-of-sample actuals (carried on every train-cell row; dedup) -----
    test = (df.groupby("shopify_customer_id")
              .agg(test_n=("test_n_orders", "first"),
                   test_sum_value=("test_sum_value", "first")).reset_index())
    test = test[test.test_n > 0].copy()                        # only returning customers are scorable
    test["actual_aov"] = test.test_sum_value / test.test_n     # realized post-cutoff mean order value

    # --- two baselines to beat, both known at the cutoff --------------------
    # (a) population mean order value over the TRAIN window (one number for everyone)
    pop_mean = df.sum_value.sum() / df.n_orders.sum()
    # (b) each customer's own RAW (unshrunk) train arithmetic mean
    gsum = df.groupby("shopify_customer_id").agg(sv=("sum_value", "sum"), no=("n_orders", "sum"))
    raw_train_mean = (gsum.sv / gsum.no).rename("raw_train_mean")

    # predictions come from cust_df.expected_aov, fit on TRAIN only
    val = (cust_df.merge(test[["shopify_customer_id", "actual_aov", "test_n"]], on="shopify_customer_id")
                  .merge(raw_train_mean, on="shopify_customer_id"))
    val["pop_mean"] = pop_mean
    print(f"scorable (>=1 post-cutoff order): {len(val):,} of {len(cust_df):,} fit customers\n")

    # median absolute % error vs the OUT-OF-SAMPLE actual, |pred/actual - 1|
    def mape(pred, actual):
        return (np.abs(pred / actual - 1)).median() * 100

    # --- 1) error by train-order tier: model vs the two baselines -----------
    val["n_tier"] = pd.cut(val.n_c, [0, 1, 2, 3, 5, 10, 10**9],
                           labels=["1", "2", "3", "4-5", "6-10", "11+"])
    print("Median abs % error vs out-of-sample actual mean order value:")
    print(f"{'train n':>8}{'cust':>7}{'pop-mean':>10}{'raw-mean':>10}{'MODEL':>9}   winner")
    for t, g in val.groupby("n_tier", observed=True):
        e_pop = mape(g.pop_mean, g.actual_aov)
        e_raw = mape(g.raw_train_mean, g.actual_aov)
        e_mod = mape(g.expected_aov, g.actual_aov)
        winner = min([("pop-mean", e_pop), ("raw-mean", e_raw), ("MODEL", e_mod)], key=lambda x: x[1])[0]
        print(f"{t:>8}{len(g):>7}{e_pop:9.1f}%{e_raw:9.1f}%{e_mod:8.1f}%   {winner}")

    # --- 2) calibration in the large (no systematic bias) -------------------
    ratio = val.expected_aov.mean() / val.actual_aov.mean()
    print(f"\nCalibration-in-the-large (scorable set):")
    print(f"  mean actual (out-of-sample) = ${val.actual_aov.mean():,.2f}")
    print(f"  mean model prediction       = ${val.expected_aov.mean():,.2f}")
    print(f"  ratio model/actual          = {ratio:.3f}   (1.00 = unbiased; >1 = over-predicting)")

    # both summary error metrics — they pull in DIFFERENT directions on skewed
    # spend, so watch the one that matches your use (summed $ -> mean/ratio;
    # per-customer accuracy -> median APE).
    med_ape  = (np.abs(val.expected_aov / val.actual_aov - 1)).median() * 100
    mean_ape = (np.abs(val.expected_aov / val.actual_aov - 1)).mean() * 100
    print(f"  median abs % error          = {med_ape:.1f}%")
    print(f"  mean   abs % error          = {mean_ape:.1f}%   (PREDICTOR={PREDICTOR!r})")

    # is any residual level bias a retransform issue or a real train->test trend?
    scor = set(val.shopify_customer_id)
    dtr = df[df.shopify_customer_id.isin(scor)]
    train_level = dtr.sum_value.sum() / dtr.n_orders.sum()
    test_level  = test.test_sum_value.sum() / test.test_n.sum()
    print(f"\nTrain vs test raw mean order value (model-free level check, scorable customers):")
    print(f"  train-window mean order value = ${train_level:,.2f}")
    print(f"  test-window  mean order value = ${test_level:,.2f}")
    print(f"  ratio test/train              = {test_level/train_level:.3f}   "
          f"(<1 => AOV fell after the cutoff: a real level shift the model can't see, not a fit error)")

    # --- 3) binned calibration: bin by prediction, compare to realized ------
    val["bin"] = pd.qcut(val.expected_aov, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    cal = val.groupby("bin", observed=True).agg(
        predicted=("expected_aov", "mean"),
        realized=("actual_aov", "mean"),
        customers=("shopify_customer_id", "size")).round(1)
    print("\nBinned calibration (want predicted ~ realized, monotonic):")
    print(cal)

    # --- how much is even predictable? out-of-sample noise vs signal --------
    # The realized mean order value carries per-order lognormal noise of about
    # sigma^2 / m (m = number of test orders). On the log scale the PREDICTABLE
    # fraction of variance is tau^2 / (tau^2 + sigma^2 / m). Average it over the
    # scorable set to get the ceiling ANY model could hit here:
    m = val.test_n.values
    predictable = (tau2 / (tau2 + sigma2 / m)).mean()
    print(f"\nPredictable ceiling: even a PERFECT model explains only "
          f"~{predictable*100:.0f}% of the out-of-sample log-variance")
    print(f"(median test orders per scorable customer = {np.median(m):.0f}; single future "
          f"orders are mostly idiosyncratic noise, so the raw scatter MUST smear)")

    # --- the decisive check: fit sharpens as we require more test orders ----
    # If the scatter is noise (not misfit), error falls and correlation rises
    # as m grows (per-customer realization noise shrinks ~ 1/sqrt(m)).
    print("\nRequiring more OUT-OF-SAMPLE orders (noise falls, signal emerges):")
    for k in [1, 2, 3, 5]:
        s = val[val.test_n >= k]
        if len(s) < 30:
            continue
        r = np.corrcoef(s.expected_aov, s.actual_aov)[0, 1]
        print(f"  test_n >= {k}: cust={len(s):>5}  median abs% err={mape(s.expected_aov, s.actual_aov):5.1f}%  "
              f"corr(pred, actual)={r:.2f}")

    # --- plots: out-of-sample scatter + binned calibration ------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].scatter(val.actual_aov, val.expected_aov, c=val.n_c.clip(upper=10),
                  cmap="viridis", s=8, alpha=0.3)
    lim = [0, np.nanpercentile(val.actual_aov, 99)]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("realized out-of-sample mean order value ($)")
    ax[0].set_ylabel("model expected AOV ($)")
    ax[0].set_title("Prediction vs out-of-sample actual"); ax[0].legend()

    ax[1].plot(cal.realized, cal.predicted, "o-", color=TCV[1], label="bins")
    lim2 = [min(cal.realized.min(), cal.predicted.min()), max(cal.realized.max(), cal.predicted.max())]
    ax[1].plot(lim2, lim2, "k--", lw=1, label="perfect calibration")
    ax[1].set_xlabel("realized (bin mean, $)"); ax[1].set_ylabel("predicted (bin mean, $)")
    ax[1].set_title("Binned calibration"); ax[1].legend()
    plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- sec 8b md
md(r"""### 8b · Does a smaller `M` help the thin (n=1, 2) tiers?

The model ties the population mean at 1&ndash;2 orders because mix-shrinkage leans those predictions almost entirely on the population. This sweep re-scores the holdout for several `M` values &mdash; only the mix term changes, so every `M` reuses the already-fit `theta_hat`, `obs_sum`, and `Abar_pop`, making it cheap. Look for an `M` that beats the `pop` row at n=1&ndash;2 **without** hurting the higher tiers or pushing the calibration ratio off 1.00. If one exists, set `MIX_SHRINK_M` at the top and re-run.""")

# ---------------------------------------------------------------- sec 8b code
code(r'''# M sweep -- re-score the holdout at several mix-shrinkage values.
# Expected AOV = exp(theta_hat) * (obs_sum + M*Abar_pop) / (n_c + M); only the M
# term depends on M, so each M reuses the already-fit pieces (no refit needed).
if "val" not in globals() or "actual_aov" not in getattr(val, "columns", []):
    print("Run Section 8 on the HOLDOUT extract first (need `val` with out-of-sample actuals).")
else:
    _mape  = lambda pred, act: float((np.abs(pred / act - 1)).median() * 100)
    tiers  = ["1", "2", "3", "4-5", "6-10", "11+"]
    Ms     = [1, 2, 3, 5, 10]
    _pred  = lambda M: np.exp(val.theta_hat) * (val.obs_sum + M * Abar_pop) / (val.n_c + M)
    def _err(pred, t):                                   # one tier cell, robust to empty tiers
        m = val.n_tier == t
        return f"{'--':>8}" if m.sum() == 0 else f"{_mape(pred[m], val.actual_aov[m]):7.1f}%"

    print("Median abs % error by train-order tier (lower = better):\n")
    print("    M " + "".join(f"{t:>8}" for t in tiers) + f"{'calib':>9}")
    print("  " + "-" * (5 + 8 * len(tiers) + 9))
    print("  pop " + "".join(_err(val.pop_mean, t) for t in tiers))   # reference (same for every M)
    for M in Ms:
        p = _pred(M)
        print(f"{M:>5} " + "".join(_err(p, t) for t in tiers) + f"{p.mean() / val.actual_aov.mean():8.3f}")
    print(f"\ncurrent MIX_SHRINK_M = {MIX_SHRINK_M:g}   |   "
          f"calib = mean predicted / mean actual over the scorable set (~1.00 = unbiased)")''')

# ---------------------------------------------------------------- sec 9 md
md(r"""## 9 · The population baseline `mu` and its spread

The prior is **featureless** — a single population baseline `mu` every customer is shrunk
toward before their own orders speak. This shows that baseline in dollars and the
between-customer spread `tau` around it (the implied distribution of the true baseline
`theta`). We tested two candidate features — `buys_exotic` and first-order department — and
both were inert (~$1 / ~0.5% of baseline variance), so the baseline carries no features:
department and exotic live solely in the order-level premiums.""")

# ---------------------------------------------------------------- sec 9 code
code(r'''import json   # self-contained: safe to run this cell on its own
# The prior is featureless: prior_mean is the SAME for every customer (the
# population baseline mu). So there's one level, not a per-feature distribution.
mu_log   = float(cust_df["prior_mean"].iloc[0])                 # constant across customers
tau      = float(np.sqrt(tau2))
base_usd = float(np.exp(mu_log) * RT)                           # $ baseline at the reference cell

print("9 · POPULATION BASELINE (featureless prior)")
print(f"  mu (log)  = {mu_log:.3f}   ~ ${base_usd:,.0f} baseline")
print(f"  tau       = {tau:.3f}   (all between-customer baseline spread; no feature explains any of it)")
print(f"  Var(prior_mean) across customers = {cust_df['prior_mean'].var(ddof=0):.4g}  (~0: single level)")

# implied population distribution of the true baseline theta ~ N(mu, tau^2)
gl   = np.linspace(mu_log - 3*tau, mu_log + 3*tau, 80)
dens = stats.norm.pdf(gl, mu_log, tau)
viz9 = {"mu_log": round(mu_log, 4), "tau": round(tau, 4), "base_usd": round(base_usd, 2),
        "usd_grid": [round(float(np.exp(x) * RT), 2) for x in gl],
        "density":  [round(float(v), 5) for v in dens]}
print("\nVIZ9_JSON=" + json.dumps(viz9))

plt.figure(figsize=(7, 4))
plt.plot(np.exp(gl) * RT, dens, color=TCV[0])
plt.axvline(base_usd, color=TCV[1], ls="--", lw=1, label=f"population baseline ${base_usd:,.0f}")
plt.title("Population distribution of the baseline (theta) — single featureless prior")
plt.xlabel("$ baseline order value (reference cell)"); plt.yticks([]); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- sec 10 md
md(r"""## 10 · Average within-customer variance (customers with n >= 2)

For each customer with at least two orders, the sample variance of their log order values around their own mean, then averaged across customers. This is the customer-level spread (it still contains their department/exotic mix, so it sits a bit above the pooled within-*cell* `sigma^2`). Also shown: the spread of these variances across the base, and one example customer whose variance is closest to the average, with the model-implied log-normal shape of their orders.""")

# ---------------------------------------------------------------- sec 10 code
code(r'''import json   # self-contained: safe to run this cell on its own
# aggregate cell sufficient stats up to the customer, then within-customer
# sample variance of log order value:  (S2 - S1^2/n) / (n - 1)
cs = (df.groupby("shopify_customer_id")
        .agg(n_c=("n_orders", "sum"), S1=("sum_log_value", "sum"), S2=("sum_log_value_sq", "sum")))
cs = cs[cs.n_c >= 2].copy()
cs["mean_log"]   = cs.S1 / cs.n_c
cs["within_var"] = ((cs.S2 - cs.S1**2 / cs.n_c) / (cs.n_c - 1)).clip(lower=0)
avg_within = cs.within_var.mean()

print("10 · WITHIN-CUSTOMER VARIANCE (n >= 2)")
print(f"customers with n>=2          = {len(cs):,}")
print(f"average within-customer var  = {avg_within:.4f}  (sd {np.sqrt(avg_within):.3f}) [log scale]")
print(f"pooled within-CELL sigma^2   = {sigma2:.4f}  (sigma {sigma:.3f})   "
      f"[customer var is larger: it still holds dept/exotic mix]")
print("percentiles:", {f"p{int(p*100)}": round(float(cs.within_var.quantile(p)), 4)
                        for p in (.1, .25, .5, .75, .9)})

# distribution of within-customer variances (histogram for the artifact)
cap = float(cs.within_var.quantile(.98))
hc, he = np.histogram(cs.within_var.clip(upper=cap), bins=30)

# example customer whose variance is closest to the average (mid-n for clarity)
cand = cs[cs.n_c.between(3, 8)].copy()
cand["d"] = (cand.within_var - avg_within).abs()
ex = cand.sort_values("d").iloc[0]; ex_id = ex.name
ex_sd = float(np.sqrt(ex.within_var))
ex_cells = (df[df.shopify_customer_id == ex_id][["order_department", "order_is_exotic", "n_orders", "sum_value"]]
              .assign(cell_mean_usd=lambda d: (d.sum_value / d.n_orders).round(2)))
print(f"\nexample customer {ex_id}: n={int(ex.n_c)}  within_var={ex.within_var:.4f} (~avg)  "
      f"geo-mean ${np.exp(ex.mean_log):,.0f}")
print(ex_cells.to_string(index=False))

# model-implied within-customer order distribution: lognormal(mean_log, within_var)
gl = np.linspace(ex.mean_log - 3.2*ex_sd, ex.mean_log + 3.2*ex_sd, 80)
dens = stats.norm.pdf(gl, ex.mean_log, ex_sd)

viz10 = {"avg_within_var": round(float(avg_within), 4),
         "avg_within_sd":  round(float(np.sqrt(avg_within)), 4),
         "sigma2_cell": round(float(sigma2), 4), "n_customers": int(len(cs)),
         "var_hist_edges": [round(float(e), 4) for e in he],
         "var_hist_counts": [int(c) for c in hc],
         "example": {"n": int(ex.n_c), "within_var": round(float(ex.within_var), 4),
                     "geomean_usd": round(float(np.exp(ex.mean_log)), 2),
                     "usd_grid": [round(float(np.exp(x)), 2) for x in gl],
                     "density":  [round(float(v), 5) for v in dens],
                     "cell_means_usd": [round(float(v), 2) for v in ex_cells.cell_mean_usd]}}
print("\nVIZ10_JSON=" + json.dumps(viz10))

plt.figure(figsize=(7, 4))
plt.plot(np.exp(gl), dens, color=TCV[0])
for m in ex_cells.cell_mean_usd:
    plt.axvline(m, color=TCV[1], ls="--", lw=1)
plt.title(f"Example within-customer order distribution (var~avg, n={int(ex.n_c)})")
plt.xlabel("$ order value"); plt.yticks([]); plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- sec 11 md
md(r"""## 11 · Average spend: exotic vs non-exotic customers

How much more an exotic buyer (`buys_exotic = 1`) spends per order than a non-exotic buyer, model-free (raw dollars) and via the model's expected AOV. The raw gap is explained by exotic buyers placing more exotic **orders** (each carrying the order-level premium) plus their department mix — there is **no separate baseline "type" effect**: we tested one (`buys_exotic` in the prior) and it moved the baseline by only ~$1, so it was dropped.""")

# ---------------------------------------------------------------- sec 11 code
code(r'''import json   # self-contained: safe to run this cell on its own
# per-customer raw spend + type flag, joined to the model's expected AOV
cust_raw = (df.groupby("shopify_customer_id")
              .agg(sv=("sum_value", "sum"), no=("n_orders", "sum"),
                   buys_exotic=("buys_exotic", "max")))
cust_raw["raw_aov"] = cust_raw.sv / cust_raw.no                    # model-free avg order value
merged = cust_raw.join(cust_df.set_index("shopify_customer_id")[["expected_aov"]])

g = (merged.groupby("buys_exotic")
           .agg(customers=("raw_aov", "size"),
                avg_orders=("no", "mean"),
                raw_avg_order_value=("raw_aov", "mean"),
                model_expected_aov=("expected_aov", "mean"),
                sv=("sv", "sum"), no_sum=("no", "sum")).reset_index())
g["pooled_order_value"] = g.sv / g.no_sum                          # order-weighted (population) AOV
g = g.drop(columns=["sv", "no_sum"])

print("11 · SPEND — exotic vs non-exotic customers")
print(g.round(2).to_string(index=False))
ne = g[g.buys_exotic == 0].iloc[0]; ex = g[g.buys_exotic == 1].iloc[0]
print(f"\nraw avg order value: exotic ${ex.raw_avg_order_value:,.0f} vs non-exotic "
      f"${ne.raw_avg_order_value:,.0f}  (x{ex.raw_avg_order_value/ne.raw_avg_order_value:.2f})")
print("the baseline carries NO exotic type effect (dropped as inert, ~$1); the whole gap is "
      "explained by exotic buyers placing more exotic orders (order-level premium) plus their "
      "department mix")

viz11 = {"groups": [
    {"label": "Non-exotic buyers", "customers": int(ne.customers),
     "avg_orders": round(float(ne.avg_orders), 2),
     "raw_avg_order_value": round(float(ne.raw_avg_order_value), 2),
     "model_expected_aov": round(float(ne.model_expected_aov), 2)},
    {"label": "Exotic buyers", "customers": int(ex.customers),
     "avg_orders": round(float(ex.avg_orders), 2),
     "raw_avg_order_value": round(float(ex.raw_avg_order_value), 2),
     "model_expected_aov": round(float(ex.model_expected_aov), 2)}],
    "ratio_raw_aov": round(float(ex.raw_avg_order_value / ne.raw_avg_order_value), 3)}
print("\nVIZ11_JSON=" + json.dumps(viz11))

plt.figure(figsize=(6, 4))
plt.bar(["Non-exotic", "Exotic"], [ne.raw_avg_order_value, ex.raw_avg_order_value], color=[TCV[2], TCV[1]])
plt.ylabel("avg order value ($)"); plt.title("Average spend: exotic vs non-exotic customers")
plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- notes
md(r"""### Notes
- `theta_c` is shrunk toward the **population baseline** (no customer features — both `buys_exotic` and first-order department were tested and proved inert), and department/exotic enter per order via `gamma_cell`, so a 2-order boot buyer is priced on the boot line while their baseline shrinks to the population — and, because `sigma_resid` is clean, keeps real weight on their own orders.
- `MIX_SHRINK_M` controls how fast a thin customer's cell mix relaxes toward the population mix; raise it to trust the observed mix less, lower it to trust it more (`M=2` here).
- Section 8 is the real validation: point `CSV_PATH` at the `pltv_holdout_validation_extract.sql` export and re-run — it scores out-of-sample against realized actuals and the two baselines. Sections 5–7 are in-sample sanity checks only.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = "/sessions/youthful-serene-mendel/mnt/outputs/eb_aov_order_level.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
