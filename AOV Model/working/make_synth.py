import numpy as np, pandas as pd
rng = np.random.default_rng(7)

depts = ["Boots", "Apparel", "Accessories", "Kids"]
# true department log-effects (Boots reference-ish) and exotic premium
gamma_dept = {"Boots":0.0, "Apparel":-0.5, "Accessories":-0.9, "Kids":-0.7}
delta_true = 0.35
sigma_true = 0.42
tau_true   = 0.30
N = 4000

rows = []
for cid in range(1, N+1):
    buys_exotic_type = rng.random() < 0.18
    # customer baseline theta_c; exotic buyers a bit higher
    theta = np.log(120) + rng.normal(0, tau_true) + (0.15 if buys_exotic_type else 0.0)
    # number of orders: skewed toward 1-3
    n = 1 + rng.poisson(1.4)
    for _ in range(n):
        d = rng.choice(depts, p=[0.5,0.28,0.15,0.07])
        e = 1 if (buys_exotic_type and rng.random() < 0.5) else 0
        logv = theta + gamma_dept[d] + delta_true*e + rng.normal(0, sigma_true)
        rows.append((cid, d, e, np.exp(logv)))
o = pd.DataFrame(rows, columns=["cid","dept","e","value"])
o = o[o.value>0]

# customer-level buys_exotic = ever bought exotic
be = o.groupby("cid")["e"].max().rename("buys_exotic")
tot = o.groupby("cid").size().rename("cust_total_orders")

g = o.groupby(["cid","dept","e"])
cell = g["value"].agg(
    n_orders="size",
    sum_value="sum").reset_index()
cell["sum_log_value"]    = g["value"].apply(lambda s: np.log(s).sum()).values
cell["sum_log_value_sq"] = g["value"].apply(lambda s: (np.log(s)**2).sum()).values
cell = cell.merge(be, on="cid").merge(tot, on="cid")
cell = cell.rename(columns={"cid":"shopify_customer_id","dept":"order_department","e":"order_is_exotic"})
cell["order_is_exotic"] = cell["order_is_exotic"].map({0:"False",1:"True"})
cell["buys_exotic"]     = cell["buys_exotic"].map({0:"False",1:"True"})
cell["first_order_date"] = "2023-06-01"
cell = cell[["shopify_customer_id","order_department","order_is_exotic","n_orders",
             "sum_log_value","sum_log_value_sq","sum_value","cust_total_orders",
             "buys_exotic","first_order_date"]]
cell.to_csv("pltv_order_level_extract.csv", index=False)
print("cells:", len(cell), "customers:", cell.shopify_customer_id.nunique())
print(cell.head())
