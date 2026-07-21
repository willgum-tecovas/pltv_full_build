import numpy as np, pandas as pd
rng=np.random.default_rng(21)
depts=["Boots","Apparel","Accessories","Kids"]; gamma={"Boots":0,"Apparel":-.5,"Accessories":-.9,"Kids":-.7}
pd_=[0.5,0.28,0.15,0.07]; delta=.35; sigma=.42; tau=.30; N=6000
rows=[]
base=pd.Timestamp("2023-01-01"); cutoff=pd.Timestamp("2025-07-16")
for cid in range(1,N+1):
    typ=rng.random()<.18
    theta=np.log(120)+rng.normal(0,tau)+(.15 if typ else 0)
    first=base+pd.Timedelta(days=int(rng.integers(0,900)))   # acquired somewhere in train span
    n=1+rng.poisson(2.0)
    for k in range(n):
        dt=first+pd.Timedelta(days=int(rng.integers(0,700)))
        d=rng.choice(depts,p=pd_); e=1 if (typ and rng.random()<.5) else 0
        rows.append((cid,dt,d,e,np.exp(theta+gamma[d]+delta*e+rng.normal(0,sigma))))
o=pd.DataFrame(rows,columns=["cid","dt","dept","e","value"])
o=o[o.dt<=pd.Timestamp("2026-07-16")]
train=o[o.dt<=cutoff].copy(); test=o[o.dt>cutoff].copy()
# train cells
g=train.groupby(["cid","dept","e"])
cell=g.value.agg(n_orders="size",sum_value="sum").reset_index()
cell["sum_log_value"]=g.value.apply(lambda s:np.log(s).sum()).values
cell["sum_log_value_sq"]=g.value.apply(lambda s:(np.log(s)**2).sum()).values
cell=cell.rename(columns={"cid":"shopify_customer_id","dept":"order_department","e":"order_is_exotic"})
be=train.groupby("cid").e.max(); tot=train.groupby("cid").size()
cell["buys_exotic"]=cell.shopify_customer_id.map(be); cell["cust_total_orders"]=cell.shopify_customer_id.map(tot)
cell["first_order_date"]="2024-01-01"
# test actuals
tn=test.groupby("cid").size(); tsv=test.groupby("cid").value.sum(); tsl=test.groupby("cid").value.apply(lambda s:np.log(s).sum())
cell["test_n_orders"]=cell.shopify_customer_id.map(tn).fillna(0).astype(int)
cell["test_sum_value"]=cell.shopify_customer_id.map(tsv).fillna(0.0)
cell["test_sum_log_value"]=cell.shopify_customer_id.map(tsl)
cell["order_is_exotic"]=cell.order_is_exotic.map({0:"False",1:"True"})
cell["buys_exotic"]=cell.buys_exotic.map({0:"False",1:"True"})
cell=cell[["shopify_customer_id","order_department","order_is_exotic","n_orders","sum_log_value",
           "sum_log_value_sq","sum_value","cust_total_orders","buys_exotic","first_order_date",
           "test_n_orders","test_sum_value","test_sum_log_value"]]
cell.to_csv("pltv_order_level_extract.csv",index=False)   # notebook reads this path
print("train cells",len(cell),"train custs",cell.shopify_customer_id.nunique(),
      "scorable(test>0)",(cell.groupby('shopify_customer_id').test_n_orders.first()>0).sum())
