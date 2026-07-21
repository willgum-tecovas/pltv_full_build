-- ============================================================================
-- Temporal-holdout validation extract for the order-level EB AOV model
-- ----------------------------------------------------------------------------
-- WHAT THIS IS FOR
-- The production extract (pltv_order_level_extract.sql) aggregates a customer's
-- ENTIRE history, so you can only compare predictions to the same orders they
-- were fit on — which is circular. This extract splits each customer's orders
-- in time so the model can be scored HONESTLY:
--
--   TRAIN window : orders on/before train_cutoff  -> fit the model, predict AOV
--   TEST  window : orders after train_cutoff       -> the actuals we score against
--
-- The train cells have the SAME schema as the production extract, so you point
-- the notebook's CSV_PATH at this file and the whole notebook fits on TRAIN.
-- Each row also carries that customer's TEST-window actuals (test_n_orders,
-- test_sum_value, test_sum_log_value); Section 8 dedups them to one row per
-- customer and scores expected_aov against the out-of-sample mean order value.
--
-- LEAKAGE GUARDS (all prediction inputs are known at train_cutoff):
--   * every train statistic uses only orders on/before train_cutoff
--   * buys_exotic is computed from TRAIN orders only (not all-time)
--   * cohort = customers ACQUIRED on/before train_cutoff, so they have train
--     history; test actuals never touch the fit
--
-- SCORABLE POPULATION: only customers with >= 1 order in the TEST window can be
--   scored (you need a realized actual). That means this validates AOV for
--   RETURNING customers; one-and-done customers have no test order and are
--   excluded from scoring (the notebook filters test_n_orders > 0).
--
-- WINDOWS (defaults): train = [2023-01-01, CURRENT_DATE - 365d];
--   test = (CURRENT_DATE - 365d, CURRENT_DATE]. One full year of holdout.
--   Adjust train_cutoff to trade train history for test length.
--
-- CONFIRM: merchandise_department values; is_exotic column name; whether
--   gross_sales_less_discount should be a returns-adjusted net field.
-- ============================================================================

WITH params AS (
  SELECT
    DATE '2023-01-01'                          AS order_floor,
    DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY) AS train_cutoff,   -- last day of TRAIN
    CURRENT_DATE()                             AS test_end        -- last day of TEST
),

-- Consumer order lines. channel_type only EXCLUDES non-consumer orders; it is
-- not a model feature and carries no premium/discount.
base_lines AS (
  SELECT
    o.shopify_customer_id,
    o.order_id,                                      -- CONFIRM column name (order grain; matches Phase 1)
    o.order_created_date,
    COALESCE(o.merchandise_department, 'Unassigned') AS merchandise_department,
    COALESCE(o.is_exotic, FALSE)                     AS is_exotic,   -- CONFIRM column name
    o.gross_sales_less_discount
  FROM `tecovas-prod-edw.core.orders` o
  WHERE o.shopify_customer_id > 0
    AND o.channel_type NOT IN ('Event', 'Corporate Sales', 'Wholesale')
),

-- Cohort = acquired within the train window (so they have training history).
cust_first AS (
  SELECT shopify_customer_id, MIN(order_created_date) AS first_order_date
  FROM base_lines
  GROUP BY 1
),
cohort AS (
  SELECT f.shopify_customer_id, f.first_order_date
  FROM cust_first f
  CROSS JOIN params p
  WHERE f.first_order_date >= p.order_floor
    AND f.first_order_date <= p.train_cutoff
),

-- Line revenue + line count per (customer, ORDER, department).
order_dept AS (
  SELECT
    b.shopify_customer_id,
    b.order_id,
    ANY_VALUE(b.order_created_date)   AS order_created_date,   -- constant within an order
    b.merchandise_department          AS dept,
    COUNT(*)                          AS dept_line_count,
    SUM(b.gross_sales_less_discount)  AS dept_rev,
    LOGICAL_OR(b.is_exotic)           AS dept_exotic
  FROM base_lines b
  JOIN cohort c USING (shopify_customer_id)
  GROUP BY 1, 2, 4            -- customer, order_id, dept  (order_created_date is ANY_VALUE)
),

-- One primary department per ORDER (count first, revenue tiebreak) and its total
-- net value = the "order value". Keep the order's date so we can split train vs
-- test below (the temporal cutoff still uses order_created_date).
order_primary AS (
  SELECT
    shopify_customer_id,
    order_id,
    ANY_VALUE(order_created_date) AS order_created_date,
    ARRAY_AGG(dept ORDER BY dept_line_count DESC, dept_rev DESC LIMIT 1)[OFFSET(0)] AS order_department,
    LOGICAL_OR(dept_exotic) AS order_is_exotic,
    SUM(dept_rev)           AS order_value
  FROM order_dept
  GROUP BY 1, 2
  HAVING SUM(dept_rev) > 0     -- positivity: LN(order_value) needs value > 0
),

-- Split the orders in time.
train_orders AS (
  SELECT op.*
  FROM order_primary op CROSS JOIN params p
  WHERE op.order_created_date <= p.train_cutoff
),
test_orders AS (
  SELECT op.*
  FROM order_primary op CROSS JOIN params p
  WHERE op.order_created_date >  p.train_cutoff
    AND op.order_created_date <= p.test_end
),

-- Customer-level TRAIN rollups. buys_exotic is TRAIN-only (leakage-safe).
train_cust AS (
  SELECT
    shopify_customer_id,
    COUNT(*)                    AS cust_total_orders,   -- train orders
    MIN(order_created_date)     AS first_order_date,
    LOGICAL_OR(order_is_exotic) AS buys_exotic          -- ever bought exotic IN TRAIN
  FROM train_orders
  GROUP BY 1
),

-- Out-of-sample actuals per customer (the scoring target).
test_actuals AS (
  SELECT
    shopify_customer_id,
    COUNT(*)                        AS test_n_orders,
    SUM(order_value)                AS test_sum_value,      -- for out-of-sample mean AOV
    SUM(LN(order_value))            AS test_sum_log_value   -- optional: log-scale scoring
  FROM test_orders
  GROUP BY 1
)

-- ===== TRAIN cells (fit grain) + carried TEST actuals (scoring) =============
-- One row per (customer, order_department, order_is_exotic) over the TRAIN
-- window. Same sufficient-stat schema as the production extract; the notebook
-- fits on these. The test_* columns are customer-level and repeat on each of a
-- customer's rows (dedup in the notebook). LEFT JOIN keeps train customers who
-- never returned — they carry test_n_orders = NULL/0 and are dropped at scoring.
SELECT
  tr.shopify_customer_id,
  tr.order_department,
  tr.order_is_exotic,
  COUNT(*)                          AS n_orders,          -- train orders in this cell
  SUM(LN(tr.order_value))           AS sum_log_value,     -- -> theta_c / gamma_d / delta
  SUM(POW(LN(tr.order_value), 2))   AS sum_log_value_sq,  -- -> sigma_resid (within cell)
  SUM(tr.order_value)               AS sum_value,         -- raw $ (train baselines / checks)
  ANY_VALUE(tc.cust_total_orders)   AS cust_total_orders, -- train order count
  ANY_VALUE(tc.buys_exotic)         AS buys_exotic,       -- TRAIN-only type flag
  ANY_VALUE(tc.first_order_date)    AS first_order_date,
  ANY_VALUE(COALESCE(ta.test_n_orders, 0))    AS test_n_orders,      -- 0 = did not return
  ANY_VALUE(COALESCE(ta.test_sum_value, 0.0)) AS test_sum_value,     -- out-of-sample $ total
  ANY_VALUE(ta.test_sum_log_value)            AS test_sum_log_value  -- out-of-sample log total
FROM train_orders tr
JOIN      train_cust  tc USING (shopify_customer_id)
LEFT JOIN test_actuals ta USING (shopify_customer_id)
GROUP BY 1, 2, 3;
