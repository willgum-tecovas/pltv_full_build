-- ============================================================================
-- Order-level AOV extract — department & exotic as ORDER-level effects
-- ----------------------------------------------------------------------------
-- WHY THIS EXISTS
-- An earlier per-customer extract charged cross-department basket variation to
-- the within-customer noise sigma: single-department customers showed
-- sigma=0.41, multi-department 0.84, pooled 0.81. That inflated
-- K = sigma^2 / tau^2 to ~16.5 and over-shrank loyal customers (a 10-order
-- customer got only 38% weight). Moving department (and exotic) to the ORDER
-- level removes that basket-mix variation from sigma, pulling it back toward
-- ~0.41, K toward ~4.3, and a 10-order customer to ~70% weight.
--
-- MODEL (fit in the notebook)
--   log(order_value)_{c,i} = theta_c + gamma_{d(i)} + delta * e(i) + eps
--                            eps ~ N(0, sigma_resid^2)
--
--   theta_c ~ N(mu + X_c * beta, tau^2)     X_c = customer features = buys_exotic
--   gamma_d = order-level department effect  (pooled / population)
--   delta   = order-level exotic effect      (pooled / population)
--
-- theta_c is the customer's DEPARTMENT/EXOTIC-ADJUSTED baseline, and
-- sigma_resid is the order-to-order noise *within a (department, exotic) cell*
-- (the true ~0.41), so shrinkage is driven by real signal, not basket mix.
--
-- EXPECTED AOV (one number per customer, to multiply Phase-1 projected orders)
--   E[AOV_c] = sum_cell  p_hat_c(cell) * exp(theta_c + gamma_cell + sigma^2 / 2)
--   where a "cell" is a (department, exotic) combination and p_hat_c(cell) is
--   the customer's cell mix, shrunk toward the population mix with M pseudo-
--   orders (see notebook). In words: value-per-order in each cell, weighted by
--   how often the customer buys that cell.
--
-- EXOTIC is carried at BOTH levels, on purpose:
--   * order_is_exotic — did that order include an exotic item. Lets the
--     notebook add an order-level term (delta) that absorbs the per-order
--     exotic premium and keeps sigma_resid clean.
--   * buys_exotic     — did the customer EVER buy an exotic item. The
--     customer-level type flag that shifts theta_c, testing "exotic buyers
--     spend more overall, even on their non-exotic orders."
--   Fit together they separate "exotic orders cost more" (delta) from "exotic
--   buyers are a higher-spending type" (buys_exotic). buys_exotic is all-time
--   (updates as they buy) and a product-choice flag, not a value-argmax, so it
--   is scoring-safe and low-leakage — same reasoning as the count-based dept.
--
-- GRAIN: one row per (shopify_customer_id, order_department, order_is_exotic).
--   The unit is one ORDER (order_id) -- matching the Phase-1 order-count model,
--   so AOV and projected orders multiply consistently. (Previously grain was the
--   purchase-day, which merged same-day back-to-back orders; order_id keeps them
--   separate.) Each order gets a single primary department by LINE COUNT (most
--   lines in the order), revenue breaking ties, plus an order_is_exotic flag
--   (any exotic line in the order). All sufficient statistics are on the log
--   scale so the Python EB fit needs nothing else.
--
-- Cohort: acquired between order_floor (2023-01-01) and join_ceiling
--   (CURRENT_DATE - 400d). Positivity: LN() needs value > 0, so days with a
--   non-positive total are dropped.
--
-- SCOPE NOTE: the channel_type filter below only EXCLUDES non-consumer orders
--   (events, corporate, wholesale) from the population. Channel is NOT a model
--   feature and carries no premium/discount here.
--
-- CONFIRM: merchandise_department values; is_exotic column name; whether
--   gross_sales_less_discount should be a returns-adjusted net field.
-- ============================================================================

WITH params AS (
  SELECT
    DATE '2023-01-01'                          AS order_floor,
    DATE_SUB(CURRENT_DATE(), INTERVAL 400 DAY) AS join_ceiling
),

-- Raw consumer order lines in scope. channel_type is used ONLY to exclude
-- non-consumer orders; it is not projected and is not a model feature.
base_lines AS (
  SELECT
    o.shopify_customer_id,
    o.shopify_order_id,                                      -- CONFIRM column name (order grain; matches Phase 1)
    o.order_created_date,
    COALESCE(o.merchandise_department, 'Unassigned') AS merchandise_department,
    COALESCE(o.is_exotic, FALSE)                     AS is_exotic,   -- CONFIRM column name
    o.gross_sales_less_discount
  FROM `tecovas-prod-edw.core.orders` o
  WHERE o.shopify_customer_id > 0
    AND o.channel_type NOT IN ('Event', 'Corporate Sales', 'Wholesale')
),

-- First order date per customer, used to define the acquisition cohort.
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
    AND f.first_order_date <= p.join_ceiling
),

-- Line revenue + line count per (customer, ORDER, department).
order_dept AS (
  SELECT
    b.shopify_customer_id,
    b.shopify_order_id,
    ANY_VALUE(b.order_created_date)   AS order_created_date,   -- constant within an order
    b.merchandise_department          AS dept,
    COUNT(*)                          AS dept_line_count,
    SUM(b.gross_sales_less_discount)  AS dept_rev,
    LOGICAL_OR(b.is_exotic)           AS dept_exotic
  FROM base_lines b
  JOIN cohort c USING (shopify_customer_id)
  GROUP BY 1, 2, 4            -- customer, order_id, dept  (order_created_date is ANY_VALUE)
),

-- Assign each ORDER ONE primary department (count first, revenue as the
-- tiebreak) and its total net value (the "order value").
order_primary AS (
  SELECT
    shopify_customer_id,
    shopify_order_id,
    ANY_VALUE(order_created_date) AS order_created_date,
    ARRAY_AGG(dept ORDER BY dept_line_count DESC, dept_rev DESC LIMIT 1)[OFFSET(0)] AS order_department,
    LOGICAL_OR(dept_exotic) AS order_is_exotic,
    SUM(dept_rev)           AS order_value
  FROM order_dept
  GROUP BY 1, 2
  HAVING SUM(dept_rev) > 0     -- positivity: LN(order_value) requires value > 0
),

-- Customer-level rollups: total orders (for the cell mix denominator) and the
-- customer-level type flag buys_exotic.
cust_tot AS (
  SELECT
    shopify_customer_id,
    COUNT(*)                    AS cust_total_orders,
    MIN(order_created_date)     AS first_order_date,
    LOGICAL_OR(order_is_exotic) AS buys_exotic
  FROM order_primary
  GROUP BY 1
)

-- ===== per (customer, order_department, order_is_exotic) log-scale stats =====
-- One row per cell. sum_log_value and sum_log_value_sq are the sufficient
-- statistics the EB fit needs; sum_value is kept for the model-free
-- arithmetic-mean validation in the notebook.
SELECT
  op.shopify_customer_id,
  op.order_department,
  op.order_is_exotic,
  COUNT(*)                         AS n_orders,         -- orders (order_id) in this cell
  SUM(LN(op.order_value))          AS sum_log_value,    -- -> theta_c / gamma_d / delta
  SUM(POW(LN(op.order_value), 2))  AS sum_log_value_sq, -- -> sigma_resid (within cell)
  SUM(op.order_value)              AS sum_value,        -- raw $ (arithmetic-mean check)
  ANY_VALUE(ct.cust_total_orders)  AS cust_total_orders,-- for the department/exotic mix
  ANY_VALUE(ct.buys_exotic)        AS buys_exotic,      -- customer-level flag (descriptive only)
  ANY_VALUE(ct.first_order_date)   AS first_order_date
FROM order_primary op
JOIN cust_tot ct USING (shopify_customer_id)
GROUP BY 1, 2, 3;
