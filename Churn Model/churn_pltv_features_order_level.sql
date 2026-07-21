-- ============================================================================
-- Churn / PLTV — ORDER-LEVEL PANEL (Phase 1 model)
-- ----------------------------------------------------------------------------
-- Grain:   one row per (shopify_customer_id, purchase-day). A customer with 4
--          purchase-days contributes 4 rows. This is a discrete-time repeat-
--          purchase hazard model: it unifies the first-order / two-order models
--          into one, with order_seq as a feature.
-- Label:   churned_400d = 1 if the customer places NO further order within 400
--          days of THIS order; 0 if they reorder within the window. Anchored to
--          each order's own date -> cohort-age-neutral (every order gets an
--          equal 400-day forward window).
-- Features: summarize the customer's state AS OF this order (everything up to &
--          including it). order_seq / cumulative spend / cadence are legitimate
--          state, NOT leakage — they're known at scoring time. The current
--          order's own return/exchange is EXCLUDED (post-order -> leakage);
--          only PRIOR orders' cumulative returns/exchanges are kept. A per-order
--          return-propensity score will replace the dropped current-order signal.
-- Maturity: only orders with order_date <= today-400d are emitted (full window
--          observed). Cumulative features still look back over FULL history
--          (incl. pre-2023) for accuracy; we just emit rows with
--          order_date in [2023-01-01, today-400d].
-- ----------------------------------------------------------------------------
-- ⚠️  TRAIN/TEST SPLIT BY shopify_customer_id (grouped), never by row — a
--     customer's rows are correlated and would leak across a random split.
-- ⚠️  Confirm CONFIRM: field names (net_sales_value, gross_sales_amount,
--     discounts, gross_product_quantity, exchanged_quantity).
-- "Order" = distinct order_created_date (purchase-day), to absorb same-day
--     split/exchange artifacts.
-- ============================================================================

WITH params AS (
  SELECT DATE_SUB(CURRENT_DATE(), INTERVAL 400 DAY) AS maturity_cutoff
),

-- 1) GLOBAL-FILTERED order lines
base_lines AS (
  SELECT
    o.shopify_customer_id,
    o.shopify_order_id,
    o.order_created_date,
    o.channel_type,
    o.store_name,
    o.merchandise_department,
    o.style,
    CAST(o.gross_product_quantity AS NUMERIC) AS order_quantity,
    CAST(o.return_quantity        AS NUMERIC) AS return_quantity,
    CAST(o.exchanged_quantity     AS NUMERIC) AS exchanged_quantity,  
    CAST(o.net_sales_value        AS NUMERIC) AS net_sales,           
    CAST(o.gross_sales_amount     AS NUMERIC) AS gross_sales,         
    CAST(o.discounts              AS NUMERIC) AS discount_amt         
  FROM `tecovas-prod-edw.core.orders` o
  WHERE o.shopify_customer_id > 0
    AND o.channel_type NOT IN ('Event', 'Corporate Sales')
),

-- 2) Product universe -> ONE row per style (prevents join fan-out)
prod AS (
  SELECT
    product_style,
    LOGICAL_OR(item_status IS NOT NULL
      AND item_status NOT IN ('Discontinued', 'Never Launched')) AS valid_style
  FROM `tecovas-prod-edw.core.products`
  GROUP BY product_style
),

-- 3) Tag lines: validity + department slot + footwear flag
lines_tagged AS (
  SELECT
    b.*,
    COALESCE(p.valid_style, FALSE) AS valid_style,
    CASE b.merchandise_department
      WHEN "Men's Footwear"   THEN 'mens_fw'
      WHEN "Women's Footwear" THEN 'womens_fw'
      WHEN "Kids' Footwear"   THEN 'kids_fw'
      WHEN "Accessories"      THEN 'acc'
      WHEN "Men's Apparel"    THEN 'mens_ap'
      WHEN "Women's Apparel"  THEN 'womens_ap'
      WHEN "(Not Provided)"   THEN 'not_provided'
      ELSE NULL
    END AS dept_slot
  FROM base_lines b
  LEFT JOIN prod p ON b.style = p.product_style
),

-- 4) Collapse to PURCHASE-DAY grain, with per-day aggregates & presence flags
pd AS (
  SELECT
    shopify_customer_id,
    order_created_date AS order_date,
    SUM(net_sales)        AS day_net,
    SUM(order_quantity)   AS day_items,
    SUM(gross_sales)      AS day_gross,
    ABS(SUM(discount_amt)) AS day_discount,
    SUM(return_quantity)    AS day_return_qty,     -- total units/lines returned that day
    SUM(exchanged_quantity) AS day_exchange_qty,   -- total units/lines exchanged that day
    -- per-day department presence (valid styles) for cumulative breadth
    LOGICAL_OR(valid_style AND dept_slot = 'mens_fw')   AS day_mens_fw,
    LOGICAL_OR(valid_style AND dept_slot = 'womens_fw') AS day_womens_fw,
    LOGICAL_OR(valid_style AND dept_slot = 'kids_fw')   AS day_kids_fw,
    LOGICAL_OR(valid_style AND dept_slot = 'acc')       AS day_acc,
    LOGICAL_OR(valid_style AND dept_slot = 'mens_ap')   AS day_mens_ap,
    LOGICAL_OR(valid_style AND dept_slot = 'womens_ap') AS day_womens_ap,
    LOGICAL_OR(channel_type = 'eComm')  AS day_ecomm,
    LOGICAL_OR(channel_type = 'Retail') AS day_retail,
    -- representative attributes for the day (earliest valid line)
    ARRAY_AGG(IF(valid_style, merchandise_department, NULL) IGNORE NULLS
              ORDER BY shopify_order_id LIMIT 1)[SAFE_OFFSET(0)] AS day_department,
    ARRAY_AGG(IF(valid_style, style, NULL) IGNORE NULLS
              ORDER BY shopify_order_id LIMIT 1)[SAFE_OFFSET(0)] AS day_style,
    ARRAY_AGG(channel_type ORDER BY shopify_order_id LIMIT 1)[SAFE_OFFSET(0)] AS day_channel,
    ARRAY_AGG(store_name   ORDER BY shopify_order_id LIMIT 1)[SAFE_OFFSET(0)] AS day_store
  FROM lines_tagged
  GROUP BY 1, 2
),

-- 5) Flag first-time-seen physical stores (for cumulative distinct-store count)
pd2 AS (
  SELECT
    pd.*,
    (day_store IS NOT NULL AND day_store <> '000 - Website') AS day_is_physical,
    (day_store IS NOT NULL AND day_store <> '000 - Website'
       AND ROW_NUMBER() OVER (PARTITION BY shopify_customer_id, day_store
                              ORDER BY order_date) = 1) AS new_physical_store
  FROM pd
),

-- 6) Windowed cumulative / sequence / cadence features (leakage-safe: each frame
--    runs UNBOUNDED PRECEDING -> CURRENT ROW, i.e. up to & including this order)
seq AS (
  SELECT
    shopify_customer_id,
    order_date,
    day_net, day_items, day_discount, day_gross,
    day_department, day_style, day_channel, day_store,

    ROW_NUMBER()                          OVER w_seq AS order_seq,
    LAG(order_date)                       OVER w_seq AS prev_order_date,
    LEAD(order_date)                      OVER w_seq AS next_order_date,       -- label only
    MIN(order_date)                       OVER w_run AS first_order_date,

    SUM(day_net)                          OVER w_run AS cum_net_spend,
    SUM(day_items)                        OVER w_run AS cum_items,
    -- TOTAL returned / exchanged units across all PRIOR orders (w_prior excludes
    -- the current order, so its own post-order returns/exchanges never leak in)
    SUM(day_return_qty)                   OVER w_prior AS cum_return_qty_prior,
    SUM(day_exchange_qty)                 OVER w_prior AS cum_exchange_qty_prior,
    SUM(CAST(new_physical_store AS INT64))OVER w_run AS cum_distinct_physical_stores,

    -- cumulative "ever bought in this dept slot by now"
    MAX(CAST(day_mens_fw   AS INT64))     OVER w_run AS cum_mens_fw,
    MAX(CAST(day_womens_fw AS INT64))     OVER w_run AS cum_womens_fw,
    MAX(CAST(day_kids_fw   AS INT64))     OVER w_run AS cum_kids_fw,
    MAX(CAST(day_acc       AS INT64))     OVER w_run AS cum_acc,
    MAX(CAST(day_mens_ap   AS INT64))     OVER w_run AS cum_mens_ap,
    MAX(CAST(day_womens_ap AS INT64))     OVER w_run AS cum_womens_ap,
    MAX(CAST(day_ecomm     AS INT64))     OVER w_run AS cum_ever_ecomm,
    MAX(CAST(day_retail    AS INT64))     OVER w_run AS cum_ever_retail,

    -- entry (order-1) static attributes, known at every order (not leakage)
    FIRST_VALUE(day_department IGNORE NULLS) OVER w_run AS entry_department,
    FIRST_VALUE(day_style      IGNORE NULLS) OVER w_run AS first_product_style,
    FIRST_VALUE(day_channel)                 OVER w_run AS acq_channel,
    FIRST_VALUE(day_store)                   OVER w_run AS acq_store_raw
  FROM pd2
  WINDOW
    w_seq AS (PARTITION BY shopify_customer_id ORDER BY order_date),
    w_run AS (PARTITION BY shopify_customer_id ORDER BY order_date
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    -- prior-orders-only frame: everything strictly BEFORE the current order
    w_prior AS (PARTITION BY shopify_customer_id ORDER BY order_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),

-- 7) Geography (static)
geo AS (
  SELECT
    shopify_customer_id,
    CASE
      WHEN customer_state IS NULL THEN NULL
      WHEN UPPER(TRIM(customer_state)) IN ('', '(NOT PROVIDED)', 'UNKNOWN', 'N/A', 'NA', 'NULL', 'NONE') THEN NULL
      ELSE UPPER(TRIM(customer_state))
    END AS state_code
  FROM `tecovas-prod-edw.base.base__customers`
)

-- ---- FINAL ASSEMBLY: one row per customer-order (matured window only) --------
SELECT
  s.shopify_customer_id,                                   -- GROUP KEY for split
  ABS(MOD(FARM_FINGERPRINT(CONCAT(CAST(s.shopify_customer_id AS STRING), 'seed_1')),
          100000)) / 100000.0                            AS split_rand,
  s.order_date,
  s.order_seq,

  -- ===== LABEL-SIDE (NOT features) =====
  s.next_order_date,                                       -- reference only
  DATE_DIFF(s.next_order_date, s.order_date, DAY) AS days_to_next_order,  -- for later time-to-event / regression
  CAST(NOT (s.next_order_date IS NOT NULL
            AND s.next_order_date <= DATE_ADD(s.order_date, INTERVAL 400 DAY)) AS INT64) AS churned_400d,

  -- ===== State / sequence / cadence (as-of this order) =====
  DATE_DIFF(s.order_date, s.first_order_date, DAY)         AS days_since_first_order,
  DATE_DIFF(s.order_date, s.prev_order_date, DAY)          AS days_since_prev_order,   -- NULL for order 1
  SAFE_DIVIDE(DATE_DIFF(s.order_date, s.first_order_date, DAY),
              NULLIF(s.order_seq - 1, 0))                  AS avg_gap_so_far,          -- NULL for order 1
  SAFE_DIVIDE(
    DATE_DIFF(s.order_date, s.prev_order_date, DAY),
    SAFE_DIVIDE(DATE_DIFF(s.order_date, s.first_order_date, DAY),
            NULLIF(s.order_seq - 1, 0)))                    AS gap_vs_cadence,

  -- ===== Monetary (cumulative + this order) =====
  s.cum_net_spend,
  SAFE_DIVIDE(s.cum_net_spend, s.order_seq)                AS cum_avg_order_value,
  SAFE_DIVIDE(s.cum_net_spend, DATE_DIFF(s.order_date, s.first_order_date, DAY)) AS spend_velocity,
  s.cum_items,
  s.day_net                                                AS this_order_spend,
  s.day_items                                              AS this_order_items,
  s.day_discount                                           AS this_order_discount_amount,
  SAFE_DIVIDE(s.day_discount, s.day_gross)                 AS this_order_discount_rate,
  (SAFE_DIVIDE(s.day_discount, s.day_gross) < 0.01)        AS paid_full_price,

  -- ===== Breadth / channel (cumulative, as-of this order) =====
  (s.cum_mens_fw + s.cum_womens_fw + s.cum_kids_fw
   + s.cum_acc + s.cum_mens_ap + s.cum_womens_ap)          AS cum_distinct_depts,
  ((s.cum_mens_fw = 1 OR s.cum_mens_ap = 1)
     AND (s.cum_womens_fw = 1 OR s.cum_womens_ap = 1))     AS cum_bought_cross_gender,
  ((s.cum_mens_fw + s.cum_womens_fw + s.cum_kids_fw) > 0
     AND (s.cum_acc + s.cum_mens_ap + s.cum_womens_ap) > 0) AS cum_footwear_plus_nonfootwear,
  s.cum_distinct_physical_stores,
  (s.cum_ever_ecomm = 1)                                   AS cum_ever_ecomm,
  (s.cum_ever_retail = 1)                                  AS cum_ever_retail,
  (s.cum_ever_ecomm = 1 AND s.cum_ever_retail = 1)         AS cum_both_channels,

  -- ===== Prior return / exchange behavior (PRIOR orders only) =====
  -- Total returned / exchanged units summed across ALL prior orders' lines.
  -- Current order's own return/exchange is EXCLUDED (post-order -> leakage);
  -- a per-order return-propensity score will be the leakage-safe replacement later.
  COALESCE(s.cum_return_qty_prior,   0) AS cum_return_qty_prior,
  COALESCE(s.cum_exchange_qty_prior, 0) AS cum_exchange_qty_prior,

  -- ===== This order's product / context =====
  s.day_department                                         AS this_order_department,
  s.day_style                                              AS this_order_style,
  s.day_channel                                            AS this_order_channel,
  IF(s.day_store IS NULL OR s.day_store = '000 - Website',
     'Online', s.day_store)                                AS this_order_store,
  EXTRACT(MONTH FROM s.order_date)                         AS order_month,

  -- ===== Entry / static (known from order 1) =====
  s.entry_department,
  s.first_product_style,
  s.acq_channel,
  IF(s.acq_store_raw IS NULL OR s.acq_store_raw = '000 - Website',
     'Online', s.acq_store_raw)                            AS acq_store,
  COALESCE(g.state_code, 'Unknown')                        AS customer_state,
  (g.state_code IS NULL)                                   AS state_missing

FROM seq s
CROSS JOIN params pr
LEFT JOIN geo g USING (shopify_customer_id)
WHERE s.first_order_date >= '2023-01-01'
  AND s.order_date <= pr.maturity_cutoff     -- only orders whose 400d window is fully observed
ORDER BY s.shopify_customer_id, s.order_date
LIMIT 100000;