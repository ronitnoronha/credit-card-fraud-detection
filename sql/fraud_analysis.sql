-- =====================================================================
-- Fraud analytics queries (DuckDB dialect, mostly ANSI SQL).
--
-- Run all of them with:   python -m src.run_sql
-- The runner creates a view `transactions` over the train + test Parquet
-- files with these extra columns:
--     split  ('train' | 'test')
--     ts     TIMESTAMP parsed from trans_date_trans_time
-- Each query starts with a "-- name: <id>" line; results are written to
-- reports/sql/<id>.csv.
-- Only columns that exist in the dataset are used. Card-level queries use
-- cc_num, which identifies one customer/card in this dataset.
-- =====================================================================

-- name: overview_by_split
-- How large is each split and how rare is fraud?
SELECT
    split,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 4)            AS fraud_rate_pct,
    COUNT(DISTINCT cc_num)                     AS n_cards,
    MIN(ts)                                    AS first_txn,
    MAX(ts)                                    AS last_txn
FROM transactions
GROUP BY split
ORDER BY split DESC;

-- name: fraud_rate_by_category
-- Fraud rate and fraud dollar volume per merchant category.
SELECT
    category,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS fraud_amount_usd
FROM transactions
GROUP BY category
ORDER BY fraud_rate_pct DESC;

-- name: amount_by_fraud_status
-- Do fraudulent transactions look different in size?
SELECT
    CASE WHEN is_fraud = 1 THEN 'fraud' ELSE 'legitimate' END AS status,
    COUNT(*)                                   AS n_transactions,
    ROUND(AVG(amt), 2)                         AS avg_amount,
    ROUND(MEDIAN(amt), 2)                      AS median_amount,
    ROUND(QUANTILE_CONT(amt, 0.90), 2)         AS p90_amount,
    ROUND(MAX(amt), 2)                         AS max_amount
FROM transactions
GROUP BY status;

-- name: transaction_counts_by_category
-- Volume mix: which categories dominate the transaction flow?
SELECT
    category,
    COUNT(*)                                                    AS n_transactions,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)          AS share_of_all_pct,
    ROUND(AVG(amt), 2)                                          AS avg_amount
FROM transactions
GROUP BY category
ORDER BY n_transactions DESC;

-- name: top_categories_by_fraud_rate
-- Top 5 categories by fraud rate, only counting categories with enough volume
-- for the rate to be meaningful (HAVING filters after aggregation).
SELECT
    category,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct,
    RANK() OVER (ORDER BY AVG(is_fraud) DESC)  AS fraud_rate_rank
FROM transactions
GROUP BY category
HAVING COUNT(*) >= 10000
ORDER BY fraud_rate_rank
LIMIT 5;

-- name: monthly_volume_and_fraud
-- Transaction volume, fraud volume and fraud rate per month.
SELECT
    DATE_TRUNC('month', ts)                    AS month,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct,
    ROUND(SUM(amt), 2)                         AS total_amount_usd
FROM transactions
GROUP BY month
ORDER BY month;

-- name: daily_volume_and_fraud
-- Daily transaction and fraud counts, with a 7-day rolling fraud count
-- (window frame over the ordered days).
WITH daily AS (
    SELECT
        CAST(ts AS DATE)                       AS day,
        COUNT(*)                               AS n_transactions,
        SUM(is_fraud)                          AS n_fraud
    FROM transactions
    GROUP BY day
)
SELECT
    day,
    n_transactions,
    n_fraud,
    SUM(n_fraud) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS fraud_7d_rolling
FROM daily
ORDER BY day;

-- name: fraud_by_hour
-- At what time of day does fraud happen?
SELECT
    EXTRACT(HOUR FROM ts)                      AS hour_of_day,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- name: amount_band_fraud_rate
-- Fraud rate by transaction-size band (CASE bucketing).
SELECT
    CASE
        WHEN amt < 10    THEN '1: < $10'
        WHEN amt < 50    THEN '2: $10-50'
        WHEN amt < 200   THEN '3: $50-200'
        WHEN amt < 500   THEN '4: $200-500'
        WHEN amt < 1000  THEN '5: $500-1000'
        ELSE                  '6: >= $1000'
    END                                        AS amount_band,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct
FROM transactions
GROUP BY amount_band
ORDER BY amount_band;

-- name: card_level_summary
-- Card (customer) level aggregation: the 20 cards with the most fraud.
SELECT
    cc_num,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 2)            AS fraud_rate_pct,
    ROUND(SUM(amt), 2)                         AS total_spend_usd,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS fraud_spend_usd,
    MIN(ts)                                    AS first_seen,
    MAX(ts)                                    AS last_seen
FROM transactions
GROUP BY cc_num
ORDER BY n_fraud DESC, fraud_spend_usd DESC
LIMIT 20;

-- name: card_fraud_exposure_distribution
-- How is fraud spread across cards? (cards with 0, 1-5, 6-10, ... frauds)
WITH per_card AS (
    SELECT cc_num, SUM(is_fraud) AS n_fraud, AVG(is_fraud) AS fraud_share
    FROM transactions
    GROUP BY cc_num
)
SELECT
    CASE
        WHEN n_fraud = 0  THEN '0'
        WHEN n_fraud <= 5 THEN '1-5'
        WHEN n_fraud <= 10 THEN '6-10'
        WHEN n_fraud <= 15 THEN '11-15'
        ELSE '16+'
    END                                        AS frauds_per_card,
    COUNT(*)                                   AS n_cards,
    SUM(CASE WHEN fraud_share = 1 THEN 1 ELSE 0 END) AS n_cards_only_fraud
FROM per_card
GROUP BY frauds_per_card
ORDER BY MIN(n_fraud);

-- name: fraud_burst_duration
-- Window functions: for every card with fraud, how long does the fraud episode
-- last, and how much is taken?
WITH fraud_txn AS (
    SELECT cc_num, ts, amt
    FROM transactions
    WHERE is_fraud = 1
),
bursts AS (
    SELECT
        cc_num,
        COUNT(*)                                            AS n_fraud,
        DATE_DIFF('hour', MIN(ts), MAX(ts))                 AS burst_hours,
        ROUND(SUM(amt), 2)                                  AS fraud_amount_usd
    FROM fraud_txn
    GROUP BY cc_num
)
SELECT
    COUNT(*)                                   AS n_cards_with_fraud,
    MEDIAN(n_fraud)                            AS median_frauds_per_card,
    MEDIAN(burst_hours)                        AS median_burst_hours,
    QUANTILE_CONT(burst_hours, 0.9)            AS p90_burst_hours,
    ROUND(MEDIAN(fraud_amount_usd), 2)         AS median_fraud_amount_per_card
FROM bursts;

-- name: time_since_previous_txn
-- LAG() window: minutes since the same card's previous transaction,
-- compared between fraud and legitimate transactions.
WITH ordered AS (
    SELECT
        is_fraud,
        DATE_DIFF('second', LAG(ts) OVER (PARTITION BY cc_num ORDER BY ts), ts) / 60.0 AS minutes_since_prev
    FROM transactions
)
SELECT
    CASE WHEN is_fraud = 1 THEN 'fraud' ELSE 'legitimate' END AS status,
    ROUND(MEDIAN(minutes_since_prev), 1)       AS median_minutes_since_prev,
    ROUND(AVG(minutes_since_prev), 1)          AS avg_minutes_since_prev
FROM ordered
WHERE minutes_since_prev IS NOT NULL
GROUP BY status;

-- name: fraud_rate_by_state
-- Geographic view. n_cards is included because a state rate built on 1-3 cards
-- is not statistically meaningful.
SELECT
    state,
    COUNT(*)                                   AS n_transactions,
    COUNT(DISTINCT cc_num)                     AS n_cards,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct
FROM transactions
GROUP BY state
ORDER BY fraud_rate_pct DESC;

-- name: fraud_by_day_of_week
-- Fraud rate by weekday (ISODOW: 1 = Monday ... 7 = Sunday).
SELECT
    ISODOW(ts)                                 AS iso_day_of_week,
    DAYNAME(ts)                                AS day_name,
    COUNT(*)                                   AS n_transactions,
    SUM(is_fraud)                              AS n_fraud,
    ROUND(100.0 * AVG(is_fraud), 3)            AS fraud_rate_pct
FROM transactions
GROUP BY iso_day_of_week, day_name
ORDER BY iso_day_of_week;

-- name: amount_histogram_by_status
-- Histogram of amounts on a log10 scale (bins of 0.1 decades) per class,
-- normalised within class so fraud and legitimate shapes are comparable.
WITH binned AS (
    SELECT
        is_fraud,
        FLOOR(LOG10(GREATEST(amt, 1)) * 10) / 10.0 AS log10_bin
    FROM transactions
)
SELECT
    CASE WHEN is_fraud = 1 THEN 'fraud' ELSE 'legitimate' END AS status,
    log10_bin,
    ROUND(POWER(10, log10_bin), 2)             AS bin_start_usd,
    COUNT(*)                                   AS n_transactions,
    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY is_fraud) AS share_within_status
FROM binned
GROUP BY is_fraud, log10_bin
ORDER BY status, log10_bin;
