# Data leakage analysis

Leakage means the model has access, during training or evaluation, to information it
would **not** have at the moment a real transaction must be scored. It makes offline
metrics look better than live performance. Every check below was run on the actual data
(`src/inspect_data.py`, `reports/data_inspection.md`, `reports/data_inspection_notes.md`).

## 1. Checks performed and results

| Check | Result | Action |
|---|---|---|
| **Direct target leakage** (a column that encodes `is_fraud`) | None found. The only columns with a perfect univariate AUC (`trans_date_trans_time`, `trans_num`) are near-unique, so a per-value fraud rate equals the label in-sample: an artefact of the scan, not a real signal. | Both excluded as raw features anyway. |
| **`merchant` names prefixed `fraud_`** | 100% of merchants in *both* classes carry the prefix. | Not leakage (an artefact of how the data was generated). `merchant` dropped for cardinality reasons (693 values). |
| **Row identifiers** | `Unnamed: 0` is the row position (restarts at 0 in each file); `trans_num` is unique per row. | Dropped. |
| **Card / customer identifier** | `cc_num` identifies one customer (each card maps to exactly one profile). Only 4 of 218 test fraud cards also had fraud in train. | Never used as a feature, because memorising card numbers cannot generalise. Used only (a) as the key for history features and (b) as the CV grouping variable. |
| **Train/test contamination** | 0 exact duplicate rows within or across splits; 0 shared `trans_num`; test starts strictly after train ends. 98% of test cards also appear in train, which is expected: the same customers keep transacting. | Out-of-time split kept as the final hold-out. |
| **Timestamp leakage** | `unix_time` duplicates `trans_date_trans_time` (shifted ~7 years). Raw timestamps would let a model learn *when* fraud happened in this particular sample. | `unix_time` dropped; raw timestamp used only to derive `hour`, `day_of_week`, `is_weekend` and history features. `month` is used for EDA only (18 months of data cannot support a seasonal pattern). |
| **Future information in engineered features** | History features are computed after sorting each card's transactions by time and use only *strictly earlier* rows (expanding statistics exclude the current row; window counts look backwards). Verified against a brute-force recomputation on a sample card. | Kept. |
| **Label information in engineered features** | No feature uses `is_fraud` of past transactions. In production a fraud label arrives days or weeks later (chargeback), so a "past fraud count" feature would be unavailable at scoring time. | By design. |
| **Preprocessing leakage** | Imputation medians, scaling and encoders are fitted inside a scikit-learn `Pipeline`, so in CV they are fitted on training folds only, and for the final model on the train split only. | Enforced in code (`src/preprocessing.py`). |
| **Resampling leakage** | If SMOTE is used it lives inside an `imblearn` pipeline, so synthetic samples are created from training folds only, never from validation or test. | Enforced in code. |
| **Cross-validation leakage via card identity** | Fraud arrives in bursts (median 10 fraud txns per card within ~2 days). With plain stratified K-fold, transactions from the *same burst* land in both training and validation folds, and card-level attributes (age, city population) let a model recognise the card, which inflates CV scores. | CV uses **`StratifiedGroupKFold` grouped by `cc_num`**: stratified on the label, but every card's transactions stay in one fold. |

## 2. Columns removed

| Column(s) | Reason |
|---|---|
| `Unnamed: 0`, `trans_num` | Row index / transaction ID, no predictive meaning. |
| `unix_time` | Redundant shifted copy of the timestamp. |
| `first`, `last`, `street`, `city`, `zip`, `job` | Personal data; with only 999 customers they act as card identifiers (memorisation risk) rather than generalisable signal. |
| `merchant` | 693 values; risk of target-encoding overfitting; category captures merchant type. |
| `state` | 5 of 51 states have ≤ 3 cards; the most extreme state rate (DE, 100%) comes from 1 card with 9 transactions. State therefore mostly encodes card identity (see `figures/05_fraud_rate_by_state.png`). |
| `lat`, `long` | Home location, effectively a card identifier. |
| `merch_lat`, `merch_long`, `merch_zipcode` | Merchant coordinates are always within ±1° of home with identical mean distance for fraud and legit (0.767° vs 0.766°), which looks like synthetic jitter. `merch_zipcode` is 15% missing with no relation to fraud. |
| `gender` | Excluded by project decision (fairness considerations in financial services; weak signal, univariate AUC 0.53). |
| `dob` | Replaced by `age` at transaction time. |

## 3. Columns / features retained

`amt`, `category`, `age`, `city_pop`, `hour`, `day_of_week`, `is_weekend`, plus backward-looking
card-history features: `card_txn_count_1h`, `card_txn_count_24h`, `secs_since_prev_txn`,
`amt_to_card_mean_ratio`, `amt_zscore_card`, `card_category_share`, `card_prior_txn_count`.

`age` and `city_pop` are customer-level attributes. They carry some genuine signal (fraud rate rises for
75+ cardholders) but with 999 customers they can also help a tree "recognise" a card. Grouped CV
measures performance honestly despite this.

## 4. Known dataset artefact (not leakage, but can inflate metrics)

75 train cards and 16 test cards contain **only** fraudulent transactions, and every card that
appears only in test is 100% fraud. A feature such as `card_prior_txn_count` can learn "new card ⇒
fraud", which is a property of the simulator rather than of real banking. Mitigation: results are
reported **with and without** the cold-start feature, and on the test set **with and without**
cards never seen in train.

## 5. Assumptions

- The data is synthetic (customer, merchant and coordinate patterns suggest a transaction
  simulator). Findings describe this dataset, not real-world fraud prevalence.
- At scoring time the bank knows the card's full transaction history up to that moment (standard
  for card authorisation systems).
- Fraud labels are assumed correct and final.
- The test period's lower fraud rate (0.39% vs 0.58%) is treated as genuine drift, not an error.
