# Data inspection — manual follow-up notes

Companion to the auto-generated `data_inspection.md`. Every number below was computed
from the downloaded data (train 1,296,675 rows / test 555,719 rows, 24 columns).

## Confirmed facts
- **Target:** `is_fraud` (0/1). Train 7,506 frauds (0.579%, ≈172:1); test 2,145 frauds (0.386%, ≈258:1).
- **Out-of-time split:** test (2020-06-21 → 2020-12-31) starts strictly after train ends
  (2019-01-01 → 2020-06-21). There are no exact duplicate rows within or across splits, and no shared `trans_num`.
- **Missing values:** only `merch_zipcode` (15.1% train / 15.1% test). Missingness carries ~no
  fraud signal (train fraud rate 0.58% missing vs 0.58% present).
- **`Unnamed: 0`:** row position, restarted at 0 in each file → meaningless.
- **`unix_time`:** `trans_date_trans_time` shifted back by ~7 years (constant offset of 220,924,800 s,
  with a one-day-different offset for some train rows) → redundant with the real timestamp.
- **`merchant`:** *every* merchant name starts with `fraud_` in both classes → generator artefact, not a label leak.
- **Card = customer:** each `cc_num` maps to exactly one profile (name, address, dob, job);
  999 cards ↔ 999 (first, last, dob) combinations.
- **Merchant coordinates:** `merch_lat/long` are always within ±1.0° of the cardholder's home
  coordinates, and mean distance is the same for fraud vs legit (0.767° vs 0.766°) → looks like
  synthetic jitter; home-to-merchant distance is unlikely to be informative.
- **Train is not sorted by time** (test is) → must sort before computing any history-based feature.

## Fraud structure (drives the feature plan)
- Fraud arrives in **bursts per card**: median 10 fraudulent transactions per affected card, and the
  first-to-last fraud span is a median of 1.9 days (90th pct 2.0 days).
- Fraud cards barely repeat across splits: only 4 of 218 test fraud cards also had fraud in train
  → memorising `cc_num` would not generalise.

## ⚠ Dataset artefact: "fraud-only" cards
- 75 train cards and 16 test cards consist **only** of fraudulent transactions (≤19 txns each).
- All 16 cards that appear only in test are 100% fraud (163 txns).
- Consequence: any feature like "number of prior transactions on this card" can partly learn
  "new/unseen card ⇒ fraud". That is a quirk of the simulator and would inflate metrics relative
  to a real bank, where new accounts are mostly legitimate.

## Leakage-scan interpretation
- `trans_date_trans_time` and `trans_num` score AUC = 1.0 only because they are (near-)unique: an
  in-sample per-value fraud rate *is* the label. That is an artefact of the scan, not a real signal. Raw timestamps and transaction IDs
  are still excluded as features.
- No column directly encodes the label.
