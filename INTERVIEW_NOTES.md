# Interview notes — Financial Transaction Fraud Detection

Explanations of every major decision in this project, written so they can be said out loud in an
interview. Numbers quoted here come from this project's own outputs (see `reports/`). The section
"This project's results" is filled from the actual training run.

---

## 1. The problem in one paragraph

Given a card transaction and what the bank already knows at that moment (the card's earlier
transactions), output the **probability that the transaction is fraudulent**, then turn that
probability into an action (approve / step-up authentication / hold for review). Fraud is rare: 0.58%
of training transactions and 0.39% of test transactions, so this is a highly **imbalanced binary
classification** problem.

---

## 2. Data understanding and leakage

### What is data leakage?
Leakage is when the model gets information during training or evaluation that it would **not have at
prediction time**. Offline metrics then look excellent, but live performance collapses. Common forms:
- **Target leakage:** a feature that is derived from the label (e.g. a "chargeback filed" flag).
- **Temporal leakage:** using information from the future (e.g. a customer's average spend computed
  over the whole dataset, including transactions after the one being scored).
- **Train/test contamination:** the same rows, or preprocessing statistics, shared between splits.
- **Group leakage:** related rows (same customer) in both training and validation folds, so the model
  is rewarded for recognising the customer rather than the pattern.

### What I checked in this dataset
- No column directly encodes the label. Two columns (`trans_num`, `trans_date_trans_time`) scored a
  perfect AUC in a univariate scan, but only because they are near-unique: a per-value fraud rate *is*
  the label in-sample. An artefact of the scan, not a real signal.
- Every merchant name begins with `fraud_` **in both classes**, which is a quirk of the data generator, not a leak.
- Train and test are split **by time** (test starts after train ends), with no duplicate rows or
  shared transaction IDs. That makes the test set an honest "future" hold-out.
- `unix_time` is the timestamp shifted by ~7 years: redundant, dropped.

### Why certain columns were removed
- **IDs** (`Unnamed: 0`, `trans_num`): no meaning; a model could only memorise them.
- **Card number** (`cc_num`): only 4 of 218 fraud cards in the test period had fraud in training, so
  memorising card numbers cannot generalise. Kept only as a grouping key.
- **Personal / identity columns** (`first`, `last`, `street`, `city`, `zip`, `job`, `lat`, `long`):
  with only 999 customers these act as customer identifiers, which invites memorisation. They are also
  personal data that a bank would minimise.
- **`state`**: 5 of 51 states have ≤ 3 customers; the most extreme state rate (Delaware, 100%) is one
  card with 9 transactions. That is noise, not geography.
- **Merchant coordinates / zip**: always within ±1° of the customer's home with identical average
  distance for fraud and legitimate (0.767° vs 0.766°), so they look like random jitter.
- **`gender`**: excluded deliberately. Weak signal (AUC 0.53) and using protected attributes in
  financial decisioning raises fairness and regulatory concerns.
- **`dob`**: converted to `age` at transaction time.

### A dataset artefact I found and controlled for
75 training cards and 16 test cards contain **only** fraudulent transactions, and **every** card that
appears for the first time in the test period is 100% fraud. A feature like "number of previous
transactions on this card" can learn "new card ⇒ fraud", which is true in this simulator but not in a
real bank. I therefore report results with and without that feature, and on the test set with and
without never-seen cards.

---

## 3. Feature engineering

### Why these features
The EDA showed four patterns, and each feature targets one of them:

| Pattern found in the data | Feature(s) |
|---|---|
| Fraud amounts are bimodal: ~$8–25 (card testing) and ~$250–1,100; legitimate median is $47 | `amt` |
| Fraud is concentrated 22:00–03:00 (≈1.4–2.9% fraud rate vs ≈0.1% during the day) | `hour`, `day_of_week`, `is_weekend` |
| Some categories are much riskier (shopping_net 1.76%, health_fitness 0.16%) | `category` |
| Fraud arrives in **bursts** (median 10 frauds per card within ~2 days) | `card_txn_count_1h`, `card_txn_count_24h`, `secs_since_prev_txn` |
| Fraud spend is unusual **for that customer** | `amt_to_card_mean_ratio`, `amt_zscore_card`, `card_category_share` |
| Age bands differ (75+ cardholders have the highest rate) | `age` |

### How leakage is avoided in behavioural features
Each card's transactions are sorted by time, and every statistic uses **strictly earlier** rows:
expanding mean/std exclude the current transaction; window counts look backwards only. No feature uses
past **labels**, because in production the bank learns a transaction was fraud only days or weeks
later (chargeback). I verified the vectorised implementation against a brute-force loop.

Processing train and test together is safe here because the features look only backwards and the
test period comes after train. A test transaction's history includes the card's train-period
transactions, which is exactly what the bank would know.

### Features I deliberately did not build
- Home-to-merchant **distance**: sounds impressive, but the data shows no difference between classes.
- **Month**: 18 months of data cannot support seasonality; a model would just learn which months of
  this sample had more fraud.

---

## 4. Preprocessing

- **Pipeline + ColumnTransformer** so imputation, scaling and encoding are learned inside `fit`. In
  cross-validation this means they are fitted on the training folds only; for the final model, on the
  train split only. Nothing is learned from validation or test data.
- **Logistic Regression** needs: imputed values, a log transform for heavily skewed columns (amount,
  counts), standardisation (so the optimiser converges and coefficients are comparable), and
  one-hot encoding of hour/weekday/category because risk is not linear in "hour = 0…23".
- **Trees** need none of that: splits are invariant to monotonic transforms, so only imputation and an
  ordinal code for `category` are applied (which also keeps SHAP plots readable).

---

## 5. Class imbalance

### What it means and why it matters
Only ~1 in 172 training transactions is fraud. A model minimising average loss can do very well by
mostly predicting "legitimate"; the minority class contributes little to the loss.

### Why accuracy is misleading
Predicting "legitimate" for everything gives **99.42% accuracy** on train and **99.61% on test**
while catching zero fraud. Accuracy is dominated by the majority class.

### Precision vs recall
- **Precision** = of the transactions we flagged, how many were fraud? Low precision means many false
  alerts: blocked customers, analyst workload, complaints.
- **Recall** = of all fraud, how much did we catch? Low recall means missed fraud, i.e. direct losses
  and chargebacks.
- They trade off through the decision threshold: lowering it catches more fraud (recall ↑) but
  flags more legitimate customers (precision ↓).
- **Recall matters most** when a missed fraud is expensive (large amounts, account takeover).
  **Precision matters most** when review capacity is limited or friction is costly (e.g. declining a
  card at checkout).

### Why PR-AUC
ROC-AUC uses the false positive *rate* = FP / all negatives. With 1.8 million negatives, even
thousands of false alerts barely move the FPR, so ROC-AUC looks great for weak models. PR-AUC (average
precision) uses precision, which directly counts false alerts relative to true catches. A random model
scores PR-AUC ≈ fraud rate (≈0.006), so it is a much more demanding and informative metric here.

### SMOTE vs class weighting
- **Class weighting** (`class_weight="balanced"`, XGBoost `scale_pos_weight`) increases the loss for
  errors on fraud cases. No new data is created; it is cheap and simple.
- **SMOTE** creates *synthetic* fraud examples by interpolating between a fraud case and its fraud
  nearest neighbours. Risks: synthetic points may not resemble real fraud, interpolating encoded
  categories (e.g. category code 3.4) is meaningless, and it is costly on 1M+ rows.
- Both shift predicted probabilities upward (the model sees more fraud than reality), so the
  probabilities are no longer calibrated to the true fraud rate. That matters when the probability
  itself is used.
- I compared them empirically rather than assuming. Result: neither helped ranking quality here.
  For XGBoost all three strategies were within one standard deviation (PR-AUC 0.925 / 0.928 / 0.930);
  for Logistic Regression class weighting *reduced* PR-AUC from 0.460 to 0.253 while ROC-AUC rose
  from 0.935 to 0.950. Weighting improved the ranking in the bulk of the data but worsened the
  top of the ranking, which is exactly where alerts are generated.

### Why SMOTE must only be applied to training data
Validation and test data must look like the real world. If SMOTE is applied before splitting, synthetic
points derived from a validation case end up in training (leakage), and validation metrics are computed
on artificial data with an unrealistic fraud rate. I use `imblearn.pipeline.Pipeline`, which runs
SMOTE during `fit` only, so within CV it touches the training folds only.

---

## 6. Models

### Why Logistic Regression as the baseline
It is simple, fast, well-understood and interpretable (coefficients), and widely accepted by model-risk
teams in banks. If a complex model cannot clearly beat it, the complexity is not justified. Its
limitation: it models log-odds as a **linear** function of the inputs, so it cannot natively capture
"small *or* large amounts are risky, medium amounts are not" or interactions like "large amount *and*
late night".

### Why Random Forest
An ensemble of deep decision trees, each trained on a bootstrap sample with random feature subsets;
predictions are averaged. It captures non-linearities and interactions automatically, is robust to
outliers and needs little preprocessing. Averaging many de-correlated trees reduces variance. It's the
natural "is non-linearity worth it?" step between LR and boosting.

### What XGBoost does
Gradient boosting builds trees **sequentially**: each new (shallow) tree is fitted to the gradient of
the loss with respect to the current predictions, i.e. it focuses on the examples the ensemble
currently gets wrong. Predictions are the sum of all trees' outputs (in log-odds), scaled by a learning
rate. XGBoost adds L1/L2 regularisation on leaf weights, second-order (Hessian) optimisation, native
missing-value handling, row/column subsampling and a fast histogram algorithm.

### Why XGBoost was selected
It achieved the highest PR-AUC in grouped cross-validation and on the out-of-time
test set, handles the imbalance via `scale_pos_weight`, is fast to train and score, and is supported by
exact TreeSHAP explanations.

### Key hyperparameters and why they matter
| Hyperparameter | Effect |
|---|---|
| `n_estimators` | number of boosting rounds; more = more capacity, slower, can overfit |
| `learning_rate` | shrinks each tree's contribution; lower needs more trees but generalises better |
| `max_depth` | depth of each tree = order of interactions it can model; deeper = more overfitting risk |
| `min_child_weight` | minimum Hessian (≈ samples) in a leaf; higher = more conservative, ignores tiny patterns |
| `subsample`, `colsample_bytree` | fraction of rows / columns per tree; randomness reduces overfitting |
| `reg_lambda`, `gamma` | L2 penalty on leaf weights; minimum loss reduction required to split |
| `scale_pos_weight` | up-weights the fraud class in the loss; ≈ negatives/positives fully re-balances |

For Random Forest: `max_depth` / `min_samples_leaf` (tree complexity), `max_features` (de-correlation),
`n_estimators`, `class_weight`.

**Search strategy:** `RandomizedSearchCV` samples random combinations from distributions. With a
fixed budget it explores more distinct values of the important parameters than a grid, and it took
minutes instead of hours.

---

## 7. Validation

### What stratified cross-validation does
K-fold CV splits the data into K parts, trains on K−1 and validates on the remaining one, rotating so
every row is validated once; the spread across folds shows stability. **Stratified** means each fold
keeps the same fraud rate as the whole dataset. Without it, some folds could get very few fraud
cases, making the metrics noisy or undefined.

### Why grouped (StratifiedGroupKFold by card)
Fraud happens in bursts on the same card. With ordinary stratified K-fold, frauds from the same burst
fall into both training and validation folds, and the model can effectively recognise the customer
(age + city population nearly identify a card). `StratifiedGroupKFold` keeps each card's
transactions in one fold, so validation measures generalisation to **unseen customers**.

### Why there is still a separate test set
All choices (features, model, hyperparameters, threshold) were made using training data only. The
later-in-time test set is evaluated **once**, at the end. It also tests robustness to drift: the
fraud rate drops from 0.58% to 0.39% between periods.

---

## 8. Threshold selection

A classifier outputs a probability; the **threshold** converts it to a decision. 0.5 is arbitrary, and
it is particularly meaningless when class weights have shifted the probabilities.

My process:
1. Pre-registered business objective (fixed before seeing results): **maximise recall subject to
   precision ≥ 50%**, i.e. at most one false alert per fraud caught, to keep an analyst queue workable.
2. Compute out-of-fold predictions on the **training** set (grouped CV), so the threshold is chosen on
   data the model did not train on and the test set stays untouched.
3. Scan thresholds; record precision, recall, F1, false positives and false negatives.
4. Pick the lowest threshold meeting the precision floor (= highest recall).
5. Apply that fixed threshold to the test set.

Alternatives I'd discuss: an explicit cost function (cost of FN ≈ fraud amount vs cost of FP ≈ review
cost + customer friction), a capacity constraint (top-N alerts per day), or a per-segment threshold.

### Risk tiers
- **Medium** (p ≥ alert threshold 0.0122, chosen by the precision ≥ 50% rule): step-up
  authentication (one-time passcode) or analyst review.
- **High** (p ≥ 0.1923, where out-of-fold precision reaches 90%): decline / hold automatically.
- **Low**: approve.

On the test period: High = 0.34% of transactions with a 93.2% fraud rate (81.3% of all fraud);
Medium = 0.32% with 14.0% fraud rate (11.7% of fraud); Low = 99.34% with 0.03% (7.0% of fraud).

**Be honest about this if asked:** my first design was "Medium = threshold that captures 95% of
fraud". The alert rule already reached 96.6% recall, so that boundary landed *above* the alert
threshold and the tiers were out of order. I replaced it with a 90%-precision rule for High. That
floor is a business choice made after seeing the curve, while the 50% alert rule was fixed in advance.
Both are configurable (`src/config.py`, env overrides).

Why the Medium band is "friction, not decline": its own precision on test is only 14%, so
declining those payments would hurt about 6 legitimate customers per fraud stopped, while a
one-time passcode costs them seconds.

---

## 9. SHAP explainability

- SHAP (SHapley Additive exPlanations) assigns each feature a contribution such that
  `base value + Σ contributions = the model's output` (log-odds) for that transaction. It is based on
  Shapley values from cooperative game theory: a feature's average marginal contribution across all
  feature orderings. TreeSHAP computes these exactly and efficiently for tree models.
- **Global view:** mean |SHAP| ranks features by average impact; the beeswarm shows direction (e.g.
  high amounts push towards fraud).
- **Local view:** a waterfall plot answers "what pushed *this* transaction's score up or down?"
- **What SHAP is not:** causal. "The amount contributed +3 log-odds" means the model relied on the
  amount for this prediction, not that the amount caused fraud. Correlated features share credit in
  ways that depend on the model, so attributions should be read together.
- Why it matters in banking: model-risk management (e.g. SR 11-7 style validation), analyst
  investigation support, and adverse-action style reason codes.

Gain importance vs SHAP: gain is a model-internal, training-set statistic with no direction and is
biased towards features used in many splits. SHAP is computed per prediction, has a direction and is
consistent.

---

## 10. Limitations (be upfront about these)

1. **Synthetic data.** The patterns (night-time fraud, fraud-only new cards, uniformly jittered merchant
   locations) come from a simulator; real fraud is more adaptive and noisier.
2. **Only 999 customers.** Customer-level attributes can act as identifiers; grouped CV mitigates the
   evaluation risk, but the model has seen few distinct customer profiles.
3. **Concept drift.** Fraud patterns change as fraudsters adapt; the model needs monitoring and
   retraining. The fraud rate already shifts between the two periods.
4. **Label delay and noise.** Real labels arrive weeks later and some fraud is never reported.
5. **Probabilities are not calibrated** when class weighting is used; thresholds were set on
   validation data, but if calibrated probabilities are needed (e.g. expected-loss calculations),
   apply Platt/isotonic calibration.
6. **No merchant-level features** (merchant risk would need target encoding with careful out-of-fold
   handling).
7. **Fairness** was addressed only by excluding gender; age remains a feature and would need a
   formal fairness review.

---

## 11. How this could be deployed in a real financial institution

1. **Real-time scoring service.** The card authorisation flow calls a low-latency model service
   (tens of milliseconds budget). XGBoost inference is fast enough on CPU.
2. **Feature store.** Behavioural features (counts in last 1h/24h, running mean/std) are maintained
   incrementally per card in a low-latency store (e.g. Redis), using the same definitions as training
   to avoid training/serving skew. Here, `features_from_card_summary` is that single shared definition.
3. **Decision layer.** Model score + business rules (velocity limits, blocklists) → approve /
   step-up / decline / queue. Thresholds are configurable per segment.
4. **Case management.** High-risk alerts go to analysts with SHAP reason codes; analyst decisions and
   chargebacks feed back as labels.
5. **Monitoring.** Population stability (PSI) on inputs and scores, alert volumes, precision of
   reviewed alerts, recall estimated from chargebacks, latency.
6. **Retraining & governance.** Periodic retraining on a rolling window with out-of-time validation;
   champion/challenger testing; model documentation, independent validation and approval (model-risk
   management); audit logging of every decision.
7. **Privacy & security.** PII minimisation (no names/addresses as features), encryption, access
   controls; card numbers tokenised.

---

## 12. This project's results (all from `reports/`)

**Grouped 5-fold CV on a 30% stratified train sample (PR-AUC, mean ± std)**
| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.460 ± 0.021 | 0.935 |
| Random Forest (tuned) | 0.887 ± 0.015 | 0.992 |
| XGBoost (tuned) | 0.931 ± 0.010 | 0.998 |

**Out-of-time test set (Jun–Dec 2020), evaluated once**
| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression @0.5 | 0.452 | 0.932 | 0.834 | 0.243 |
| Random Forest @0.5 | 0.892 | 0.994 | 0.973 | 0.720 |
| XGBoost @0.5 | 0.921 | 0.998 | 0.977 | 0.746 |
| **XGBoost @0.0122 (chosen)** | 0.921 | 0.998 | **0.545** | **0.930** |

At the chosen threshold on test: 1,994 frauds caught, 151 missed, 1,662 false alerts among 553,574
legitimate transactions (0.30% of them).

**Talking points**
- *"Why is ROC-AUC 0.93 for a model that misses 76% of fraud at 0.5?"* With 550k negatives the
  false-positive rate stays tiny. PR-AUC (0.45) shows the real weakness, which is why PR-AUC is primary.
- *"Why is the threshold so low (0.012)?"* The model was trained without re-weighting
  (`scale_pos_weight=1`), so its probabilities sit near the true base rate (~0.6%). A score of 0.012 is
  already ~2× the base rate. The threshold is relative to the score distribution, not an absolute
  "confidence level".
- *"Did the precision target hold out-of-sample?"* Yes: 50% was targeted on train OOF, and the test
  set came out at 54.5%. Recall fell from 96.6% (OOF) to 93.0% (test), consistent with drift and the
  lower test fraud rate.
- *"Did tuning help?"* Barely: XGBoost +0.001 PR-AUC (within noise); Random Forest +0.018. I report that
  rather than overselling tuning.
- *"Is the result driven by the fraud-only-card artefact?"* Test PR-AUC 0.921 → 0.912 without unseen
  cards, → 0.917 without `card_prior_txn_count`. Small effect.
- *"What matters most to the model?"* By mean |SHAP|: amount, hour, card history length, category,
  amount relative to the card's average. `card_txn_count_1h` has the largest *gain* share (22.6%) but a
  small mean |SHAP|: decisive for burst transactions, irrelevant for the rest. It's a good example of
  why one importance metric is not enough.
- *"A bug you caught?"* pandas 3 stores timestamps in microseconds; my first conversion to seconds
  assumed nanoseconds, which made "transactions in last 24h" equal to the card's entire history.
  Summary statistics looked implausible, so I fixed the conversion and added a brute-force check.
  Lesson: sanity-check engineered features against a naive implementation.
