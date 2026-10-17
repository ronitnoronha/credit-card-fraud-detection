# Data

The raw dataset is **not committed** (~500 MB of CSV). It is downloaded automatically.

- Source: Hugging Face dataset [`pointe77/credit-card-transaction`](https://huggingface.co/datasets/pointe77/credit-card-transaction)
- Files: `credit_card_transaction_train.csv` (1,296,675 rows), `credit_card_transaction_test.csv` (555,719 rows), 24 columns each.
- The data is synthetic (simulated cardholders, merchants and transactions).

## How it is loaded

`src/data_loader.py` calls `datasets.load_dataset("pointe77/credit-card-transaction")` on first use and
caches each split as Parquet:

```
data/raw/train.parquet
data/raw/test.parquet
data/processed/features.parquet   # engineered features (built by src/features.py)
```

All of `data/raw/` and `data/processed/` are git-ignored.

Offline alternative: put the two CSVs in `data/raw/` (any names; each file becomes one split; name them
`train.csv` / `test.csv`) and the loader will read them instead.

If the download fails, check network access to `huggingface.co`; set `HF_TOKEN` if you hit rate limits.

## Columns (as found in the data)

| Column | Type | Notes |
|---|---|---|
| `Unnamed: 0` | int | row index (dropped) |
| `trans_date_trans_time` | string datetime | transaction timestamp |
| `cc_num` | int | card number = customer key (grouping only) |
| `merchant` | string | 693 values, all prefixed `fraud_` (generator artefact) |
| `category` | string | 14 merchant categories |
| `amt` | float | amount in USD |
| `first`, `last`, `gender`, `street`, `city`, `state`, `zip`, `job`, `dob` | various | cardholder profile |
| `lat`, `long`, `city_pop` | numeric | cardholder location / city population |
| `trans_num` | string | unique transaction ID (dropped) |
| `unix_time` | int | shifted copy of the timestamp (dropped) |
| `merch_lat`, `merch_long`, `merch_zipcode` | numeric | merchant location (`merch_zipcode` 15% missing) |
| `is_fraud` | int | **target** (1 = fraud) |
