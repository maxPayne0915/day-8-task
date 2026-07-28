# Message Delivery Failure — Student A

Predicting whether an SMS/message will fail to deliver, using a synthetic 30,000-row message log (message metadata, network provider, timing, cost, delivery outcome).

## Why this matters

Message failures cost money (resent messages, refunds) and hurt trust in whatever product is sending them. If we can flag likely failures *before* sending — or at least understand which providers/times/campaigns are risky — that's actionable for routing decisions and cost control.

## The data

30,300 raw rows, one message per row. It was messy in realistic ways:
- `recipient_id` and `network_provider` had missing values (~250–580 rows each)
- ~300 exact duplicate rows
- `message_parts` had implausible outliers (up to 90 parts on a message that should realistically be 1–3)
- `cost` had negative values and a few extreme outliers (one single message at 4,141.50 vs. a normal range of ~20–100)
- `sent_at` needed parsing into real datetimes

All of this is documented and cleaned in `day2_cleaning.ipynb`.

## Approach

| Notebook | What it does |
|---|---|
| `day1_explore.ipynb` | First look: shape, dtypes, `describe()`, baseline failure rate (~8.56%) |
| `day2_cleaning.ipynb` | Fixes missing values, duplicates, outliers in `message_parts`/`cost`, parses `sent_at` |
| `day3_eda.ipynb` | Groupby analysis — which provider, hour, campaign, account drives failures |
| `day4_features.ipynb` | Builds `hour`, `is_weekend`, one-hot `network_provider`, `parts_bucket`; drops leaky columns (`delivered_at`, `error_code`) |
| `day5-7_model.ipynb` | Train/test split, dummy baseline, logistic regression, and honest evaluation (precision/recall/F1/AUC, cross-validation) |
| `sql_practice.ipynb` | Re-answers the Day 3 EDA questions in SQL (SQLite), plus a window-function ranking of providers |

## Main finding

**Provider and time-of-day matter a lot; the baseline logistic regression model doesn't yet act on it.**

- `Provider_D` fails ~14.7% of the time vs. ~6.7% for the best provider (`Provider_A`) — confirmed in both pandas (Day 3) and SQL.
- Failure rate roughly doubles in the evening window (18:00–22:00) compared to the rest of the day.
- Despite that signal existing in the data, a first-pass logistic regression matched the "always predict delivered" dummy baseline exactly — 91.5% accuracy, but **0 out of 503 actual failures caught** in the test set. AUC came out to ~0.60 (5-fold cross-validated), meaning there's real but weak signal that a 0.5 decision threshold doesn't surface.

**Honest takeaway: accuracy alone was misleading here.** On an imbalanced target (~8.5% failure rate), a model can look "good" while being functionally useless for the actual goal (catching failures before they happen).

## Limits

- Trained on **synthetic data** — patterns may not reflect a real messaging provider's behavior.
- The model was evaluated with a default 0.5 threshold and no class weighting — it never had a real chance to predict the minority class. This isn't evidence that the features are useless (AUC ~0.60 says otherwise), just that this specific setup didn't use them well.
- No hyperparameter tuning, no comparison against other model types (tree-based models, etc.) yet.

## What I'd try next with more time

- `class_weight="balanced"` (or similar) in logistic regression, and re-check recall on the failure class
- Try a tree-based model (e.g. `HistGradientBoostingClassifier`) which handles imbalance and non-linearity better out of the box
- Add the `campaign_id`/`network_provider` interaction found in Day 3 EDA (e.g. `Provider_D` × campaign 12 → 30% failure) as an explicit feature
- Threshold tuning instead of the default 0.5, based on the actual cost of a missed failure vs. a false alarm

## How to run

1. Place the raw CSV at `data/A_message_delivery.csv` (not included in this repo — see `.gitignore`).
2. Run notebooks in order: `day1_explore` → `day2_cleaning` (produces `clean.csv`) → `day3_eda` → `day4_features` (produces the feature table) → `day5-7_model`.
3. `sql_practice.ipynb` can be run any time after `day2_cleaning.ipynb`, since it only needs `clean.csv`.

All notebooks run top-to-bottom without errors.
