# Reconciliation dataset — data dictionary

Generated for Razorpay AI Buildathon, Track 04 (AI Finance Controller). Synthetic,
seeded (seed=42, reproducible), field names mirror Razorpay's real settlement API.
`generate_recon_data.py` is included — rerun it any time to regenerate or to change
`N_ORDERS` / noise ratios.

## The problem structure — two match levels

**Level 1 (order ↔ settlement row):** joins cleanly on `order_id`/`payment_id` — both
systems share the ID. The challenge here is *not* fuzzy matching, it's correctly
handling refunds (one order → two settlement rows) and orphans (settlement rows with
no order at all).

**Level 2 (settlement batch ↔ bank statement line):** the hard, realistic problem.
The bank only gives you a narration string, a date, and an amount — no clean foreign
key. This is where tolerance windows, UTR parsing, and sum-of-batch matching matter.

## Files

| File | Rows | What it is |
|---|---|---|
| `orders_ledger.csv` | 62 | Your internal ledger/OMS. `order_id`, `payment_id`, `amount` (paise), `method`, `created_date`. |
| `settlement_report.csv` | 72 | Razorpay's settlement recon report. `type` is `payment`/`refund`/`adjustment`. `net_amount` = `amount - fee - tax`. Batched under `settlement_id`. |
| `bank_statement.csv` | 14 | One NEFT credit per settlement batch (minus the deliberately-missing one). `narration` contains the UTR — except where noted below. |
| `ground_truth_order_to_settlement.csv` | — | **Held out.** Level-1 true mapping. Use only to score, never feed to your pipeline. |
| `ground_truth_settlement_to_bank.csv` | — | **Held out.** Level-2 true mapping, including which two batches got merged. |
| `ground_truth_true_exceptions.csv` | 6 | Records with **no real counterpart**. Your exception report should contain exactly these, reason `UNEXPLAINED`. |
| `ground_truth_resolvable_challenges.csv` | 4 | Records that **do** have a match but are hard to find. These should end up in your match rate, not your exception list — see below. |
| `labeled_pairs_for_training.csv` | 225 (60 pos / 165 neg) | ML-ready. Candidate order↔settlement pairs with engineered features and a binary label, for the harder no-shared-ID scenario. |

## The two ground-truth files are different tests

This is the part most submissions get wrong, so it's worth being explicit:

- **`ground_truth_true_exceptions.csv`** (6 records: 2 orders that never settle, 3
  orphan adjustment rows, 1 settlement batch that never hits the bank feed) — a
  *correct* pipeline reports these as exceptions. If your pipeline force-matches any
  of these, that's a false match, which is worse than an honest exception.
- **`ground_truth_resolvable_challenges.csv`** (4 records) — a *correct* pipeline
  still finds these:
  - `ROUNDING` — bank amount off by a few paise from settlement net → needs a
    tolerance band, not exact-equality matching.
  - `MISSING_UTR` — UTR dropped from the narration → needs a fallback to
    amount+date matching.
  - `SETTLEMENT_LAG` — lands one business day after the standard T+2 window →
    needs a wider date tolerance, not a fixed diff.
  - `MERGED_BATCH` — one bank credit equals the sum of *two* settlement batches →
    needs sum-of-N candidate matching, not strict 1:1.

  If your pipeline reports any of these four as exceptions, that's a false
  negative — the fix is almost always "widen tolerance" or "consider batch sums,"
  not "give up and flag it."

Report both numbers separately in your pitch: **match rate** (did you resolve the
resolvable ones) and **exception precision** (did your exception list match the true
6, no more, no less). That combination is exactly what the track's bar asks for.

## Using `labeled_pairs_for_training.csv`

Features: `amount_diff`, `amount_diff_pct`, `date_diff_days`, `method_match`,
`desc_similarity`. Label: `1` = true match, `0` = hard negative (a plausible but
wrong candidate within a 7-day/40%-amount window — not a random row, so the
classifier has to actually learn something).

```python
import pandas as pd
from xgboost import XGBClassifier
import shap

df = pd.read_csv("labeled_pairs_for_training.csv")
X = df[["amount_diff", "amount_diff_pct", "date_diff_days", "method_match", "desc_similarity"]]
y = df["label"]

model = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss")
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
# use this on your pipeline's borderline/low-confidence matches at inference time —
# not on every record, it's the explanation for cases you're already unsure about
```

225 rows is illustrative-scale, enough to demo a working confidence-scoring layer
on top of your rules — not enough to claim a rigorously validated model. Say that
plainly in your pitch if you use it; a small honestly-labeled model beats an
oversold one.

## Known simplification

Business-day math skips Saturday and Sunday only. Razorpay's real rule also
excludes the 2nd/4th Saturday and bank holidays — not modeled here, mention it as a
scoping note if asked.
