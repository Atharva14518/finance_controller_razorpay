"""
Synthetic reconciliation dataset generator — Razorpay AI Buildathon (Track 04: AI Finance Controller)

Produces three raw source tables (mirroring Razorpay's real settlement API schema),
a held-out ground truth mapping (NEVER fed to the matching pipeline, only used to score it),
and an ML-ready labeled candidate-pairs table for training a match/no-match classifier
(e.g. XGBoost) in the harder case where records don't share a clean join key.

Reproducible: seeded RNG.
"""
import random
import string
import datetime as dt
import difflib
import json

import pandas as pd
import numpy as np
from faker import Faker

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

OUT_DIR = "/mnt/user-data/outputs"

METHODS = ["card", "upi", "netbanking", "wallet"]
BANK_NAMES = ["HDFC", "ICICI", "KOTAK", "AXIS", "SBI"]
MDR_RATE = 0.02   # Razorpay's standard merchant discount rate
GST_RATE = 0.18   # GST on MDR
START_DATE = dt.date(2026, 7, 1)
N_ORDERS = 62


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rand_id(prefix, length=14):
    chars = string.ascii_letters + string.digits
    return prefix + "".join(random.choices(chars, k=length))


def rand_utr(length=16):
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def next_business_day(d, n):
    """Add n business days, skipping Sat/Sun (simplified vs. Razorpay's
    '2nd/4th Saturday + bank holiday' rule — noted as a simplification)."""
    while n > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


# --------------------------------------------------------------------------- #
# 1. base orders — the internal ledger / OMS side (ground truth origin)
# --------------------------------------------------------------------------- #
orders = []
for i in range(N_ORDERS):
    order_date = START_DATE + dt.timedelta(days=random.randint(0, 20))
    orders.append({
        "order_id": rand_id("order_"),
        "payment_id": rand_id("pay_"),
        "amount": random.randint(500, 50000) * 100,  # paise
        "currency": "INR",
        "method": random.choice(METHODS),
        "customer_email": fake.email(),
        "created_date": order_date.isoformat(),
    })
orders_df = pd.DataFrame(orders)

# pick special-case orders
refund_idx = set(random.sample(range(N_ORDERS), k=int(N_ORDERS * 0.15)))
never_settled_idx = set(
    random.sample([i for i in range(N_ORDERS) if i not in refund_idx], k=2)
)

# --------------------------------------------------------------------------- #
# 2. settlement report rows (mirrors Razorpay's real recon report fields)
# --------------------------------------------------------------------------- #
settlement_rows = []
gt_order_to_settlement = []

for i, o in orders_df.iterrows():
    if i in never_settled_idx:
        continue  # order exists but never settles -> ledger-side exception
    order_dt = dt.date.fromisoformat(o["created_date"])
    settle_date = next_business_day(order_dt, 2)  # Razorpay standard T+2
    fee = round(o["amount"] * MDR_RATE)
    tax = round(fee * GST_RATE)
    net = o["amount"] - fee - tax

    settlement_rows.append({
        "entity_id": o["payment_id"],
        "type": "payment",
        "amount": o["amount"],
        "fee": fee,
        "tax": tax,
        "net_amount": net,
        "order_id": o["order_id"],
        "payment_id": o["payment_id"],
        "method": o["method"],
        "settled_date": settle_date.isoformat(),
        "settlement_id": None,  # filled after batching below
    })
    gt_order_to_settlement.append({
        "order_id": o["order_id"], "entity_id": o["payment_id"], "relation": "payment"
    })

    if i in refund_idx:
        refund_amount = round(o["amount"] * random.uniform(0.2, 0.6))
        refund_date = next_business_day(settle_date, 1)
        refund_entity = rand_id("rfnd_")
        settlement_rows.append({
            "entity_id": refund_entity,
            "type": "refund",
            "amount": refund_amount,
            "fee": 0,
            "tax": 0,
            "net_amount": -refund_amount,
            "order_id": o["order_id"],
            "payment_id": o["payment_id"],
            "method": o["method"],
            "settled_date": refund_date.isoformat(),
            "settlement_id": None,
        })
        gt_order_to_settlement.append({
            "order_id": o["order_id"], "entity_id": refund_entity, "relation": "refund"
        })

# orphan settlement rows — adjustments with no order behind them
N_ORPHANS = 3
for _ in range(N_ORPHANS):
    settle_date = START_DATE + dt.timedelta(days=random.randint(2, 22))
    amt = random.randint(200, 5000) * 100
    settlement_rows.append({
        "entity_id": rand_id("adj_"),
        "type": "adjustment",
        "amount": amt,
        "fee": 0,
        "tax": 0,
        "net_amount": amt,
        "order_id": None,
        "payment_id": None,
        "method": None,
        "settled_date": settle_date.isoformat(),
        "settlement_id": None,
    })

settlement_df = pd.DataFrame(settlement_rows)

# batch settlement rows into daily settlement_id groups (Razorpay settles per business day)
date_to_sid = {d: rand_id("setl_", 12) for d in settlement_df["settled_date"].unique()}
settlement_df["settlement_id"] = settlement_df["settled_date"].map(date_to_sid)
sid_to_utr = {sid: rand_utr() for sid in settlement_df["settlement_id"].unique()}

# --------------------------------------------------------------------------- #
# 3. bank statement — one NEFT credit per settlement batch, with injected noise
# --------------------------------------------------------------------------- #
batch_totals = (
    settlement_df.groupby("settlement_id")
    .agg(net_total=("net_amount", "sum"), settled_date=("settled_date", "first"))
    .reset_index()
)
batch_totals["utr"] = batch_totals["settlement_id"].map(sid_to_utr)

batch_ids = list(batch_totals["settlement_id"])
random.shuffle(batch_ids)
rounding_batch   = batch_ids[0]
missing_utr_batch = batch_ids[1]
lag_batch        = batch_ids[2]
merge_batches    = batch_ids[3:5]   # two batches paid out as ONE bank credit
missing_batch    = batch_ids[5]     # a batch that never lands in the bank feed
noise_batches = {
    rounding_batch: "ROUNDING_NOISE",
    missing_utr_batch: "MISSING_UTR",
    lag_batch: "SETTLEMENT_LAG_EXTRA_DAY",
    merge_batches[0]: "MERGED_BATCH_PRIMARY",
    merge_batches[1]: "MERGED_BATCH_ABSORBED",
    missing_batch: "NEVER_IN_BANK_FEED",
}

bank_rows = []
gt_settlement_to_bank = []
skip_ids = {missing_batch, merge_batches[1]}

for _, row in batch_totals.iterrows():
    sid = row["settlement_id"]
    if sid in skip_ids:
        continue

    amount = row["net_total"]
    utr = row["utr"]
    bank_date = dt.date.fromisoformat(row["settled_date"])
    narration_utr = utr

    if sid == rounding_batch:
        amount += random.choice([-3, -2, -1, 1, 2, 3])          # paise-level rounding drift
    if sid == missing_utr_batch:
        narration_utr = ""                                       # UTR dropped from narration
    if sid == lag_batch:
        bank_date += dt.timedelta(days=1)                        # extra settlement lag
    if sid == merge_batches[0]:
        other_total = batch_totals.loc[
            batch_totals["settlement_id"] == merge_batches[1], "net_total"
        ].values[0]
        amount += other_total                                    # two batches, one bank line

    bank_name = random.choice(BANK_NAMES)
    narration = f"NEFT CR: {bank_name} {narration_utr} RAZORPAY SETTLEMENT".replace("  ", " ").strip()
    bank_row_id = rand_id("bnk_", 10)

    bank_rows.append({
        "bank_row_id": bank_row_id,
        "date": bank_date.isoformat(),
        "amount": amount,
        "narration": narration,
    })

    gt_settlement_to_bank.append({
        "settlement_id": sid, "bank_row_id": bank_row_id,
        "note": noise_batches.get(sid, "clean")
    })
    if sid == merge_batches[0]:
        gt_settlement_to_bank.append({
            "settlement_id": merge_batches[1], "bank_row_id": bank_row_id,
            "note": "MERGED_BATCH_ABSORBED"
        })

bank_df = pd.DataFrame(bank_rows)

# --------------------------------------------------------------------------- #
# 4a. TRUE exceptions — records that genuinely have no counterpart. A correct
#     pipeline should report exactly these, with reason UNEXPLAINED, and
#     should NOT force a match for any of them.
# --------------------------------------------------------------------------- #
true_exceptions = []
for i in never_settled_idx:
    true_exceptions.append({
        "side": "ledger", "record_id": orders_df.loc[i, "order_id"],
        "true_reason": "UNEXPLAINED", "detail": "order never appears in settlement report"
    })
for _, r in settlement_df[settlement_df["type"] == "adjustment"].iterrows():
    true_exceptions.append({
        "side": "settlement", "record_id": r["entity_id"],
        "true_reason": "UNEXPLAINED", "detail": "adjustment row with no source order"
    })
true_exceptions.append({
    "side": "settlement", "record_id": missing_batch,
    "true_reason": "UNEXPLAINED", "detail": "settlement batch never lands in bank feed"
})
true_exceptions_df = pd.DataFrame(true_exceptions)

# --------------------------------------------------------------------------- #
# 4b. RESOLVABLE challenges — these DO have a true match, but a naive pipeline
#     will wrongly report them as exceptions unless it handles the noise
#     correctly. Use this file to check your match-rate is counting these as
#     matched, not to check your exception list contains them.
# --------------------------------------------------------------------------- #
resolvable_challenges = [
    {"side": "bank", "record_id": rounding_batch, "challenge_type": "ROUNDING",
     "detail": "bank amount differs from settlement net by a few paise — needs a tolerance band, not exact match"},
    {"side": "bank", "record_id": missing_utr_batch, "challenge_type": "MISSING_UTR",
     "detail": "UTR absent from narration — must fall back to amount+date matching"},
    {"side": "bank", "record_id": lag_batch, "challenge_type": "SETTLEMENT_LAG",
     "detail": "lands one business day after the standard T+2 window — needs a wider date tolerance"},
    {"side": "bank", "record_id": merge_batches[0], "challenge_type": "MERGED_BATCH",
     "detail": "one bank credit equals the sum of TWO settlement batches — needs sum-of-N matching, not 1:1"},
]
resolvable_challenges_df = pd.DataFrame(resolvable_challenges)

# --------------------------------------------------------------------------- #
# 5. ML-ready labeled candidate pairs (harder version: NO shared ID field,
#    only amount / date / method / free-text description — the situation you
#    actually face when a source system doesn't expose clean join keys)
# --------------------------------------------------------------------------- #
side_a = orders_df.copy()
side_a["date"] = pd.to_datetime(side_a["created_date"])
side_a["description"] = side_a.apply(
    lambda r: f"Order {r['order_id'][-6:]} via {r['method']}", axis=1
)

payment_settlements = settlement_df[settlement_df["type"] == "payment"].copy()
side_b = payment_settlements.merge(
    orders_df[["order_id", "payment_id"]], on=["order_id", "payment_id"], how="left"
)
side_b["date"] = pd.to_datetime(side_b["settled_date"])
side_b["description"] = side_b.apply(
    lambda r: f"Settlement {r['entity_id'][-6:]} net payout", axis=1
)

pairs = []
b_pool = side_b.to_dict("records")

for _, a in side_a.iterrows():
    true_b = side_b[side_b["order_id"] == a["order_id"]]
    if true_b.empty:
        continue
    true_b = true_b.iloc[0]

    def make_row(a, b, label):
        amount_diff = abs(a["amount"] - b["net_amount"])
        date_diff_days = abs((a["date"] - b["date"]).days)
        return {
            "order_id": a["order_id"],
            "candidate_entity_id": b["entity_id"],
            "amount_a": a["amount"],
            "amount_b": b["net_amount"],
            "amount_diff": amount_diff,
            "amount_diff_pct": round(amount_diff / a["amount"], 4),
            "date_diff_days": date_diff_days,
            "method_match": int(a["method"] == b["method"]),
            "desc_similarity": round(
                difflib.SequenceMatcher(None, a["description"], b["description"]).ratio(), 3
            ),
            "label": label,
        }

    pairs.append(make_row(a, true_b, 1))

    # hard negatives: other settlement rows within a plausible amount/date window
    decoys = [
        b for b in b_pool
        if b["order_id"] != a["order_id"]
        and abs(b["net_amount"] - a["amount"]) < a["amount"] * 0.4
        and abs((pd.to_datetime(b["date"]) - a["date"]).days) <= 7
    ]
    random.shuffle(decoys)
    for b in decoys[:3]:
        pairs.append(make_row(a, {**b, "date": pd.to_datetime(b["date"])}, 0))

pairs_df = pd.DataFrame(pairs)

# --------------------------------------------------------------------------- #
# 6. write everything out
# --------------------------------------------------------------------------- #
orders_df.drop(columns=[]).to_csv(f"{OUT_DIR}/orders_ledger.csv", index=False)
settlement_df.to_csv(f"{OUT_DIR}/settlement_report.csv", index=False)
bank_df.to_csv(f"{OUT_DIR}/bank_statement.csv", index=False)
true_exceptions_df.to_csv(f"{OUT_DIR}/ground_truth_true_exceptions.csv", index=False)
resolvable_challenges_df.to_csv(f"{OUT_DIR}/ground_truth_resolvable_challenges.csv", index=False)
pairs_df.to_csv(f"{OUT_DIR}/labeled_pairs_for_training.csv", index=False)

pd.DataFrame(gt_order_to_settlement).to_csv(f"{OUT_DIR}/ground_truth_order_to_settlement.csv", index=False)
pd.DataFrame(gt_settlement_to_bank).to_csv(f"{OUT_DIR}/ground_truth_settlement_to_bank.csv", index=False)

print("orders_df:", orders_df.shape)
print("settlement_df:", settlement_df.shape)
print("bank_df:", bank_df.shape)
print("true_exceptions_df:", true_exceptions_df.shape)
print("resolvable_challenges_df:", resolvable_challenges_df.shape)
print("pairs_df:", pairs_df.shape, "| positives:", pairs_df["label"].sum(), "| negatives:", (pairs_df["label"]==0).sum())
print("\nnoise batches:", json.dumps(noise_batches, indent=2))
