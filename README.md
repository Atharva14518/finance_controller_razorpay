# Reconciliation Agent — Razorpay AI Buildathon (Track 04)

Multi-source reconciliation pipeline with **measured match rate** and **honest exception reporting**. One LLM-backed agent, invoked only on gray-zone cases the rules engine cannot resolve.

## Results (on synthetic dataset)

| Metric | Score |
|---|---|
| Level 1 match rate | **60/60 orders** (100%) |
| Level 2 match rate | **15/15 settlement batches** (100%) |
| Exception precision | **6/6** (100%) |
| Exception recall | **6/6** (100%) |
| False matches | **0** |
| LLM calls | **0** (all resolvable via deterministic + rules) |

## Architecture

```
orders_ledger ──┐
                ├── Level 1 (key join) ──► Order ↔ Settlement
settlement ─────┘                              │
                                               ▼
bank_statement ────── Level 2 cascade ──► Settlement Batch ↔ Bank
                         │
                         ├─ 2a: UTR + exact amount + date
                         ├─ 2b: Rules (rounding, missing UTR, lag, merged batch)
                         ├─ 2c: XGBoost + SHAP (gray zone)
                         └─ 2d: LangChain/Groq agent (borderline only)
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS: XGBoost needs OpenMP
brew install libomp

# Run pipeline
python -m pipeline.runner

# Run tests against ground truth
pytest tests/ -v

# Start API + dashboard
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000
```

### LLM agent (optional)

Set `GROQ_API_KEY` to enable the gray-zone agent:

```bash
export GROQ_API_KEY=your_key
export GROQ_MODEL=moonshotai/kimi-k2-instruct  # optional
```

Without the key, the pipeline uses deterministic fallback for borderline cases.

## Project structure

```
razorpay/
├── data/                    # symlinks to source CSVs
├── pipeline/
│   ├── ingest.py            # load CSVs
│   ├── level1_match.py      # order ↔ settlement
│   ├── level2_match.py      # settlement ↔ bank (4 stages)
│   ├── ml_scorer.py         # XGBoost + SHAP
│   ├── agent.py             # LangChain agent (gray zone)
│   ├── reporter.py          # metrics + exceptions
│   └── runner.py            # orchestration
├── api/                     # FastAPI backend
├── dashboard/               # HTML dashboard
├── audit/audit_log.db       # SQLite audit trail
└── tests/test_pipeline.py   # ground-truth scoring
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/run` | Trigger full reconciliation |
| GET | `/results/{run_id}` | Match rates + stage breakdown |
| GET | `/exceptions/{run_id}` | Exception report |
| GET | `/audit/{run_id}` | Audit log (filterable) |
| GET | `/health` | Liveness check |

## Pitch demo script

1. **MERGED_BATCH** (`setl_rsWw6SEMCiay`): show sum-of-two-batches rule in audit log
2. **True exception** (`adj_1iCdzzGypueBxH`): orphan adjustment flagged UNEXPLAINED
3. **Stage breakdown**: "LLM touched 0 records — rules resolved all 4 injected challenges"

## Known scoping notes

1. **Business-day math:** Skips Saturday/Sunday only. Real Razorpay rules also exclude 2nd/4th Saturday and RBI bank holidays.
2. **ML dataset:** 225 rows is illustrative-scale. XGBoost is a confidence signal, not a validated production model.
3. **Merged batch dates:** Absorbed batch may settle on a different day than the bank credit; matching uses sum-of-N with at least one batch anchoring near the bank date.
4. **LLM constraint:** Agent tools only expose structured records. Abstains with `INSUFFICIENT_DATA` when retrieval confidence is low.

## Ground truth (held out — scoring only)

Never fed to the pipeline:

- `ground_truth_order_to_settlement.csv`
- `ground_truth_settlement_to_bank.csv`
- `ground_truth_true_exceptions.csv` (6 records)
- `ground_truth_resolvable_challenges.csv` (4 records)
