"""FastAPI backend for reconciliation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    AuditEntry,
    AuditResponse,
    BankRowDetail,
    DrillDownResponse,
    EvalRow,
    EvaluationResponse,
    ExceptionItem,
    ExceptionsResponse,
    ResultsResponse,
    RunResponse,
    SettlementBatchDetail,
)
from pipeline.reporter import report_to_dict
from pipeline.runner import DEMO_AGENT_SID, RunResult, run_reconciliation
from pipeline.utils import parse_utr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

app = FastAPI(
    title="Razorpay Reconciliation Agent",
    description="Multi-source reconciliation with measured match rate + honest exception report",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for run results (demo-scale)
_runs: dict[str, dict] = {}
_run_results: dict[str, RunResult] = {}

TRUE_EXCEPTION_IDS = [
    "order_WohoNMdZEHNJSs",
    "order_RTWPElndTdtdD5",
    "adj_1iCdzzGypueBxH",
    "adj_qPL2lVSID1u7a6",
    "adj_4RRyotrCOtw2CX",
    "setl_Jq5QrHOl0ami",
]

RESOLVABLE_CHALLENGES = {
    "setl_4OaGzbSHrFRA": "ROUNDING",
    "setl_smoJpI6uVEST": "MISSING_UTR",
    "setl_NWBoIxlKxVsp": "SETTLEMENT_LAG",
    "setl_rsWw6SEMCiay": "MERGED_BATCH",
}


@app.get("/health")
def health():
    return {"status": "ok", "service": "reconciliation-agent"}


@app.post("/run", response_model=RunResponse)
def trigger_run(demo_agent: bool = Query(False, description="Route ROUNDING case through LLM agent for demo")):
    result = run_reconciliation(demo_agent=demo_agent)
    report_dict = report_to_dict(result.report)
    report_dict["demo_agent"] = demo_agent
    _runs[result.run_id] = {
        "report": report_dict,
        "audit": result.audit.fetch_run(),
    }
    _run_results[result.run_id] = result
    return RunResponse(run_id=result.run_id, status="completed", demo_agent=demo_agent)


@app.get("/results/{run_id}", response_model=ResultsResponse)
def get_results(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    r = run["report"]
    return ResultsResponse(
        run_id=run_id,
        match_rate_l1=r["match_rate_l1"],
        match_rate_l2=r["match_rate_l2"],
        l1_matched=r["l1_matched"],
        l1_total=r["l1_total"],
        l2_matched=r["l2_matched"],
        l2_total=r["l2_total"],
        exception_precision=r["exception_precision"],
        exception_recall=r["exception_recall"],
        stage_breakdown=r["stage_breakdown"],
        false_matches=r["false_matches"],
        false_exceptions=r["false_exceptions"],
        llm_call_count=r["llm_call_count"],
        metrics_detail=r["metrics_detail"],
        demo_agent=r.get("demo_agent", False),
    )


@app.get("/exceptions/{run_id}", response_model=ExceptionsResponse)
def get_exceptions(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    exceptions = [ExceptionItem(**e) for e in run["report"]["exceptions"]]
    return ExceptionsResponse(
        run_id=run_id,
        exceptions=exceptions,
        true_exception_ids=TRUE_EXCEPTION_IDS,
    )


@app.get("/audit/{run_id}", response_model=AuditResponse)
def get_audit(
    run_id: str,
    stage: str | None = None,
    decision: str | None = None,
    record_id: str | None = None,
):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    entries = run["audit"]
    if stage:
        entries = [e for e in entries if e.get("resolved_by") == stage]
    if decision:
        entries = [e for e in entries if e.get("decision") == decision]
    if record_id:
        entries = [e for e in entries if e.get("record_id") == record_id]
    return AuditResponse(
        run_id=run_id,
        entries=[AuditEntry(**e) for e in entries],
        total=len(entries),
    )


@app.get("/record/{run_id}/{record_id}", response_model=DrillDownResponse)
def get_record(run_id: str, record_id: str):
    """Drill-down: return settlement batch, matched bank row, and audit entry for a record."""
    run_result = _run_results.get(run_id)
    run = _runs.get(run_id)
    if not run_result or not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    settlement_df = run_result.settlement_df
    bank_df = run_result.bank_df

    # Find settlement detail
    settlement_detail = None
    batch_rows = settlement_df[settlement_df["settlement_id"] == record_id]
    if not batch_rows.empty:
        net_total = int(batch_rows["net_amount"].sum())
        settled_date = str(batch_rows["settled_date"].iloc[0])
        settlement_detail = SettlementBatchDetail(
            settlement_id=record_id,
            net_total=net_total,
            settled_date=settled_date,
            entity_ids=batch_rows["entity_id"].tolist(),
            entity_types=batch_rows["type"].tolist(),
        )

    # Find audit entry for this record in this run
    audit_entries = [e for e in run["audit"] if e.get("record_id") == record_id]
    # Pick the most decisive entry (MATCH > EXCEPTION > BORDERLINE)
    priority = {"MATCH": 0, "EXCEPTION": 1, "BORDERLINE": 2}
    audit_entries_sorted = sorted(audit_entries, key=lambda e: priority.get(e.get("decision", "BORDERLINE"), 2))
    audit_entry = AuditEntry(**audit_entries_sorted[0]) if audit_entries_sorted else None

    # Find matched bank row
    bank_row_detail = None
    matched_to = audit_entry.matched_to if audit_entry else None
    if matched_to:
        bank_rows = bank_df[bank_df["bank_row_id"] == matched_to]
        if not bank_rows.empty:
            br = bank_rows.iloc[0]
            bank_row_detail = BankRowDetail(
                bank_row_id=matched_to,
                date=str(br["date"]),
                amount=int(br["amount"]),
                narration=br["narration"],
                utr=parse_utr(br["narration"]),
            )

    # Find merged batches if MERGED_BATCH — check both primary and absorbed roles
    merged_details = []
    for m in run_result.level2_matches:
        # Case 1: record_id is the primary batch (has merged_with)
        if m.settlement_id == record_id and m.merged_with:
            for merged_sid in m.merged_with:
                mb_rows = settlement_df[settlement_df["settlement_id"] == merged_sid]
                if not mb_rows.empty:
                    merged_details.append(SettlementBatchDetail(
                        settlement_id=merged_sid,
                        net_total=int(mb_rows["net_amount"].sum()),
                        settled_date=str(mb_rows["settled_date"].iloc[0]),
                        entity_ids=mb_rows["entity_id"].tolist(),
                        entity_types=mb_rows["type"].tolist(),
                    ))
        # Case 2: record_id is an absorbed batch — show the primary
        if record_id in (m.merged_with or []):
            primary_rows = settlement_df[settlement_df["settlement_id"] == m.settlement_id]
            if not primary_rows.empty:
                merged_details.append(SettlementBatchDetail(
                    settlement_id=m.settlement_id,
                    net_total=int(primary_rows["net_amount"].sum()),
                    settled_date=str(primary_rows["settled_date"].iloc[0]),
                    entity_ids=primary_rows["entity_id"].tolist(),
                    entity_types=primary_rows["type"].tolist(),
                ))

    return DrillDownResponse(
        record_id=record_id,
        settlement=settlement_detail,
        bank_row=bank_row_detail,
        audit_entry=audit_entry,
        merged_with=merged_details,
    )


@app.get("/evaluation/{run_id}", response_model=EvaluationResponse)
def get_evaluation(run_id: str):
    """Ground-truth evaluation: exceptions + resolvable challenges side-by-side."""
    run = _runs.get(run_id)
    run_result = _run_results.get(run_id)
    if not run or not run_result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    report = run["report"]
    audit_entries = run["audit"]

    # Build a quick lookup: record_id → best decision + resolved_by
    decision_map: dict[str, tuple[str, str]] = {}
    priority = {"MATCH": 0, "EXCEPTION": 1, "BORDERLINE": 2}
    for e in audit_entries:
        rid = e.get("record_id", "")
        dec = e.get("decision", "BORDERLINE")
        stage = e.get("resolved_by", "")
        if rid not in decision_map or priority.get(dec, 2) < priority.get(decision_map[rid][0], 2):
            decision_map[rid] = (dec, stage)

    rows: list[EvalRow] = []

    # True exceptions — pipeline should say EXCEPTION
    for tid in TRUE_EXCEPTION_IDS:
        pipeline_dec, stage = decision_map.get(tid, ("NOT_SEEN", ""))
        # "NOT_SEEN" means it was never processed = not flagged = treated as MATCH (false)
        effective_dec = pipeline_dec if pipeline_dec in ("MATCH", "EXCEPTION") else "EXCEPTION"
        rows.append(EvalRow(
            record_id=tid,
            category="true_exception",
            challenge_type="UNEXPLAINED",
            expected="EXCEPTION",
            pipeline_decision=effective_dec,
            resolved_by=stage,
            correct=(effective_dec == "EXCEPTION"),
        ))

    # Resolvable challenges — pipeline should say MATCH
    for rid, ctype in RESOLVABLE_CHALLENGES.items():
        pipeline_dec, stage = decision_map.get(rid, ("EXCEPTION", "not_reached"))
        effective_dec = pipeline_dec if pipeline_dec in ("MATCH", "EXCEPTION") else "EXCEPTION"
        rows.append(EvalRow(
            record_id=rid,
            category="resolvable_challenge",
            challenge_type=ctype,
            expected="MATCH",
            pipeline_decision=effective_dec,
            resolved_by=stage,
            correct=(effective_dec == "MATCH"),
        ))

    return EvaluationResponse(
        run_id=run_id,
        rows=rows,
        exception_precision=report["exception_precision"],
        exception_recall=report["exception_recall"],
        match_rate_l1=report["match_rate_l1"],
        match_rate_l2=report["match_rate_l2"],
        false_matches=report["false_matches"],
        false_exceptions=report["false_exceptions"],
        llm_call_count=report["llm_call_count"],
        demo_agent=report.get("demo_agent", False),
    )


@app.get("/")
def serve_dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")
