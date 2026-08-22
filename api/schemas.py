"""Pydantic schemas for API request/response."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ExceptionItem(BaseModel):
    record_id: str
    side: str
    reason_code: str
    agent_rationale: str
    confidence: float
    resolved_by: str


class RunResponse(BaseModel):
    run_id: str
    status: str
    demo_agent: bool = False


class ResultsResponse(BaseModel):
    run_id: str
    match_rate_l1: float
    match_rate_l2: float
    l1_matched: int
    l1_total: int
    l2_matched: int
    l2_total: int
    exception_precision: float
    exception_recall: float
    stage_breakdown: dict[str, int]
    false_matches: list[str]
    false_exceptions: list[str]
    llm_call_count: int
    agent_fallback_count: int = 0
    metrics_detail: dict
    demo_agent: bool = False


class ExceptionsResponse(BaseModel):
    run_id: str
    exceptions: list[ExceptionItem]
    true_exception_ids: list[str]


class AuditEntry(BaseModel):
    id: int
    run_id: str
    timestamp: str
    record_id: str
    side: str
    matched_to: str | None
    decision: str
    resolved_by: str
    confidence: float | None
    reason_code: str | None
    rationale: str | None
    shap_json: str | None
    tool_trace_json: str | None = None


class AuditResponse(BaseModel):
    run_id: str
    entries: list[AuditEntry]
    total: int


# --- Drill-down ---

class SettlementBatchDetail(BaseModel):
    settlement_id: str
    net_total: int
    settled_date: str
    entity_ids: list[str]
    entity_types: list[str]


class BankRowDetail(BaseModel):
    bank_row_id: str
    date: str
    amount: int
    narration: str
    utr: str


class DrillDownResponse(BaseModel):
    record_id: str
    settlement: SettlementBatchDetail | None
    bank_row: BankRowDetail | None
    audit_entry: AuditEntry | None
    merged_with: list[SettlementBatchDetail] = []


# --- Evaluation ---

class EvalRow(BaseModel):
    record_id: str
    category: str          # "true_exception" or "resolvable_challenge"
    challenge_type: str    # UNEXPLAINED, ROUNDING, MISSING_UTR, SETTLEMENT_LAG, MERGED_BATCH
    expected: str          # EXCEPTION or MATCH
    pipeline_decision: str  # EXCEPTION or MATCH
    resolved_by: str
    correct: bool


class EvaluationResponse(BaseModel):
    run_id: str
    rows: list[EvalRow]
    exception_precision: float
    exception_recall: float
    match_rate_l1: float
    match_rate_l2: float
    false_matches: list[str]
    false_exceptions: list[str]
    llm_call_count: int
    agent_fallback_count: int = 0
    demo_agent: bool
