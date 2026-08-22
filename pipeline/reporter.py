"""Match rate calculation, exception report, and ground-truth scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pipeline.level1_match import Level1Result
from pipeline.level2_match import Level2Result

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExceptionRecord:
    record_id: str
    side: str
    reason_code: str
    agent_rationale: str
    confidence: float
    resolved_by: str


@dataclass
class ReconciliationReport:
    run_id: str
    match_rate_l1: float
    match_rate_l2: float
    l1_matched: int
    l1_total: int
    l2_matched: int
    l2_total: int
    exception_precision: float
    exception_recall: float
    exceptions: list[ExceptionRecord]
    stage_breakdown: dict[str, int]
    false_matches: list[str]
    false_exceptions: list[str]
    llm_call_count: int
    agent_fallback_count: int = 0
    metrics_detail: dict = field(default_factory=dict)


def _load_ground_truth() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_l1 = pd.read_csv(PROJECT_ROOT / "ground_truth_order_to_settlement.csv")
    gt_l2 = pd.read_csv(PROJECT_ROOT / "ground_truth_settlement_to_bank.csv")
    gt_exc = pd.read_csv(PROJECT_ROOT / "ground_truth_true_exceptions.csv")
    return gt_l1, gt_l2, gt_exc


def build_exception_report(
    level1: Level1Result,
    level2: Level2Result,
) -> list[ExceptionRecord]:
    exceptions: list[ExceptionRecord] = []

    for order_id in level1.unmatched_orders:
        exceptions.append(
            ExceptionRecord(
                record_id=order_id,
                side="ledger",
                reason_code="UNEXPLAINED",
                agent_rationale=(
                    f"Order {order_id} appears in orders_ledger but has no corresponding "
                    "payment row in settlement_report."
                ),
                confidence=0.0,
                resolved_by="level1_deterministic",
            )
        )

    for eid in level1.orphan_settlements:
        exceptions.append(
            ExceptionRecord(
                record_id=eid,
                side="settlement",
                reason_code="UNEXPLAINED",
                agent_rationale=f"Adjustment row {eid} has no source order.",
                confidence=0.0,
                resolved_by="level1_deterministic",
            )
        )

    for sid in level2.unresolved_batches:
        exceptions.append(
            ExceptionRecord(
                record_id=sid,
                side="settlement",
                reason_code="UNEXPLAINED",
                agent_rationale=(
                    f"Settlement batch {sid} never matched to any bank statement row."
                ),
                confidence=0.0,
                resolved_by="level2_unresolved",
            )
        )

    return exceptions


def compute_report(
    run_id: str,
    level1: Level1Result,
    level2: Level2Result,
    orders_df: pd.DataFrame,
) -> ReconciliationReport:
    gt_l1, gt_l2, gt_exc = _load_ground_truth()

    exceptions = build_exception_report(level1, level2)
    flagged_ids = {e.record_id for e in exceptions}
    true_exception_ids = set(gt_exc["record_id"].tolist())

    true_positives_exc = flagged_ids & true_exception_ids
    exception_precision = (
        len(true_positives_exc) / len(flagged_ids) if flagged_ids else 0.0
    )
    exception_recall = len(true_positives_exc) / len(true_exception_ids)

    # Level 1 match rate: unique orders in ground truth that we matched
    gt_orders = set(gt_l1["order_id"].unique())
    matched_orders = {m.order_id for m in level1.matches}
    l1_tp = len(gt_orders & matched_orders)
    l1_fn = len(gt_orders - matched_orders)
    match_rate_l1 = l1_tp / (l1_tp + l1_fn) if (l1_tp + l1_fn) else 0.0

    # Level 2 match rate: settlement batches in ground truth
    gt_batches = set(gt_l2["settlement_id"].unique())
    matched_batches = {m.settlement_id for m in level2.matches}
    for m in level2.matches:
        matched_batches.update(m.merged_with)
    l2_tp = len(gt_batches & matched_batches)
    l2_fn = len(gt_batches - matched_batches)
    match_rate_l2 = l2_tp / (l2_tp + l2_fn) if (l2_tp + l2_fn) else 0.0

    # False matches: matched records that are true exceptions
    false_matches = list(matched_batches & true_exception_ids)
    false_matches += [o for o in matched_orders if o in true_exception_ids]

    # False exceptions: resolvable challenges we failed to match
    gt_resolvable = pd.read_csv(PROJECT_ROOT / "ground_truth_resolvable_challenges.csv")
    resolvable_ids = set(gt_resolvable["record_id"].tolist())
    false_exceptions = list(resolvable_ids - matched_batches)

    stage_breakdown = dict(level2.stage_counts)
    stage_breakdown["level1_deterministic"] = len(level1.matches)

    return ReconciliationReport(
        run_id=run_id,
        match_rate_l1=round(match_rate_l1, 4),
        match_rate_l2=round(match_rate_l2, 4),
        l1_matched=l1_tp,
        l1_total=l1_tp + l1_fn,
        l2_matched=l2_tp,
        l2_total=l2_tp + l2_fn,
        exception_precision=round(exception_precision, 4),
        exception_recall=round(exception_recall, 4),
        exceptions=exceptions,
        stage_breakdown=stage_breakdown,
        false_matches=false_matches,
        false_exceptions=false_exceptions,
        llm_call_count=level2.llm_call_count,
        agent_fallback_count=level2.agent_fallback_count,
        metrics_detail={
            "true_exceptions_found": sorted(true_positives_exc),
            "false_exception_flags": sorted(flagged_ids - true_exception_ids),
            "missed_exceptions": sorted(true_exception_ids - flagged_ids),
        },
    )


def report_to_dict(report: ReconciliationReport) -> dict:
    return {
        "run_id": report.run_id,
        "match_rate_l1": report.match_rate_l1,
        "match_rate_l2": report.match_rate_l2,
        "l1_matched": report.l1_matched,
        "l1_total": report.l1_total,
        "l2_matched": report.l2_matched,
        "l2_total": report.l2_total,
        "exception_precision": report.exception_precision,
        "exception_recall": report.exception_recall,
        "exceptions": [
            {
                "record_id": e.record_id,
                "side": e.side,
                "reason_code": e.reason_code,
                "agent_rationale": e.agent_rationale,
                "confidence": e.confidence,
                "resolved_by": e.resolved_by,
            }
            for e in report.exceptions
        ],
        "stage_breakdown": report.stage_breakdown,
        "false_matches": report.false_matches,
        "false_exceptions": report.false_exceptions,
        "llm_call_count": report.llm_call_count,
        "agent_fallback_count": report.agent_fallback_count,
        "metrics_detail": report.metrics_detail,
    }
