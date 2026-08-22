"""Orchestrate full reconciliation pipeline run."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from pipeline.agent import AgentAdjudicator
from pipeline.audit import AuditLogger
from pipeline.ingest import load_all
from pipeline.level1_match import run_level1
from pipeline.level2_match import apply_ml_and_agent, run_level2_deterministic_and_rules
from pipeline.ml_scorer import MLScorer
from pipeline.reporter import ReconciliationReport, compute_report, report_to_dict

# Load .env so GROQ_API_KEY is available when running via uvicorn
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# The ROUNDING case — skipped by rules in demo mode so it hits the LLM agent
DEMO_AGENT_SID = "setl_4OaGzbSHrFRA"


@dataclass
class RunResult:
    run_id: str
    report: ReconciliationReport
    level1_matches: list
    level2_matches: list
    audit: AuditLogger
    orders_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    settlement_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bank_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    demo_agent: bool = False


def run_reconciliation(run_id: str | None = None, demo_agent: bool = False) -> RunResult:
    run_id = run_id or str(uuid.uuid4())[:8]
    audit = AuditLogger(run_id)

    orders_df, settlement_df, bank_df = load_all()

    level1 = run_level1(orders_df, settlement_df, audit)

    skip_sid = DEMO_AGENT_SID if demo_agent else None
    level2 = run_level2_deterministic_and_rules(settlement_df, bank_df, audit, demo_agent_skip_sid=skip_sid)

    ml_scorer = MLScorer()
    agent = AgentAdjudicator(settlement_df, bank_df)
    level2 = apply_ml_and_agent(level2, settlement_df, bank_df, audit, ml_scorer, agent, demo_agent=demo_agent)

    report = compute_report(run_id, level1, level2, orders_df)

    return RunResult(
        run_id=run_id,
        report=report,
        level1_matches=level1.matches,
        level2_matches=level2.matches,
        audit=audit,
        orders_df=orders_df,
        settlement_df=settlement_df,
        bank_df=bank_df,
        demo_agent=demo_agent,
    )


def run_and_print() -> RunResult:
    result = run_reconciliation()
    d = report_to_dict(result.report)
    print(f"\n=== Reconciliation Run {result.run_id} ===")
    print(f"Level 1 Match Rate: {d['l1_matched']}/{d['l1_total']} ({d['match_rate_l1']:.1%})")
    print(f"Level 2 Match Rate: {d['l2_matched']}/{d['l2_total']} ({d['match_rate_l2']:.1%})")
    print(f"Exception Precision: {d['exception_precision']:.1%}")
    print(f"Exception Recall: {d['exception_recall']:.1%}")
    print(f"LLM Calls: {d['llm_call_count']}")
    print(f"Stage Breakdown: {d['stage_breakdown']}")
    print(f"False Matches: {d['false_matches']}")
    print(f"False Exceptions (missed resolvable): {d['false_exceptions']}")
    print(f"Exceptions ({len(d['exceptions'])}):")
    for e in d["exceptions"]:
        print(f"  - {e['record_id']} ({e['side']}): {e['reason_code']}")
    return result


if __name__ == "__main__":
    run_and_print()
