#!/usr/bin/env python3
"""Print the three pitch-demo moments with audit evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.runner import run_reconciliation


def main() -> None:
    result = run_reconciliation()
    audit = result.audit.fetch_run()
    report = result.report

    print("\n" + "=" * 60)
    print("PITCH DEMO — Three Moments")
    print("=" * 60)

    # Moment 1: MERGED_BATCH
    print("\n[1] TRUE POSITIVE — MERGED_BATCH (setl_rsWw6SEMCiay)")
    merged = [e for e in audit if e.get("reason_code") == "MERGED_BATCH"]
    for e in merged[:2]:
        print(f"  {e['record_id']} → {e['matched_to']} via {e['resolved_by']}")
        print(f"  {e['rationale']}")

    # Moment 2: True exception
    print("\n[2] TRUE EXCEPTION — Orphan adjustment (adj_1iCdzzGypueBxH)")
    orphan = [e for e in audit if e.get("record_id") == "adj_1iCdzzGypueBxH"]
    for e in orphan:
        print(f"  decision={e['decision']} reason={e['reason_code']}")
        print(f"  {e['rationale']}")

    print(f"\n  Total exceptions: {len(report.exceptions)} (precision {report.exception_precision:.0%})")
    for ex in report.exceptions:
        print(f"    • {ex.record_id} ({ex.side})")

    # Moment 3: Agent / LLM
    print("\n[3] GRAY-ZONE AGENT")
    agent_rows = [e for e in audit if e.get("resolved_by") == "level2_agent"]
    if agent_rows:
        for e in agent_rows[:1]:
            print(f"  Agent handled: {e['record_id']}")
            if e.get("shap_json"):
                print(f"  SHAP: {e['shap_json']}")
            print(f"  {e['rationale']}")
    else:
        print("  LLM calls: 0 — all records resolved before gray zone.")
        print("  Stage breakdown (rules did the work):")
        for stage, count in sorted(report.stage_breakdown.items()):
            print(f"    {stage}: {count}")

    print("\n" + "-" * 60)
    print("HEADLINE NUMBERS")
    print("-" * 60)
    print(f"  L1 match rate:  {report.l1_matched}/{report.l1_total} ({report.match_rate_l1:.0%})")
    print(f"  L2 match rate:  {report.l2_matched}/{report.l2_total} ({report.match_rate_l2:.0%})")
    print(f"  Exception precision: {report.exception_precision:.0%}")
    print(f"  Exception recall:    {report.exception_recall:.0%}")
    print(f"  False matches:       {len(report.false_matches)}")
    print(f"  Resolvable missed:   {report.false_exceptions}")
    print(f"  LLM calls:           {report.llm_call_count}")
    print()


if __name__ == "__main__":
    main()
