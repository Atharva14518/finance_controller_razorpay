"""Automated scoring against ground truth."""

import pytest

from pipeline.runner import run_reconciliation

TRUE_EXCEPTIONS = {
    "order_WohoNMdZEHNJSs",
    "order_RTWPElndTdtdD5",
    "adj_1iCdzzGypueBxH",
    "adj_qPL2lVSID1u7a6",
    "adj_4RRyotrCOtw2CX",
    "setl_Jq5QrHOl0ami",
}

RESOLVABLE_CHALLENGES = {
    "setl_4OaGzbSHrFRA",
    "setl_smoJpI6uVEST",
    "setl_NWBoIxlKxVsp",
    "setl_rsWw6SEMCiay",
}


@pytest.fixture(scope="module")
def result():
    return run_reconciliation()


def test_level1_match_rate(result):
    assert result.report.l1_matched == 60
    assert result.report.match_rate_l1 == 1.0


def test_level2_match_rate(result):
    assert result.report.l2_matched == 15
    assert result.report.match_rate_l2 == 1.0


def test_exception_precision(result):
    flagged = {e.record_id for e in result.report.exceptions}
    assert flagged == TRUE_EXCEPTIONS
    assert result.report.exception_precision == 1.0


def test_exception_recall(result):
    assert result.report.exception_recall == 1.0


def test_no_false_matches(result):
    assert result.report.false_matches == []


def test_resolvable_challenges_matched(result):
    matched_batches = {m.settlement_id for m in result.level2_matches}
    for m in result.level2_matches:
        matched_batches.update(m.merged_with)
    for sid in RESOLVABLE_CHALLENGES:
        assert sid in matched_batches, f"Resolvable challenge {sid} not matched"


def test_stage_breakdown_has_deterministic(result):
    breakdown = result.report.stage_breakdown
    assert breakdown.get("level1_deterministic", 0) == 60
    assert breakdown.get("level2_deterministic_utr", 0) >= 10


def test_audit_log_populated(result):
    entries = result.audit.fetch_run()
    assert len(entries) > 50
    decisions = {e["decision"] for e in entries}
    assert "MATCH" in decisions
    assert "EXCEPTION" in decisions
