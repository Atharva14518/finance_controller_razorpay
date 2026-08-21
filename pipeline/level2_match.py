"""Level 2 — Settlement batch ↔ Bank matching (deterministic, rules, ML, agent)."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from pipeline.audit import AuditLogger
from pipeline.utils import business_days_between, parse_utr


@dataclass
class Level2Match:
    settlement_id: str
    bank_row_id: str
    resolved_by: str
    reason_code: str
    confidence: float = 1.0
    rationale: str = ""
    merged_with: list[str] = field(default_factory=list)


@dataclass
class Level2Result:
    matches: list[Level2Match]
    unresolved_batches: list[str]
    sid_to_utr: dict[str, str]
    stage_counts: dict[str, int]
    gray_zone_pairs: list[tuple[str, str, dict]] = field(default_factory=list)
    llm_call_count: int = 0


def _batch_totals(settlement_df: pd.DataFrame) -> pd.DataFrame:
    return (
        settlement_df.groupby("settlement_id")
        .agg(net_total=("net_amount", "sum"), settled_date=("settled_date", "first"))
        .reset_index()
    )


def _build_sid_to_utr(
    batch_totals: pd.DataFrame,
    bank_df: pd.DataFrame,
) -> dict[str, str]:
    """Infer settlement_id → UTR from exact amount + date matches with UTR present."""
    sid_to_utr: dict[str, str] = {}
    for _, bank in bank_df.iterrows():
        utr = parse_utr(bank["narration"])
        if not utr:
            continue
        bank_date = bank["date"]
        for _, batch in batch_totals.iterrows():
            sid = batch["settlement_id"]
            if sid in sid_to_utr:
                continue
            if batch["net_total"] != bank["amount"]:
                continue
            if business_days_between(bank_date, batch["settled_date"]) <= 4:
                sid_to_utr[sid] = utr
    return sid_to_utr


def _desc_similarity(settlement_id: str, narration: str) -> float:
    return round(difflib.SequenceMatcher(None, settlement_id, narration or "").ratio(), 3)


def _pair_features(batch: pd.Series, bank: pd.Series) -> dict:
    amount_diff = abs(int(batch["net_total"]) - int(bank["amount"]))
    net_total = int(batch["net_total"])
    amount_diff_pct = round(amount_diff / max(abs(net_total), 1), 4)
    date_diff = abs((bank["date"] - batch["settled_date"]).days)
    return {
        "amount_diff": amount_diff,
        "amount_diff_pct": amount_diff_pct,
        "date_diff_days": date_diff,
        "method_match": 0,
        "desc_similarity": _desc_similarity(batch["settlement_id"], bank["narration"]),
    }


def _record_match(
    matches: list[Level2Match],
    matched_batches: set[str],
    matched_banks: set[str],
    stage_counts: dict[str, int],
    audit: AuditLogger,
    *,
    settlement_id: str,
    bank_row_id: str,
    resolved_by: str,
    reason_code: str,
    confidence: float,
    rationale: str,
    merged_with: list[str] | None = None,
) -> None:
    matches.append(
        Level2Match(
            settlement_id=settlement_id,
            bank_row_id=bank_row_id,
            resolved_by=resolved_by,
            reason_code=reason_code,
            confidence=confidence,
            rationale=rationale,
            merged_with=merged_with or [],
        )
    )
    matched_batches.add(settlement_id)
    stage_counts[resolved_by] = stage_counts.get(resolved_by, 0) + 1
    audit.log(
        record_id=settlement_id,
        side="settlement_batch",
        decision="MATCH",
        resolved_by=resolved_by,
        matched_to=bank_row_id,
        confidence=confidence,
        reason_code=reason_code,
        rationale=rationale,
    )
    if merged_with:
        for sid in merged_with:
            matched_batches.add(sid)
            stage_counts[resolved_by] = stage_counts.get(resolved_by, 0) + 1
            audit.log(
                record_id=sid,
                side="settlement_batch",
                decision="MATCH",
                resolved_by=resolved_by,
                matched_to=bank_row_id,
                confidence=confidence,
                reason_code=reason_code,
                rationale=f"Merged batch absorbed into {bank_row_id} with {settlement_id}.",
            )


def run_level2_deterministic_and_rules(
    settlement_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    audit: AuditLogger,
    demo_agent_skip_sid: str | None = None,
) -> Level2Result:
    """Run deterministic + rules cascade.

    demo_agent_skip_sid: if set, this settlement_id is intentionally skipped by
    all rules stages so it falls into the gray zone and reaches the LLM agent.
    Used only for demo mode — does not affect normal run accuracy.
    """
    batch_totals = _batch_totals(settlement_df)
    sid_to_utr = _build_sid_to_utr(batch_totals, bank_df)

    matches: list[Level2Match] = []
    matched_batches: set[str] = set()
    matched_banks: set[str] = set()
    stage_counts: dict[str, int] = {}

    batch_lookup = {row["settlement_id"]: row for _, row in batch_totals.iterrows()}
    bank_lookup = {row["bank_row_id"]: row for _, row in bank_df.iterrows()}

    # --- Stage 2a: Deterministic UTR + exact amount + date <= 2 business days ---
    for _, bank in bank_df.iterrows():
        if bank["bank_row_id"] in matched_banks:
            continue
        utr = parse_utr(bank["narration"])
        if not utr:
            continue
        bank_date = bank["date"]
        for sid, batch in batch_lookup.items():
            if sid in matched_batches:
                continue
            if abs(batch["net_total"] - bank["amount"]) != 0:
                continue
            if business_days_between(bank_date, batch["settled_date"]) > 2:
                continue
            batch_utr = sid_to_utr.get(sid)
            if batch_utr and batch_utr != utr:
                continue
            if not batch_utr:
                sid_to_utr[sid] = utr
            _record_match(
                matches,
                matched_batches,
                matched_banks,
                stage_counts,
                audit,
                settlement_id=sid,
                bank_row_id=bank["bank_row_id"],
                resolved_by="level2_deterministic_utr",
                reason_code="CLEAN",
                confidence=1.0,
                rationale=f"UTR {utr}, exact amount, within 2 business days.",
            )
            matched_banks.add(bank["bank_row_id"])
            break

    # --- Stage 2b: Rules-based cascade ---

    # ROUNDING
    for _, bank in bank_df.iterrows():
        if bank["bank_row_id"] in matched_banks:
            continue
        bank_date = bank["date"]
        for sid, batch in batch_lookup.items():
            if sid in matched_batches:
                continue
            # Demo mode: let this sid fall through to the agent
            if demo_agent_skip_sid and sid == demo_agent_skip_sid:
                continue
            amt_diff = abs(batch["net_total"] - bank["amount"])
            if amt_diff == 0 or amt_diff > 5:
                continue
            if business_days_between(bank_date, batch["settled_date"]) > 3:
                continue
            _record_match(
                matches,
                matched_batches,
                matched_banks,
                stage_counts,
                audit,
                settlement_id=sid,
                bank_row_id=bank["bank_row_id"],
                resolved_by="level2_rules_rounding",
                reason_code="ROUNDING",
                confidence=0.95,
                rationale=f"Amount diff {amt_diff} paise within tolerance band.",
            )
            matched_banks.add(bank["bank_row_id"])
            break

    # MISSING_UTR
    for _, bank in bank_df.iterrows():
        if bank["bank_row_id"] in matched_banks:
            continue
        if parse_utr(bank["narration"]):
            continue
        bank_date = bank["date"]
        for sid, batch in batch_lookup.items():
            if sid in matched_batches:
                continue
            amt_diff = abs(batch["net_total"] - bank["amount"])
            if amt_diff > 5:
                continue
            if business_days_between(bank_date, batch["settled_date"]) > 4:
                continue
            _record_match(
                matches,
                matched_batches,
                matched_banks,
                stage_counts,
                audit,
                settlement_id=sid,
                bank_row_id=bank["bank_row_id"],
                resolved_by="level2_rules_missing_utr",
                reason_code="MISSING_UTR",
                confidence=0.95,
                rationale="No UTR in narration; matched on amount+date fallback.",
            )
            matched_banks.add(bank["bank_row_id"])
            break

    # SETTLEMENT_LAG
    for _, bank in bank_df.iterrows():
        if bank["bank_row_id"] in matched_banks:
            continue
        utr = parse_utr(bank["narration"])
        if not utr:
            continue
        bank_date = bank["date"]
        for sid, batch in batch_lookup.items():
            if sid in matched_batches:
                continue
            batch_utr = sid_to_utr.get(sid)
            if not batch_utr or batch_utr != utr:
                continue
            if abs(batch["net_total"] - bank["amount"]) > 5:
                continue
            bdays = business_days_between(bank_date, batch["settled_date"])
            if bdays <= 2 or bdays > 4:
                continue
            _record_match(
                matches,
                matched_batches,
                matched_banks,
                stage_counts,
                audit,
                settlement_id=sid,
                bank_row_id=bank["bank_row_id"],
                resolved_by="level2_rules_lag",
                reason_code="SETTLEMENT_LAG",
                confidence=0.92,
                rationale=f"UTR match with {bdays} business-day lag (extended window).",
            )
            matched_banks.add(bank["bank_row_id"])
            break

    # MERGED_BATCH — sum-of-N: at least one batch must anchor near bank date;
    # absorbed batches may settle on a different day (real-world merged payout).
    for _, bank in bank_df.iterrows():
        if bank["bank_row_id"] in matched_banks:
            continue
        bank_date = bank["date"]
        unmatched_sids = [s for s in batch_lookup if s not in matched_batches]
        found = False
        for i, sid1 in enumerate(unmatched_sids):
            for sid2 in unmatched_sids[i + 1 :]:
                b1 = batch_lookup[sid1]
                b2 = batch_lookup[sid2]
                combined = b1["net_total"] + b2["net_total"]
                if abs(combined - bank["amount"]) > 5:
                    continue
                d1 = business_days_between(bank_date, b1["settled_date"])
                d2 = business_days_between(bank_date, b2["settled_date"])
                if min(d1, d2) > 3:
                    continue
                _record_match(
                    matches,
                    matched_batches,
                    matched_banks,
                    stage_counts,
                    audit,
                    settlement_id=sid1,
                    bank_row_id=bank["bank_row_id"],
                    resolved_by="level2_rules_merged_batch",
                    reason_code="MERGED_BATCH",
                    confidence=0.93,
                    rationale=f"Sum of {sid1} + {sid2} equals bank amount.",
                    merged_with=[sid2],
                )
                matched_banks.add(bank["bank_row_id"])
                found = True
                break
            if found:
                break

    unresolved = [sid for sid in batch_lookup if sid not in matched_batches]
    gray_zone_pairs: list[tuple[str, str, dict]] = []
    for sid in unresolved:
        batch = batch_lookup[sid]
        for _, bank in bank_df.iterrows():
            if bank["bank_row_id"] in matched_banks:
                continue
            features = _pair_features(batch, bank)
            gray_zone_pairs.append((sid, bank["bank_row_id"], features))

    for sid in unresolved:
        audit.log(
            record_id=sid,
            side="settlement_batch",
            decision="BORDERLINE",
            resolved_by="level2_pending",
            reason_code="PENDING",
            rationale=f"Batch {sid} unresolved after rules cascade.",
            confidence=0.5,
        )

    return Level2Result(
        matches=matches,
        unresolved_batches=unresolved,
        sid_to_utr=sid_to_utr,
        stage_counts=stage_counts,
        gray_zone_pairs=gray_zone_pairs,
    )


def apply_ml_and_agent(
    level2_result: Level2Result,
    settlement_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    audit: AuditLogger,
    ml_scorer,
    agent_adjudicator,
    demo_agent: bool = False,
) -> Level2Result:
    """Run stages 2c (ML) and 2d (agent) on unresolved batches.

    demo_agent: if True, raises ML auto-match threshold to 0.99 so borderline records
    go to the LLM agent instead of being auto-resolved by ML.
    """
    batch_totals = _batch_totals(settlement_df)
    batch_lookup = {row["settlement_id"]: row for _, row in batch_totals.iterrows()}
    bank_lookup = {row["bank_row_id"]: row for _, row in bank_df.iterrows()}
    matched_banks = {m.bank_row_id for m in level2_result.matches}

    still_unresolved = set(level2_result.unresolved_batches)
    llm_calls = 0
    # In demo mode raise threshold so gray-zone records reach the agent
    ml_auto_match_threshold = 0.99 if demo_agent else 0.85

    # Group gray-zone pairs by settlement batch
    pairs_by_batch: dict[str, list[tuple[str, dict]]] = {}
    for sid, bank_id, features in level2_result.gray_zone_pairs:
        if sid not in still_unresolved:
            continue
        pairs_by_batch.setdefault(sid, []).append((bank_id, features))

    for sid in list(still_unresolved):
        # Filter candidates: skip bank rows already claimed in this loop
        all_candidates = pairs_by_batch.get(sid, [])
        candidates = [(bank_id, features) for bank_id, features in all_candidates
                      if bank_id not in matched_banks]
        if not candidates:
            continue

        best_match = None
        best_conf = 0.0
        best_shap: dict[str, float] = {}
        best_bank_id = None
        best_features = None

        for bank_id, features in candidates:
            conf, shap_vals = ml_scorer.score(features)
            if conf > best_conf:
                best_conf = conf
                best_shap = shap_vals
                best_bank_id = bank_id
                best_features = features

        if best_conf > ml_auto_match_threshold and best_bank_id:
            level2_result.matches.append(
                Level2Match(
                    settlement_id=sid,
                    bank_row_id=best_bank_id,
                    resolved_by="level2_ml_high_confidence",
                    reason_code="ML_MATCH",
                    confidence=best_conf,
                    rationale=f"XGBoost confidence {best_conf:.3f} above 0.85 threshold.",
                )
            )
            level2_result.stage_counts["level2_ml_high_confidence"] = (
                level2_result.stage_counts.get("level2_ml_high_confidence", 0) + 1
            )
            still_unresolved.discard(sid)
            matched_banks.add(best_bank_id)
            audit.log(
                record_id=sid,
                side="settlement_batch",
                decision="MATCH",
                resolved_by="level2_ml_high_confidence",
                matched_to=best_bank_id,
                confidence=best_conf,
                reason_code="ML_MATCH",
                rationale=f"ML auto-match at confidence {best_conf:.3f}.",
                shap_json=best_shap,
            )
            continue

        if best_conf < 0.15:
            continue

        # Gray zone — agent adjudication
        if best_bank_id and agent_adjudicator:
            llm_calls += 1
            decision = agent_adjudicator.adjudicate(
                settlement_id=sid,
                bank_row_id=best_bank_id,
                shap_values=best_shap,
                features=best_features or {},
            )
            tool_trace = decision.get("tool_trace", [])
            if decision["decision"] == "MATCH":
                level2_result.matches.append(
                    Level2Match(
                        settlement_id=sid,
                        bank_row_id=best_bank_id,
                        resolved_by="level2_agent",
                        reason_code=decision["reason"],
                        confidence=decision["confidence"],
                        rationale=decision["rationale"],
                    )
                )
                level2_result.stage_counts["level2_agent"] = (
                    level2_result.stage_counts.get("level2_agent", 0) + 1
                )
                still_unresolved.discard(sid)
                matched_banks.add(best_bank_id)
                audit.log(
                    record_id=sid,
                    side="settlement_batch",
                    decision="MATCH",
                    resolved_by="level2_agent",
                    matched_to=best_bank_id,
                    confidence=decision["confidence"],
                    reason_code=decision["reason"],
                    rationale=decision["rationale"],
                    shap_json=best_shap,
                    tool_trace=tool_trace,
                )
            else:
                audit.log(
                    record_id=sid,
                    side="settlement_batch",
                    decision="EXCEPTION",
                    resolved_by="level2_agent",
                    matched_to=best_bank_id,
                    confidence=decision["confidence"],
                    reason_code=decision["reason"],
                    rationale=decision["rationale"],
                    shap_json=best_shap,
                    tool_trace=tool_trace,
                )

    # Mark remaining unresolved as exceptions
    for sid in still_unresolved:
        audit.log(
            record_id=sid,
            side="settlement_batch",
            decision="EXCEPTION",
            resolved_by="level2_unresolved",
            reason_code="UNEXPLAINED",
            rationale=f"Settlement batch {sid} could not be matched to any bank row.",
            confidence=0.0,
        )

    level2_result.unresolved_batches = list(still_unresolved)
    level2_result.llm_call_count = llm_calls
    return level2_result
