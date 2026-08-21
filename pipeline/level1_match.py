"""Level 1 — Order ↔ Settlement key-based matching."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pipeline.audit import AuditLogger


@dataclass
class Level1Match:
    order_id: str
    payment_entity_ids: list[str] = field(default_factory=list)
    refund_entity_ids: list[str] = field(default_factory=list)
    match_type: str = "PAYMENT_ONLY"  # PAYMENT_ONLY | PAYMENT_WITH_REFUND


@dataclass
class Level1Result:
    matches: list[Level1Match]
    unmatched_orders: list[str]
    orphan_settlements: list[str]
    unmatched_settlement_entities: list[str]


def run_level1(
    orders_df: pd.DataFrame,
    settlement_df: pd.DataFrame,
    audit: AuditLogger,
) -> Level1Result:
    matches: list[Level1Match] = []
    unmatched_orders: list[str] = []
    matched_entity_ids: set[str] = set()

    for _, order in orders_df.iterrows():
        order_id = order["order_id"]
        rows = settlement_df[settlement_df["order_id"] == order_id]

        if rows.empty:
            unmatched_orders.append(order_id)
            audit.log(
                record_id=order_id,
                side="ledger",
                decision="EXCEPTION",
                resolved_by="level1_deterministic",
                reason_code="UNEXPLAINED",
                rationale=f"Order {order_id} has no corresponding rows in settlement report.",
                confidence=0.0,
            )
            continue

        payments = rows[rows["type"] == "payment"]
        refunds = rows[rows["type"] == "refund"]

        if len(payments) != 1:
            unmatched_orders.append(order_id)
            audit.log(
                record_id=order_id,
                side="ledger",
                decision="EXCEPTION",
                resolved_by="level1_deterministic",
                reason_code="UNEXPLAINED",
                rationale=f"Order {order_id} expected exactly one payment row, found {len(payments)}.",
                confidence=0.0,
            )
            continue

        payment_id = payments.iloc[0]["entity_id"]
        refund_ids = refunds["entity_id"].tolist()
        match_type = "PAYMENT_WITH_REFUND" if refund_ids else "PAYMENT_ONLY"

        match = Level1Match(
            order_id=order_id,
            payment_entity_ids=[payment_id],
            refund_entity_ids=refund_ids,
            match_type=match_type,
        )
        matches.append(match)

        all_entities = [payment_id] + refund_ids
        for eid in all_entities:
            matched_entity_ids.add(eid)
            audit.log(
                record_id=order_id,
                side="ledger",
                decision="MATCH",
                resolved_by="level1_deterministic",
                matched_to=eid,
                reason_code=match_type,
                rationale=f"Key join on order_id matched settlement entity {eid}.",
                confidence=1.0,
            )

    orphan_settlements: list[str] = []
    for _, row in settlement_df[settlement_df["type"] == "adjustment"].iterrows():
        if pd.isna(row["order_id"]) or row["order_id"] == "":
            eid = row["entity_id"]
            orphan_settlements.append(eid)
            audit.log(
                record_id=eid,
                side="settlement",
                decision="EXCEPTION",
                resolved_by="level1_deterministic",
                reason_code="UNEXPLAINED",
                rationale=f"Adjustment row {eid} has no source order.",
                confidence=0.0,
            )

    unmatched_settlement_entities: list[str] = []
    for _, row in settlement_df[settlement_df["type"].isin(["payment", "refund"])].iterrows():
        eid = row["entity_id"]
        if eid not in matched_entity_ids:
            unmatched_settlement_entities.append(eid)
            audit.log(
                record_id=eid,
                side="settlement",
                decision="EXCEPTION",
                resolved_by="level1_deterministic",
                reason_code="UNEXPLAINED",
                rationale=f"Settlement entity {eid} has no matching order in ledger.",
                confidence=0.0,
            )

    return Level1Result(
        matches=matches,
        unmatched_orders=unmatched_orders,
        orphan_settlements=orphan_settlements,
        unmatched_settlement_entities=unmatched_settlement_entities,
    )
