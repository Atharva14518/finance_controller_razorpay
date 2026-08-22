"""LangChain + Groq agent for gray-zone settlement ↔ bank adjudication."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from pipeline.utils import business_days_between, parse_utr


class AgentDecision(BaseModel):
    decision: str = Field(description="MATCH or EXCEPTION")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(
        description="ROUNDING, MISSING_UTR, SETTLEMENT_LAG, MERGED_BATCH, INSUFFICIENT_DATA, UNEXPLAINED"
    )
    rationale: str
    used_fallback: bool = False


class AgentAdjudicator:
    def __init__(
        self,
        settlement_df: pd.DataFrame,
        bank_df: pd.DataFrame,
        batch_totals: pd.DataFrame | None = None,
    ):
        self.settlement_df = settlement_df
        self.bank_df = bank_df
        self.batch_totals = batch_totals if batch_totals is not None else (
            settlement_df.groupby("settlement_id")
            .agg(net_total=("net_amount", "sum"), settled_date=("settled_date", "first"))
            .reset_index()
        )
        self._llm_tools = None
        self._tools_map = {}
        self._init_agent()

    def _init_agent(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return
        try:
            from langchain_core.tools import tool
            from langchain_groq import ChatGroq

            settlement_df = self.settlement_df
            bank_df = self.bank_df
            batch_totals = self.batch_totals

            def lookup_settlement_batch(settlement_id: str) -> dict:
                """Return net_total, settled_date, constituent entity_ids for a settlement batch."""
                batch = batch_totals[batch_totals["settlement_id"] == settlement_id]
                if batch.empty:
                    return {"error": f"Settlement batch {settlement_id} not found"}
                rows = settlement_df[settlement_df["settlement_id"] == settlement_id]
                return {
                    "settlement_id": settlement_id,
                    "net_total": int(batch.iloc[0]["net_total"]),
                    "settled_date": str(batch.iloc[0]["settled_date"]),
                    "entity_ids": rows["entity_id"].tolist(),
                }

            def lookup_bank_row(bank_row_id: str) -> dict:
                """Return date, amount, narration, utr for a bank statement row."""
                row = bank_df[bank_df["bank_row_id"] == bank_row_id]
                if row.empty:
                    return {"error": f"Bank row {bank_row_id} not found"}
                r = row.iloc[0]
                return {
                    "bank_row_id": bank_row_id,
                    "date": str(r["date"]),
                    "amount": int(r["amount"]),
                    "narration": r["narration"],
                    "utr": parse_utr(r["narration"]),
                }

            def check_tolerance_rule(
                amount_diff_paise: int, date_diff_days: int, rule: str = "ROUNDING"
            ) -> dict:
                """Check whether a candidate pair passes a named tolerance rule (ROUNDING, MISSING_UTR, SETTLEMENT_LAG, MERGED_BATCH)."""
                rules = {
                    "ROUNDING": {"max_amount": 5, "max_bdays": 3},
                    "MISSING_UTR": {"max_amount": 5, "max_bdays": 4},
                    "SETTLEMENT_LAG": {"max_amount": 5, "max_bdays": 4},
                    "MERGED_BATCH": {"max_amount": 5, "max_bdays": 3},
                }
                cfg = rules.get(str(rule).upper(), {"max_amount": 5, "max_bdays": 3})
                passes = (
                    amount_diff_paise <= cfg["max_amount"]
                    and date_diff_days <= cfg["max_bdays"]
                )
                return {
                    "passes": passes,
                    "reasoning": (
                        f"Rule {rule}: amount_diff={amount_diff_paise} paise (max {cfg['max_amount']}), "
                        f"date_diff={date_diff_days} days (max {cfg['max_bdays']}) → {'PASS' if passes else 'FAIL'}"
                    ),
                }

            def submit_decision(
                decision: str, confidence: float, reason: str, rationale: str
            ) -> dict:
                """Submit final reconciliation decision with decision (MATCH or EXCEPTION), confidence (0.0-1.0), reason (ROUNDING, etc), and rationale."""
                return {
                    "status": "submitted",
                    "decision": decision,
                    "confidence": confidence,
                    "reason": reason,
                    "rationale": rationale,
                }

            def JSON(
                decision: str, confidence: float, reason: str, rationale: str
            ) -> dict:
                """Submit final JSON reconciliation decision with decision (MATCH or EXCEPTION), confidence (0.0-1.0), reason (ROUNDING, etc), and rationale."""
                return submit_decision(decision, confidence, reason, rationale)

            self._tools_map = {
                "lookup_settlement_batch": lookup_settlement_batch,
                "lookup_bank_row": lookup_bank_row,
                "check_tolerance_rule": check_tolerance_rule,
                "submit_decision": submit_decision,
                "JSON": JSON,
            }

            t1 = tool(lookup_settlement_batch)
            t2 = tool(lookup_bank_row)
            t3 = tool(check_tolerance_rule)
            t4 = tool(submit_decision)
            t5 = tool(JSON)

            llm = ChatGroq(
                model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
                temperature=0,
                groq_api_key=api_key,
            )
            self._llm_tools = llm.bind_tools([t1, t2, t3, t4, t5])
        except Exception:
            self._llm_tools = None

    @property
    def is_available(self) -> bool:
        return self._llm_tools is not None

    def adjudicate(
        self,
        settlement_id: str,
        bank_row_id: str,
        shap_values: dict[str, float],
        features: dict,
    ) -> dict[str, Any]:
        if self._llm_tools is None:
            return self._fallback_adjudicate(settlement_id, bank_row_id, features)

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        shap_str = json.dumps(shap_values, indent=2)
        feat_str = json.dumps(features, indent=2)
        system_prompt = (
            "You are a fintech reconciliation auditor. You will be given a candidate match\n"
            "between a Razorpay settlement batch and a bank statement line that the rules engine\n"
            "could not confidently resolve.\n\n"
            "Your job: determine whether this is a true match, using ONLY the data you retrieve\n"
            "with your tools (lookup_settlement_batch, lookup_bank_row, check_tolerance_rule).\n"
            "Do NOT fabricate amounts, dates, or UTRs. If the data is insufficient to be confident,\n"
            "return EXCEPTION with reason INSUFFICIENT_DATA.\n\n"
            "When finished checking, output your final decision by calling submit_decision or providing a JSON object:\n"
            '{"decision": "MATCH" or "EXCEPTION", "confidence": float 0.0-1.0, "reason": "ROUNDING" | "MISSING_UTR" | "SETTLEMENT_LAG" | "MERGED_BATCH" | "INSUFFICIENT_DATA" | "UNEXPLAINED", "rationale": "1-2 sentences"}'
        )
        user_prompt = (
            f"Candidate match to verify:\n"
            f"  settlement_id: {settlement_id}\n"
            f"  bank_row_id: {bank_row_id}\n"
            f"  ML features: {feat_str}\n"
            f"  SHAP values: {shap_str}\n\n"
            f"Use your tools to verify this candidate pair."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        tool_trace = []
        final_tool_decision = None

        try:
            for _ in range(5):
                resp = self._llm_tools.invoke(messages)
                messages.append(resp)

                if not resp.tool_calls:
                    break

                for tc in resp.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})
                    fn = self._tools_map.get(tool_name)
                    out = fn(**tool_args) if fn else {"error": f"Unknown tool {tool_name}"}

                    if tool_name not in ("submit_decision", "JSON"):
                        tool_trace.append({
                            "tool": tool_name,
                            "input": tool_args,
                            "output": out,
                        })
                    else:
                        final_tool_decision = tool_args

                    tool_id = tc.get("id", "call_1")
                    messages.append(ToolMessage(tool_call_id=tool_id, content=json.dumps(out)))

                if final_tool_decision:
                    break

            if final_tool_decision:
                decision = self._format_decision_dict(final_tool_decision, settlement_id, bank_row_id, features)
                decision["tool_trace"] = tool_trace
                decision["used_fallback"] = False
                return decision

            # If no tool decision was submitted, parse from final text content
            final_text = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                    final_text = msg.content
                    break

            decision = self._parse_decision(final_text, settlement_id, bank_row_id, features)
            decision["tool_trace"] = tool_trace
            decision["used_fallback"] = False
            return decision
        except Exception as exc:
            return self._fallback_adjudicate(
                settlement_id, bank_row_id, features, error=str(exc)
            )

    def _format_decision_dict(
        self, raw: dict, settlement_id: str, bank_row_id: str, features: dict
    ) -> dict[str, Any]:
        valid_reasons = {"ROUNDING", "MISSING_UTR", "SETTLEMENT_LAG", "MERGED_BATCH", "INSUFFICIENT_DATA", "UNEXPLAINED"}
        raw_reason = str(raw.get("reason", "UNEXPLAINED")).upper()
        reason = raw_reason if raw_reason in valid_reasons else ("ROUNDING" if "ROUND" in raw_reason else "UNEXPLAINED")
        raw_dec = str(raw.get("decision", "EXCEPTION")).upper()
        decision = "MATCH" if "MATCH" in raw_dec else "EXCEPTION"
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.7))))
        except (ValueError, TypeError):
            confidence = 0.7
        rationale = str(raw.get("rationale", f"Agent adjudicated candidate {settlement_id} ↔ {bank_row_id}."))

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "rationale": rationale,
            "used_fallback": False,
        }

    def _parse_decision(
        self, output: str, settlement_id: str, bank_row_id: str, features: dict
    ) -> dict[str, Any]:
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(output[start:end])
                return self._format_decision_dict(parsed, settlement_id, bank_row_id, features)
        except (json.JSONDecodeError, ValueError):
            pass
        return self._fallback_adjudicate(settlement_id, bank_row_id, features)

    def _fallback_adjudicate(
        self,
        settlement_id: str,
        bank_row_id: str,
        features: dict,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Deterministic fallback when LLM unavailable — uses tolerance rules."""
        batch = self.batch_totals[
            self.batch_totals["settlement_id"] == settlement_id
        ]
        bank = self.bank_df[self.bank_df["bank_row_id"] == bank_row_id]
        if batch.empty or bank.empty:
            return {
                "decision": "EXCEPTION",
                "confidence": 0.0,
                "reason": "INSUFFICIENT_DATA",
                "rationale": error or "Could not retrieve batch or bank row.",
                "tool_trace": [],
                "used_fallback": True,
            }

        amt_diff = abs(int(batch.iloc[0]["net_total"]) - int(bank.iloc[0]["amount"]))
        bdays = business_days_between(bank.iloc[0]["date"], batch.iloc[0]["settled_date"])
        utr = parse_utr(bank.iloc[0]["narration"])

        if amt_diff <= 5 and bdays <= 4:
            reason = "MISSING_UTR" if not utr else ("ROUNDING" if amt_diff > 0 else "UNEXPLAINED")
            return {
                "decision": "MATCH",
                "confidence": 0.75,
                "reason": reason,
                "rationale": (
                    f"Fallback rule match: amount_diff={amt_diff} paise, "
                    f"{bdays} business days (LLM unavailable)."
                ),
                "tool_trace": [],
                "used_fallback": True,
            }

        return {
            "decision": "EXCEPTION",
            "confidence": 0.3,
            "reason": "INSUFFICIENT_DATA",
            "rationale": error or f"Fallback could not confirm match for {settlement_id}.",
            "tool_trace": [],
            "used_fallback": True,
        }
