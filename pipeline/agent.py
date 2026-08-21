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


class AgentAdjudicator:
    def __init__(
        self,
        settlement_df: pd.DataFrame,
        bank_df: pd.DataFrame,
        batch_totals: pd.DataFrame | None = None,
    ):
        self.settlement_df = settlement_df
        self.bank_df = bank_df
        self.batch_totals = batch_totals or (
            settlement_df.groupby("settlement_id")
            .agg(net_total=("net_amount", "sum"), settled_date=("settled_date", "first"))
            .reset_index()
        )
        self._llm = None
        self._agent = None
        self._tools = []
        self._init_agent()

    def _init_agent(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.tools import tool
            from langchain_groq import ChatGroq

            settlement_df = self.settlement_df
            bank_df = self.bank_df
            batch_totals = self.batch_totals

            @tool
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

            @tool
            def lookup_bank_row(bank_row_id: str) -> dict:
                """Return date, amount, narration for a bank statement row."""
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

            @tool
            def check_tolerance_rule(
                amount_diff_paise: int, date_diff_days: int, rule: str
            ) -> dict:
                """Check whether a candidate pair passes a named tolerance rule."""
                rules = {
                    "ROUNDING": {"max_amount": 5, "max_bdays": 3},
                    "MISSING_UTR": {"max_amount": 5, "max_bdays": 4},
                    "SETTLEMENT_LAG": {"max_amount": 5, "max_bdays": 4},
                    "MERGED_BATCH": {"max_amount": 5, "max_bdays": 3},
                }
                cfg = rules.get(rule.upper())
                if not cfg:
                    return {"passes": False, "reasoning": f"Unknown rule: {rule}"}
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

            self._tools = [
                lookup_settlement_batch,
                lookup_bank_row,
                check_tolerance_rule,
            ]

            llm = ChatGroq(
                model=os.environ.get("GROQ_MODEL", "moonshotai/kimi-k2-instruct"),
                temperature=0,
                groq_api_key=api_key,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a fintech reconciliation auditor. You will be given a candidate match
between a Razorpay settlement batch and a bank statement line that the rules engine
could not confidently resolve.

Your job: determine whether this is a true match, using ONLY the data you can retrieve
with your tools. Do NOT fabricate amounts, dates, or UTRs. If the data is insufficient
to be confident, say so and return EXCEPTION.

Return a JSON object with exactly these fields:
- "decision": "MATCH" or "EXCEPTION"
- "confidence": float 0.0-1.0
- "reason": one of ["ROUNDING", "MISSING_UTR", "SETTLEMENT_LAG", "MERGED_BATCH", "INSUFFICIENT_DATA", "UNEXPLAINED"]
- "rationale": 1-2 sentences explaining what you looked at and why you decided

Never guess. If confidence < 0.7, return EXCEPTION with reason INSUFFICIENT_DATA.""",
                    ),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(llm, self._tools, prompt)
            self._agent = AgentExecutor(agent=agent, tools=self._tools, verbose=False, return_intermediate_steps=True)
            self._llm = llm
        except ImportError:
            pass

    @property
    def is_available(self) -> bool:
        return self._agent is not None

    def adjudicate(
        self,
        settlement_id: str,
        bank_row_id: str,
        shap_values: dict[str, float],
        features: dict,
    ) -> dict[str, Any]:
        if self._agent is None:
            return self._fallback_adjudicate(settlement_id, bank_row_id, features)

        shap_str = json.dumps(shap_values, indent=2)
        feat_str = json.dumps(features, indent=2)
        prompt = (
            f"Candidate match:\n"
            f"  settlement_id: {settlement_id}\n"
            f"  bank_row_id: {bank_row_id}\n"
            f"  ML features: {feat_str}\n"
            f"  SHAP values: {shap_str}\n\n"
            f"Use your tools to verify this pair and return your decision as JSON."
        )
        try:
            result = self._agent.invoke({"input": prompt})
            output = result.get("output", "")

            # Extract tool call trace from intermediate steps
            tool_trace = []
            for action, observation in result.get("intermediate_steps", []):
                tool_trace.append({
                    "tool": action.tool,
                    "input": action.tool_input,
                    "output": observation,
                })

            decision = self._parse_decision(output, settlement_id, bank_row_id, features)
            decision["tool_trace"] = tool_trace
            return decision
        except Exception as exc:
            return self._fallback_adjudicate(
                settlement_id, bank_row_id, features, error=str(exc)
            )

    def _parse_decision(
        self, output: str, settlement_id: str, bank_row_id: str, features: dict
    ) -> dict[str, Any]:
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(output[start:end])
                decision = AgentDecision(**parsed)
                return decision.model_dump()
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
            }

        return {
            "decision": "EXCEPTION",
            "confidence": 0.3,
            "reason": "INSUFFICIENT_DATA",
            "rationale": error or f"Fallback could not confirm match for {settlement_id}.",
            "tool_trace": [],
        }
