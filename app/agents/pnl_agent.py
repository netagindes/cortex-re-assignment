"""
Agent responsible for profit & loss aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from app import tools
from app.prompts.pnl_prompt import PNL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PnLAgent:
    """
    Performs simple aggregations over the assets dataset.

    `SYSTEM_PROMPT` documents the contract followed by any LLM-driven P&L agent.
    """

    SYSTEM_PROMPT: str = PNL_SYSTEM_PROMPT

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        working_task = dict(task)
        comparison_periods = working_task.pop("comparison_periods", None) or []
        if comparison_periods:
            return self._run_comparison(working_task, comparison_periods)
        logger.info("PnLAgent invoked with filters %s", {k: v for k, v in working_task.items() if v})
        response = tools.compute_portfolio_pnl(**working_task)
        logger.info("PnLAgent total pnl=%s", response["value"])
        return response

    def _run_comparison(self, base_task: Dict[str, Any], comparison_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(comparison_periods) != 2:
            return {
                "status": "error",
                "errors": ["comparison_period_count_invalid"],
                "message": "Please provide exactly two periods to compare.",
            }

        base_filters = {key: base_task.get(key) for key in ("entity_name", "property_name", "tenant_name")}
        period_results: List[Dict[str, Any]] = []

        for idx, period in enumerate(comparison_periods):
            filters = dict(base_filters)
            filters.update(
                level=period.get("level"),
                label=period.get("label"),
                year=period.get("year"),
                quarter=period.get("quarter"),
                month=period.get("month"),
            )
            logger.info(
                "PnLAgent comparison period %s filters %s",
                idx + 1,
                {k: v for k, v in filters.items() if v},
            )
            result = tools.compute_portfolio_pnl(**filters)
            if result.get("status") == "no_data":
                return {
                    "status": "error",
                    "errors": ["no_data_for_period"],
                    "message": result.get("message"),
                    "period": period.get("label"),
                }
            period_results.append(result)

        first, second = period_results
        summary_a = first.get("totals_summary") or {}
        summary_b = second.get("totals_summary") or {}

        delta_revenue = summary_b.get("total_revenue", 0.0) - summary_a.get("total_revenue", 0.0)
        delta_expenses = summary_b.get("total_expenses", 0.0) - summary_a.get("total_expenses", 0.0)
        delta_noi = summary_b.get("net_operating_income", 0.0) - summary_a.get("net_operating_income", 0.0)
        baseline_noi = summary_a.get("net_operating_income", 0.0)
        percent_change = None
        if baseline_noi not in (0, 0.0):
            percent_change = (delta_noi / baseline_noi) * 100.0

        comparison_payload = {
            "periods": [
                {
                    "label": first.get("label"),
                    "formatted": first.get("formatted"),
                    "totals_summary": summary_a,
                    "result": first,
                },
                {
                    "label": second.get("label"),
                    "formatted": second.get("formatted"),
                    "totals_summary": summary_b,
                    "result": second,
                },
            ],
            "delta": {
                "total_revenue": delta_revenue,
                "total_expenses": delta_expenses,
                "net_operating_income": delta_noi,
                "noi_percent_change": percent_change,
            },
        }

        property_name = base_filters.get("property_name") or "the portfolio"
        label_a = first.get("label")
        label_b = second.get("label")
        message = (
            f"Compared {property_name} P&L for {label_a} vs {label_b}. "
            f"Difference in NOI: {tools.format_currency(delta_noi)}"
        )

        return {
            "status": "ok",
            "mode": "comparison",
            "comparison": comparison_payload,
            "comparison_periods": [period.get("label") for period in comparison_periods if period.get("label")],
            "message": message,
        }


__all__ = ["PnLAgent"]

