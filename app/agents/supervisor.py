"""
Supervisor agent responsible for classifying and routing user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import re

from app import tools
from app.agents.intent_parser import IntentParseResult, IntentParser
from app.agents.request_types import RequestType, request_definition_for
from app.knowledge import PropertyMatch
from app.graph.state import ClarificationItem, QueryContext
from app.prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class SupervisorDecision:
    request_type: RequestType
    addresses: List[str] = field(default_factory=list)
    address_matches: List[Dict[str, Any]] = field(default_factory=list)
    suggested_addresses: List[str] = field(default_factory=list)
    candidate_terms: List[str] = field(default_factory=list)
    unresolved_terms: List[str] = field(default_factory=list)
    missing_addresses: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    period: Optional[str] = None
    period_level: Optional[str] = None
    entity_name: Optional[str] = None
    property_name: Optional[str] = None
    tenant_name: Optional[str] = None
    tenant_candidates: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    year: Optional[int] = None
    quarter: Optional[str] = None
    month: Optional[str] = None
    measurement_id: Optional[str] = None
    comparison_periods: List[Dict[str, Any]] = field(default_factory=list)
    aggregation_level: Optional[str] = None
    clarifications: List[ClarificationItem] = field(default_factory=list)
    awaiting_user_reply: bool = False


@dataclass
class SupervisorAgent:
    """
    Intent-aware classifier that routes user queries to the appropriate agent.

    The `SYSTEM_PROMPT` field surfaces the canonical supervisor instructions so any
    future LLM-driven Supervisor node can bind the same routing/clarification rules.
    """

    SYSTEM_PROMPT: str = SUPERVISOR_SYSTEM_PROMPT
    intent_parser: IntentParser = field(default_factory=IntentParser)
    _cached_input: Optional[str] = field(default=None, init=False, repr=False)
    _cached_result: Optional[IntentParseResult] = field(default=None, init=False, repr=False)

    def classify(self, user_input: str) -> RequestType:
        return self._parse(user_input).request_type

    def analyze(self, context_or_input: QueryContext | str) -> SupervisorDecision:
        if isinstance(context_or_input, QueryContext):
            context = context_or_input
        else:
            context = QueryContext(user_input=context_or_input, request_type=RequestType.GENERAL)

        if context.awaiting_user_reply and context.clarifications:
            logger.info(
                "Handling follow-up answer for field=%s",
                context.clarifications[-1].field,
            )
            return self._handle_follow_up(context)

        return self._classify_new_request(context)

    def _classify_new_request(self, context: QueryContext) -> SupervisorDecision:
        user_input = context.user_input
        parse_result = self._parse(user_input)
        request_type = parse_result.request_type
        definition = request_definition_for(request_type)

        max_matches = 4 if request_type == RequestType.PRICE_COMPARISON else 2
        property_resolution = tools.resolve_properties(user_input, max_matches=max_matches)
        matches = property_resolution.matches
        dataset_matches = [match for match in matches if match.metadata]
        explicit_matches = [match for match in dataset_matches if match.reason == "Alias or direct match"]
        resolved_matches = explicit_matches or dataset_matches
        address_limit = 2 if request_type == RequestType.PRICE_COMPARISON else 1
        primary_matches = resolved_matches[:address_limit]
        addresses = [match.address for match in primary_matches]

        period_info = tools.extract_period_hint(user_input)
        period = period_info.get("label")
        period_level = period_info.get("level")
        year = period_info.get("year")
        quarter = period_info.get("quarter")
        month = period_info.get("month")
        tenants = tools.extract_tenant_names(user_input)

        property_name = None
        entity_name = parse_result.entity_name
        if primary_matches:
            primary = primary_matches[0]
            property_name = primary.property_name or primary.address
            entity_name = entity_name or primary.metadata.get("entity")
        elif request_type in {RequestType.ASSET_DETAILS, RequestType.PRICE_COMPARISON} and parse_result.address_terms:
            property_name = parse_result.address_terms[0]

        tenant_name = parse_result.tenant_name or (tenants[0] if tenants else None)
        missing: List[str] = list(parse_result.missing_fields or [])

        comparison_markers = bool(parse_result.comparison_markers)
        text_lower = user_input.lower()
        if not comparison_markers:
            comparison_markers = bool(re.search(r"\b(compare|versus|vs\.?|between|difference)\b", text_lower))

        is_pnl_comparison = request_type == RequestType.PNL and comparison_markers
        comparison_periods = tools.extract_comparison_periods(user_input, max_periods=2) if is_pnl_comparison else []
        if is_pnl_comparison and comparison_periods:
            period = comparison_periods[0]["label"]
            period_level = comparison_periods[0].get("level")
            year = None
            quarter = None
            month = None
        elif period_info.get("label"):
            period = period_info.get("label")
            period_level = period_info.get("level")

        if request_type == RequestType.PRICE_COMPARISON and len(addresses) < 2 and "second_property" not in missing:
            missing.append("second_property")
        if request_type == RequestType.ASSET_DETAILS and not addresses and "property" not in missing:
            missing.append("property")
        if request_type == RequestType.PNL and not any([period, year, quarter, month]) and not comparison_periods:
            missing.append("period")
        aggregation_required = (
            request_type == RequestType.PNL
            and bool(property_name)
            and not tenant_name
            and (
                period_level == "year"
                or (year is not None and not quarter and not month)
            )
        )
        if aggregation_required and "aggregation_level" not in missing:
            missing.append("aggregation_level")

        if is_pnl_comparison:
            if len(comparison_periods) != 2 and "comparison_periods" not in missing:
                missing.append("comparison_periods")
            explicit_property_count = len(explicit_matches)
            if explicit_property_count > 1 and "property_selection" not in missing:
                missing.append("property_selection")
            if not resolved_matches and "property" not in missing:
                missing.append("property")

        needs_clarification = bool(missing)
        serialized_matches = [self._serialize_match(match) for match in matches]
        suggested_addresses = self._suggest_addresses(matches, addresses)
        notes = list(parse_result.notes)
        if property_resolution.unresolved_terms:
            notes.append(f"Unresolved mentions: {', '.join(property_resolution.unresolved_terms)}")
        if property_resolution.missing_assets:
            notes.append(f"Not in dataset: {', '.join(property_resolution.missing_assets)}")

        logger.info(
            "Supervisor analysis: request_type=%s addresses=%s period=%s",
            request_type.value if isinstance(request_type, RequestType) else request_type,
            addresses,
            period,
        )
        return SupervisorDecision(
            request_type=request_type,
            addresses=addresses,
            address_matches=serialized_matches,
            suggested_addresses=suggested_addresses,
            candidate_terms=property_resolution.candidate_terms,
            unresolved_terms=property_resolution.unresolved_terms,
            missing_addresses=property_resolution.missing_assets,
            notes=notes,
            period=period,
            period_level=period_level,
            property_name=property_name,
            entity_name=entity_name,
            tenant_name=tenant_name,
            tenant_candidates=tenants,
            missing_requirements=missing,
            needs_clarification=needs_clarification,
            year=year,
            quarter=quarter,
            month=month,
            measurement_id=definition.measurement_id,
            comparison_periods=comparison_periods,
        )

    def _handle_follow_up(self, context: QueryContext) -> SupervisorDecision:
        last_item = context.clarifications[-1]
        answer = context.user_input.strip()
        if not answer:
            logger.info("Empty clarification answer; staying in awaiting mode.")
            context.awaiting_user_reply = True
            return self._decision_from_context(context, context.clarification_reasons or [last_item.field])

        handled = False
        requires_additional = False

        if last_item.field == "aggregation_level":
            normalized = self._normalize_aggregation_level(answer)
            if normalized:
                context.aggregation_level = normalized
                last_item.value = normalized
                handled = True
                logger.info("Clarification resolved: aggregation_level=%s", normalized)
        elif last_item.field == "period":
            period_info, requires_additional = self._normalize_period_answer(answer)
            if period_info:
                last_item.value = period_info.get("label") or period_info.get("granularity")
                if not requires_additional:
                    self._apply_period_info(context, period_info)
                    handled = True
                    logger.info("Clarification resolved: period=%s", context.period)
                else:
                    context.period_level = period_info.get("granularity")
                    handled = True
                    logger.info("Clarification updated granularity=%s", context.period_level)
        else:
            # Default behavior: capture the raw answer.
            last_item.value = answer
            handled = True
            if last_item.field == "tenant_name":
                context.tenant_name = answer
            elif last_item.field == "property_name":
                resolution = tools.resolve_properties(answer, max_matches=1)
                if resolution.matches:
                    match = resolution.matches[0]
                    context.property_name = match.property_name or match.address
                    context.addresses = [match.address]
                    context.address_matches = [self._serialize_match(match)]
                    handled = True
                    logger.info("Clarification resolved: property=%s", context.property_name)

        if handled:
            context.clarifications.pop()
            context.awaiting_user_reply = False
        else:
            logger.info("Unable to map clarification answer '%s' to field '%s'", answer, last_item.field)
            context.awaiting_user_reply = True
            return self._decision_from_context(context, context.clarification_reasons or [last_item.field])

        missing = self._compute_missing_requirements(context)
        if requires_additional and "period" not in missing:
            missing.append("period")

        context.clarification_reasons = missing
        context.needs_clarification = bool(missing)
        context.clarification_needed = context.needs_clarification
        if not context.needs_clarification:
            context.clarifications = []

        return self._decision_from_context(context, missing)

    def _decision_from_context(self, context: QueryContext, missing: List[str]) -> SupervisorDecision:
        request_type = context.request_type if isinstance(context.request_type, RequestType) else RequestType.GENERAL
        return SupervisorDecision(
            request_type=request_type,
            addresses=list(context.addresses),
            address_matches=list(context.address_matches),
            suggested_addresses=list(context.suggested_addresses),
            candidate_terms=list(context.candidate_terms),
            unresolved_terms=list(context.unresolved_terms),
            missing_addresses=list(context.missing_addresses),
            notes=list(context.notes),
            period=context.period,
            period_level=context.period_level,
            property_name=context.property_name,
            entity_name=context.entity_name,
            tenant_name=context.tenant_name,
            tenant_candidates=[],
            missing_requirements=list(missing),
            needs_clarification=bool(missing),
            year=context.year,
            quarter=context.quarter,
            month=context.month,
            measurement_id=context.request_measurement,
            comparison_periods=list(context.comparison_periods),
            aggregation_level=context.aggregation_level,
            clarifications=list(context.clarifications),
            awaiting_user_reply=context.awaiting_user_reply,
        )

    def _compute_missing_requirements(self, context: QueryContext) -> List[str]:
        request_type = context.request_type if isinstance(context.request_type, RequestType) else RequestType.GENERAL
        missing: List[str] = []
        if request_type == RequestType.PNL:
            if not context.property_name:
                missing.append("property")
            if not self._has_period(context):
                missing.append("period")
            if context.comparison_periods and len(context.comparison_periods) != 2:
                missing.append("comparison_periods")
            if not context.tenant_name and context.aggregation_level is None:
                missing.append("aggregation_level")
        elif request_type == RequestType.ASSET_DETAILS:
            if not context.addresses:
                missing.append("property")
        elif request_type == RequestType.PRICE_COMPARISON:
            if len(context.addresses) < 2:
                missing.append("second_property")
        return list(dict.fromkeys(missing))

    @staticmethod
    def _has_period(context: QueryContext) -> bool:
        return bool(
            context.period
            or context.year is not None
            or context.quarter
            or context.month
            or context.comparison_periods
        )

    @staticmethod
    def _normalize_aggregation_level(answer: str) -> Optional[str]:
        normalized = answer.strip().lower()
        if normalized.startswith("tenant"):
            return "tenant"
        if normalized.startswith("prop"):
            return "property"
        if normalized.startswith("comb") or normalized.startswith("port") or normalized.startswith("total"):
            return "combined"
        return None

    def _normalize_period_answer(self, answer: str) -> tuple[Optional[Dict[str, Any]], bool]:
        info = tools.extract_period_hint(answer)
        if info.get("label"):
            return info, False
        normalized = answer.strip().lower()
        if normalized in {"year", "annual", "yearly"}:
            return {"granularity": "year"}, True
        if normalized in {"quarter", "quarterly"}:
            return {"granularity": "quarter"}, True
        if normalized in {"month", "monthly"}:
            return {"granularity": "month"}, True
        return None, False

    @staticmethod
    def _apply_period_info(context: QueryContext, info: Dict[str, Any]) -> None:
        context.period = info.get("label")
        context.period_level = info.get("level")
        context.year = info.get("year")
        context.quarter = info.get("quarter")
        context.month = info.get("month")

    def _suggest_addresses(self, matches: List[PropertyMatch], resolved: List[str]) -> List[str]:
        suggestions: List[str] = []
        for match in matches:
            label = match.property_name or match.address
            if label and label not in suggestions:
                suggestions.append(label)
        remaining = max(0, 4 - len(suggestions))
        if len(suggestions) < 2 and remaining:
            extras = tools.suggest_alternative_properties(exclude=resolved, limit=remaining)
            for extra in extras:
                if extra not in suggestions:
                    suggestions.append(extra)
        return suggestions

    def _parse(self, user_input: str) -> IntentParseResult:
        if self._cached_input == user_input and self._cached_result is not None:
            return self._cached_result
        result = self.intent_parser.parse(user_input)
        self._cached_input = user_input
        self._cached_result = result
        return result

    @staticmethod
    def _serialize_match(match: PropertyMatch) -> Dict[str, Any]:
        return {
            "address": match.address,
            "property_name": match.property_name,
            "confidence": match.confidence,
            "reason": match.reason,
            "metadata": match.metadata,
        }


__all__ = ["SupervisorAgent", "SupervisorDecision", "RequestType"]

