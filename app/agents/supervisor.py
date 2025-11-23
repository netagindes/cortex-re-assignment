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

    def analyze(self, user_input: str) -> SupervisorDecision:
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
        granularity_required = (
            request_type == RequestType.PNL
            and bool(property_name)
            and not tenant_name
            and (
                period_level == "year"
                or (year is not None and not quarter and not month)
            )
        )
        if granularity_required and "granularity" not in missing:
            missing.append("granularity")

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

