"""
Supervisor agent responsible for classifying and routing user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app import tools
from app.agents.intent_parser import IntentParseResult, IntentParser
from app.agents.request_types import RequestType
from app.knowledge import PropertyMatch

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


@dataclass
class SupervisorAgent:
    """
    Intent-aware classifier that routes user queries to the appropriate agent.
    """

    intent_parser: IntentParser = field(default_factory=IntentParser)
    _cached_input: Optional[str] = field(default=None, init=False, repr=False)
    _cached_result: Optional[IntentParseResult] = field(default=None, init=False, repr=False)

    def classify(self, user_input: str) -> RequestType:
        return self._parse(user_input).request_type

    def analyze(self, user_input: str) -> SupervisorDecision:
        parse_result = self._parse(user_input)
        request_type = parse_result.request_type

        max_matches = 4 if request_type == "price_comparison" else 2
        property_resolution = tools.resolve_properties(user_input, max_matches=max_matches)
        matches = property_resolution.matches
        primary_matches = matches[:2]
        addresses = [match.address for match in primary_matches]

        period_info = tools.extract_period_hint(user_input)
        period = period_info.get("label")
        period_level = period_info.get("level")
        year = period_info.get("year")
        quarter = period_info.get("quarter")
        month = period_info.get("month")
        tenants = tools.extract_tenant_names(user_input)

        property_name = None
        if primary_matches:
            property_name = primary_matches[0].property_name or primary_matches[0].address
        elif request_type in {"asset_details", "price_comparison"} and parse_result.address_terms:
            property_name = parse_result.address_terms[0]

        tenant_name = parse_result.tenant_name or (tenants[0] if tenants else None)
        missing: List[str] = list(parse_result.missing_fields or [])

        if request_type == "price_comparison" and len(addresses) < 2 and "second_property" not in missing:
            missing.append("second_property")
        if request_type == "asset_details" and not addresses and "property" not in missing:
            missing.append("property")
        if request_type == "pnl" and not any([period, year, quarter, month]):
            missing.append("period")
        granularity_required = (
            request_type == "pnl"
            and bool(property_name)
            and not tenant_name
            and (
                period_level == "year"
                or (year is not None and not quarter and not month)
            )
        )
        if granularity_required and "granularity" not in missing:
            missing.append("granularity")

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
            request_type,
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
            tenant_name=tenant_name,
            tenant_candidates=tenants,
            missing_requirements=missing,
            needs_clarification=needs_clarification,
            year=year,
            quarter=quarter,
            month=month,
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

