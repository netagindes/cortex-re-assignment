"""
Basic FastAPI service acting as the entrypoint for the multi-agent system.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Mapping, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.request_types import RequestType, normalize_request_type
from app.graph.state import GraphState, QueryContext
from app.graph.workflow import build_workflow
from app.logging_utils import PipelineLogEntry, PipelineLogger, setup_logging
from app.tools import format_currency

setup_logging()

compiled_workflow = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global compiled_workflow
    workflow = build_workflow()
    compiled_workflow = workflow.compile()
    yield
    compiled_workflow = None


app = FastAPI(title="Cortex Asset Agent API", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    note: str | None = None
    logs: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] | None = None
    log_markdown: str | None = None


@app.get("/health", response_model=dict)
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    if compiled_workflow is None:
        raise HTTPException(status_code=503, detail="Workflow is not initialized.")

    request_id = str(uuid.uuid4())
    pipeline_logger = PipelineLogger("api.chat", context={"request_id": request_id})
    pipeline_logger.info("Received chat request", user_message=payload.message)

    context = QueryContext(user_input=payload.message, request_type=RequestType.GENERAL)
    state = GraphState(context=context, logger=pipeline_logger)

    raw_state = compiled_workflow.invoke(state)
    result_state = _ensure_graph_state(raw_state, pipeline_logger)
    response_text, note = _format_response(result_state)
    metadata = _build_metadata(result_state)

    pipeline_logger.info("Responding to client", note=note)
    return ChatResponse(
        response=response_text,
        note=note,
        logs=pipeline_logger.as_text_lines(),
        metadata=metadata,
        log_markdown=pipeline_logger.as_markdown(),
    )


def _ensure_graph_state(
    state_like: GraphState | Mapping[str, Any],
    logger: PipelineLogger,
) -> GraphState:
    if isinstance(state_like, GraphState):
        if state_like.logger is None:
            state_like.logger = logger
        return state_like

    if not isinstance(state_like, Mapping):
        raise TypeError("Workflow returned an unexpected state type.")

    context = state_like.get("context")
    if isinstance(context, dict):
        try:
            user_input = context["user_input"]
            request_type = normalize_request_type(context["request_type"])
        except KeyError as exc:
            raise ValueError("Workflow state context is missing required fields.") from exc
        context = QueryContext(
            user_input=user_input,
            request_type=request_type,
            addresses=list(context.get("addresses") or []),
            suggested_addresses=list(context.get("suggested_addresses") or []),
            address_matches=list(context.get("address_matches") or []),
            candidate_terms=list(context.get("candidate_terms") or []),
            unresolved_terms=list(context.get("unresolved_terms") or []),
            missing_addresses=list(context.get("missing_addresses") or []),
            notes=list(context.get("notes") or []),
            period=context.get("period"),
            period_level=context.get("period_level"),
            entity_name=context.get("entity_name"),
            property_name=context.get("property_name"),
            tenant_name=context.get("tenant_name"),
            year=context.get("year"),
            quarter=context.get("quarter"),
            month=context.get("month"),
            needs_clarification=bool(context.get("needs_clarification")),
            clarification_reasons=list(context.get("clarification_reasons") or []),
            request_measurement=context.get("request_measurement"),
        )
    if not isinstance(context, QueryContext):
        raise TypeError("Workflow returned state without a valid context.")

    diagnostics = _normalize_diagnostics(state_like.get("diagnostics"))

    return GraphState(
        context=context,
        result=state_like.get("result"),
        diagnostics=diagnostics,
        logger=logger,
        pnl_result=state_like.get("pnl_result"),
    )


def _normalize_diagnostics(entries: Any) -> List[PipelineLogEntry]:
    if not entries:
        return []
    normalized: List[PipelineLogEntry] = []
    for entry in entries:
        if isinstance(entry, PipelineLogEntry):
            normalized.append(entry)
        elif isinstance(entry, dict):
            normalized.append(PipelineLogEntry(**entry))
    return normalized


def _format_response(state: GraphState) -> Tuple[str, str | None]:
    result = state.result
    if isinstance(result, str):
        return result, None

    if not isinstance(result, dict):
        return ("I wasn't able to produce an answer. Please try rephrasing.", None)

    request_type = state.context.request_type

    if request_type == RequestType.PRICE_COMPARISON and {"property_a", "property_b"} <= result.keys():
        lines: List[str] = []
        for key in ("property_a", "property_b"):
            entry = result[key]
            lines.append(f"{entry['address']}: {format_currency(entry['price'])}")
        diff_line = f"Difference: {result['difference_formatted']}"
        if result.get("percent_delta") is not None:
            diff_line += f" ({result['percent_delta']:.2f}% vs second property)"
        lines.append(diff_line)
        return ("\n".join(lines), result.get("note"))

    if request_type == RequestType.PNL and {"label", "formatted"} <= result.keys():
        lines = [f"{result['label']}: {result['formatted']}"]
        summary = result.get("totals_summary") or {}
        if summary:
            lines.append(
                "Summary: "
                f"Revenue {format_currency(summary.get('total_revenue', 0.0))}, "
                f"Expenses {format_currency(summary.get('total_expenses', 0.0))}"
            )
        breakdown = result.get("breakdown") or []
        if breakdown:
            lines.append("Top contributors:")
            for row in breakdown:
                lines.append(f"- {row['address']}: {format_currency(row['pnl'])}")
        return ("\n".join(lines), result.get("note"))

    if request_type == RequestType.ASSET_DETAILS:
        pairs = []
        for field in ("address", "city", "state", "price", "pnl", "tenant_name"):
            if field in result:
                value = result[field]
                if field in {"price", "pnl"}:
                    value = format_currency(float(value))
                pairs.append(f"{field.replace('_', ' ').title()}: {value}")
        if not pairs:
            pairs = [f"{k}: {v}" for k, v in result.items()]
        return ("\n".join(pairs), result.get("note"))

    if "message" in result:
        note = result.get("note")
        details = result.get("details")
        if note is None and isinstance(details, str):
            note = details
        return result["message"], note

    return (
        "I wasn't able to match your request to a known operation. Please provide more details.",
        None,
    )


def _build_metadata(state: GraphState) -> Dict[str, Any] | None:
    """
    Provide a structured payload for UI/clients that want more than plain text.
    """

    req_type = state.context.request_type.value if isinstance(state.context.request_type, RequestType) else state.context.request_type
    base: Dict[str, Any] = {
        "request_type": req_type,
        "addresses": state.context.addresses,
        "suggested_addresses": state.context.suggested_addresses,
        "address_matches": state.context.address_matches,
        "candidate_terms": state.context.candidate_terms,
        "unresolved_terms": state.context.unresolved_terms,
        "missing_addresses": state.context.missing_addresses,
        "notes": state.context.notes,
        "period": state.context.period,
        "period_level": state.context.period_level,
        "entity_name": state.context.entity_name,
        "property_name": state.context.property_name,
        "tenant_name": state.context.tenant_name,
        "year": state.context.year,
        "quarter": state.context.quarter,
        "month": state.context.month,
        "request_measurement": state.context.request_measurement,
    }

    result = state.result
    if isinstance(result, dict):
        base["result"] = result
    elif isinstance(result, str):
        base["result"] = {"message": result}
    else:
        base["result"] = None

    return base
