"""
Basic FastAPI service acting as the entrypoint for the multi-agent system.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Mapping, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

    context = QueryContext(user_input=payload.message, request_type="general")
    state = GraphState(context=context, logger=pipeline_logger)

    raw_state = compiled_workflow.invoke(state)
    result_state = _ensure_graph_state(raw_state, pipeline_logger)
    response_text, note = _format_response(result_state)

    pipeline_logger.info("Responding to client", note=note)
    return ChatResponse(
        response=response_text,
        note=note,
        logs=pipeline_logger.as_text_lines(),
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
            request_type = context["request_type"]
        except KeyError as exc:
            raise ValueError("Workflow state context is missing required fields.") from exc
        context = QueryContext(
            user_input=user_input,
            request_type=request_type,
            addresses=list(context.get("addresses") or []),
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
    if "message" in result and len(result) == 1:
        return result["message"], result.get("note")

    if request_type == "price_comparison" and {"property_a", "property_b"} <= result.keys():
        lines: List[str] = []
        for key in ("property_a", "property_b"):
            entry = result[key]
            lines.append(f"{entry['address']}: {format_currency(entry['price'])}")
        diff_line = f"Difference: {result['difference_formatted']}"
        if result.get("percent_delta") is not None:
            diff_line += f" ({result['percent_delta']:.2f}% vs second property)"
        lines.append(diff_line)
        return ("\n".join(lines), result.get("note"))

    if request_type == "pnl" and {"label", "formatted"} <= result.keys():
        lines = [f"{result['label']}: {result['formatted']}"]
        breakdown = result.get("breakdown") or []
        if breakdown:
            lines.append("Top contributors:")
            for row in breakdown:
                lines.append(f"- {row['address']}: {format_currency(row['pnl'])}")
        return ("\n".join(lines), result.get("note"))

    if request_type == "asset_details":
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

    return (
        "I wasn't able to match your request to a known operation. Please provide more details.",
        None,
    )
