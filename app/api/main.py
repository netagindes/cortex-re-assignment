"""
Basic FastAPI service acting as the entrypoint for the multi-agent system.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.logging_utils import PipelineLogger, setup_logging

setup_logging()

app = FastAPI(title="Cortex Asset Agent API", version="0.1.0")


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
    """
    Placeholder endpoint that will later call into the LangGraph workflow.
    """

    request_id = str(uuid.uuid4())
    pipeline_logger = PipelineLogger("api.chat", context={"request_id": request_id})
    pipeline_logger.info("Received chat request", message=payload.message)
    pipeline_logger.info("Generating placeholder response while LangGraph integration is pending")

    response_text = f"(Stubbed response) I received: {payload.message}"
    pipeline_logger.info("Responding to client")

    return ChatResponse(
        response=response_text,
        note="Agent orchestration not yet wired to this endpoint.",
        logs=pipeline_logger.as_text_lines(),
    )

