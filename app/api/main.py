"""
Basic FastAPI service acting as the entrypoint for the multi-agent system.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cortex Asset Agent API", version="0.1.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    note: str | None = None


@app.get("/health", response_model=dict)
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Placeholder endpoint that will later call into the LangGraph workflow.
    """

    return ChatResponse(
        response=f"(Stubbed response) I received: {payload.message}",
        note="Agent orchestration not yet wired to this endpoint.",
    )

