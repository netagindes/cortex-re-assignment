"""
Streamlit chatbot interface that communicates with the FastAPI service.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import streamlit as st

# API URL provided via environment (docker-compose) or default localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")
_PLACEHOLDER_NOTE = "Agent orchestration not yet wired to this endpoint."


def main() -> None:
    st.set_page_config(page_title="Real Estate Asset Manager", layout="wide")
    st.title("AI Asset Manager Assistant")
    st.caption("Chat with the FastAPI service powered by LangGraph agents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        _append_message(
            role="assistant",
            content="Hi! Ask me to compare prices, summarize P&L, or describe any asset.",
        )
    if "pipeline_logs" not in st.session_state:
        st.session_state.pipeline_logs = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    chat_col, log_col = st.columns([2, 1], gap="large")

    with chat_col:
        st.subheader("Conversation (oldest → newest)")
        history_container = st.container()

        prompt = st.chat_input("How can I help with your assets today?")
        if prompt:
            _append_message("user", prompt)
            with st.spinner("Contacting API..."):
                reply, logs, markdown, conv_id = _call_api(prompt, st.session_state.conversation_id)
            _append_message("assistant", reply)
            st.session_state.pipeline_logs = logs
            st.session_state.pipeline_markdown = markdown
            if conv_id:
                st.session_state.conversation_id = conv_id

        with history_container:
            _render_chat_history()

    with log_col:
        _render_log_panel()


def _call_api(message: str, conversation_id: str | None) -> tuple[str, list[str], str | None, str | None]:
    try:
        with httpx.Client(timeout=30.0) as client:
            payload: Dict[str, Any] = {"message": message}
            if conversation_id:
                payload["conversation_id"] = conversation_id
            response = client.post(f"{API_URL}/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            base = data.get("response", "No response received.")
            note = data.get("note")
            if isinstance(note, str) and note.strip() == _PLACEHOLDER_NOTE:
                note = None
            logs = list(data.get("logs") or [])
            if not logs:
                logs = ["Processing steps are not available for this request."]
            if note:
                logs.append(f"Note: {note}")
            conv_id = data.get("conversation_id") or conversation_id
            return base, logs, data.get("log_markdown"), conv_id
    except httpx.RequestError as exc:
        return (f"Failed to reach API: {exc}", [f"RequestError: {exc}"], None, conversation_id)
    except httpx.HTTPStatusError as exc:
        detail = f"API error {exc.response.status_code}: {exc.response.text}"
        return (detail, [detail], None, conversation_id)


def _render_log_panel() -> None:
    st.subheader("Processing steps")
    logs = st.session_state.get("pipeline_logs") or []
    if logs:
        st.code("\n".join(logs))
        markdown = st.session_state.get("pipeline_markdown") or ""
        if markdown:
            st.download_button(
                "Download pipeline trace",
                data=markdown,
                file_name="pipeline_logs.md",
                mime="text/markdown",
            )
    else:
        st.info("No logs yet. Submit a prompt to view the pipeline trace.")


def _render_chat_history() -> None:
    ordered_messages = _ordered_messages()
    for idx, message in enumerate(ordered_messages, start=1):
        role = message.get("role", "assistant")
        header = _format_message_header(idx, role, message.get("timestamp"))
        with st.chat_message(role):
            if header:
                st.caption(header)
            st.write(message.get("content", ""))


def _append_message(role: str, content: str) -> None:
    messages = st.session_state.setdefault("messages", [])
    messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": _current_timestamp(),
        }
    )


def _ordered_messages() -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = st.session_state.get("messages", [])
    for message in messages:
        message.setdefault("timestamp", _current_timestamp())
    return sorted(messages, key=lambda msg: msg.get("timestamp", ""))


def _format_message_header(index: int, role: str, timestamp: str | None) -> str:
    parts = [f"{index}.", role.title()]
    if timestamp:
        human_time = timestamp.replace("T", " ").split("+")[0].split(".")[0]
        parts.append(f"{human_time} UTC")
    return " • ".join(parts)


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
