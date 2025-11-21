"""
Streamlit chatbot interface that communicates with the FastAPI service.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

# API URL provided via environment (docker-compose) or default localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")


def main() -> None:
    st.set_page_config(page_title="Real Estate Asset Manager", layout="wide")
    st.title("AI Asset Manager Assistant")
    st.caption("Chat with the FastAPI service powered by LangGraph agents.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! Ask me to compare prices, summarize P&L, or describe any asset.",
            }
        ]
    if "pipeline_logs" not in st.session_state:
        st.session_state.pipeline_logs = []

    chat_col, log_col = st.columns([2, 1], gap="large")

    with chat_col:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("How can I help with your assets today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Contacting API..."):
                    reply, logs = _call_api(prompt)
                st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.pipeline_logs = logs

    with log_col:
        _render_log_panel()


def _call_api(message: str) -> tuple[str, list[str]]:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{API_URL}/chat", json={"message": message})
            response.raise_for_status()
            data = response.json()
            base = data.get("response", "No response received.")
            note = data.get("note")
            logs = data.get("logs") or []
            reply = f"{base}\n\n_{note}_" if note else base
            return reply, logs
    except httpx.RequestError as exc:
        return (f"Failed to reach API: {exc}", [f"RequestError: {exc}"])
    except httpx.HTTPStatusError as exc:
        detail = f"API error {exc.response.status_code}: {exc.response.text}"
        return (detail, [detail])


def _render_log_panel() -> None:
    st.subheader("Processing steps")
    logs = st.session_state.get("pipeline_logs") or []
    if logs:
        st.code("\n".join(logs))
    else:
        st.info("No logs yet. Submit a prompt to view the pipeline trace.")


if __name__ == "__main__":
    main()

