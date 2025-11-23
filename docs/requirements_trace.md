##
# Requirements Trace & Coverage Map

This document tracks how the current implementation aligns with the assignment's mandatory requirements. Every section points to the concrete modules/tests that implement the behavior plus any known gaps.

## Requirement 1 – Architecture & LangGraph (Status: ✅ Met, pending general-knowledge node)

| Aspect | Implementation | Notes |
| --- | --- | --- |
| Multi-agent topology | `app/graph/workflow.py` wires Supervisor → {price, pnl, asset_details, general_knowledge, clarification} nodes using LangGraph. | Clarification node still handles unsupported flows. |
| Specialist agents | `app/agents/*.py` (PriceComparisonAgent, PnLAgent, AssetDetailsAgent, GeneralKnowledgeAgent, ClarificationAgent) encapsulate tool access & business logic. | General agent answers dataset/ledger/how-to prompts. |
| Shared state/logging | `app/graph/state.py` and `app/logging_utils.py` keep context + pipeline diagnostic stream for UI/API. |  |
| UI/API entrypoints | `/chat` FastAPI (`app/api/main.py`) + Streamlit UI (`app/ui/streamlit_app.py`). |  |

## Requirement 2 – Natural-language Input (Status: ✅ Met)

- Streamlit chat input (`app/ui/streamlit_app.py`) and FastAPI `/chat` accept arbitrary user text.
- `app/agents/intent_parser.py` extracts intents, addresses, timeframes, tenants.
- Property/tenant/period helpers in `app/tools.py` turn free text into structured filters.

## Requirement 3 – Processing Pipeline (Status: ⚠️ Partial)

| Sub-requirement | Coverage | References |
| --- | --- | --- |
| Detect request type (price, P&L, asset, general, etc.) | Supervisor + IntentParser categorize into `"price_comparison"`, `"pnl"`, `"asset_details"`, `"general"`, `"clarification"`. | `app/agents/supervisor.py`, `app/agents/intent_parser.py`. |
| Extract details (addresses, ledgers, timeframes, financial data) | `tools.resolve_properties`, `tools.extract_period_hint`, `tools.extract_tenant_names`; Supervisor stores in `GraphState`. | `app/tools.py`, `app/graph/state.py`. |
| Retrieve from dataset | `app/data_layer.py`, `app/knowledge/property_memory.py`, and helpers in `tools.py`. |  |
| Perform calculations / assemble answers | `PnLAgent` (NOI math), `PriceComparisonAgent` (valuation diff), `AssetDetailsAgent` (record snapshot). |  |
| Step-by-step confirmation | Pipeline logs captured and surfaced in UI; response formatting in `app/api/main.py` presents structured lines. | Expand to explicit “step list” later. |
| Robust to vague/incomplete inputs | ClarificationAgent prompts for missing data; supervisor notes missing assets/periods. | Need richer fallback taxonomy & general-knowledge handling. |

## Requirement 4 – Clear Output (Status: ✅ Met)

- `/chat` builds concise strings per request type (`_format_response` in `app/api/main.py`).
- UI presents message + diagnostic log trace.

## Requirement 5 – Error Handling & Fallback (Status: ⚠️ Partial)

| Scenario | Current Behavior | Location | Gap |
| --- | --- | --- | --- |
| Address not in dataset | Supervisor flags `missing_addresses`; Price node responds with suggestions. | `app/agents/supervisor.py`, `_build_missing_property_message`. | Need shared error taxonomy + reuse in all agents. |
| Missing financial data | PnLAgent returns `"no_data"` with guidance. | `app/agents/pnl_agent.py`, `app/tools.py`. | Works; ensure other agents reuse pattern. |
| Price data missing | Price node reports dataset limitation. | `app/graph/workflow.py` (`_build_price_capability_message`). | Must offer alternative analytics per assignment (planned). |
| Ambiguous/unsupported instructions | ClarificationAgent requests details; Supervisor defaults to clarification. | `app/agents/clarification_agent.py`. | Need dedicated fallback + general knowledge path. |
| Input not expected format | Rules + property resolver attempt normalization; missing info triggers clarification. | `app/tools.py`. | Add explicit validation errors. |
| Multi-agent fallback logic | Supervisor routes to Clarification; no explicit "supervising agent" escalation. | `app/graph/workflow.py`. | Add error taxonomy + re-route later. |

## Requirement 6 – Documentation / Reasoning Trace (Status: ⚠️ Partial, logger counts once tagged)

- README outlines architecture, dataset, sample prompts.
- PipelineLogger stores timestamped diagnostics per request and UI displays them.
- Upcoming work: tag logs with agent/requirement metadata and export as markdown to treat as design documentation.

## Query-Type Coverage & Classification

| Query Type | Description | Current Handling | Coverage Level |
| --- | --- | --- | --- |
| `price_comparison` | Compare value of two properties. | Supervisor routes to PriceComparisonAgent; falls back when price column missing. | Partial (dataset lacks valuation column). |
| `pnl` | Profit & loss for entity/property/tenant/time range. | Fully handled via PnLAgent, `tools.compute_portfolio_pnl`, tested in `tests/test_pnl_aggregation.py`, `tests/test_agents.py`, `tests/test_end_to_end.py`. | Full. |
| `asset_details` | Snapshot of a single property. | AssetDetailsAgent returns record dictionaries; API formats human-readable list. | Full. |
| `general` | General knowledge / ledger explanations. | Supervisor routes to `GeneralKnowledgeAgent`, which answers P&L concepts, ledger code questions, and dataset summaries. | Partial (LLM-free explanations only). |
| `clarification` / `fallback` | Ambiguous, incomplete, unsupported inputs. | ClarificationAgent prompts for details; supervisor notes missing fields. | Partial (needs taxonomy + fallback). |

### Currently surfaced query flows

1. Price comparison (addresses resolved, price data check).
2. Portfolio/entity/property/tenant P&L with period detection.
3. Asset detail lookup with alias support.
4. Clarification for missing data.

General knowledge and explicit fallback/unsupported flows are not yet implemented and remain the top-priority gaps.

## Phrase → Agent Mapping Cheat Sheet

| Phrase / Marker | Routed Agent | Extraction Helpers |
| --- | --- | --- |
| “compare”, “versus”, “worth”, “valuation” | PriceComparisonAgent | `resolve_properties`, `_COMPARISON_PATTERNS`. |
| “p&l”, “profit”, “income”, “loss”, explicit years/quarters/months | PnLAgent | `extract_period_hint`, `extract_tenant_names`. |
| “details”, “tell me about”, “describe property” | AssetDetailsAgent | `resolve_properties`, alias dictionary. |
| “what does ledger…”, “explain”, “general knowledge” | ❌ No dedicated agent yet → falls back to Clarification. |
| Short / vague instructions (≤3 words) or missing required fields | ClarificationAgent | Supervisor `missing_requirements`. |

## Measurement Hooks & Upcoming Work

- Supervisor task evals (`tests/supervisor_tasks_evals.jsonl`) and full chat evals (`tests/pnl_chat_evals.jsonl`) validate existing flows.
- `tests/request_type_chat_evals.jsonl` exercises each request type (price, P&L clarifications, asset details, general knowledge) end-to-end, ensuring coverage beyond P&L.
- Future “request health” test will iterate over a centralized request-type registry to ensure each query type has: (a) routing, (b) specialist, (c) eval coverage.
- Logging upgrades will tag each PipelineLogger entry with `agent`, `request_type`, and `requirement_section` so exported traces serve as lightweight documentation, satisfying requirement 6 when complete.

