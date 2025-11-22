# Cortex Real Estate Agent

Prototype LangGraph-based multi-agent system for real-estate asset management.

## Project Structure

```
real_estate_agent/
├── app/
│   ├── agents/                 # Supervisor + specialized agents
│   ├── api/                    # FastAPI service entrypoint
│   ├── graph/                  # State objects and LangGraph workflow
│   ├── ui/                     # Streamlit entrypoint
│   ├── config.py               # Paths, environment helpers
│   ├── data_layer.py           # Access to assets.parquet
│   └── tools.py                # Shared formatting utilities
├── data/
│   └── assets.parquet          # Provided dataset (not tracked)
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_end_to_end.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── pyproject.toml
└── README.md
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Terminal 1 - FastAPI service
uvicorn app.api.main:app --reload

# Terminal 2 - Streamlit UI (speaks to API_URL)
API_URL=http://localhost:8000 streamlit run app/ui/streamlit_app.py

# Run tests
python -m pytest
```

### Docker Compose

```bash
docker compose up --build
```

## Assignment Goals

- Classify natural-language queries into price comparison, P&L, asset details, etc.
- Fetch and aggregate data from `data/assets.parquet`.
- Orchestrate agent behavior with LangGraph (coming online behind the API).
- Provide a simple UI that communicates with a FastAPI service.

## Current Capabilities

- **Supervisor routing** extracts property, tenant, and period hints (year / quarter / month) and keeps track of any missing requirements so the clarification agent can ask targeted follow-ups.
- **PnL agent** performs structured filtering (entity/property/tenant + time frames) and returns revenue, expenses, and net operating income using the ledger hierarchy contained in `assets.parquet`.
- **Clarification agent** produces contextual prompts (e.g., “Please share two property names …”) whenever the supervisor detects missing information.
- **API metadata**: every `/chat` response includes the original logs plus a metadata payload so clients/diagnostics UIs can display structured summaries without reparsing text.

> **Note on pricing data**  
> The provided dataset contains ledger rows (revenues and expenses) but no property valuation column. The price-comparison agent therefore reports that price data is unavailable once two properties are supplied. This keeps the workflow predictable while making it clear to the user why an answer cannot be produced.

## Configuration

- **Environment variables**: store API keys inside `.env` (loaded by `app/config.py`). The sample project only requires `OPENAI_API_KEY` if you intend to swap the naive router for an LLM.
- **Address aliases**: if you want to map natural-language addresses (e.g., “123 Main St”) to portfolio names (“Building 120”), create a JSON file such as:

  ```json
  {
    "123 main st": "Building 120",
    "456 oak ave": "Building 160"
  }
  ```

  Then point `ADDRESS_ALIAS_FILE` to it (or place it at `data/address_aliases.json`). The supervisor will automatically use the supplied mapping while still supporting fuzzy matches against known property names.

## Testing & Quality Gates

- Unit tests target the supervisor, toolkit helpers, and P&L agent (`tests/test_agents.py`, `tests/test_tools.py`).
- End-to-end tests (`tests/test_end_to_end.py`) compile the LangGraph workflow and assert that:
  - The graph builds successfully.
  - A P&L query such as “What is the total P&L for 2025?” returns the expected USD 589 524.86 net operating income for that year.
  - Price comparison queries reach the correct node (even though valuations are not present in the dataset).

Run the full suite with:

```bash
python -m pytest
```

## Sample Prompts

You can try the following messages through the API or Streamlit UI:

1. **Portfolio P&L**  
   “What is the total P&L for 2025?” → returns revenue/expense/net income totals and top contributing properties.
2. **Tenant drill-down**  
   “Show Tenant 14 performance for 2025-Q1.” → leverages tenant extraction plus quarter filtering.
3. **Asset details**  
   “Tell me about Building 180.” → returns the most recent ledger snapshot for that property.

When the system cannot satisfy a request (e.g., valuations that are missing from the dataset), it explains what additional information is needed or why the data is unavailable.
