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
