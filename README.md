# Developer Note

## Overview

This project uses a local dataset (the assets parquet file) to support a multi-agent system acting as an AI asset menager assistant, designed to perform financial calculations, asset lookups, and general assistance. Implemented using Streamlit client app, FastAPI service, LangGraph-powered workflow, OpenAI API for LLM reasoning.


## Estimated Workflow

- 1–2 hours — Review the assignment and explore the dataset.
- 1 hour — Plan data flow, identify key use cases, outline architecture
- 1 hour — Set up the environment, repository, and initial Streamlit app
- 1–2 hours — Design the data layer, define global/local state, create supporting components
- 2 hours — Implement the assistant and multi-agent LangGraph workflow
- 1 hour — Final review and submission


## 📌 Assets Dataset

The dataset can be partitioned into several logical domains:

1. **Tenant Information** - Contains inconsistencies (e.g., zero values), which must be handled gracefully
   - Represents client or property metadata 
   - Contains inconsistencies (e.g., zero values), which must be handled gracefully

2. **Ledger Information** - A hierarchical financial chart-of-accounts structure (by order):
   - ger Type (e.g., revenue / expenses)
   - Ledger Group (category)
   - Ledger Type/Subtype (subcategory)
   - Ledger Code (description)

3. **Timeframe Values** - Inherited hierarchy (month → quarter → year) with consistent formatting:
   - Year: YYYY 
   - Quarter: YY-QN 
   - Month: YY-MM

4. **Ledger Values**
   - Numerical “profit” amounts
   - Aggregatable into financial calculations


## 🧠 System Design — Multi-Agent Architecture

### Top Layer - Supervisor / Orchestrator Agent

#### Responsibilities:
- Receive and classify all user messages
  - Identify request type:
	- Financial calculation (P&L)
    - Asset details
    - Price comparison
    - General knowledge
    - Unsupported queries
  - Dispatch request to the correct specialist agent
  - Aggregate responses into a final user answer
  - Manage conversation loops and stopping conditions:

#### Decision flow:
1. If the result is clear → return to user
2. If flagged as ambiguous → ask for clarification
3. On error → return a helpful fallback (e.g., missing asset, invalid timeframe, unsupported instruction)

### Middle Layer — Specialist Domain Agents

#### 1. PriceComparisonAgent
- Retrieves ledger values for each asset 
- Identifies missing/unavailable assets → raise error 
- Ensures at least two assets exist for comparison 
- Returns data for each provided asset and performs comparison

#### 2. PnLAgent (expends to FinancialCalcAgent)
- Interprets timeframe (absolute or relative to current date)
- Retrieves all relevant ledger values per asset 
- Handles:
  - Missing asset → raise error 
  - Missing timeframe → return partial result + error 
  - Ambiguous input → raise error 
  - Calculate P&L (expends to more financial calculations)
- Returns:
  - Total P&L 
  - Breakdown per asset

#### 3. AssetDetailsAgent
- Retrieves full asset information using any identifier (address, ledger info, tenant info, timeframe)
- Raises errors for missing or invalid assets
- Returns complete, structured asset details

#### 4. GeneralKnowledgeAgent
- Handles non-property-specific queries:
- Current date/time 
- P&L formula explanation (expends)
- Event actions (e.g., “a tenant is stuck in the elevator”)
- Referral to appropriate professionals (e.g., plumber)

#### 5. FallbackAgent
- Generates clarification prompts
- Determines input relevance
- Manages ambiguous follow-up interactions


### Bottom layer – Tools

Core Data Tools
	• Property and entity discovery
	• Generic ledger filters
	• Tenant filters

Financial Logic
	• Core P&L utilities
	• Comparison engines
	• Scenario simulation tools

Asset Understanding Tools
	• Output formatting
	• Validation/normalization helpers
	• Timeframe parser
	• Missing-data detection

Conversation & Meta-Tools
	• Loggers
	• Explanation helpers
	• State summarization tools


## 🔥 Challenges and Implemented Solutions

**Hallucination**
- Hallucination in calculation → Clear step-by-step instruction 
- Fake response for non-existing asset request → Define as system demand.

**Clarification / Fallback Issues**
- Some queries remained unresolved even after clarification

	→ Improved fallback mechanism and message structure.

**Invalid Input Handling**
- Difficulty recognizing non-existent assets

	→ Added data validation logic and stricter prompting for the agents.

**Over-compression in P&L logic** 
- Attempt to "solve" without actually calculate

	→ Incorrect Updated system prompt to improve separation of responsibilities.

**General Task Coverage**
- Non-financial or general-knowledge questions weren't handled.

	→ Created/expanded GeneralKnowledgeAgent and added optional web search capabilities.


## 💡 Personal Note

The initial implementation was developed entirely with the assistance of Cursor. My next step is to refine the system to produce clearer, more consistent, and better-monitored results.


## ✅ Next steps

[ ] Project reconstruction

[ ] Overcome unresolved obstacles

[ ] Create a virtual DB

[ ] Model comparison and cost evaluation

[ ] Agentic framework improvements - ScenarioAgent


## 📦 Project Packaging
✔ The project uses a local dataset located at: `data/assets.parquet`

✔ Before running, ensure that `OPENAI_API_KEY` is set in your environment

✔ Start all backend services using Docker:  

```bash
  docker compose up --build -d
```

✔ Launch the Streamlit app from the project root directory:

```bash
./run_streamlit.sh
```
