# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T1D-AI is a Type 1 Diabetes management platform with ML-powered glucose predictions, AI insights, and real-time monitoring. It's deployed on Azure App Service with a React frontend and FastAPI backend.

## Commands

### Development

```bash
# Full stack with Docker (recommended)
docker-compose up

# Backend only (from backend/src/)
cd backend && source venv/bin/activate && cd src && uvicorn main:app --reload --port 8000

# Frontend only
cd frontend && npm run dev

# Run backend tests
cd backend && pytest tests/ -v
pytest tests/test_insights.py -v  # Single test file
```

### Deployment

**Do not deploy without explicit user permission.**

```bash
# 1. Build frontend
cd frontend && npm run build

# 2. Copy to backend static
rm -rf backend/static/* && cp -r frontend/dist/* backend/static/

# 3. Build and push Docker
cd backend
sg docker -c "docker build -t knowledge2aiacr.azurecr.io/t1d-ai:latest ."
sg docker -c "docker push knowledge2aiacr.azurecr.io/t1d-ai:latest"

# 4. Restart Azure App Service
az webapp stop --name t1d-ai --resource-group rg-knowledge2ai-eastus && sleep 5
az webapp start --name t1d-ai --resource-group rg-knowledge2ai-eastus

# 5. Verify (after ~90s)
curl -s https://t1d-ai.azurewebsites.net/health
```

### Linting

```bash
cd frontend && npm run lint
```

## Architecture

### Tech Stack
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui + Zustand
- **Backend**: FastAPI + Python 3.12 + PyTorch
- **Database**: Azure CosmosDB Serverless (partition key: `/userId`)
- **AI**: Azure OpenAI GPT-4.1 for insights
- **ML**: LSTM/TFT models for glucose prediction, ISF learning

### Key Data Flow

1. **Real-time glucose**: Gluroo sync (5-min background task) → CosmosDB → WebSocket push to frontend
2. **Predictions**: Linear predictor (always available) + LSTM + TFT models
3. **Insights**: Pattern detection + anomaly detection → GPT-4.1 recommendations

### Backend Structure

```
backend/src/
├── main.py              # FastAPI app, middleware, lifespan events
├── config.py            # Pydantic settings from environment
├── api/v1/              # REST endpoints (glucose, treatments, predictions, insights, etc.)
├── services/            # Business logic (prediction_service, insight_service, iob_cob_service, etc.)
├── ml/                  # ML models and inference
│   ├── models/          # TFT, LSTM, absorption models, physics-informed models
│   └── inference/       # bg_inference, isf_inference
├── database/            # CosmosDB client and repositories
└── auth/                # JWT + Azure AD B2C authentication
```

### Frontend Structure

```
frontend/src/
├── pages/               # Route pages (Dashboard, Settings, Onboarding, etc.)
├── components/          # UI components organized by domain (glucose/, insights/, treatments/)
├── stores/              # Zustand stores (authStore, glucoseStore)
├── hooks/               # Custom hooks (useGlucose, useWebSocket)
└── lib/api.ts           # Centralized Axios client with auth headers
```

### Design Patterns

- **Singleton services**: `get_prediction_service()`, `get_cosmos_manager()` - cached instances
- **Repository pattern**: Data access through repository classes
- **Dependency injection**: FastAPI `Depends()` for auth and services
- **Zustand with persistence**: Frontend state persisted to localStorage

## Important Implementation Details

### Diabetes-Specific Parameters

The system uses personalized insulin kinetics (configured in `config.py`):
- `insulin_half_life_minutes: 54.0` - Child's faster insulin (vs 81 min adult)
- `carb_half_life_minutes: 45.0`
- Partition key for all user data is `/userId`

### Background Tasks

- Gluroo sync runs every 5 minutes (`run_gluroo_sync_loop` in `main.py`)
- Managed via asyncio task in lifespan context

### Static File Serving

Production serves frontend from `backend/static/`. The `spa_fallback` route in `main.py` handles React Router paths.

### Rate Limiting

- Default: 60 req/min per IP
- Health endpoints (`/health`, `/ready`, `/`) are exempt

## Azure Resources

All resources are shared with other projects (no additional fixed cost):
- CosmosDB: `knowledge2ai-cosmos-serverless` → database `T1D-AI-DB`
- Container Registry: `knowledge2aiacr`
- App Service Plan: `asp-knowledge2ai-eastus`
- OpenAI: `jadericdawson-4245-resource` → deployment `H4D_Assistant_gpt-4.1`

## Environment Variables

Required variables are in `backend/src/config.py`. Key ones:
- `COSMOS_ENDPOINT`, `COSMOS_KEY`, `COSMOS_DATABASE`
- `AZURE_OPENAI_ENDPOINT` (or `GPT41_ENDPOINT`)
- `JWT_SECRET_KEY`
- `STORAGE_CONNECTION_STRING`

View Azure settings: `az webapp config appsettings list --name t1d-ai --resource-group rg-knowledge2ai-eastus -o table`
