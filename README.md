# T1D-AI: Type 1 Diabetes Management Platform

A cloud-hosted Type 1 Diabetes management platform with ML predictions, AI-powered insights, and real-time glucose monitoring.

## Features

- **Real-time Glucose Monitoring** - WebSocket-based live updates from Gluroo/Nightscout
- **ML-Powered Predictions** - LSTM neural network for 15/30/45-minute glucose predictions
- **AI Insights** - GPT-4.1 powered personalized recommendations
- **IOB/COB Calculations** - Accurate insulin and carbs on board tracking
- **ISF Prediction** - Dynamic insulin sensitivity factor estimation
- **Pattern Detection** - Automated detection of dawn phenomenon, nocturnal lows, etc.
- **Beautiful Dashboard** - Modern React UI with glassmorphism design

## Tech Stack

### Backend
- **FastAPI** - High-performance async Python API
- **PyTorch** - LSTM models for glucose prediction
- **Azure CosmosDB** - Serverless document database
- **Azure OpenAI** - GPT-4.1 for insights generation

### Frontend
- **React 18** - Modern React with hooks
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Beautiful component library
- **Recharts** - Interactive glucose charts
- **Zustand** - Lightweight state management

### Infrastructure
- **Azure App Service** - Scalable hosting
- **Docker** - Containerized deployment
- **Azure Blob Storage** - ML model storage

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 20+
- Docker & Docker Compose
- Azure subscription (for cloud services)

### Local Development

1. **Clone the repository**
   ```bash
   cd /home/jadericdawson/Documents/AI/T1D-AI
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials
   ```

3. **Start with Docker Compose**
   ```bash
   docker-compose up
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Manual Setup (without Docker)

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cd src
uvicorn main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

### Glucose Data
- `GET /api/v1/glucose/current` - Current glucose with predictions
- `GET /api/v1/glucose/history` - Historical readings
- `GET /api/v1/glucose/stats` - Statistics (avg, TIR, etc.)

### Predictions
- `GET /api/v1/predictions/bg` - BG predictions (linear + LSTM)
- `GET /api/v1/predictions/isf` - ISF prediction
- `GET /api/v1/predictions/accuracy` - Model accuracy stats

### Calculations
- `GET /api/v1/calculations/iob` - Insulin on Board
- `GET /api/v1/calculations/cob` - Carbs on Board
- `GET /api/v1/calculations/dose` - Recommended dose

### AI Insights
- `GET /api/v1/insights/` - Get insights
- `POST /api/v1/insights/generate` - Generate new insights
- `GET /api/v1/insights/patterns` - Detected patterns
- `GET /api/v1/insights/anomalies` - Anomaly detection
- `GET /api/v1/insights/weekly-summary` - Weekly report

### WebSocket
- `WS /api/v1/ws/glucose/{user_id}` - Real-time glucose stream

## ML Models

### BG Predictor (LSTM)
- **Architecture**: 3-layer LSTM, 128 hidden units
- **Input**: 26 features (glucose, IOB, COB, time, etc.)
- **Output**: 15/30/45 minute predictions
- **File**: `bg_predictor_3step_v2.pth`

### ISF Net (LSTM)
- **Architecture**: 3-layer LSTM with softplus output
- **Purpose**: Dynamic insulin sensitivity estimation
- **File**: `best_isf_net.pth`

## Configuration

### Diabetes Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INSULIN_HALF_LIFE` | 81.0 min | Novolog insulin half-life |
| `CARB_HALF_LIFE` | 45.0 min | Carb absorption half-life |
| `CARB_BG_FACTOR` | 4.0 mg/dL/g | BG rise per gram of carbs |
| `TARGET_BG` | 100 mg/dL | Target blood glucose |

### Alert Thresholds

| Threshold | Default | Description |
|-----------|---------|-------------|
| `HIGH_BG_THRESHOLD` | 180 mg/dL | High alert |
| `LOW_BG_THRESHOLD` | 70 mg/dL | Low alert |
| `CRITICAL_HIGH_THRESHOLD` | 250 mg/dL | Critical high |
| `CRITICAL_LOW_THRESHOLD` | 54 mg/dL | Critical low |

## Production Deployment

### Docker Production Build
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Azure App Service
1. Build and push Docker images to Azure Container Registry
2. Create App Service using existing plan (`asp-knowledge2ai-eastus`)
3. Configure environment variables from `.env.production.example`

## Testing

```bash
cd backend
pytest tests/ -v
```

## Project Structure

```
T1D-AI/
├── backend/
│   ├── src/
│   │   ├── api/v1/          # REST endpoints
│   │   ├── services/        # Business logic
│   │   ├── ml/              # ML models & inference
│   │   ├── database/        # CosmosDB repositories
│   │   └── models/          # Pydantic schemas
│   ├── tests/               # Pytest tests
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── hooks/           # Custom hooks
│   │   ├── stores/          # Zustand stores
│   │   └── lib/             # Utilities
│   └── Dockerfile
├── models/                  # ML model files (gitignored)
├── data/                    # Data files (gitignored)
├── docker-compose.yml       # Development
└── docker-compose.prod.yml  # Production
```

## Azure Resource Usage

This project reuses existing Azure resources to minimize costs:

| Resource | Existing Name | Usage |
|----------|---------------|-------|
| CosmosDB | `knowledge2ai-cosmos-serverless` | Separate database `T1D-AI-DB` |
| Storage | `knowledge2aistorage` | Containers `t1d-ai-*` |
| OpenAI | `jadericdawson-4245-resource` | GPT-4.1 deployment |
| App Service | `asp-knowledge2ai-eastus` | New web app `t1d-ai` |

**Total Additional Fixed Cost: $0** (all pay-per-use)

## Security

- JWT-based authentication
- Azure AD B2C integration (optional)
- Rate limiting (60 req/min default)
- Security headers (X-Frame-Options, CSP, etc.)
- CORS whitelist
- Secrets management via environment variables

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original Dexcom reader: `dexcom_reader_ML_complete`
- UI inspiration: KnowlEdge2 AI, JaderBot
- ML models trained on personal CGM data
