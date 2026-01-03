# T1D-AI - Azure Deployment Guide

## IMPORTANT: AI ASSISTANTS READ THIS

**DO NOT deploy without explicit user permission.** Only run deployment commands when the user specifically asks to deploy. Making code changes does NOT mean deploy - wait for the user to say "deploy" or "push to production".

---

## Overview

This app uses **Docker container deployment** to Azure App Service. Docker cache is GOOD - it makes builds fast by only rebuilding changed layers.

## Architecture

- **Frontend**: React (Vite) - built and served as static files from FastAPI
- **Backend**: FastAPI with uvicorn
- **Container Registry**: Azure Container Registry (knowledge2aiacr.azurecr.io)
- **App Service**: t1d-ai.azurewebsites.net (shares plan with knowledge2aiv2)
- **Database**: CosmosDB Serverless (T1D-AI-DB database)

---

## Quick Deploy (With Verification at Each Step)

**Run each step and verify before moving to the next.**

```bash
# ============================================
# STEP 1: Build Frontend
# ============================================
cd /home/jadericdawson/Documents/AI/T1D-AI/frontend
npm run build

# ============================================
# STEP 2: Copy to backend/static (CRITICAL!)
# ============================================
rm -rf /home/jadericdawson/Documents/AI/T1D-AI/backend/static/*
cp -r dist/* /home/jadericdawson/Documents/AI/T1D-AI/backend/static/

# VERIFY: Note the JS hash - you'll check this at the end
ls /home/jadericdawson/Documents/AI/T1D-AI/backend/static/assets/*.js
# Example output: index-CXFh6UwZ.js  <-- REMEMBER THIS HASH

# ============================================
# STEP 3: Login to Azure Container Registry
# ============================================
az acr login --name knowledge2aiacr
# VERIFY: Should say "Login Succeeded"

# ============================================
# STEP 4: Build Docker image (uses cache - fast!)
# ============================================
cd /home/jadericdawson/Documents/AI/T1D-AI/backend
sg docker -c "docker build -t knowledge2aiacr.azurecr.io/t1d-ai:latest ."

# VERIFY: Look for these lines near the end:
#   Step 8/12 : COPY src/ ./src/
#   ---> [NOT "Using cache" if src/ changed]
#   Step 9/12 : COPY static/ ./static/
#   ---> [NOT "Using cache" if static/ changed]
#
# If BOTH say "Using cache" but you changed files, use --no-cache (see troubleshooting)

# ============================================
# STEP 5: Push to Azure Container Registry
# ============================================
sg docker -c "docker push knowledge2aiacr.azurecr.io/t1d-ai:latest"

# VERIFY: Should show multiple "Pushed" lines, ending with:
#   latest: digest: sha256:xxxxx size: xxxx
# NOTE THE DIGEST - you can verify Azure has it

# ============================================
# STEP 6: Force Azure to pull new image
# ============================================
az webapp stop --name t1d-ai --resource-group rg-knowledge2ai-eastus
sleep 5
az webapp start --name t1d-ai --resource-group rg-knowledge2ai-eastus

# ============================================
# STEP 7: Wait for container startup
# ============================================
echo "Waiting 90 seconds for container startup..."
sleep 90

# ============================================
# STEP 8: VERIFY deployment succeeded
# ============================================
# Check health
curl -s https://t1d-ai.azurewebsites.net/health
# Should return: {"status":"healthy","version":"1.0.0"}

# Check JS hash matches Step 2
curl -s https://t1d-ai.azurewebsites.net/ | grep -o 'index-[^"]*\.js'
# MUST match the hash from Step 2!

# Quick comparison command:
echo "Local:" && ls /home/jadericdawson/Documents/AI/T1D-AI/backend/static/assets/*.js | xargs basename
echo "Live:" && curl -s https://t1d-ai.azurewebsites.net/ | grep -o 'index-[^"]*\.js'
```

---

## If Verification Fails

### JS Hash Doesn't Match

**Step A: Check if Docker detected the changes**
```bash
# Rebuild and watch the output carefully
cd /home/jadericdawson/Documents/AI/T1D-AI/backend
sg docker -c "docker build -t knowledge2aiacr.azurecr.io/t1d-ai:latest ." 2>&1 | grep -A1 "COPY static"
```

If it says "Using cache" for the static layer but files DID change:
```bash
# Force rebuild of src/static layers only (still uses pip cache)
sg docker -c "docker build --no-cache -t knowledge2aiacr.azurecr.io/t1d-ai:latest ."
```

**Step B: Make sure push completed**
```bash
# Re-push and verify
sg docker -c "docker push knowledge2aiacr.azurecr.io/t1d-ai:latest"
```

**Step C: Force Azure to pull fresh**
```bash
# Sometimes you need to wait longer or restart multiple times
az webapp stop --name t1d-ai --resource-group rg-knowledge2ai-eastus
sleep 10
az webapp start --name t1d-ai --resource-group rg-knowledge2ai-eastus
sleep 120  # Wait longer
curl -s https://t1d-ai.azurewebsites.net/ | grep -o 'index-[^"]*\.js'
```

---

## When to Use --no-cache

Only use `--no-cache` when:
1. Docker says "Using cache" for layers you KNOW changed
2. Something is corrupted and you want a clean slate

**Don't use it routinely** - it rebuilds pip dependencies (5+ minutes) every time.

```bash
# Nuclear option - full rebuild
sg docker -c "docker build --no-cache -t knowledge2aiacr.azurecr.io/t1d-ai:latest ."
```

---

## First-Time Setup (Create Azure Resources)

### Prerequisites
- Azure CLI installed and logged in (`az login`)
- Docker installed
- Node.js 20+ installed
- Python 3.12+ installed

### Step 1: Create Web App on Existing App Service Plan
```bash
# Create web app using existing plan (FREE - shared plan)
az webapp create \
  --name t1d-ai \
  --resource-group rg-knowledge2ai-eastus \
  --plan asp-knowledge2ai-eastus \
  --runtime "PYTHON:3.12"

# VERIFY: Should create successfully
az webapp show --name t1d-ai --resource-group rg-knowledge2ai-eastus --query "state" -o tsv
# Should output: Running
```

### Step 2: Create CosmosDB Database (Reusing Existing Account)
```bash
# Create new database in existing CosmosDB account
az cosmosdb sql database create \
  --account-name knowledge2ai-cosmos-serverless \
  --resource-group rg-knowledge2ai-eastus \
  --name T1D-AI-DB

# Create containers
az cosmosdb sql container create \
  --account-name knowledge2ai-cosmos-serverless \
  --resource-group rg-knowledge2ai-eastus \
  --database-name T1D-AI-DB \
  --name users \
  --partition-key-path /id

az cosmosdb sql container create \
  --account-name knowledge2ai-cosmos-serverless \
  --resource-group rg-knowledge2ai-eastus \
  --database-name T1D-AI-DB \
  --name glucose_readings \
  --partition-key-path /userId \
  --default-ttl 31536000

az cosmosdb sql container create \
  --account-name knowledge2ai-cosmos-serverless \
  --resource-group rg-knowledge2ai-eastus \
  --database-name T1D-AI-DB \
  --name treatments \
  --partition-key-path /userId \
  --default-ttl 31536000

az cosmosdb sql container create \
  --account-name knowledge2ai-cosmos-serverless \
  --resource-group rg-knowledge2ai-eastus \
  --database-name T1D-AI-DB \
  --name insights \
  --partition-key-path /userId \
  --default-ttl 7776000
```

### Step 3: Configure Container Deployment
```bash
# Get ACR credentials
az acr credential show --name knowledge2aiacr

# Set container image
az webapp config container set \
  --name t1d-ai \
  --resource-group rg-knowledge2ai-eastus \
  --container-image-name knowledge2aiacr.azurecr.io/t1d-ai:latest \
  --container-registry-url https://knowledge2aiacr.azurecr.io \
  --container-registry-user knowledge2aiacr \
  --container-registry-password "<password>"

# Set port
az webapp config appsettings set \
  --name t1d-ai \
  --resource-group rg-knowledge2ai-eastus \
  --settings WEBSITES_PORT=8000
```

### Step 4: Configure Environment Variables
```bash
# Get CosmosDB key
az cosmosdb keys list \
  --name knowledge2ai-cosmos-serverless \
  --resource-group rg-knowledge2ai-eastus \
  --query "primaryMasterKey" -o tsv

# Set all environment variables
az webapp config appsettings set \
  --name t1d-ai \
  --resource-group rg-knowledge2ai-eastus \
  --settings \
    APP_NAME="T1D-AI" \
    APP_VERSION="1.0.0" \
    DEBUG="false" \
    COSMOS_ENDPOINT="https://knowledge2ai-cosmos-serverless.documents.azure.com:443/" \
    COSMOS_KEY="<your-cosmos-key>" \
    COSMOS_DATABASE="T1D-AI-DB" \
    STORAGE_ACCOUNT_URL="https://knowledge2aistorage.blob.core.windows.net" \
    GPT41_ENDPOINT="https://jadericdawson-4245-resource.openai.azure.com/" \
    GPT41_DEPLOYMENT="H4D_Assistant_gpt-4.1" \
    AZURE_OPENAI_API_VERSION="2024-12-01-preview" \
    CORS_ORIGINS="https://t1d-ai.azurewebsites.net" \
    RATE_LIMIT_PER_MINUTE="60"
```

---

## Troubleshooting

### Check Container Logs
```bash
az webapp log tail --name t1d-ai --resource-group rg-knowledge2ai-eastus
```

### Check What Image Azure Is Running
```bash
az webapp config container show --name t1d-ai --resource-group rg-knowledge2ai-eastus
```

### CRITICAL: Webapp Running Wrong Image Tag
If the webapp is running an old image tag instead of `latest`:
```bash
# Check current tag
az webapp show --name t1d-ai --resource-group rg-knowledge2ai-eastus --query "siteConfig.linuxFxVersion" -o tsv
# BAD: DOCKER|knowledge2aiacr.azurecr.io/t1d-ai:old-tag
# GOOD: DOCKER|knowledge2aiacr.azurecr.io/t1d-ai:latest

# Fix it - set to latest tag
az webapp config container set --name t1d-ai --resource-group rg-knowledge2ai-eastus \
  --container-image-name knowledge2aiacr.azurecr.io/t1d-ai:latest

# Then stop/start to pull fresh
az webapp stop --name t1d-ai --resource-group rg-knowledge2ai-eastus
sleep 5
az webapp start --name t1d-ai --resource-group rg-knowledge2ai-eastus
```

### Check ACR for Latest Image
```bash
az acr repository show-tags --name knowledge2aiacr --repository t1d-ai --orderby time_desc --top 5
```

### 503 Service Unavailable
Container failed to start. Check logs:
```bash
az webapp log tail --name t1d-ai --resource-group rg-knowledge2ai-eastus
```

Common causes:
- Missing environment variables (check COSMOS_KEY, etc.)
- Port mismatch (should be 8000)
- Python import errors (check logs for traceback)

---

## Important Files

| File | Purpose |
|------|---------|
| `backend/Dockerfile` | Docker build configuration |
| `backend/requirements.txt` | Python dependencies (change triggers full rebuild) |
| `backend/static/` | Frontend build output (copied from frontend/dist) |
| `backend/src/` | Python backend code |
| `frontend/src/` | React frontend source |

---

## Local Development

### Option 1: Docker Compose
```bash
cd /home/jadericdawson/Documents/AI/T1D-AI
docker-compose up
```

### Option 2: Run Directly
```bash
# Terminal 1: Backend
cd /home/jadericdawson/Documents/AI/T1D-AI/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd /home/jadericdawson/Documents/AI/T1D-AI/frontend
npm install
npm run dev
```

---

## Environment Variables

All environment variables are configured in Azure App Service settings. Key variables:

| Variable | Description |
|----------|-------------|
| `COSMOS_ENDPOINT` | CosmosDB endpoint URL |
| `COSMOS_KEY` | CosmosDB primary key |
| `COSMOS_DATABASE` | Database name (T1D-AI-DB) |
| `GPT41_ENDPOINT` | Azure OpenAI endpoint |
| `GPT41_DEPLOYMENT` | GPT-4.1 deployment name |
| `CORS_ORIGINS` | Allowed CORS origins |
| `RATE_LIMIT_PER_MINUTE` | API rate limit |

To update:
```bash
az webapp config appsettings set --name t1d-ai --resource-group rg-knowledge2ai-eastus \
  --settings KEY=value
```

To view all settings:
```bash
az webapp config appsettings list --name t1d-ai --resource-group rg-knowledge2ai-eastus -o table
```

---

## Azure Resources Used (All Shared - No Additional Fixed Cost)

| Resource | Name | Usage |
|----------|------|-------|
| **Resource Group** | `rg-knowledge2ai-eastus` | Shared with Knowledge2AI |
| **App Service Plan** | `asp-knowledge2ai-eastus` | Shared (B1 tier) |
| **Container Registry** | `knowledge2aiacr` | Shared |
| **CosmosDB Account** | `knowledge2ai-cosmos-serverless` | Separate database: `T1D-AI-DB` |
| **Storage Account** | `knowledge2aistorage` | Separate containers: `t1d-ai-*` |
| **Azure OpenAI** | `jadericdawson-4245-resource` | Shared GPT-4.1 deployment |

**Total Additional Fixed Cost: $0** (all resources are pay-per-use)

---

## Things NOT Captured in Code (External Config)

These are configured outside the codebase and won't be in git:

| Config | Location | Example |
|--------|----------|---------|
| Azure App Settings | Azure Portal / CLI | API keys, secrets |
| CosmosDB data | Azure Portal | User data |
| Blob Storage data | Azure Portal | ML models, user files |
| DNS Settings | Domain registrar | Custom domain (if any) |

---

## ML Model Deployment

ML models can be:
1. **Embedded in container** - Copy to `backend/models/` before Docker build
2. **Loaded from Blob Storage** - Configure `MODELS_BLOB_URL` environment variable

For embedded models:
```bash
# Copy models before building Docker image
cp /path/to/bg_predictor_3step_v2.pth /home/jadericdawson/Documents/AI/T1D-AI/backend/models/
cp /path/to/*.pkl /home/jadericdawson/Documents/AI/T1D-AI/backend/models/

# Update Dockerfile to copy models (already configured)
# Then build and deploy as normal
```
