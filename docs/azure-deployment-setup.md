# Azure Deployment Setup Guide - T1D-AI

This guide explains how the automated deployment works for the T1D-AI project on Azure.

---

## Azure Resources

T1D-AI shares Azure resources with other projects (no additional fixed cost):

| Resource | Name | Purpose |
|----------|------|---------|
| Container Registry | `knowledge2aiacr` | Stores Docker images |
| App Service | `t1d-ai` | Hosts the application |
| App Service Plan | `asp-knowledge2ai-eastus` | Shared compute |
| Resource Group | `rg-knowledge2ai-eastus` | Resource container |
| CosmosDB | `knowledge2ai-cosmos-serverless` | Database (T1D-AI-DB) |
| OpenAI | `jadericdawson-4245-resource` | GPT-4.1 for AI insights |

---

## Project Structure

```
T1D-AI/
├── frontend/          # React app (Vite + TypeScript)
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── backend/           # FastAPI app (Python 3.12)
│   ├── src/           # Source code
│   ├── static/        # Frontend build goes here
│   ├── Dockerfile
│   └── requirements.txt
├── deploy.sh          # One-command deployment
├── DEPLOYMENT.md      # Deployment documentation
├── CLAUDE.md          # Project instructions for Claude Code
└── docs/
    ├── git-workflow-guide.md
    └── azure-deployment-setup.md
```

---

## Deployment Workflow

### Quick Deploy (One Command)

```bash
./deploy.sh
```

This script:
1. Builds the React frontend
2. Copies build to `backend/static/`
3. Builds Docker image
4. Pushes to Azure Container Registry
5. Restarts Azure App Service
6. Verifies deployment with health check

### Manual Steps (if needed)

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

---

## Configuration

### deploy.sh Configuration (lines 25-30)

```bash
PROJECT_DIR="/home/jadericdawson/Documents/AI/T1D-AI"
ACR_NAME="knowledge2aiacr"
ACR_IMAGE="knowledge2aiacr.azurecr.io/t1d-ai:latest"
APP_NAME="t1d-ai"
RESOURCE_GROUP="rg-knowledge2ai-eastus"
APP_URL="https://t1d-ai.azurewebsites.net"
```

### Environment Variables

View current Azure settings:
```bash
az webapp config appsettings list --name t1d-ai --resource-group rg-knowledge2ai-eastus -o table
```

Key environment variables (defined in Azure Portal or via CLI):
- `COSMOS_ENDPOINT`, `COSMOS_KEY`, `COSMOS_DATABASE`
- `AZURE_OPENAI_ENDPOINT` (or `GPT41_ENDPOINT`)
- `JWT_SECRET_KEY`
- `STORAGE_CONNECTION_STRING`

---

## Health Endpoint

The backend includes a health endpoint at `/health`:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

This is used by:
- Azure App Service for health probes
- `deploy.sh` for deployment verification
- Monitoring tools

---

## Git Workflow

The `deploy.sh` script is Git-agnostic. Use feature branches for safe development:

```bash
# 1. Start from stable main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes & commit
git add -A
git commit -m "feat: Description of changes"

# 4. Deploy feature branch to test
./deploy.sh

# 5a. If it works - merge to main
git checkout main
git merge feature/your-feature-name
git push origin main

# 5b. If it breaks - rollback to main
git checkout main    # Local files instantly revert
./deploy.sh          # Redeploy stable main
```

See `docs/git-workflow-guide.md` for detailed instructions.

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./deploy.sh` | Deploy current code to Azure |
| `git checkout main && ./deploy.sh` | Rollback to main |
| `curl https://t1d-ai.azurewebsites.net/health` | Check app health |
| `az webapp log tail --name t1d-ai --resource-group rg-knowledge2ai-eastus` | View live logs |

---

## Troubleshooting

### Health check fails

```bash
# Check container logs
az webapp log tail --name t1d-ai --resource-group rg-knowledge2ai-eastus

# Verify container is running
az webapp show --name t1d-ai --resource-group rg-knowledge2ai-eastus --query state
```

### JS hash mismatch

Azure may still be pulling the new image. Wait 1-2 minutes and retry:
```bash
curl -s https://t1d-ai.azurewebsites.net/ | grep -o 'index-[^"]*\.js'
```

### Docker permission issues (Linux)

If you get permission errors with Docker:
```bash
sg docker -c "docker build -t knowledge2aiacr.azurecr.io/t1d-ai:latest ."
sg docker -c "docker push knowledge2aiacr.azurecr.io/t1d-ai:latest"
```

Or add yourself to the docker group:
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Azure CLI login issues

```bash
az login
az account show  # Confirm correct subscription
az acr login --name knowledge2aiacr
```

---

## Local Development

### Backend (from backend/src/)

```bash
cd backend && source venv/bin/activate && cd src
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend && npm run dev
```

### Full Stack with Docker

```bash
docker-compose up
```

---

## Related Documentation

- `DEPLOYMENT.md` - Step-by-step deployment docs
- `CLAUDE.md` - Project overview for Claude Code
- `docs/git-workflow-guide.md` - Git feature branch workflow
