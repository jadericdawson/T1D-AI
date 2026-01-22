#!/bin/bash
# ============================================================================
# T1D-AI - One-Command Deployment Script
# ============================================================================
# Usage: ./deploy.sh
#
# This script deploys your current code to Azure. It does NOT touch Git/GitHub.
# Handle Git manually (commit, push, branches) when you're ready.
#
# What this script does:
#   1. Builds the React frontend
#   2. Copies build to backend/static
#   3. Builds Docker image
#   4. Pushes to Azure Container Registry
#   5. Restarts Azure App Service
#   6. Waits for startup
#   7. Verifies deployment succeeded
# ============================================================================

set -e  # Exit immediately if any command fails

# ----------------------------------------------------------------------------
# Configuration - T1D-AI Specific
# ----------------------------------------------------------------------------
PROJECT_DIR="/home/jadericdawson/Documents/AI/T1D-AI"
ACR_NAME="knowledge2aiacr"
ACR_IMAGE="knowledge2aiacr.azurecr.io/t1d-ai:latest"
APP_NAME="t1d-ai"
RESOURCE_GROUP="rg-knowledge2ai-eastus"
APP_URL="https://t1d-ai.azurewebsites.net"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
print_step() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}  STEP $1: $2${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ----------------------------------------------------------------------------
# Pre-flight Checks
# ----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}T1D-AI - Deployment Script${NC}"
echo -e "${BLUE}==========================${NC}"
echo ""
echo "This will deploy your CURRENT LOCAL CODE to Azure."
echo "Git is not touched - commit manually when ready."
echo ""

# Check we're in the right directory
if [ ! -f "$PROJECT_DIR/DEPLOYMENT.md" ]; then
    print_error "Not in project directory. Expected: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"
print_success "Project directory confirmed"

# Show current git status (informational only)
echo ""
echo "Current Git branch: $(git branch --show-current)"
echo "Uncommitted changes: $(git status --porcelain | wc -l) files"
echo ""

# ----------------------------------------------------------------------------
# STEP 1: Build Frontend
# ----------------------------------------------------------------------------
print_step "1/7" "Building React Frontend"

cd "$PROJECT_DIR/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_warning "node_modules not found, running npm install..."
    npm install
fi

npm run build

if [ ! -d "dist" ]; then
    print_error "Frontend build failed - dist/ not created"
    exit 1
fi

print_success "Frontend built successfully"

# ----------------------------------------------------------------------------
# STEP 2: Copy Frontend to Backend
# ----------------------------------------------------------------------------
print_step "2/7" "Copying Frontend to Backend Static"

cd "$PROJECT_DIR"

# Clear old static files
rm -rf backend/static/*

# Copy new build
cp -r frontend/dist/* backend/static/

# Copy logo if it exists
if [ -f "frontend/public/logo.svg" ]; then
    cp frontend/public/logo.svg backend/static/
fi

# Record the JS hash for verification later
JS_FILE=$(ls backend/static/assets/index-*.js 2>/dev/null | head -1)
if [ -n "$JS_FILE" ]; then
    LOCAL_JS_HASH=$(basename "$JS_FILE")
    print_success "Frontend copied. JS hash: $LOCAL_JS_HASH"
else
    print_error "No JS file found in build"
    exit 1
fi

# ----------------------------------------------------------------------------
# STEP 3: Login to Azure Container Registry
# ----------------------------------------------------------------------------
print_step "3/7" "Logging into Azure Container Registry"

az acr login --name "$ACR_NAME"

print_success "Logged into ACR"

# ----------------------------------------------------------------------------
# STEP 4: Build Docker Image
# ----------------------------------------------------------------------------
print_step "4/7" "Building Docker Image"

cd "$PROJECT_DIR/backend"

# Build with docker (using sg for group permissions on Linux)
sg docker -c "docker build -t $ACR_IMAGE ."

print_success "Docker image built"

# ----------------------------------------------------------------------------
# STEP 5: Push to Azure Container Registry
# ----------------------------------------------------------------------------
print_step "5/7" "Pushing Image to Azure Container Registry"

sg docker -c "docker push $ACR_IMAGE"

print_success "Image pushed to ACR"

# ----------------------------------------------------------------------------
# STEP 6: Restart Azure App Service
# ----------------------------------------------------------------------------
print_step "6/7" "Restarting Azure App Service"

echo "Stopping app..."
az webapp stop --name "$APP_NAME" --resource-group "$RESOURCE_GROUP"

echo "Waiting 5 seconds..."
sleep 5

echo "Starting app..."
az webapp start --name "$APP_NAME" --resource-group "$RESOURCE_GROUP"

print_success "App Service restarted"

# ----------------------------------------------------------------------------
# STEP 7: Wait and Verify
# ----------------------------------------------------------------------------
print_step "7/7" "Waiting for Startup and Verifying Deployment"

echo "Waiting 90 seconds for container startup..."
echo ""

# Countdown timer
for i in {90..1}; do
    printf "\r  Time remaining: %02d seconds" $i
    sleep 1
done
echo ""
echo ""

# Check health endpoint
echo "Checking health endpoint..."
HEALTH_RESPONSE=$(curl -s --max-time 10 "$APP_URL/health" || echo "FAILED")

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_success "Health check passed"
else
    print_warning "Health check returned: $HEALTH_RESPONSE"
fi

# Check JS hash matches
echo ""
echo "Verifying JS hash..."
LIVE_JS_HASH=$(curl -s --max-time 10 "$APP_URL/" | grep -o 'index-[^"]*\.js' | head -1)

echo "  Local:  $LOCAL_JS_HASH"
echo "  Live:   $LIVE_JS_HASH"

if [ "$LOCAL_JS_HASH" = "$LIVE_JS_HASH" ]; then
    print_success "JS hash matches - deployment verified!"
else
    print_warning "JS hash mismatch - Azure may still be pulling the new image"
    echo ""
    echo "  Try waiting a minute and checking manually:"
    echo "  curl -s $APP_URL/ | grep -o 'index-[^\"]*\.js'"
fi

# ----------------------------------------------------------------------------
# Done!
# ----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "  App URL: $APP_URL"
echo ""
echo "  Remember: Git was NOT touched. When ready, commit manually:"
echo "    git add -A"
echo "    git commit -m \"Your message\""
echo "    git push"
echo ""
