#!/bin/bash
# Start MLflow tracking server on port 5002
#
# Usage: ./scripts/start_mlflow.sh
#
# The MLflow UI will be available at http://localhost:5002
# All T1D-AI model experiments will be tracked here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
MLRUNS_DIR="$BACKEND_DIR/mlruns"

# Create mlruns directory if it doesn't exist
mkdir -p "$MLRUNS_DIR"

echo "Starting MLflow server..."
echo "  Port: 5002"
echo "  Backend store: $MLRUNS_DIR"
echo "  UI: http://localhost:5002"
echo ""
echo "Tracking experiments:"
echo "  - T1D-AI/TFT-GlucosePredictor"
echo "  - T1D-AI/IOB-Personalized"
echo "  - T1D-AI/COB-Personalized"
echo "  - T1D-AI/BG-Pressure"
echo "  - T1D-AI/ISF-Personalized"
echo "  - T1D-AI/LSTM-GlucosePredictor"
echo ""

mlflow server \
    --host 0.0.0.0 \
    --port 5002 \
    --backend-store-uri "file://$MLRUNS_DIR" \
    --default-artifact-root "file://$MLRUNS_DIR"
