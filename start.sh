#!/bin/bash
# ============================================
# Chadpocalypse 3DMODEL Pod Start Script
# Auto-starts TRELLIS.2 API server on boot
# ============================================

echo "=== Chadpocalypse 3DMODEL Pod Starting ==="
echo "Time: $(date)"

# Create directories
mkdir -p /workspace/outputs /workspace/logs

# Install API dependencies (fast if already cached)
echo "Installing API dependencies..."
pip install fastapi uvicorn python-multipart aiofiles 2>/dev/null

# Pull latest code from repo
REPO_DIR="/workspace/3DMODEL"
if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating repo..."
    cd "$REPO_DIR" && git pull 2>/dev/null
else
    echo "Repo not found, should have been cloned by Docker command."
fi

# Start TRELLIS.2 API server on port 8000
echo "Starting TRELLIS.2 API server on port 8000..."
cd /content/TRELLIS.2
nohup python "$REPO_DIR/trellis_server.py" > /workspace/logs/trellis_api.log 2>&1 &
API_PID=$!
echo "API server PID: $API_PID"

echo "=== Pod ready (model loading takes 2-5 min) ==="
echo "=== API docs available at port 8000/docs ==="
echo "=== Check logs: cat /workspace/logs/trellis_api.log ==="

# Keep container alive
sleep infinity
