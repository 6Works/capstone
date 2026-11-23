#!/usr/bin/env bash
echo "========================================================"
echo "  6-Stage Conversational Voice Agent Pipeline"
echo "  (Isolated Conda Environment: capstone_voice)"
echo "========================================================"

CONDA_ENV_PY="$HOME/.conda/envs/capstone_voice/bin/python"
if [ ! -f "$CONDA_ENV_PY" ]; then
    CONDA_ENV_PY="$HOME/.conda/envs/capstone_voice/python.exe"
fi

if [ -f "$CONDA_ENV_PY" ]; then
    PY_CMD="$CONDA_ENV_PY"
    echo "[Environment] Using Conda env: capstone_voice"
else
    PY_CMD="python"
    echo "[Warning] Conda env not found, using default python."
fi

echo ""
echo "Select execution mode:"
echo "  [1] Start Web UI (http://localhost:8000)"
echo "  [2] Run Terminal Automated Demo"
echo "  [3] Run Live Microphone Loop"
echo "  [4] Run Automated Pytest Test Suite"
echo "========================================================"
read -p "Enter choice [1-4]: " choice

if [ "$choice" = "1" ]; then
    echo "Starting Web Server at http://localhost:8000..."
    "$PY_CMD" server.py
elif [ "$choice" = "2" ]; then
    "$PY_CMD" main.py --mode demo
elif [ "$choice" = "3" ]; then
    "$PY_CMD" main.py --mode mic
elif [ "$choice" = "4" ]; then
    "$PY_CMD" -m pytest test_pipeline.py -v
else
    "$PY_CMD" server.py
fi
