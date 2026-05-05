#!/bin/bash
# Usage: ./unlock_model.sh [MODEL_DIR]
DEFAULT_MODEL_DIR="/home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL_DIR="${1:-$DEFAULT_MODEL_DIR}"

echo "Step 0: Find and kill any vmtouch daemon(s)"
pids=$(ps aux | grep "[v]mtouch" | awk '{print $2}')
if [[ -n "$pids" ]]; then
    echo "Killing vmtouch daemon(s): $pids"
    kill $pids
    sleep 1
else
    echo "No vmtouch daemon found"
fi

echo "All safetensors pages are now unlocked (daemon terminated)."
