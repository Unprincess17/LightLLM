#!/bin/bash
# Usage: ./lock_model_daemon.sh [MODEL_DIR]
# Default: Qwen3-VL-30B-A3B snapshot

DEFAULT_MODEL_DIR="/home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL_DIR="${1:-$DEFAULT_MODEL_DIR}"

# Allow unlimited mlock
ulimit -l unlimited

echo "Locking all safetensors in $MODEL_DIR into host RAM using daemon..."

# Find all safetensors and resolve symlinks
SHARDS=$(find "$MODEL_DIR" -name "*.safetensors" | while read f; do
    realpath "$f"
done)

# Lock shards in parallel, daemon mode, file-size based
for shard in $SHARDS; do
    size=$(stat -c%s "$shard")
    echo "Starting daemon to lock $shard ($((size/1024/1024)) MB)..."
    # Run vmtouch in background for each shard
    vmtouch -dl -m "$size" -v "$shard" &
done
