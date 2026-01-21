#!/bin/bash
# =============================================================================
# LoRA Serving Launcher Script
# =============================================================================
# Usage:
#   ./start_server.sh [OPTIONS]
#
# Options:
#   --model_dir PATH      Base model directory (required)
#   --lora_dir PATH       LoRA adapter directory (optional)
#   --port PORT           Server port (default: 8080)
#   --tp TP               Tensor parallel degree (default: 1)
#   --host HOST           Server host (default: 127.0.0.1)
#   --enable_multimodal   Enable multimodal support
#   --help                Show this help message
#
# Examples:
#   # Start server with base model only
#   ./start_server.sh --model_dir /path/to/qwen3-vl-2b --port 8080
#
#   # Start server with LoRA adapter
#   ./start_server.sh --model_dir /path/to/qwen3-vl-2b --lora_dir /path/to/lora_adapter --port 8080
#
#   # Start server with multiple LoRA adapters
#   ./start_server.sh --model_dir /path/to/qwen3-vl-2b --lora_dir /path/to/lora1 --port 8080
# =============================================================================

set -e

# Default values
MODEL_DIR="/home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
LORA_DIR="/home/shufan/Qwen-VL-FT/work/lora_dummy"
PORT=8040
TP=2
HOST="0.0.0.0"
ENABLE_MULTIMODAL=true
LORA_MAX_SIZE=1024
COMPUTE_ON_CPU=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --lora_dir)
            LORA_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --enable_multimodal)
            ENABLE_MULTIMODAL=true
            shift
            ;;
        --lora_max_size)
            LORA_MAX_SIZE="$2"
            shift 2
            ;;
        --compute_on_cpu)
            COMPUTE_ON_CPU=true
            shift
            ;;
        --help|-h)
            cat "$0" | grep -E '^[A-Z#]|^[a-zA-Z_]+:'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_DIR" ]]; then
    echo "ERROR: --model_dir is required"
    echo "Usage: $0 --model_dir /path/to/model [--lora_dir /path/to/lora] [--port PORT]"
    exit 1
fi

# Check model directory exists
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check LoRA directory exists if specified
if [[ -n "$LORA_DIR" && ! -d "$LORA_DIR" ]]; then
    echo "ERROR: LoRA directory not found: $LORA_DIR"
    exit 1
fi

# Get absolute paths
MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
[[ -n "$LORA_DIR" ]] && LORA_DIR=$(cd "$LORA_DIR" && pwd)

echo "=============================================="
echo "LightLLM LoRA Server"
echo "=============================================="
echo "Base Model: $MODEL_DIR"
[[ -n "$LORA_DIR" ]] && echo "LoRA Adapter: $LORA_DIR"
echo "Host: $HOST"
echo "Port: $PORT"
echo "TP: $TP"
echo "=============================================="

# Build command
CMD="python -m lightllm.server.api_server \
    --model_dir $MODEL_DIR \
    --host $HOST \
    --port $PORT \
    --tp $TP"

if [[ -n "$LORA_DIR" ]]; then
    CMD="$CMD --lora_dir $LORA_DIR --lora_max_size $LORA_MAX_SIZE"
fi

if [[ "$ENABLE_MULTIMODAL" == "true" ]]; then
    CMD="$CMD --enable_multimodal"
fi

if [[ "$COMPUTE_ON_CPU" == "true" ]]; then
    CMD="$CMD --compute_on_cpu"
fi

# Add common optimizations and disable cudagraph (requires cupy)
CMD="$CMD --mem_fraction 0.7 --batch_max_tokens 4096 --disable_cudagraph"

echo ""
echo "Starting server..."
echo "Command: $CMD"
echo ""

# Execute
eval "$CMD"
