#!/bin/bash
# =============================================================================
# LoRA Serving Launcher Script
# =============================================================================
# Usage:
#   ./start_server.sh [OPTIONS]
#
# Options:
#   --model_dir PATH       Base model directory (required)
#   --lora_dir PATH        LoRA adapter directory (optional)
#   --port PORT            Server port (default: 8080)
#   --tp TP                Tensor parallel degree (default: 1)
#   --host HOST            Server host (default: 127.0.0.1)
#   --enable_multimodal    Enable multimodal support
#   --lora_max_size SIZE   Max LoRA size (default: 1024)
#   --compute_device STR   Configure LoRA storage and compute locations
#                          Format: 'vl_storage:{gpu|cpu},vl_compute:{gpu|cpu|off},attn_storage:{gpu|cpu},attn_compute:{gpu|cpu|off},moe_storage:{gpu|cpu},moe_compute:{gpu|cpu|off}'
#                          Short format 'moe:cpu' sets both storage and compute
#                          Example: 'moe:cpu' - MoE on CPU (both storage and compute)
#                          Example: 'vl_storage:cpu,vl_compute:gpu' - VL weights on CPU, compute on GPU
#   --force_slow_lora_path Force slow path for LoRA (per-expert baseline)
#   --max_req_total_len    Max request total length
#   --mem_fraction         Memory fraction (default: 0.6)
#   --batch_max_tokens     Batch max tokens (default: 4096)
#   --help                 Show this help message
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
#
#   # Run MoE LoRA on CPU only
#   ./start_server.sh --model_dir /path/to/qwen3-vl-2b --compute_device "moe:cpu"
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

### Baseline 1: Store on CPU, Compute on GPU ###
# COMPUTE_DEVICE="vl_storage:cpu,vl_compute:gpu,attn_storage:cpu,attn_compute:gpu,moe_storage:cpu,moe_compute:gpu"

### Baseline 2: Store on CPU, Compute on CPU ###
# COMPUTE_DEVICE="vl_storage:cpu,vl_compute:cpu,attn_storage:cpu,attn_compute:cpu,moe_storage:cpu,moe_compute:cpu"

### Baseline 3: Store on GPU, compute on GPU ###
 COMPUTE_DEVICE="vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:gpu,moe_compute:gpu"

### Baseline 4: Store on CPU, compute Attn on CPU, MoE off ### TODO: it's fails.
# COMPUTE_DEVICE="vl_storage:cpu,vl_compute:gpu,attn_storage:cpu,attn_compute:cpu,moe_storage:off,moe_compute:off"

### Proposed: Store on CPU, compute Attn on GPU, MoE on CPU ###
# COMPUTE_DEVICE="vl_storage:cpu,vl_compute:gpu,attn_storage:cpu,attn_compute:gpu,moe_storage:cpu,moe_compute:cpu"

FORCE_SLOW_LORA_PATH=true
MAX_REQ_TOTAL_LEN=8192
MEM_FRACTION=0.6
BATCH_MAX_TOKENS=4096

# Environment variables
LOADWORKER=8
LIGHTLLM_LOGGING="DEBUG"
MOE_MODE="TP"
MOCK_PREFILL_LOGITS="TRUE"
#MOCK_PREFILL_LOGITS="FALSE"

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
        --compute_device)
            COMPUTE_DEVICE="$2"
            shift 2
            ;;
        --force_slow_lora_path)
            FORCE_SLOW_LORA_PATH=true
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

echo "Cache model in memory (vmtouch)"
find -L "$MODEL_DIR" -type f \( -name "*.safetensors" -o -name "*.bin" \) -print0 | xargs -0 vmtouch -vt

# Build command
CMD="python -m lightllm.server.api_server \
    --model_dir $MODEL_DIR \
    --host $HOST \
    --port $PORT \
    --tp $TP \
    --batch_max_tokens $BATCH_MAX_TOKENS \
    --max_req_total_len $MAX_REQ_TOTAL_LEN \
    --trust_remote_code \
    --mem_fraction $MEM_FRACTION \
    --disable_cudagraph"

if [[ -n "$LORA_DIR" ]]; then
    CMD="$CMD --lora_dir $LORA_DIR --lora_max_size $LORA_MAX_SIZE"
fi

if [[ "$ENABLE_MULTIMODAL" == "true" ]]; then
    CMD="$CMD --enable_multimodal"
fi

if [[ -n "$COMPUTE_DEVICE" ]]; then
    CMD="$CMD --compute_device $COMPUTE_DEVICE"
fi

if [[ "$FORCE_SLOW_LORA_PATH" == "true" ]]; then
    CMD="$CMD --force_slow_lora_path"
fi

# Export environment variables
export LOADWORKER=$LOADWORKER
export LIGHTLLM_LOGGING=$LIGHTLLM_LOGGING
export MOE_MODE=$MOE_MODE
export MOCK_PREFILL_LOGITS=$MOCK_PREFILL_LOGITS

echo ""
echo "Starting server..."
echo "Command: $CMD"
echo ""
echo "Environment:"
echo "  LOADWORKER=$LOADWORKER"
echo "  LIGHTLLM_LOGGING=$LIGHTLLM_LOGGING"
echo "  MOE_MODE=$MOE_MODE"
echo "  MOCK_PREFILL_LOGITS=$MOCK_PREFILL_LOGITS"
echo ""

# Execute
eval "$CMD"
