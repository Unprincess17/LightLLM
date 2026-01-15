#!/bin/bash
# =============================================================================
# Start MoE LoRA Server Script
#
# This script starts the LightLLM server with MoE model and LoRA support.
# Supports both merged and detached LoRA modes, with GPU and CPU offload options.
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Model configuration
MODEL_NAME=${MODEL_NAME:-"Qwen3-VL-30B-A3B-Instruct"}
MODEL_PATH=${MODEL_PATH:-"/home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"}
TP_SIZE=${TP_SIZE:-1}
# For MoE, we need to specify EP or TP mode
MOE_MODE=${MOE_MODE:-"TP"}  # Options: "TP" or "EP"

# LoRA configuration
LORA_DIR=${LORA_DIR:-""}  # Path to LoRA adapter directory (optional)
LORA_MODE=${LORA_MODE:-"detached"}  # Options: "merged" or "detached"
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-1.0}
COMPUTE_ON_CPU=${COMPUTE_ON_CPU:-false}  # true for CPU offload mode

# Server configuration
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8040}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
MAX_REQ_TOKENS=${MAX_REQ_TOKENS:-8192}

# GPU configuration
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
}

check_model_path() {
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model path does not exist: $MODEL_PATH"
        return 1
    fi
    log_info "Model path exists: $MODEL_PATH"
    return 0
}

check_lora_dir() {
    if [ -n "$LORA_DIR" ] && [ ! -d "$LORA_DIR" ]; then
        log_error "LoRA directory does not exist: $LORA_DIR"
        return 1
    fi
    if [ -n "$LORA_DIR" ]; then
        log_info "LoRA directory: $LORA_DIR"
    else
        log_warn "No LoRA directory specified - running without LoRA"
    fi
    return 0
}

print_config() {
    echo ""
    echo "============================================================"
    echo "MoE LoRA Server Configuration"
    echo "============================================================"
    echo "Model:        $MODEL_NAME"
    echo "Model path:   $MODEL_PATH"
    echo "TP size:      $TP_SIZE"
    echo "MoE mode:     $MOE_MODE"
    echo "------------------------------------------------------------"
    echo "LoRA mode:    $LORA_MODE"
    echo "LoRA dir:     ${LORA_DIR:-'(none)'}"
    echo "LoRA rank:    $LORA_RANK"
    echo "LoRA alpha:   $LORA_ALPHA"
    echo "CPU offload:  $COMPUTE_ON_CPU"
    echo "------------------------------------------------------------"
    echo "Host:         $HOST"
    echo "Port:         $PORT"
    echo "Max batch:    $MAX_BATCH_SIZE"
    echo "Max tokens:   $MAX_REQ_TOKENS"
    echo "------------------------------------------------------------"
    echo "GPU:          $CUDA_VISIBLE_DEVICES"
    echo "============================================================"
    echo ""
}

# =============================================================================
# Build Arguments
# =============================================================================

build_args() {
    local args=""

    # Model config
    args="$args --model_dir $MODEL_PATH"
    args="$args --model_name $MODEL_NAME"
    args="$args --tp $TP_SIZE"

    # LoRA config
    if [ -n "$LORA_DIR" ]; then
        args="$args --lora_dir $LORA_DIR"
        args="$args --lora_max_size $LORA_RANK"
    fi

    # Server config
    args="$args --host $HOST"
    args="$args --port $PORT"
    args="$args --batch_max_tokens $MAX_BATCH_SIZE"
    args="$args --max_req_total_len $MAX_REQ_TOKENS"

    # Other options
    args="$args --trust_remote_code"
    args="$args --enable_multimodal"
    args="$args --mem_fraction 0.9"

    echo "$args"
}

# =============================================================================
# Start Server
# =============================================================================

start_server() {
    print_config

    # Set environment
    export CUDA_VISIBLE_DEVICES

    log_info "Starting MoE LoRA server..."

    local args=$(build_args)
    local cmd="python -m lightllm.server.api_server $args"

    log_info "Command: $cmd"
    echo ""

    exec $cmd
}

# =============================================================================
# Quick Start Functions
# =============================================================================

start_no_lora() {
    log_info "Starting server WITHOUT LoRA..."
    LORA_DIR=""
    start_server
}

start_detached_gpu() {
    log_info "Starting server with detached LoRA (GPU mode)..."
    LORA_MODE="detached"
    COMPUTE_ON_CPU="false"
    start_server
}

start_detached_cpu() {
    log_info "Starting server with detached LoRA (CPU offload mode)..."
    LORA_MODE="detached"
    COMPUTE_ON_CPU="true"
    start_server
}

start_merged() {
    log_info "Starting server with merged LoRA..."
    LORA_MODE="merged"
    start_server
}

# =============================================================================
# Main Entry Point
# =============================================================================

usage() {
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_NAME          Model name (default: Qwen3-VL-30B-A3B-Instruct)"
    echo "  MODEL_PATH          Path to model directory"
    echo "  TP_SIZE             Tensor parallel size (default: 1)"
    echo "  MOE_MODE            MoE mode: TP or EP (default: TP)"
    echo "  LORA_DIR            Path to LoRA adapter directory"
    echo "  LORA_MODE           LoRA mode: merged or detached (default: detached)"
    echo "  LORA_RANK           LoRA rank (default: 64)"
    echo "  LORA_ALPHA          LoRA alpha (default: 1.0)"
    echo "  COMPUTE_ON_CPU      Compute LoRA on CPU: true or false (default: false)"
    echo "  HOST                Server host (default: 0.0.0.0)"
    echo "  PORT                Server port (default: 8080)"
    echo "  MAX_BATCH_SIZE      Maximum batch size (default: 32)"
    echo "  MAX_REQ_TOKENS      Maximum request tokens (default: 8192)"
    echo "  CUDA_VISIBLE_DEVICES GPU devices (default: 0)"
    echo ""
    echo "Quick Start Examples:"
    echo ""
    echo "  # Start without LoRA"
    echo "  $0"
    echo ""
    echo "  # Start with detached LoRA on GPU"
    export LORA_DIR=/path/to/lora_adapter
    echo "  LORA_DIR=\$LORA_DIR $0"
    echo ""
    echo "  # Start with detached LoRA on CPU"
    export LORA_DIR=/path/to/lora_adapter
    export COMPUTE_ON_CPU=true
    echo "  LORA_DIR=\$LORA_DIR COMPUTE_ON_CPU=true $0"
    echo ""
    echo "  # Start with merged LoRA"
    export LORA_DIR=/path/to/lora_adapter
    export LORA_MODE=merged
    echo "  LORA_DIR=\$LORA_DIR LORA_MODE=merged $0"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

# Check prerequisites
check_gpu
check_model_path || exit 1
check_lora_dir || log_warn "LoRA directory check failed, but continuing..."

# Start server
start_server
