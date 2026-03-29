#!/bin/bash
# =============================================================================
# LoRA Serving Launcher Script
# =============================================================================
# Usage:
#   ./start_server.sh [OPTIONS]
#
# Options:
#   --model_dir PATH       Base model directory (required)
#   --lora_dir PATH        Single LoRA adapter directory (optional)
#   --lora_dirs PATHS      Comma-separated LoRA adapter directories (optional)
#   --no_lora              Disable loading default or explicit LoRA adapters
#   --port PORT            Server port (default: 8080)
#   --tp TP                Tensor parallel degree (default: 1)
#   --host HOST            Server host (default: 127.0.0.1)
#   --enable_multimodal    Enable multimodal support
#   --lora_max_size SIZE   Max LoRA size (default: 1024)
#   --compute_device STR   Configure LoRA storage and compute locations
#                          Format: 'vl_storage:{gpu|cpu},vl_compute:{gpu|cpu|off},attn_storage:{gpu|cpu},attn_compute:{gpu|cpu|off},moe_storage:{gpu|cpu},moe_compute:{gpu|cpu|hybrid|off}'
#                          Short format 'moe:cpu' sets both storage and compute
#                          Example: 'moe:cpu' - MoE on CPU (both storage and compute)
#                          Example: 'vl_storage:cpu,vl_compute:gpu' - VL weights on CPU, compute on GPU
#   --force_slow_lora_path Force per-expert LoRA path (needed when fused MoE path cannot inject LoRA)
#   --no_force_slow_lora_path Disable forced slow path (advanced; only if your model path injects LoRA without it)
#   --max_req_total_len    Max request total length
#   --mem_fraction         Memory fraction (default: 0.6)
#   --batch_max_tokens     Batch max tokens (default: 4096)
#   --colora_cache_budget_mb MB        COLoRA GPU hot cache budget
#   --colora_promote_min_hits N        COLoRA min accesses before promotion
#   --colora_promote_window N          COLoRA utility window
#   --colora_max_promote_per_step N    COLoRA max promotions per step
#   --colora_decay F                   COLoRA utility decay
#   --colora_miss_policy STR           COLoRA miss policy (cpu_first|load_then_run)
#   --colora_async_fallback 0|1        Enable async CPU fallback in hybrid mode
#   --colora_cpu_workers N             Async CPU fallback worker threads
#   --colora_cpu_queue_depth N         Async CPU fallback queue depth
#   --colora_cpu_batch_timeout_us N    Queue wait budget before sync degrade
#   --colora_speculative_dispatch      Enable COLoRA speculative dispatch MVP
#   --no_colora_speculative_dispatch   Disable COLoRA speculative dispatch MVP
#   --colora_spec_layer_whitelist CSV  Comma-separated speculative layer whitelist
#   --adapter_expert_profile Enable adapter x expert routing profiling log
#   --adapter_expert_log_path PATH Log path for adapter x expert routing profiling
#   --router_trace Enable ordered per-token router trace logging
#   --router_trace_path PATH Log path for ordered router trace JSONL
#   --router_trace_phases CSV Phase filter for router trace, e.g. 'decode' or 'prefill,decode'
#   --help                 Show this help message
#
# =============================================================================

set -e

usage() {
    cat <<'USAGE'
Usage: ./start_server.sh [OPTIONS]

Options:
  --model_dir PATH
  --lora_dir PATH
  --lora_dirs PATHS
  --no_lora
  --port PORT
  --tp TP
  --host HOST
  --enable_multimodal
  --lora_max_size SIZE
  --compute_device STR
  --force_slow_lora_path
  --no_force_slow_lora_path (default: force_slow enabled)
  --max_req_total_len N
  --mem_fraction F
  --batch_max_tokens N
  --colora_cache_budget_mb MB
  --colora_promote_min_hits N
  --colora_promote_window N
  --colora_max_promote_per_step N
  --colora_decay F
  --colora_miss_policy STR
  --colora_async_fallback 0|1
  --colora_cpu_workers N
  --colora_cpu_queue_depth N
  --colora_cpu_batch_timeout_us N
  --colora_speculative_dispatch
  --no_colora_speculative_dispatch
  --colora_spec_layer_whitelist CSV
  --adapter_expert_profile
  --adapter_expert_log_path PATH
  --router_trace
  --router_trace_path PATH
  --router_trace_phases CSV
  --help|-h
USAGE
}

# Env
export MOE_PROFILING=1

# Default values
MODEL_DIR="/home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
DEFAULT_LORA_DIRS=()
for i in {0..9}; do
    DEFAULT_LORA_DIRS+=("/home/shufan/Qwen-VL-FT/work/lora_dummy_${i}")
done
LORA_DIR=$(IFS=,; echo "${DEFAULT_LORA_DIRS[*]}")
USE_LORA=true
PORT=8040
TP=2
HOST="0.0.0.0"
ENABLE_MULTIMODAL=true
LORA_MAX_SIZE=1024

<<EOF
# This two baselines are not implemented yet.
### TODO: it's fails.
### Baseline 1: Store on CPU, compute Attn on CPU, MoE off 
# COMPUTE_DEVICE="vl_storage:cpu,vl_compute:gpu,attn_storage:cpu,attn_compute:cpu,moe_storage:off,moe_compute:off"

### TODO: This is a little faulty. To fast, and contains CUDA kernel.
### Baseline 2: Store on CPU, Compute on CPU ###
# COMPUTE_DEVICE="vl_storage:cpu,vl_compute:cpu,attn_storage:cpu,attn_compute:cpu,moe_storage:cpu,moe_compute:cpu"
EOF

### Baseline 3: Store on CPU, Compute on GPU ###
# COMPUTE_DEVICE="vl_storage:cpu,vl_compute:gpu,attn_storage:cpu,attn_compute:gpu,moe_storage:cpu,moe_compute:gpu"

### Baseline 4: Store on GPU, compute on GPU ###
# COMPUTE_DEVICE="vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:gpu,moe_compute:gpu"

### Optional baseline (CPU MoE compute):
# COMPUTE_DEVICE="vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:cpu"

### COLoRA default: GPU hit + CPU miss for MoE LoRA ###
COMPUTE_DEVICE="vl_storage:gpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid"

FORCE_SLOW_LORA_PATH=true
MAX_REQ_TOTAL_LEN=8192
MEM_FRACTION=0.6
BATCH_MAX_TOKENS=4096
COLORA_CACHE_BUDGET_MB=8192
COLORA_PROMOTE_MIN_HITS=2
COLORA_PROMOTE_WINDOW=128
COLORA_MAX_PROMOTE_PER_STEP=8
COLORA_DECAY=0.9
COLORA_DEFERRED_PROMOTION_DELTA_STEPS=4
COLORA_PROMOTION_EMA_ALPHA=0.5
COLORA_MISS_POLICY="cpu_first"
COLORA_ASYNC_FALLBACK=1
COLORA_CPU_WORKERS=4
COLORA_CPU_QUEUE_DEPTH=256
COLORA_CPU_BATCH_TIMEOUT_US=50
COLORA_TEMPORAL_PREFETCH=0
COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST=""
COLORA_TEMPORAL_HOT_CACHE_SLOTS=64
COLORA_SPECULATIVE_DISPATCH=0
COLORA_SPEC_LAYER_WHITELIST=""

# Environment variables
LOADWORKER=8
LIGHTLLM_LOGGING="${LIGHTLLM_LOGGING:-DEBUG}"
MOE_MODE="TP"
MOCK_PREFILL_LOGITS="TRUE"
#MOCK_PREFILL_LOGITS="FALSE"
MOE_ADAPTER_EXPERT_PROFILING=0
MOE_ADAPTER_EXPERT_LOG_PATH="/tmp/moe_adapter_expert_profile.log"
MOE_ROUTER_TRACE=0
MOE_ROUTER_TRACE_PATH="/tmp/moe_router_trace.jsonl"
MOE_ROUTER_TRACE_PHASES="prefill,decode"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --lora_dir)
            LORA_DIR="$2"
            USE_LORA=true
            shift 2
            ;;
        --lora_dirs)
            LORA_DIR="$2"
            USE_LORA=true
            shift 2
            ;;
        --no_lora)
            LORA_DIR=""
            USE_LORA=false
            shift
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
        --no_force_slow_lora_path)
            FORCE_SLOW_LORA_PATH=false
            shift
            ;;
        --max_req_total_len)
            MAX_REQ_TOTAL_LEN="$2"
            shift 2
            ;;
        --mem_fraction)
            MEM_FRACTION="$2"
            shift 2
            ;;
        --batch_max_tokens)
            BATCH_MAX_TOKENS="$2"
            shift 2
            ;;
        --colora_cache_budget_mb)
            COLORA_CACHE_BUDGET_MB="$2"
            shift 2
            ;;
        --colora_promote_min_hits)
            COLORA_PROMOTE_MIN_HITS="$2"
            shift 2
            ;;
        --colora_promote_window)
            COLORA_PROMOTE_WINDOW="$2"
            shift 2
            ;;
        --colora_max_promote_per_step)
            COLORA_MAX_PROMOTE_PER_STEP="$2"
            shift 2
            ;;
        --colora_decay)
            COLORA_DECAY="$2"
            shift 2
            ;;
        --colora_deferred_promotion_delta_steps)
            COLORA_DEFERRED_PROMOTION_DELTA_STEPS="$2"
            shift 2
            ;;
        --colora_promotion_ema_alpha)
            COLORA_PROMOTION_EMA_ALPHA="$2"
            shift 2
            ;;
        --colora_miss_policy)
            COLORA_MISS_POLICY="$2"
            shift 2
            ;;
        --colora_async_fallback)
            COLORA_ASYNC_FALLBACK="$2"
            shift 2
            ;;
        --colora_cpu_workers)
            COLORA_CPU_WORKERS="$2"
            shift 2
            ;;
        --colora_cpu_queue_depth)
            COLORA_CPU_QUEUE_DEPTH="$2"
            shift 2
            ;;
        --colora_cpu_batch_timeout_us)
            COLORA_CPU_BATCH_TIMEOUT_US="$2"
            shift 2
            ;;
        --colora_temporal_prefetch)
            COLORA_TEMPORAL_PREFETCH=1
            shift
            ;;
        --colora_temporal_prefetch_layer_whitelist)
            COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST="$2"
            shift 2
            ;;
        --colora_temporal_hot_cache_slots)
            COLORA_TEMPORAL_HOT_CACHE_SLOTS="$2"
            shift 2
            ;;
        --colora_speculative_dispatch)
            COLORA_SPECULATIVE_DISPATCH=1
            shift
            ;;
        --no_colora_speculative_dispatch)
            COLORA_SPECULATIVE_DISPATCH=0
            shift
            ;;
        --colora_spec_layer_whitelist)
            COLORA_SPEC_LAYER_WHITELIST="$2"
            shift 2
            ;;
        --adapter_expert_profile)
            MOE_ADAPTER_EXPERT_PROFILING=1
            shift
            ;;
        --adapter_expert_log_path)
            MOE_ADAPTER_EXPERT_LOG_PATH="$2"
            shift 2
            ;;
        --router_trace)
            MOE_ROUTER_TRACE=1
            shift
            ;;
        --router_trace_path)
            MOE_ROUTER_TRACE_PATH="$2"
            shift 2
            ;;
        --router_trace_phases)
            MOE_ROUTER_TRACE_PHASES="$2"
            shift 2
            ;;
        --help|-h)
            usage
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
if [[ "$USE_LORA" == "true" && -n "$LORA_DIR" ]]; then
    IFS=',' read -r -a LORA_DIR_ARRAY <<< "$LORA_DIR"
    ABS_LORA_DIRS=()
    for ADAPTER_DIR in "${LORA_DIR_ARRAY[@]}"; do
        ADAPTER_DIR="${ADAPTER_DIR#"${ADAPTER_DIR%%[![:space:]]*}"}"
        ADAPTER_DIR="${ADAPTER_DIR%"${ADAPTER_DIR##*[![:space:]]}"}"
        [[ -z "$ADAPTER_DIR" ]] && continue
        if [[ ! -d "$ADAPTER_DIR" ]]; then
            echo "ERROR: LoRA directory not found: $ADAPTER_DIR"
            exit 1
        fi
        ABS_LORA_DIRS+=("$(cd "$ADAPTER_DIR" && pwd)")
    done
    LORA_DIR=$(IFS=,; echo "${ABS_LORA_DIRS[*]}")
fi

# Get absolute paths
MODEL_DIR=$(cd "$MODEL_DIR" && pwd)

echo "=============================================="
echo "LightLLM LoRA Server"
echo "=============================================="
echo "Base Model: $MODEL_DIR"
if [[ "$USE_LORA" == "true" && -n "$LORA_DIR" ]]; then
    echo "LoRA Adapter(s): $LORA_DIR"
else
    echo "LoRA Adapter(s): disabled"
fi
echo "Host: $HOST"
echo "Port: $PORT"
echo "TP: $TP"
echo "=============================================="

# Clean up shared memory segments
ipcs -m | awk -v user="$USER" '$3 == user && $6 == "0" && $2 ~ /^[0-9]+$/ {print $2}' | xargs -r -n 1 ipcrm -m

# cache
echo "Cache model in memory (vmtouch)"
find  "$MODEL_DIR" -name "*safetensors" | xargs -n 1 realpath | xargs vmtouch -vt

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

if [[ "$USE_LORA" == "true" && -n "$LORA_DIR" ]]; then
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

if [[ "$FORCE_SLOW_LORA_PATH" != "true" && "$COMPUTE_DEVICE" == *"moe_compute:hybrid"* ]]; then
    echo "WARNING: FORCE_SLOW_LORA_PATH is disabled while moe_compute:hybrid is set."
    echo "         If fused MoE path cannot inject LoRA on your model, results may be invalid."
fi

CMD="$CMD --colora_cache_budget_mb $COLORA_CACHE_BUDGET_MB \
    --colora_promote_min_hits $COLORA_PROMOTE_MIN_HITS \
    --colora_promote_window $COLORA_PROMOTE_WINDOW \
    --colora_max_promote_per_step $COLORA_MAX_PROMOTE_PER_STEP \
    --colora_decay $COLORA_DECAY \
    --colora_deferred_promotion_delta_steps $COLORA_DEFERRED_PROMOTION_DELTA_STEPS \
    --colora_promotion_ema_alpha $COLORA_PROMOTION_EMA_ALPHA \
    --colora_miss_policy $COLORA_MISS_POLICY \
    --colora_async_fallback $COLORA_ASYNC_FALLBACK \
    --colora_cpu_workers $COLORA_CPU_WORKERS \
    --colora_cpu_queue_depth $COLORA_CPU_QUEUE_DEPTH \
    --colora_cpu_batch_timeout_us $COLORA_CPU_BATCH_TIMEOUT_US"

COLORA_SPEC_LAYER_WHITELIST="${COLORA_SPEC_LAYER_WHITELIST//[[:space:]]/}"
COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST="${COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST//[[:space:]]/}"
if [[ "$COLORA_TEMPORAL_PREFETCH" == "1" ]]; then
    CMD="$CMD --colora_temporal_prefetch"
fi
if [[ -n "$COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST" ]]; then
    CMD="$CMD --colora_temporal_prefetch_layer_whitelist $COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST"
fi
if [[ -n "$COLORA_TEMPORAL_HOT_CACHE_SLOTS" ]]; then
    CMD="$CMD --colora_temporal_hot_cache_slots $COLORA_TEMPORAL_HOT_CACHE_SLOTS"
fi
if [[ "$COLORA_SPECULATIVE_DISPATCH" == "1" ]]; then
    CMD="$CMD --colora_speculative_dispatch"
fi

if [[ -n "$COLORA_SPEC_LAYER_WHITELIST" ]]; then
    CMD="$CMD --colora_spec_layer_whitelist $COLORA_SPEC_LAYER_WHITELIST"
fi

# Export environment variables
export LOADWORKER=$LOADWORKER
export LIGHTLLM_LOGGING=$LIGHTLLM_LOGGING
export MOE_MODE=$MOE_MODE
export MOCK_PREFILL_LOGITS=$MOCK_PREFILL_LOGITS
export MOE_ADAPTER_EXPERT_PROFILING=$MOE_ADAPTER_EXPERT_PROFILING
export MOE_ADAPTER_EXPERT_LOG_PATH=$MOE_ADAPTER_EXPERT_LOG_PATH
export MOE_ROUTER_TRACE=$MOE_ROUTER_TRACE
export MOE_ROUTER_TRACE_PATH=$MOE_ROUTER_TRACE_PATH
export MOE_ROUTER_TRACE_PHASES=$MOE_ROUTER_TRACE_PHASES

echo ""
echo "Starting server..."
echo "Command: $CMD"
echo ""
echo "Environment:"
echo "  LOADWORKER=$LOADWORKER"
echo "  LIGHTLLM_LOGGING=$LIGHTLLM_LOGGING"
echo "  MOE_MODE=$MOE_MODE"
echo "  MOCK_PREFILL_LOGITS=$MOCK_PREFILL_LOGITS"
echo "  MOE_ADAPTER_EXPERT_PROFILING=$MOE_ADAPTER_EXPERT_PROFILING"
echo "  MOE_ADAPTER_EXPERT_LOG_PATH=$MOE_ADAPTER_EXPERT_LOG_PATH"
echo "  MOE_ROUTER_TRACE=$MOE_ROUTER_TRACE"
echo "  MOE_ROUTER_TRACE_PATH=$MOE_ROUTER_TRACE_PATH"
echo "  MOE_ROUTER_TRACE_PHASES=$MOE_ROUTER_TRACE_PHASES"
echo "  USE_LORA=$USE_LORA"
echo "  FORCE_SLOW_LORA_PATH=$FORCE_SLOW_LORA_PATH"
echo "  COMPUTE_DEVICE=$COMPUTE_DEVICE"
echo "  COLORA_CACHE_BUDGET_MB=$COLORA_CACHE_BUDGET_MB"
echo "  COLORA_PROMOTE_MIN_HITS=$COLORA_PROMOTE_MIN_HITS"
echo "  COLORA_PROMOTE_WINDOW=$COLORA_PROMOTE_WINDOW"
echo "  COLORA_MAX_PROMOTE_PER_STEP=$COLORA_MAX_PROMOTE_PER_STEP"
echo "  COLORA_DECAY=$COLORA_DECAY"
echo "  COLORA_DEFERRED_PROMOTION_DELTA_STEPS=$COLORA_DEFERRED_PROMOTION_DELTA_STEPS"
echo "  COLORA_PROMOTION_EMA_ALPHA=$COLORA_PROMOTION_EMA_ALPHA"
echo "  COLORA_MISS_POLICY=$COLORA_MISS_POLICY"
echo "  COLORA_ASYNC_FALLBACK=$COLORA_ASYNC_FALLBACK"
echo "  COLORA_CPU_WORKERS=$COLORA_CPU_WORKERS"
echo "  COLORA_CPU_QUEUE_DEPTH=$COLORA_CPU_QUEUE_DEPTH"
echo "  COLORA_CPU_BATCH_TIMEOUT_US=$COLORA_CPU_BATCH_TIMEOUT_US"
echo "  COLORA_TEMPORAL_PREFETCH=$COLORA_TEMPORAL_PREFETCH"
echo "  COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST=$COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST"
echo "  COLORA_TEMPORAL_HOT_CACHE_SLOTS=$COLORA_TEMPORAL_HOT_CACHE_SLOTS"
echo "  COLORA_SPECULATIVE_DISPATCH=$COLORA_SPECULATIVE_DISPATCH"
echo "  COLORA_SPEC_LAYER_WHITELIST=$COLORA_SPEC_LAYER_WHITELIST"
echo ""


eval "exec $CMD"
