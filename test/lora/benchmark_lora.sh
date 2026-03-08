#!/bin/bash
set -e

# =============================================================================
# Automated MoE LoRA Profiling Script
# =============================================================================

usage() {
    cat <<'USAGE'
Usage: ./benchmark_lora.sh [OPTIONS]

Core options:
  --delay SEC
  --max_tokens N
  --decode_target_tokens N
  --ignore_eos | --no_ignore_eos
  --adapter_ids CSV
  --poisson_lambda F
  --poisson_seed N
  --adapter_expert_profile [0|1] | --no_adapter_expert_profile
  --top_k_slowest N
  --print_per_request | --no_print_per_request
  --per_request_log_path PATH | --no_per_request_log

Server pass-through options:
  --model_dir PATH
  --lora_dirs CSV
  --tp N
  --compute_device STR
  --force_slow_lora_path | --no_force_slow_lora_path (default: enabled)
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
USAGE
}


# Default values
SETUP_DELAY=10
MAX_WAIT=1200
OUTPUT_PREFIX="moe_offload_profile"
TEST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$TEST_SCRIPT_DIR/test_moe_lora_api.py"
MAX_TOKENS=1
DECODE_TARGET_TOKENS=16
IGNORE_EOS=1
SERVER_SCRIPT="$TEST_SCRIPT_DIR/start_server.sh"
SERVER_HOST="localhost"
SERVER_PORT=8040
SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"
ADAPTER_IDS="lora_dummy_0,lora_dummy_1,lora_dummy_2,lora_dummy_3,lora_dummy_4,lora_dummy_5,lora_dummy_6,lora_dummy_7,lora_dummy_8,lora_dummy_9"
POISSON_LAMBDA=3.0
POISSON_SEED=42
ADAPTER_EXPERT_PROFILE=1
ADAPTER_EXPERT_LOG_PATH="/tmp/moe_adapter_expert_profile.log"
PRINT_PER_REQUEST=0
TOP_K_SLOWEST=10
PER_REQUEST_LOG_PATH="/tmp/moe_per_request_metrics_$(date +%Y%m%d_%H%M%S).jsonl"
MODEL_DIR=""
LORA_DIRS=""
TP=""
COMPUTE_DEVICE=""
FORCE_SLOW_LORA_PATH="1"
MAX_REQ_TOTAL_LEN=""
MEM_FRACTION=""
BATCH_MAX_TOKENS=""
COLORA_CACHE_BUDGET_MB=""
COLORA_PROMOTE_MIN_HITS=""
COLORA_PROMOTE_WINDOW=""
COLORA_MAX_PROMOTE_PER_STEP=""
COLORA_DECAY=""
COLORA_MISS_POLICY=""
COLORA_ASYNC_FALLBACK=""
COLORA_CPU_WORKERS=""
COLORA_CPU_QUEUE_DEPTH=""
COLORA_CPU_BATCH_TIMEOUT_US=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --delay) SETUP_DELAY="$2"; shift 2 ;;
        --output) OUTPUT_PREFIX="$2"; shift 2 ;;
        --test_script) TEST_SCRIPT="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --decode_target_tokens) DECODE_TARGET_TOKENS="$2"; shift 2 ;;
        --ignore_eos) IGNORE_EOS=1; shift ;;
        --no_ignore_eos) IGNORE_EOS=0; shift ;;
        --adapter_ids) ADAPTER_IDS="$2"; shift 2 ;;
        --poisson_lambda) POISSON_LAMBDA="$2"; shift 2 ;;
        --poisson_seed) POISSON_SEED="$2"; shift 2 ;;
        --adapter_expert_profile)
            if [[ -n "${2:-}" && "$2" != --* ]]; then
                ADAPTER_EXPERT_PROFILE="$2"
                shift 2
            else
                ADAPTER_EXPERT_PROFILE=1
                shift
            fi
            ;;
        --no_adapter_expert_profile) ADAPTER_EXPERT_PROFILE=0; shift ;;
        --adapter_expert_log_path) ADAPTER_EXPERT_LOG_PATH="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --lora_dirs) LORA_DIRS="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --compute_device) COMPUTE_DEVICE="$2"; shift 2 ;;
        --force_slow_lora_path) FORCE_SLOW_LORA_PATH="1"; shift ;;
        --no_force_slow_lora_path) FORCE_SLOW_LORA_PATH="0"; shift ;;
        --max_req_total_len) MAX_REQ_TOTAL_LEN="$2"; shift 2 ;;
        --mem_fraction) MEM_FRACTION="$2"; shift 2 ;;
        --batch_max_tokens) BATCH_MAX_TOKENS="$2"; shift 2 ;;
        --colora_cache_budget_mb) COLORA_CACHE_BUDGET_MB="$2"; shift 2 ;;
        --colora_promote_min_hits) COLORA_PROMOTE_MIN_HITS="$2"; shift 2 ;;
        --colora_promote_window) COLORA_PROMOTE_WINDOW="$2"; shift 2 ;;
        --colora_max_promote_per_step) COLORA_MAX_PROMOTE_PER_STEP="$2"; shift 2 ;;
        --colora_decay) COLORA_DECAY="$2"; shift 2 ;;
        --colora_miss_policy) COLORA_MISS_POLICY="$2"; shift 2 ;;
        --colora_async_fallback) COLORA_ASYNC_FALLBACK="$2"; shift 2 ;;
        --colora_cpu_workers) COLORA_CPU_WORKERS="$2"; shift 2 ;;
        --colora_cpu_queue_depth) COLORA_CPU_QUEUE_DEPTH="$2"; shift 2 ;;
        --colora_cpu_batch_timeout_us) COLORA_CPU_BATCH_TIMEOUT_US="$2"; shift 2 ;;
        --print_per_request) PRINT_PER_REQUEST=1; shift ;;
        --no_print_per_request) PRINT_PER_REQUEST=0; shift ;;
        --top_k_slowest) TOP_K_SLOWEST="$2"; shift 2 ;;
        --per_request_log_path) PER_REQUEST_LOG_PATH="$2"; shift 2 ;;
        --no_per_request_log) PER_REQUEST_LOG_PATH=""; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Cleanup function
cleanup() {
    echo "[Cleanup] Sending signal to all processes in group..."
    trap - INT TERM EXIT # Disable traps to avoid recursion

    # 1. Send SIGINT to the group. 
    # This tells nsys to "stop and save" and the server to "gracefully exit."
    kill -INT -$$ 2>/dev/null

    pkill -9 -f "lightllm.server|lightllm::|gunicorn" && \
    pkill -9 -f "multiprocessing.resource_tracker|multiprocessing.spawn"
    ipcs -m | awk -v user="$USER" '$3 == user && $6 == "0" && $2 ~ /^[0-9]+$/ {print $2}' | xargs -r -n 1 ipcrm -m

    echo "[Cleanup] Waiting for nsys to finalize report (max 15s)..."
    # Wait for the specific nsys process to finish
    # nsys post-processing can take a while for large MoE models
    for i in {1..15}; do
        if ! pgrep -x "nsys" > /dev/null; then
            echo "[Cleanup] nsys finished."
            break
        fi
        sleep 1
    done
}
trap cleanup EXIT

# Step 0: clean up old processes if any
echo "cleanup processes"
pgrep -f "lightllm.server|lightllm::|gunicorn|multiprocessing.resource_tracker|multiprocessing.spawn" && echo "Killing old processes..." && \
pkill -9 -f "lightllm.server|lightllm::|gunicorn" && \
pkill -9 -f "multiprocessing.resource_tracker|multiprocessing.spawn"

echo > $ADAPTER_EXPERT_LOG_PATH

sleep 5

echo "=============================================="
echo "Starting MoE Profiling"
echo "=============================================="
echo "Adapter IDs: $ADAPTER_IDS"
echo "Poisson lambda: $POISSON_LAMBDA"
echo "Poisson seed: $POISSON_SEED"
echo "Max tokens (fallback): $MAX_TOKENS"
echo "Decode target tokens: $DECODE_TARGET_TOKENS"
echo "Ignore EOS: $IGNORE_EOS"
echo "Adapter expert profile: $ADAPTER_EXPERT_PROFILE"
echo "Adapter expert profile log: $ADAPTER_EXPERT_LOG_PATH"
echo "Top-K slowest requests: $TOP_K_SLOWEST"
echo "Print per-request lines: $PRINT_PER_REQUEST"
[[ -n "$MODEL_DIR" ]] && echo "Server model_dir override: $MODEL_DIR"
[[ -n "$LORA_DIRS" ]] && echo "Server lora_dirs override: $LORA_DIRS"
[[ -n "$TP" ]] && echo "Server TP override: $TP"
[[ -n "$COMPUTE_DEVICE" ]] && echo "Server compute_device override: $COMPUTE_DEVICE"
[[ -n "$FORCE_SLOW_LORA_PATH" ]] && echo "Server force_slow_lora_path override: $FORCE_SLOW_LORA_PATH"
[[ -n "$COLORA_MISS_POLICY" ]] && echo "COLoRA miss policy override: $COLORA_MISS_POLICY"
[[ -n "$COLORA_ASYNC_FALLBACK" ]] && echo "COLoRA async fallback override: $COLORA_ASYNC_FALLBACK"
[[ -n "$COLORA_CPU_WORKERS" ]] && echo "COLoRA CPU workers override: $COLORA_CPU_WORKERS"
[[ -n "$COLORA_CPU_QUEUE_DEPTH" ]] && echo "COLoRA CPU queue depth override: $COLORA_CPU_QUEUE_DEPTH"
[[ -n "$COLORA_CPU_BATCH_TIMEOUT_US" ]] && echo "COLoRA CPU batch timeout (us) override: $COLORA_CPU_BATCH_TIMEOUT_US"
if [[ -n "$PER_REQUEST_LOG_PATH" ]]; then
    echo "Per-request metrics log: $PER_REQUEST_LOG_PATH"
else
    echo "Per-request metrics log: disabled"
fi

COMMON_TEST_ARGS=(
    --max_tokens "$MAX_TOKENS"
    --adapter_ids "$ADAPTER_IDS"
    --poisson_lambda "$POISSON_LAMBDA"
    --poisson_seed "$POISSON_SEED"
    --top_k_slowest "$TOP_K_SLOWEST"
)
if [[ -n "$DECODE_TARGET_TOKENS" ]]; then
    COMMON_TEST_ARGS+=(--decode_target_tokens "$DECODE_TARGET_TOKENS")
fi
if [[ "$IGNORE_EOS" == "1" ]]; then
    COMMON_TEST_ARGS+=(--ignore_eos)
else
    COMMON_TEST_ARGS+=(--no_ignore_eos)
fi
if [[ "$PRINT_PER_REQUEST" == "1" ]]; then
    COMMON_TEST_ARGS+=(--print_per_request)
fi
if [[ -n "$PER_REQUEST_LOG_PATH" ]]; then
    COMMON_TEST_ARGS+=(--per_request_log_path "$PER_REQUEST_LOG_PATH")
fi

# Step 1: Start nsys profiling with server
echo "[1/5] Launching server"
SERVER_ARGS=(--host "$SERVER_HOST" --port "$SERVER_PORT")
if [[ "$ADAPTER_EXPERT_PROFILE" == "1" ]]; then
    SERVER_ARGS+=(--adapter_expert_profile --adapter_expert_log_path "$ADAPTER_EXPERT_LOG_PATH")
fi
if [[ -n "$MODEL_DIR" ]]; then
    SERVER_ARGS+=(--model_dir "$MODEL_DIR")
fi
if [[ -n "$LORA_DIRS" ]]; then
    SERVER_ARGS+=(--lora_dirs "$LORA_DIRS")
fi
if [[ -n "$TP" ]]; then
    SERVER_ARGS+=(--tp "$TP")
fi
if [[ -n "$COMPUTE_DEVICE" ]]; then
    SERVER_ARGS+=(--compute_device "$COMPUTE_DEVICE")
fi
if [[ "$FORCE_SLOW_LORA_PATH" == "1" ]]; then
    SERVER_ARGS+=(--force_slow_lora_path)
elif [[ "$FORCE_SLOW_LORA_PATH" == "0" ]]; then
    SERVER_ARGS+=(--no_force_slow_lora_path)
fi
if [[ -n "$MAX_REQ_TOTAL_LEN" ]]; then
    SERVER_ARGS+=(--max_req_total_len "$MAX_REQ_TOTAL_LEN")
fi
if [[ -n "$MEM_FRACTION" ]]; then
    SERVER_ARGS+=(--mem_fraction "$MEM_FRACTION")
fi
if [[ -n "$BATCH_MAX_TOKENS" ]]; then
    SERVER_ARGS+=(--batch_max_tokens "$BATCH_MAX_TOKENS")
fi
if [[ -n "$COLORA_CACHE_BUDGET_MB" ]]; then
    SERVER_ARGS+=(--colora_cache_budget_mb "$COLORA_CACHE_BUDGET_MB")
fi
if [[ -n "$COLORA_PROMOTE_MIN_HITS" ]]; then
    SERVER_ARGS+=(--colora_promote_min_hits "$COLORA_PROMOTE_MIN_HITS")
fi
if [[ -n "$COLORA_PROMOTE_WINDOW" ]]; then
    SERVER_ARGS+=(--colora_promote_window "$COLORA_PROMOTE_WINDOW")
fi
if [[ -n "$COLORA_MAX_PROMOTE_PER_STEP" ]]; then
    SERVER_ARGS+=(--colora_max_promote_per_step "$COLORA_MAX_PROMOTE_PER_STEP")
fi
if [[ -n "$COLORA_DECAY" ]]; then
    SERVER_ARGS+=(--colora_decay "$COLORA_DECAY")
fi
if [[ -n "$COLORA_MISS_POLICY" ]]; then
    SERVER_ARGS+=(--colora_miss_policy "$COLORA_MISS_POLICY")
fi
if [[ -n "$COLORA_ASYNC_FALLBACK" ]]; then
    SERVER_ARGS+=(--colora_async_fallback "$COLORA_ASYNC_FALLBACK")
fi
if [[ -n "$COLORA_CPU_WORKERS" ]]; then
    SERVER_ARGS+=(--colora_cpu_workers "$COLORA_CPU_WORKERS")
fi
if [[ -n "$COLORA_CPU_QUEUE_DEPTH" ]]; then
    SERVER_ARGS+=(--colora_cpu_queue_depth "$COLORA_CPU_QUEUE_DEPTH")
fi
if [[ -n "$COLORA_CPU_BATCH_TIMEOUT_US" ]]; then
    SERVER_ARGS+=(--colora_cpu_batch_timeout_us "$COLORA_CPU_BATCH_TIMEOUT_US")
fi
bash "$SERVER_SCRIPT" "${SERVER_ARGS[@]}" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Record start time
start_time=$SECONDS

# Step 2: Wait for server to be healthy
echo "[2/5] Waiting for server to be healthy..."

while (( SECONDS - start_time < MAX_WAIT )); do
    if nc -vz "$SERVER_HOST" "$SERVER_PORT" 2>/dev/null; then
        echo ""
        echo "Server is UP (port $SERVER_PORT open)"
        break
    fi
    echo -n "."
    sleep 5
done

# Check if server is ready (break sets server_ready=true)
if [[ $(($SECONDS - start_time)) -ge $MAX_WAIT ]]; then
    echo "ERROR: Server failed to start within ${MAX_WAIT}s."
    exit 1
fi

# Step 3: Wait another delay before sending requests
echo "Waiting additional ${SETUP_DELAY}s before sending test request..."
sleep "$SETUP_DELAY"

# Step 4: Send test request (nsys is now capturing)
echo "[3/5] Sending test request..."
echo > benchmark_lora.log
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 1 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 1 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 1 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 2 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 4 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 8 2>&1 | tee -a benchmark_lora.log

sleep 5
python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 16 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 32 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 64 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 128 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 256 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 512 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 1024 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 2048 2>&1 | tee -a benchmark_lora.log
