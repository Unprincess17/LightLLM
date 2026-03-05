#!/bin/bash
set -e

# =============================================================================
# Automated MoE LoRA Profiling Script
# =============================================================================


# Default values
SETUP_DELAY=10
MAX_WAIT=1200
OUTPUT_PREFIX="moe_offload_profile"
TEST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$TEST_SCRIPT_DIR/test_moe_lora_api.py"
MAX_TOKENS=1
DECODE_TARGET_TOKENS=8
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
        --adapter_expert_profile) ADAPTER_EXPERT_PROFILE="$2"; shift 2 ;;
        --adapter_expert_log_path) ADAPTER_EXPERT_LOG_PATH="$2"; shift 2 ;;
        --print_per_request) PRINT_PER_REQUEST=1; shift ;;
        --no_print_per_request) PRINT_PER_REQUEST=0; shift ;;
        --top_k_slowest) TOP_K_SLOWEST="$2"; shift 2 ;;
        --per_request_log_path) PER_REQUEST_LOG_PATH="$2"; shift 2 ;;
        --no_per_request_log) PER_REQUEST_LOG_PATH=""; shift ;;
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
python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 1 2>&1 | tee -a benchmark_lora.log

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

# sleep 5
# python "$TEST_SCRIPT" "${COMMON_TEST_ARGS[@]}" --num_requests 16 2>&1 | tee -a benchmark_lora.log

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
