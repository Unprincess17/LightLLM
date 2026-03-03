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
SERVER_SCRIPT="$TEST_SCRIPT_DIR/start_server.sh"
SERVER_HOST="localhost"
SERVER_PORT=8040
SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --delay) SETUP_DELAY="$2"; shift 2 ;;
        --output) OUTPUT_PREFIX="$2"; shift 2 ;;
        --test_script) TEST_SCRIPT="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
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

sleep 5

echo "=============================================="
echo "Starting MoE Profiling"
echo "=============================================="

# Step 1: Start nsys profiling with server
echo "[1/5] Launching server"
bash "$SERVER_SCRIPT" --host "$SERVER_HOST" --port "$SERVER_PORT" &
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
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 1 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 2 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 4 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 8 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 16 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 32 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 64 2>&1 | tee -a benchmark_lora.log

sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 128 2>&1 | tee -a benchmark_lora.log

sleep 5
python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 256 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 512 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 1024 2>&1 | tee -a benchmark_lora.log

# sleep 5
# python "$TEST_SCRIPT" --max_tokens "$MAX_TOKENS" --num_requests 2048 2>&1 | tee -a benchmark_lora.log