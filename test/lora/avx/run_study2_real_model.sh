#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_SCRIPT="$ROOT_DIR/test/lora/start_server.sh"
CLIENT_SCRIPT="$ROOT_DIR/test/lora/test_moe_lora_api.py"
PARSER_SCRIPT="$ROOT_DIR/test/lora/avx/parse_study2_nsys_sqlite.py"

PORT=8040
HOST="localhost"
N_VALUES="1,8,32,64,128,256"
LAYER_ID=0
MAX_TOKENS=1
SETUP_DELAY=5
STEP_DELAY=2
ADAPTER_IDS="lora_dummy_0"
POISSON_LAMBDA=0.0
POISSON_SEED=42
OUTPUT_PREFIX="$ROOT_DIR/test/lora/avx/study2_real_model"

# Keep these consistent with your current detached CPU-MoE path.
COMPUTE_DEVICE="vl_storage:cpu,vl_compute:gpu,attn_storage:cpu,attn_compute:gpu,moe_storage:cpu,moe_compute:cpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --n-values) N_VALUES="$2"; shift 2 ;;
        --layer-id) LAYER_ID="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --setup-delay) SETUP_DELAY="$2"; shift 2 ;;
        --step-delay) STEP_DELAY="$2"; shift 2 ;;
        --adapter-ids) ADAPTER_IDS="$2"; shift 2 ;;
        --poisson-lambda) POISSON_LAMBDA="$2"; shift 2 ;;
        --poisson-seed) POISSON_SEED="$2"; shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        --compute-device) COMPUTE_DEVICE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys is not found in PATH."
    exit 1
fi

if ! command -v nc >/dev/null 2>&1; then
    echo "ERROR: nc (netcat) is required."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

echo "=============================================="
echo "Study2 Real-Model Profiling"
echo "=============================================="
echo "Output prefix: $OUTPUT_PREFIX"
echo "Target layer: $LAYER_ID"
echo "N values: $N_VALUES"
echo "Server: $HOST:$PORT"
echo "Adapters: $ADAPTER_IDS"
echo "=============================================="

echo "[Cleanup] Clearing stale server processes if any..."
pkill -9 -f "lightllm.server|lightllm::|gunicorn" 2>/dev/null || true
pkill -9 -f "multiprocessing.resource_tracker|multiprocessing.spawn" 2>/dev/null || true

TMP_RUN_SCRIPT="$(mktemp /tmp/study2_real_model_run.XXXXXX.sh)"
trap 'rm -f "$TMP_RUN_SCRIPT"' EXIT

cat > "$TMP_RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SERVER_SCRIPT="$SERVER_SCRIPT"
CLIENT_SCRIPT="$CLIENT_SCRIPT"
HOST="$HOST"
PORT="$PORT"
N_VALUES="$N_VALUES"
MAX_TOKENS="$MAX_TOKENS"
SETUP_DELAY="$SETUP_DELAY"
STEP_DELAY="$STEP_DELAY"
ADAPTER_IDS="$ADAPTER_IDS"
POISSON_LAMBDA="$POISSON_LAMBDA"
POISSON_SEED="$POISSON_SEED"
COMPUTE_DEVICE="$COMPUTE_DEVICE"
LAYER_ID="$LAYER_ID"

export MOE_STUDY2_PROFILE=1
export MOE_STUDY2_LAYER="$LAYER_ID"
export MOE_COALESCING_PACKER=1

bash "\$SERVER_SCRIPT" --host "\$HOST" --port "\$PORT" --compute_device "\$COMPUTE_DEVICE" --force_slow_lora_path &
SERVER_PID=\$!

terminate_server_tree() {
    local pid="\$1"
    if [[ -z "\$pid" ]]; then
        return 0
    fi

    # Try graceful stop first.
    kill -INT "\$pid" 2>/dev/null || true
    for _ in \$(seq 1 20); do
        if ! kill -0 "\$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    # Fallback: terminate known descendants and force-kill root pid.
    pkill -TERM -P "\$pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "\$pid" 2>/dev/null || true
    kill -KILL "\$pid" 2>/dev/null || true
}

cleanup() {
    terminate_server_tree "\$SERVER_PID"
    wait "\$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in \$(seq 1 300); do
    if nc -z "\$HOST" "\$PORT" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! nc -z "\$HOST" "\$PORT" >/dev/null 2>&1; then
    echo "ERROR: Server did not become ready in time."
    exit 1
fi

sleep "\$SETUP_DELAY"

IFS=',' read -r -a N_ARR <<< "\$N_VALUES"
for N in "\${N_ARR[@]}"; do
    N="\${N//[[:space:]]/}"
    [[ -z "\$N" ]] && continue
    echo "[Run] num_requests=\$N"
    python "\$CLIENT_SCRIPT" \
      --url "http://\$HOST:\$PORT" \
      --max_tokens "\$MAX_TOKENS" \
      --adapter_ids "\$ADAPTER_IDS" \
      --poisson_lambda "\$POISSON_LAMBDA" \
      --poisson_seed "\$POISSON_SEED" \
      --num_requests "\$N"
    sleep "\$STEP_DELAY"
done

sleep 2
terminate_server_tree "\$SERVER_PID"
wait "\$SERVER_PID" 2>/dev/null || true
trap - EXIT
EOF

chmod +x "$TMP_RUN_SCRIPT"

echo "[1/3] Running nsys profile..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output "$OUTPUT_PREFIX" \
    "$TMP_RUN_SCRIPT"

echo "[2/3] Exporting sqlite..."
nsys export \
    --type sqlite \
    --force-overwrite=true \
    --output "${OUTPUT_PREFIX}.sqlite" \
    "${OUTPUT_PREFIX}.nsys-rep"

echo "[3/3] Parsing Study2 metrics..."
python "$PARSER_SCRIPT" \
    --sqlite "${OUTPUT_PREFIX}.sqlite" \
    --mode real \
    --layer "$LAYER_ID" \
    --agg median \
    --output-csv "${OUTPUT_PREFIX}.csv"

echo "Done."
echo "Report: ${OUTPUT_PREFIX}.nsys-rep"
echo "SQLite: ${OUTPUT_PREFIX}.sqlite"
echo "CSV: ${OUTPUT_PREFIX}.csv"
