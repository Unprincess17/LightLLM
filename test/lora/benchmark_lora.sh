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
  --warmup_decode_target_tokens N
  --ignore_eos | --no_ignore_eos
  --adapter_ids CSV
  --warmup_adapter_trace_path PATH
  --measure_adapter_trace_path PATH
  --warmup_num_requests N
  --measure_num_requests N
  --max_concurrent_requests N        Max concurrent requests per phase (passed to client)
  --rps F                            Target requests per second (default: 3.0; 0 = burst)
  --phase_gap_s SEC
  --poisson_lambda F
  --poisson_seed N
  --adapter_expert_profile [0|1] | --no_adapter_expert_profile
  --top_k_slowest N
  --print_per_request | --no_print_per_request
  --per_request_log_path PATH | --no_per_request_log
  --phase STR
  --mode_label STR           # colora_min | colora_full | load_then_run
  --colora_stats_path PATH

Server pass-through options:
  --model_dir PATH
  --lora_dirs CSV
  --tp N
  --compute_device STR
  --force_slow_lora_path | --no_force_slow_lora_path (default: enabled)
  --max_req_total_len N
  --mem_fraction F
  --batch_max_tokens N
  --trust_remote_code | --no_trust_remote_code
  --disable_cudagraph | --no_disable_cudagraph
  --enable_multimodal | --no_enable_multimodal
  --lora_max_size N
  --mock_prefill_logits | --no_mock_prefill_logits
  --colora_cache_budget_mb MB
  --colora_promote_min_hits N
  --colora_promote_window N
  --colora_max_promote_per_step N
  --colora_decay F
  --colora_deferred_promotion_delta_steps N
  --colora_promotion_ema_alpha F
  --colora_miss_policy STR
  --colora_overlap_mode full|no_overlap
  --colora_async_fallback 0|1
  --colora_cpu_workers N
  --colora_cpu_queue_depth N
  --colora_cpu_batch_timeout_us N
  --colora_temporal_prefetch | --no_colora_temporal_prefetch
  --colora_temporal_prefetch_layer_whitelist CSV
  --colora_temporal_hot_cache_slots N
  --colora_speculative_dispatch | --no_colora_speculative_dispatch
  --colora_spec_layer_whitelist CSV
  --server_log_path PATH
  --server_stdout_log PATH
  --server_host HOST          Bind / health-check host (default: localhost)
  --server_port PORT          Bind port for server and client URL (default: 8040)
USAGE
}


# Default values
SETUP_DELAY=10
MAX_WAIT=2000
OUTPUT_PREFIX="moe_offload_profile"
TEST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$TEST_SCRIPT_DIR/test_moe_lora_api.py"
MAX_TOKENS=1
DECODE_TARGET_TOKENS=2
WARMUP_DECODE_TARGET_TOKENS=""
IGNORE_EOS=1
SERVER_SCRIPT="$TEST_SCRIPT_DIR/start_server.sh"
SERVER_HOST="localhost"
SERVER_PORT=8040
ADAPTER_IDS="lora_dummy_0,lora_dummy_1,lora_dummy_2,lora_dummy_3,lora_dummy_4,lora_dummy_5,lora_dummy_6,lora_dummy_7,lora_dummy_8,lora_dummy_9"
ADAPTER_IDS_SET="0"
POISSON_LAMBDA=3.0
POISSON_SEED=42
WARMUP_ADAPTER_TRACE_PATH=""
MEASURE_ADAPTER_TRACE_PATH=""
WARMUP_NUM_REQUESTS=""
MEASURE_NUM_REQUESTS=""
MAX_CONCURRENT_REQUESTS=""
RPS=""
PHASE_GAP_S="10"
ADAPTER_EXPERT_PROFILE=0
ADAPTER_EXPERT_LOG_PATH="/tmp/moe_adapter_expert_profile.log"
PRINT_PER_REQUEST=0
TOP_K_SLOWEST=10
PER_REQUEST_LOG_PATH="/tmp/moe_per_request_metrics_$(date +%Y%m%d_%H%M%S).jsonl"
PHASE=""
COLORA_STATS_PATH=""
MODEL_DIR=""
LORA_DIRS=""
LORA_CLONE_COUNT=""
TP=""
COMPUTE_DEVICE=""
FORCE_SLOW_LORA_PATH="1"
MAX_REQ_TOTAL_LEN=""
MEM_FRACTION=""
BATCH_MAX_TOKENS=""
TRUST_REMOTE_CODE=""
DISABLE_CUDAGRAPH=""
ENABLE_MULTIMODAL=""
LORA_MAX_SIZE=""
MOCK_PREFILL_LOGITS=""
COLORA_CACHE_BUDGET_MB=""
COLORA_PROMOTE_MIN_HITS="1"
COLORA_PROMOTE_WINDOW="512"
COLORA_MAX_PROMOTE_PER_STEP="32"
COLORA_DECAY=""
COLORA_DEFERRED_PROMOTION_DELTA_STEPS=""
COLORA_PROMOTION_EMA_ALPHA=""
COLORA_MISS_POLICY=""
COLORA_OVERLAP_MODE=""
COLORA_ASYNC_FALLBACK=""
COLORA_CPU_WORKERS="8"
COLORA_CPU_QUEUE_DEPTH="512"
COLORA_CPU_BATCH_TIMEOUT_US="200"
COLORA_TEMPORAL_PREFETCH="0"
COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST=""
COLORA_TEMPORAL_HOT_CACHE_SLOTS=""
COLORA_SPECULATIVE_DISPATCH="0"
COLORA_SPEC_LAYER_WHITELIST=""
COLORA_REQUEST_SKIP=""
COLORA_MAX_CONTINUATIONS=""
COLORA_HIT_INDEXING=""
SERVER_LOG_PATH=""
SERVER_PID=""
SERVER_STDOUT_LOG=""

terminate_server_tree() {
    local pid="$1"
    if [[ -z "$pid" ]]; then
        return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    echo "[Cleanup] Stopping server tree rooted at PID $pid..."

    # Prefer signaling the whole process group to stop launcher + workers together.
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 2

    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

# SIGKILL immediately after measurement can corrupt Nsight .qdstrm; try SIGTERM first.
# Gunicorn respawns workers on worker SIGKILL; kill gunicorn masters early in the SIGKILL phase.
finish_profiling_children() {
    local max_wait_s="${1:-90}"
    local _pat="lightllm.server|lightllm::|gunicorn|python.*api_server|multiprocessing.resource_tracker|multiprocessing.spawn|multiprocessing.forkserver"
    pkill -TERM -f "lightllm.server|lightllm::|gunicorn|python.*api_server" 2>/dev/null || true
    pkill -TERM -f "multiprocessing.resource_tracker|multiprocessing.spawn|multiprocessing.forkserver" 2>/dev/null || true
    local w=0
    while (( w < max_wait_s )); do
        if ! pgrep -f "$_pat" >/dev/null; then
            return 0
        fi
        sleep 1
        w=$((w + 1))
    done
    echo "[Cleanup] Some processes still alive after ${max_wait_s}s; sending SIGKILL..."
    pkill -KILL -f "gunicorn" 2>/dev/null || true
    sleep 2
    local r _p
    for r in $(seq 1 20); do
        while read -r _p; do
            [[ -z "$_p" || "$_p" == "$$" ]] && continue
            kill -KILL "$_p" 2>/dev/null || true
        done < <(pgrep -f "$_pat" 2>/dev/null || true)
        pkill -KILL -f "gunicorn" 2>/dev/null || true
        if ! pgrep -f "$_pat" >/dev/null; then
            return 0
        fi
        sleep 1
    done
    pkill -KILL -f "lightllm.server|lightllm::|gunicorn|python.*api_server" 2>/dev/null || true
    pkill -KILL -f "multiprocessing.resource_tracker|multiprocessing.spawn|multiprocessing.forkserver" 2>/dev/null || true
    sleep 1
    if pgrep -f "$_pat" >/dev/null; then
        echo "[Cleanup] WARNING: processes may still be alive; run: pgrep -af 'lightllm|gunicorn'"
        pgrep -af "lightllm|gunicorn" 2>/dev/null || true
    fi
}

build_phase_args() {
    local -n out_arr="$1"
    local decode_target_tokens="$2"
    local adapter_trace_path="$3"
    local num_requests="$4"
    local top_k_slowest="$5"
    local per_request_log_path="$6"
    local print_per_request="$7"
    local prompt_namespace="$8"
    local phase_name="$9"
    local max_concurrent_requests="${10}"
    local rps="${11}"

    out_arr=(
        --url "$SERVER_URL"
        --max_tokens "$MAX_TOKENS"
        --adapter_ids "$ADAPTER_IDS"
        --poisson_lambda "$POISSON_LAMBDA"
        --poisson_seed "$POISSON_SEED"
        --top_k_slowest "$top_k_slowest"
        --prompt_namespace "$prompt_namespace"
    )
    if [[ -n "$max_concurrent_requests" ]]; then
        out_arr+=(--max_concurrent_requests "$max_concurrent_requests")
    fi
    if [[ -n "$decode_target_tokens" ]]; then
        out_arr+=(--decode_target_tokens "$decode_target_tokens")
    fi
    if [[ "$IGNORE_EOS" == "1" ]]; then
        out_arr+=(--ignore_eos)
    else
        out_arr+=(--no_ignore_eos)
    fi
    if [[ -n "$adapter_trace_path" ]]; then
        out_arr+=(--adapter_trace_path "$adapter_trace_path")
    fi
    if [[ "$print_per_request" == "1" ]]; then
        out_arr+=(--print_per_request)
    fi
    if [[ -n "$per_request_log_path" ]]; then
        out_arr+=(--per_request_log_path "$per_request_log_path")
    fi
    if [[ -n "$num_requests" ]]; then
        out_arr+=(--num_requests "$num_requests")
    fi
    if [[ -n "$phase_name" ]]; then
        out_arr+=(--phase "$phase_name")
    fi
    if [[ -n "$rps" ]]; then
        out_arr+=(--rps "$rps")
    fi
}

run_client_phase() {
    local phase_name="$1"
    shift

    echo "===== ${phase_name} =====" | tee -a benchmark_lora.log

    # Print exact client command
    printf "Running client command: python %q" "$TEST_SCRIPT"
    for arg in "$@"; do
        printf " %q" "$arg"
    done
    printf "\n" | tee -a benchmark_lora.log

    python "$TEST_SCRIPT" "$@" 2>&1 | tee -a benchmark_lora.log
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --delay) SETUP_DELAY="$2"; shift 2 ;;
        --output) OUTPUT_PREFIX="$2"; shift 2 ;;
        --test_script) TEST_SCRIPT="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --decode_target_tokens) DECODE_TARGET_TOKENS="$2"; shift 2 ;;
        --warmup_decode_target_tokens) WARMUP_DECODE_TARGET_TOKENS="$2"; shift 2 ;;
        --ignore_eos) IGNORE_EOS=1; shift ;;
        --no_ignore_eos) IGNORE_EOS=0; shift ;;
        --adapter_ids) ADAPTER_IDS="$2"; ADAPTER_IDS_SET="1"; shift 2 ;;
        --warmup_adapter_trace_path) WARMUP_ADAPTER_TRACE_PATH="$2"; shift 2 ;;
        --measure_adapter_trace_path) MEASURE_ADAPTER_TRACE_PATH="$2"; shift 2 ;;
        --warmup_num_requests) WARMUP_NUM_REQUESTS="$2"; shift 2 ;;
        --measure_num_requests) MEASURE_NUM_REQUESTS="$2"; shift 2 ;;
        --max_concurrent_requests) MAX_CONCURRENT_REQUESTS="$2"; shift 2 ;;
        --rps) RPS="$2"; shift 2 ;;
        --phase_gap_s) PHASE_GAP_S="$2"; shift 2 ;;
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
        --lora_clone_count) LORA_CLONE_COUNT="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --compute_device) COMPUTE_DEVICE="$2"; shift 2 ;;
        --force_slow_lora_path) FORCE_SLOW_LORA_PATH="1"; shift ;;
        --no_force_slow_lora_path) FORCE_SLOW_LORA_PATH="0"; shift ;;
        --max_req_total_len) MAX_REQ_TOTAL_LEN="$2"; shift 2 ;;
        --mem_fraction) MEM_FRACTION="$2"; shift 2 ;;
        --batch_max_tokens) BATCH_MAX_TOKENS="$2"; shift 2 ;;
        --trust_remote_code) TRUST_REMOTE_CODE="1"; shift ;;
        --no_trust_remote_code) TRUST_REMOTE_CODE="0"; shift ;;
        --disable_cudagraph) DISABLE_CUDAGRAPH="1"; shift ;;
        --no_disable_cudagraph) DISABLE_CUDAGRAPH="0"; shift ;;
        --enable_multimodal) ENABLE_MULTIMODAL="1"; shift ;;
        --no_enable_multimodal) ENABLE_MULTIMODAL="0"; shift ;;
        --lora_max_size) LORA_MAX_SIZE="$2"; shift 2 ;;
        --mock_prefill_logits) MOCK_PREFILL_LOGITS="1"; shift ;;
        --no_mock_prefill_logits) MOCK_PREFILL_LOGITS="0"; shift ;;
        --colora_cache_budget_mb) COLORA_CACHE_BUDGET_MB="$2"; shift 2 ;;
        --colora_promote_min_hits) COLORA_PROMOTE_MIN_HITS="$2"; shift 2 ;;
        --colora_promote_window) COLORA_PROMOTE_WINDOW="$2"; shift 2 ;;
        --colora_max_promote_per_step) COLORA_MAX_PROMOTE_PER_STEP="$2"; shift 2 ;;
        --colora_decay) COLORA_DECAY="$2"; shift 2 ;;
        --colora_deferred_promotion_delta_steps) COLORA_DEFERRED_PROMOTION_DELTA_STEPS="$2"; shift 2 ;;
        --colora_promotion_ema_alpha) COLORA_PROMOTION_EMA_ALPHA="$2"; shift 2 ;;
        --colora_miss_policy) COLORA_MISS_POLICY="$2"; shift 2 ;;
        --colora_overlap_mode) COLORA_OVERLAP_MODE="$2"; shift 2 ;;
        --colora_async_fallback) COLORA_ASYNC_FALLBACK="$2"; shift 2 ;;
        --colora_cpu_workers) COLORA_CPU_WORKERS="$2"; shift 2 ;;
        --colora_cpu_queue_depth) COLORA_CPU_QUEUE_DEPTH="$2"; shift 2 ;;
        --colora_cpu_batch_timeout_us) COLORA_CPU_BATCH_TIMEOUT_US="$2"; shift 2 ;;
        --colora_temporal_prefetch) COLORA_TEMPORAL_PREFETCH="1"; shift ;;
        --no_colora_temporal_prefetch) COLORA_TEMPORAL_PREFETCH="0"; shift ;;
        --colora_temporal_prefetch_layer_whitelist) COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST="$2"; shift 2 ;;
        --colora_temporal_hot_cache_slots) COLORA_TEMPORAL_HOT_CACHE_SLOTS="$2"; shift 2 ;;
        --colora_speculative_dispatch) COLORA_SPECULATIVE_DISPATCH="1"; shift ;;
        --no_colora_speculative_dispatch) COLORA_SPECULATIVE_DISPATCH="0"; shift ;;
        --colora_spec_layer_whitelist) COLORA_SPEC_LAYER_WHITELIST="$2"; shift 2 ;;
		--colora_request_skip) COLORA_REQUEST_SKIP="$2"; shift 2 ;;
		--colora_max_continuations) COLORA_MAX_CONTINUATIONS="$2"; shift 2 ;;
		--colora_hit_indexing) COLORA_HIT_INDEXING="$2"; shift 2 ;;
        --server_log_path) SERVER_LOG_PATH="$2"; shift 2 ;;
        --server_stdout_log) SERVER_STDOUT_LOG="$2"; shift 2 ;;
        --server_host) SERVER_HOST="$2"; shift 2 ;;
        --server_port) SERVER_PORT="$2"; shift 2 ;;
        --print_per_request) PRINT_PER_REQUEST=1; shift ;;
        --no_print_per_request) PRINT_PER_REQUEST=0; shift ;;
        --top_k_slowest) TOP_K_SLOWEST="$2"; shift 2 ;;
        --per_request_log_path) PER_REQUEST_LOG_PATH="$2"; shift 2 ;;
        --no_per_request_log) PER_REQUEST_LOG_PATH=""; shift ;;
        --phase) PHASE="$2"; shift 2 ;;
        --mode_label) MODE_LABEL="$2"; shift 2 ;;
        --colora_stats_path) COLORA_STATS_PATH="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"

if [[ -z "$SERVER_LOG_PATH" ]]; then
    SERVER_LOG_PATH="/tmp/${OUTPUT_PREFIX}_server.log"
fi
mkdir -p "$(dirname "$SERVER_LOG_PATH")"
COLORA_SPEC_LAYER_WHITELIST="${COLORA_SPEC_LAYER_WHITELIST//[[:space:]]/}"
COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST="${COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST//[[:space:]]/}"

if [[ -n "$WARMUP_ADAPTER_TRACE_PATH" && ! -f "$WARMUP_ADAPTER_TRACE_PATH" ]]; then
    echo "ERROR: warmup adapter trace not found: $WARMUP_ADAPTER_TRACE_PATH"
    exit 1
fi
if [[ -n "$MEASURE_ADAPTER_TRACE_PATH" && ! -f "$MEASURE_ADAPTER_TRACE_PATH" ]]; then
    echo "ERROR: measurement adapter trace not found: $MEASURE_ADAPTER_TRACE_PATH"
    exit 1
fi
if [[ -z "$WARMUP_DECODE_TARGET_TOKENS" ]]; then
    WARMUP_DECODE_TARGET_TOKENS="$DECODE_TARGET_TOKENS"
fi
if [[ "$ADAPTER_IDS_SET" != "1" && -z "$LORA_DIRS" ]]; then
    DEFAULT_DISCOVERED_ADAPTER_IDS=()
    shopt -s nullglob
    DISCOVERED_DEFAULT_LORA_DIRS=(/home/shufan/Qwen-VL-FT/work/lora_dummy_[0-9]*)
    shopt -u nullglob
    if (( ${#DISCOVERED_DEFAULT_LORA_DIRS[@]} > 0 )); then
        while IFS= read -r discovered_dir; do
            DEFAULT_DISCOVERED_ADAPTER_IDS+=("$(basename "$discovered_dir")")
        done < <(printf '%s\n' "${DISCOVERED_DEFAULT_LORA_DIRS[@]}" | sort -V)
        ADAPTER_IDS="$(IFS=,; echo "${DEFAULT_DISCOVERED_ADAPTER_IDS[*]}")"
    fi
fi
if [[ -n "$LORA_DIRS" && "$ADAPTER_IDS_SET" != "1" ]]; then
    DERIVED_ADAPTER_IDS=()
    IFS=',' read -r -a _RAW_LORA_DIRS <<< "$LORA_DIRS"
    for raw_dir in "${_RAW_LORA_DIRS[@]}"; do
        trimmed_dir="${raw_dir#"${raw_dir%%[![:space:]]*}"}"
        trimmed_dir="${trimmed_dir%"${trimmed_dir##*[![:space:]]}"}"
        if [[ -z "$trimmed_dir" ]]; then
            continue
        fi
        DERIVED_ADAPTER_IDS+=("$(basename "$trimmed_dir")")
    done
    if (( ${#DERIVED_ADAPTER_IDS[@]} > 0 )); then
        ADAPTER_IDS="$(IFS=,; echo "${DERIVED_ADAPTER_IDS[*]}")"
    fi
fi

if [[ -n "$LORA_CLONE_COUNT" && "$LORA_CLONE_COUNT" -gt 1 && "$ADAPTER_IDS_SET" != "1" ]]; then
    CLONE_IDS=()
    for i in $(seq 1 "$LORA_CLONE_COUNT"); do
        CLONE_IDS+=("$i")
    done
    ADAPTER_IDS="$(IFS=,; echo "${CLONE_IDS[*]}")"
fi

# Cleanup function
cleanup() {
    echo "[Cleanup] Tearing down benchmark processes..."
    trap - INT TERM EXIT # Disable traps to avoid recursion

    # if [[ -n "$SERVER_PID" ]]; then
    #     # Fast kill - skip the 20-second wait in terminate_server_tree
    #     kill -INT "$SERVER_PID" 2>/dev/null || true
    #     sleep 2
    #     # Hard kill immediately if still running
    #     pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
    #     kill -KILL "$SERVER_PID" 2>/dev/null || true
    #     wait "$SERVER_PID" 2>/dev/null || true
    #     SERVER_PID=""
    # fi

    finish_profiling_children 30

    # Fast shared memory cleanup with 5s timeout
    echo "[Cleanup] Removing shared memory segments..."
    timeout 5 bash -c 'ipcs -m | awk -v user="$USER" '\''$3 == user && $6 == "0" && $2 ~ /^[0-9]+$/ {print $2}'\'' | xargs -r -n 1 ipcrm -m' 2>/dev/null || true

    echo "[Cleanup] Complete"
    exit 0 # Force immediate exit to prevent nsys hang
}
trap cleanup EXIT

# Step 0: clean up old processes if any
echo "cleanup processes"
if pgrep -f "lightllm.server|lightllm::|gunicorn|multiprocessing.resource_tracker|multiprocessing.spawn" >/dev/null; then
    echo "Killing old processes..."
    pkill -9 -f "lightllm.server|lightllm::|gunicorn" 2>/dev/null || true
    pkill -9 -f "multiprocessing.resource_tracker|multiprocessing.spawn" 2>/dev/null || true
fi

echo > $ADAPTER_EXPERT_LOG_PATH

sleep 5

echo "=============================================="
echo "Starting MoE Profiling"
echo "=============================================="
echo "Adapter IDs: $ADAPTER_IDS"
echo "Number of adapters: $(echo "$ADAPTER_IDS" | tr ',' '\n' | wc -l)"
echo "Poisson lambda: $POISSON_LAMBDA"
echo "Poisson seed: $POISSON_SEED"
echo "Max tokens (fallback): $MAX_TOKENS"
echo "Decode target tokens: $DECODE_TARGET_TOKENS"
echo "Warmup decode target tokens: $WARMUP_DECODE_TARGET_TOKENS"
echo "Ignore EOS: $IGNORE_EOS"
echo "Warmup prompt namespace: Warmup-Req"
[[ -n "$WARMUP_ADAPTER_TRACE_PATH" ]] && echo "Warmup adapter trace: $WARMUP_ADAPTER_TRACE_PATH"
[[ -n "$MEASURE_ADAPTER_TRACE_PATH" ]] && echo "Measurement adapter trace: $MEASURE_ADAPTER_TRACE_PATH"
[[ -n "$WARMUP_NUM_REQUESTS" ]] && echo "Warmup num requests override: $WARMUP_NUM_REQUESTS"
[[ -n "$MEASURE_NUM_REQUESTS" ]] && echo "Measurement num requests override: $MEASURE_NUM_REQUESTS"
echo "Phase gap (s): $PHASE_GAP_S"
[[ -n "$MAX_CONCURRENT_REQUESTS" ]] && echo "Max concurrent requests: $MAX_CONCURRENT_REQUESTS"
[[ -n "$RPS" ]] && echo "Target RPS: $RPS"
echo "Adapter expert profile: $ADAPTER_EXPERT_PROFILE"
echo "Adapter expert profile log: $ADAPTER_EXPERT_LOG_PATH"
echo "Top-K slowest requests: $TOP_K_SLOWEST"
echo "Print per-request lines: $PRINT_PER_REQUEST"
[[ -n "$MODEL_DIR" ]] && echo "Server model_dir override: $MODEL_DIR"
[[ -n "$LORA_DIRS" ]] && echo "Server lora_dirs override: $LORA_DIRS"
[[ -n "$LORA_CLONE_COUNT" ]] && echo "Server lora_clone_count override: $LORA_CLONE_COUNT"
[[ -n "$TP" ]] && echo "Server TP override: $TP"
[[ -n "$COMPUTE_DEVICE" ]] && echo "Server compute_device override: $COMPUTE_DEVICE"
[[ -n "$FORCE_SLOW_LORA_PATH" ]] && echo "Server force_slow_lora_path override: $FORCE_SLOW_LORA_PATH"
[[ -n "$COLORA_MISS_POLICY" ]] && echo "COLoRA miss policy override: $COLORA_MISS_POLICY"
[[ -n "$COLORA_ASYNC_FALLBACK" ]] && echo "COLoRA async fallback override: $COLORA_ASYNC_FALLBACK"
[[ -n "$COLORA_CPU_WORKERS" ]] && echo "COLoRA CPU workers override: $COLORA_CPU_WORKERS"
[[ -n "$COLORA_CPU_QUEUE_DEPTH" ]] && echo "COLoRA CPU queue depth override: $COLORA_CPU_QUEUE_DEPTH"
[[ -n "$COLORA_CPU_BATCH_TIMEOUT_US" ]] && echo "COLoRA CPU batch timeout (us) override: $COLORA_CPU_BATCH_TIMEOUT_US"
[[ -n "$COLORA_DEFERRED_PROMOTION_DELTA_STEPS" ]] && echo "COLoRA deferred promotion delta override: $COLORA_DEFERRED_PROMOTION_DELTA_STEPS"
[[ -n "$COLORA_PROMOTION_EMA_ALPHA" ]] && echo "COLoRA promotion EMA alpha override: $COLORA_PROMOTION_EMA_ALPHA"
echo "COLoRA temporal prefetch override: $COLORA_TEMPORAL_PREFETCH"
[[ -n "$COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST" ]] && echo "COLoRA temporal prefetch layer whitelist: $COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST"
[[ -n "$COLORA_TEMPORAL_HOT_CACHE_SLOTS" ]] && echo "COLoRA temporal hot cache slots: $COLORA_TEMPORAL_HOT_CACHE_SLOTS"
echo "COLoRA speculative dispatch override: $COLORA_SPECULATIVE_DISPATCH"
[[ -n "$COLORA_SPEC_LAYER_WHITELIST" ]] && echo "COLoRA speculative layer whitelist: $COLORA_SPEC_LAYER_WHITELIST"
[[ -n "$COLORA_REQUEST_SKIP" ]] && echo "COLoRA request-level skip override: $COLORA_REQUEST_SKIP"
[[ -n "$COLORA_MAX_CONTINUATIONS" ]] && echo "COLoRA max continuations override: $COLORA_MAX_CONTINUATIONS"
echo "Server log path: $SERVER_LOG_PATH"
if [[ -n "$PER_REQUEST_LOG_PATH" ]]; then
    echo "Per-request metrics log: $PER_REQUEST_LOG_PATH"
else
    echo "Per-request metrics log: disabled"
fi
if [[ -n "$COLORA_STATS_PATH" ]]; then
    echo "CoLoRA stats output: $COLORA_STATS_PATH"
fi

WARMUP_ENABLED=0
if [[ -n "$WARMUP_ADAPTER_TRACE_PATH" || -n "$WARMUP_NUM_REQUESTS" ]]; then
    WARMUP_ENABLED=1
fi

WARMUP_TEST_ARGS=()
build_phase_args \
    WARMUP_TEST_ARGS \
    "$WARMUP_DECODE_TARGET_TOKENS" \
    "$WARMUP_ADAPTER_TRACE_PATH" \
    "$WARMUP_NUM_REQUESTS" \
    "0" \
    "" \
    "0" \
    "Warmup-Req" \
    "warmup" \
    "$MAX_CONCURRENT_REQUESTS" \
    "$RPS"

MEASURE_TEST_ARGS=()
MEASURE_PROMPT_NAMESPACE="Req"
if [[ "$WARMUP_ENABLED" == "1" ]]; then
    MEASURE_PROMPT_NAMESPACE="Measure-Req"
fi
echo "Measurement prompt namespace: $MEASURE_PROMPT_NAMESPACE"
build_phase_args \
    MEASURE_TEST_ARGS \
    "$DECODE_TARGET_TOKENS" \
    "$MEASURE_ADAPTER_TRACE_PATH" \
    "$MEASURE_NUM_REQUESTS" \
    "$TOP_K_SLOWEST" \
    "$PER_REQUEST_LOG_PATH" \
    "$PRINT_PER_REQUEST" \
    "$MEASURE_PROMPT_NAMESPACE" \
    "measurement" \
    "$MAX_CONCURRENT_REQUESTS" \
    "$RPS"

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
if [[ -n "$LORA_CLONE_COUNT" ]]; then
    SERVER_ARGS+=(--lora_clone_count "$LORA_CLONE_COUNT")
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
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    SERVER_ARGS+=(--trust_remote_code)
elif [[ "$TRUST_REMOTE_CODE" == "0" ]]; then
    SERVER_ARGS+=(--no_trust_remote_code)
fi
if [[ "$DISABLE_CUDAGRAPH" == "1" ]]; then
    SERVER_ARGS+=(--disable_cudagraph)
elif [[ "$DISABLE_CUDAGRAPH" == "0" ]]; then
    SERVER_ARGS+=(--no_disable_cudagraph)
fi
if [[ "$ENABLE_MULTIMODAL" == "1" ]]; then
    SERVER_ARGS+=(--enable_multimodal)
elif [[ "$ENABLE_MULTIMODAL" == "0" ]]; then
    SERVER_ARGS+=(--no_enable_multimodal)
fi
if [[ -n "$LORA_MAX_SIZE" ]]; then
    SERVER_ARGS+=(--lora_max_size "$LORA_MAX_SIZE")
fi
if [[ "$MOCK_PREFILL_LOGITS" == "1" ]]; then
    SERVER_ARGS+=(--mock_prefill_logits)
elif [[ "$MOCK_PREFILL_LOGITS" == "0" ]]; then
    SERVER_ARGS+=(--no_mock_prefill_logits)
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
if [[ -n "$COLORA_DEFERRED_PROMOTION_DELTA_STEPS" ]]; then
    SERVER_ARGS+=(--colora_deferred_promotion_delta_steps "$COLORA_DEFERRED_PROMOTION_DELTA_STEPS")
fi
if [[ -n "$COLORA_PROMOTION_EMA_ALPHA" ]]; then
    SERVER_ARGS+=(--colora_promotion_ema_alpha "$COLORA_PROMOTION_EMA_ALPHA")
fi
if [[ -n "$COLORA_MISS_POLICY" ]]; then
    SERVER_ARGS+=(--colora_miss_policy "$COLORA_MISS_POLICY")
fi
if [[ -n "$COLORA_OVERLAP_MODE" ]]; then
    SERVER_ARGS+=(--colora_overlap_mode "$COLORA_OVERLAP_MODE")
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
if [[ "$COLORA_TEMPORAL_PREFETCH" == "1" ]]; then
    SERVER_ARGS+=(--colora_temporal_prefetch)
fi
if [[ -n "$COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST" ]]; then
    SERVER_ARGS+=(--colora_temporal_prefetch_layer_whitelist "$COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST")
fi
if [[ -n "$COLORA_TEMPORAL_HOT_CACHE_SLOTS" ]]; then
    SERVER_ARGS+=(--colora_temporal_hot_cache_slots "$COLORA_TEMPORAL_HOT_CACHE_SLOTS")
fi
if [[ "$COLORA_SPECULATIVE_DISPATCH" == "1" ]]; then
    SERVER_ARGS+=(--colora_speculative_dispatch)
fi
if [[ -n "$COLORA_SPEC_LAYER_WHITELIST" ]]; then
    SERVER_ARGS+=(--colora_spec_layer_whitelist "$COLORA_SPEC_LAYER_WHITELIST")
fi
if [[ -n "$COLORA_REQUEST_SKIP" ]]; then
    SERVER_ARGS+=(--colora_request_skip "$COLORA_REQUEST_SKIP")
fi
if [[ -n "$COLORA_MAX_CONTINUATIONS" ]]; then
    SERVER_ARGS+=(--colora_max_continuations "$COLORA_MAX_CONTINUATIONS")
fi
if [[ -n "$COLORA_HIT_INDEXING" ]]; then
    SERVER_ARGS+=(--colora_hit_indexing "$COLORA_HIT_INDEXING")
fi
# Redirect server output
if [[ -n "$SERVER_STDOUT_LOG" ]]; then
    # User specified explicit output location for stdout/stderr.
    # Use process substitution so $! is the server launcher PID (not tee PID).
    bash "$SERVER_SCRIPT" "${SERVER_ARGS[@]}" > >(tee -a "$SERVER_STDOUT_LOG") 2>&1 &
elif [[ -z "$SERVER_LOG_PATH" || "$SERVER_LOG_PATH" == "/dev/null" ]]; then
    # Output only to stdout/stderr
    bash "$SERVER_SCRIPT" "${SERVER_ARGS[@]}" &
else
    # Default: output to both server log file AND stdout.
    # Use process substitution so shutdown targets the real server launcher tree.
    bash "$SERVER_SCRIPT" "${SERVER_ARGS[@]}" > >(tee "$SERVER_LOG_PATH") 2>&1 &
fi
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Record start time
start_time=$SECONDS

# Step 2: Wait for server to be healthy
echo "[2/5] Waiting for server to be healthy..."

while (( SECONDS - start_time < MAX_WAIT )); do
    # First check if process is still alive
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: Server process exited early (PID $SERVER_PID)."
        echo "Check server log at: $SERVER_LOG_PATH"
        exit 1
    fi

    # First check port open
    if nc -vz "$SERVER_HOST" "$SERVER_PORT" 2>/dev/null; then
        # Port is open, try health checks
        # Try /healthz first
        if curl -sf "http://$SERVER_HOST:$SERVER_PORT/healthz" >/dev/null 2>&1; then
            echo ""
            echo "Server is HEALTHY (healthz check passed)"
            break
        fi
        # Fallback to /health
        if curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" >/dev/null 2>&1; then
            echo ""
            echo "Server is HEALTHY (health check passed)"
            break
        fi
    fi
    echo -n "."
    sleep 5
done

# Check if we timed out
if [[ $(($SECONDS - start_time)) -ge $MAX_WAIT ]]; then
    echo "ERROR: Server failed to become healthy within ${MAX_WAIT}s."
    echo "Check server log at: $SERVER_LOG_PATH"
    exit 1
fi

# Step 3: Wait another delay before sending requests
echo "Waiting additional ${SETUP_DELAY}s before sending test request..."
sleep "$SETUP_DELAY"

# Step 4: Send warmup + measured requests (nsys is now capturing)
echo "[3/5] Sending benchmark traffic..."
echo > benchmark_lora.log
if [[ "$WARMUP_ENABLED" == "1" ]]; then
    run_client_phase "Warmup (excluded from measurement)" "${WARMUP_TEST_ARGS[@]}"
    # Reset cumulative CoLoRA stats so measurement phase starts clean
    if curl -sf "http://$SERVER_HOST:$SERVER_PORT/colora_stats" -o /dev/null 2>/dev/null; then
        echo "CoLoRA stats reset after warmup"
    fi

    # Warm cache: block-promote all adapters used in warmup (only for colora_min)
    if [[ -n "$WARMUP_ADAPTER_TRACE_PATH" && -f "$WARMUP_ADAPTER_TRACE_PATH" && "${MODE_LABEL:-}" == "colora_min" ]]; then
        echo "Warming cache from warmup adapters..."
        WARMUP_ADAPTER_IDS=$(python3 -c "
import json, sys, re
seen = set()
with open('$WARMUP_ADAPTER_TRACE_PATH') as f:
    for line in f:
        aid = json.loads(line).get('adapter_id')
        if aid and str(aid).lower() not in ('none', 'null', 'base', ''):
            # Handle both numeric IDs and lora_dummy_XX format
            m = re.match(r'lora_dummy_(\d+)', str(aid))
            if m:
                seen.add(m.group(1))
            else:
                seen.add(str(aid))
print(json.dumps(list(seen)))
")
        if [[ "$WARMUP_ADAPTER_IDS" != "[]" ]]; then
            echo "Promoting adapter IDs: $WARMUP_ADAPTER_IDS"
            if curl -sf -X POST "http://$SERVER_HOST:$SERVER_PORT/colora_promote_adapters" \
                 -H "Content-Type: application/json" \
                 -d "{\"adapter_ids\": $WARMUP_ADAPTER_IDS}" \
                 -o /tmp/colora_warm.json 2>/dev/null; then
                cat /tmp/colora_warm.json
                echo ""
                echo "Cache warm complete"
            else
                echo "WARNING: /colora_promote_adapters failed or endpoint unavailable"
            fi
        fi
    fi

    # Freeze promotion (and prefetch) for measurement so cache residency stays fixed.
    # Only needed for colora_min; colora_full keeps promoting, load_then_run does not use promotion.
    if [[ "${MODE_LABEL:-}" == "colora_min" ]]; then
        if curl -sf -X POST "http://$SERVER_HOST:$SERVER_PORT/colora_config" \
             -H "Content-Type: application/json" \
             -d '{"deferred_promotion_delta_steps":0,"temporal_prefetch":false}' \
             -o /dev/null 2>/dev/null; then
            echo "CoLoRA promotion/prefetch frozen for measurement"
        fi
    fi

    if [[ "$PHASE_GAP_S" != "0" ]]; then
        echo "Sleeping ${PHASE_GAP_S}s between warmup and measurement..." | tee -a benchmark_lora.log
        sleep "$PHASE_GAP_S"
    fi
fi

if [[ -n "$MEASURE_ADAPTER_TRACE_PATH" || -n "$MEASURE_NUM_REQUESTS" ]]; then
    run_client_phase "Measurement" "${MEASURE_TEST_ARGS[@]}"
else
    REQUEST_COUNTS=(16)
    for i in "${!REQUEST_COUNTS[@]}"; do
        num_requests="${REQUEST_COUNTS[$i]}"
        run_client_phase "Measurement num_requests=${num_requests}" "${MEASURE_TEST_ARGS[@]}" --num_requests "$num_requests"
        if (( i < ${#REQUEST_COUNTS[@]} - 1 )); then
            sleep 5
        fi
    done
fi

# Fetch CoLoRA stats from server before shutdown
if [[ -n "$COLORA_STATS_PATH" ]]; then
    echo "[3.5/5] Fetching CoLoRA stats from server..."
    if curl -sf "http://$SERVER_HOST:$SERVER_PORT/colora_stats" -o "$COLORA_STATS_PATH" 2>/dev/null; then
        echo "CoLoRA stats saved to: $COLORA_STATS_PATH"
    else
        echo "WARNING: Failed to fetch /colora_stats (endpoint may not exist for this policy); writing empty JSON"
        echo '{}' > "$COLORA_STATS_PATH"
    fi
fi

echo "[4/5] Stopping server before exit..."
terminate_server_tree "$SERVER_PID"
if [[ -n "$SERVER_PID" ]]; then
    wait "$SERVER_PID" 2>/dev/null || true
fi
SERVER_PID=""
echo "[5/5] Server log saved to: $SERVER_LOG_PATH"

echo "[Cleanup] Stopping remaining benchmark processes (graceful first, for valid nsys traces)..."
finish_profiling_children 90

# Wait for CUDA context teardown
sleep 5

# Bounded wait — avoids infinite hang if pgrep matches an unrelated long-lived process
_SPIN=0
while pgrep -f "lightllm.server|python.*api_server" >/dev/null && (( _SPIN < 45 )); do
    echo "Waiting for processes to exit..."
    sleep 2
    _SPIN=$((_SPIN + 1))
done
if pgrep -f "lightllm.server|python.*api_server" >/dev/null; then
    echo "[Cleanup] WARNING: pgrep still matches after ${_SPIN} waits; try: pgrep -af 'lightllm|api_server'"
    finish_profiling_children 15
fi

echo "[5/5] Benchmark complete."
echo "[Cleanup] If nsys is still running, it is usually waiting on traced children; try: nsys profile --wait=primary ... (faster shutdown, may drop orphan trace) or: pgrep -af lightllm"

# Disable EXIT trap since we already did full cleanup
trap - EXIT

# Force exit to ensure nsys doesn't hang
exit 0
