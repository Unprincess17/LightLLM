#!/bin/bash
#
# Launch cross_node_benchmark.py on UM253
# Usage: ./run_benchmark.sh [extra-args...]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/cross_node_benchmark.py"

# Check prerequisites
python -c "import torch; import torch.distributed" || {
    echo "ERROR: torch.distributed not available"
    exit 1
}

# CUDA check
python -c "import torch; assert torch.cuda.is_available()" || {
    echo "ERROR: CUDA required for this benchmark"
    exit 1
}

# Machine IP detection
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "Detected local IP: ${LOCAL_IP}"

# Default: this is UM253 (rank 0)
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-10.10.1.3}"  # UM251
MASTER_PORT="${MASTER_PORT:-29500}"
OUTPUT_DIR="${OUTPUT_DIR:-results/cross_node_benchmark}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-100}"

echo "============================================================"
echo "Cross-Node Benchmark (UM253 side)"
echo "============================================================"
echo "Rank: ${RANK}"
echo "Server (UM251): ${MASTER_ADDR}:${MASTER_PORT}"
echo "Output: ${OUTPUT_DIR}"
echo "Warmup: ${WARMUP}, Iters: ${ITERS}"
echo "============================================================"

python "${BENCHMARK_SCRIPT}" \
    --rank "${RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    "$@"

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "============================================================"
