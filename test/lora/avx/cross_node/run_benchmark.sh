#!/bin/bash
#
# Launch cross_node_benchmark.py on UM253
# Usage: ./run_benchmark.sh [extra-args...]
#

# Detect IB interface
detect_ib_iface() {
    ip -o link show 2>/dev/null | awk -F': ' '/ib[sp]/ {print $2; exit}'
}
GLOO_IFACE="${GLOO_SOCKET_IFNAME:-$(detect_ib_iface)}"
export GLOO_SOCKET_IFNAME="${GLOO_IFACE}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/cross_node_benchmark.py"

# Check prerequisites
python -c "import torch; import torch.distributed" || {
    echo "ERROR: torch.distributed not available"
    exit 1
}
command -v ib_write_bw >/dev/null || {
    echo "ERROR: ib_write_bw not found; install perftest"
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
MASTER_ADDR="${MASTER_ADDR:-10.10.1.1}"  # UM253 (this node, rank 0)
SERVER_ADDR="${SERVER_ADDR:-10.10.1.3}"  # UM251 IB IP
MASTER_PORT="${MASTER_PORT:-29500}"
SERVER_PORT="${SERVER_PORT:-29501}"
OUTPUT_DIR="${OUTPUT_DIR:-results/cross_node_benchmark}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-100}"

echo "============================================================"
echo "Cross-Node Benchmark (UM253 side)"
echo "============================================================"
echo "Rank: ${RANK}"
echo "Master (rendezvous): ${MASTER_ADDR}:${MASTER_PORT}"
echo "Server (UM251): ${SERVER_ADDR}:${SERVER_PORT}"
echo "Output: ${OUTPUT_DIR}"
echo "Warmup: ${WARMUP}, Iters: ${ITERS}"
echo "EP traffic requires a remote ib_write_bw server on ${SERVER_ADDR}:18515:"
echo "  ib_write_bw -d mlx5_0 -p 18515 --duration=999999 -b"
echo "============================================================"

python "${BENCHMARK_SCRIPT}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --server-addr "${SERVER_ADDR}" \
    --server-port "${SERVER_PORT}" \
    --ep-ssh-host "${EP_SSH_HOST:-}" \
    --output-dir "${OUTPUT_DIR}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    "$@"

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "============================================================"
