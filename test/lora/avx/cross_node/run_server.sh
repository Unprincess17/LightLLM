#!/bin/bash
#
# Launch cross_node_server.py on UM251
# Usage: ./run_server.sh [extra-args...]
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
SERVER_SCRIPT="${SCRIPT_DIR}/cross_node_server.py"

# RDMA device detection
detect_rdma_device() {
    if [ -d /sys/class/infiniband/mlx5_0 ]; then
        echo "mlx5_0"
    elif [ -d /sys/class/infiniband/mlx5_1 ]; then
        echo "mlx5_1"
    elif [ -d /sys/class/infiniband/mlx5_2 ]; then
        echo "mlx5_2"
    elif [ -d /sys/class/infiniband/rocep0s6f0 ]; then
        echo "rocep0s6f0"
    else
        echo "ib0"
    fi
}

# Check if torch.distributed is available
python -c "import torch; import torch.distributed" || {
    echo "ERROR: torch.distributed not available"
    exit 1
}

# Detect this machine's IP for the rendezvous
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "Detected local IP: ${LOCAL_IP}"

# Default: this is UM251 (rank 1)
RANK="${RANK:-1}"
MASTER_ADDR="${MASTER_ADDR:-10.10.1.1}"  # UM253 IB IP (rank 0)
MASTER_PORT="${MASTER_PORT:-29500}"
LISTEN_PORT="${LISTEN_PORT:-29501}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
INTERMEDIATE_DIM="${INTERMEDIATE_DIM:-1536}"

echo "============================================================"
echo "Cross-Node Server (UM251 side)"
echo "============================================================"
echo "Rank: ${RANK}"
echo "Master (UM253): ${MASTER_ADDR}:${MASTER_PORT}"
echo "Listen port: ${LISTEN_PORT}"
echo "============================================================"

python "${SERVER_SCRIPT}" \
    --rank "${RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --listen-port "${LISTEN_PORT}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --intermediate-dim "${INTERMEDIATE_DIM}" \
    "$@"
