pkill -9 -f "lightllm.server|lightllm::|gunicorn" 2>/dev/null || true
pkill -9 -f "multiprocessing.resource_tracker|multiprocessing.spawn" 2>/dev/null || true

echo "[Cleanup] Removing shared memory segments..."
timeout 5 bash -c 'ipcs -m | awk -v user="$USER" '\''$3 == user && $6 == "0" && $2 ~ /^[0-9]+$/ {print $2}'\'' | xargs -r -n 1 ipcrm -m' 2>/dev/null || true
