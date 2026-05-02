import os
import asyncio
import torch.multiprocessing as mp
import multiprocessing
import threading
import inspect
import setproctitle
from datetime import timedelta
from typing import Dict, List, Tuple, Union
from lightllm.server.router.model_infer.mode_backend import (
    ChunkedPrefillBackend,
    FirstTokenConstraintBackend,
    OutlinesConstraintBackend,
    ReturnPromptLogProbBackend,
    RewardModelBackend,
    TokenHealingBackend,
    XgrammarBackend,
    DPChunkedPrefillBackend,
    DiversehBackend,
    DecodeNode,
    DPForDecodeNode,
    ChunckedPrefillForPrefillNode,
    DPChunkedForPrefillNode,
    NIXLChunckedPrefillForPrefillNode,
    NIXLDPChunkedForPrefillNode,
    NIXLDecodeNode,
    NIXLDPForDecodeNode,
)
from lightllm.server.router.model_infer.mode_backend.redundancy_expert_manager import RedundancyExpertManager
from lightllm.server.core.objs import RpcShmParams, RpcShmResults, ShmSyncStatusArray
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.log_utils import init_logger
from lightllm.utils.auto_shm_cleanup import register_cleanup_callback
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)


class ModelRpcServer:
    def __init__(
        self,
        args,
        rank: int,
        rank_in_node: int,
        node_world_size: int,
        rpc_event: multiprocessing.Event,
        rpc_finished_event: multiprocessing.Event,
        info_queue: mp.Queue,
    ):
        super().__init__()
        self.args: StartArgs = args
        self.node_world_size = node_world_size
        self.info_queue = info_queue
        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event

        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()
        self.rpc_shm_sync_status = ShmSyncStatusArray(self.node_world_size)
        self.rpc_shm_sync_status.create_or_link_shm()

        self.rank = rank
        self.rank_in_node = rank_in_node
        logger.info(f"Initialized RPC server for rank {self.rank}.")
        register_cleanup_callback(self.cleanup_shared_memory)

        self.rpc_loop_thread = threading.Thread(target=self.rpc_loop, daemon=True)
        self.rpc_loop_thread.start()
        return

    def rpc_loop(self):
        error_count = 0
        while True:
            try:
                self.rpc_event.wait()
                func_name, args = self.rpc_shm_params.read_func_params()

                ans = getattr(self, func_name)(*args)
                if ans is not None and self.rank_in_node == 0:
                    self.rpc_shm_results.write_func_result(func_name=func_name, ret=ans)

                # 下面得执行顺序不可随意交换, 否则容易出现同步或者死锁问题。
                self.rpc_shm_sync_status.add_mark(self.rank_in_node)
                while not self.rpc_shm_sync_status.run_finished():
                    pass

                self.rpc_event.clear()

                self.rpc_shm_sync_status.add_mark1(self.rank_in_node)
                while not self.rpc_shm_sync_status.run_finished1():
                    pass

                if self.rank_in_node == 0:
                    self.rpc_finished_event.set()

            except BaseException as e:
                logger.exception(str(e))
                error_count += 1

            if error_count >= 1:
                logger.error("infer process error to exit")
                self.cleanup_shared_memory()
                os._exit(-1)

        return

    def init_model(self, kvargs):
        # 填充真正的 rank_id 参数
        kvargs["rank_id"] = self.rank
        self.world_size = kvargs["world_size"]
        return_all_prompt_logprobs = self.args.return_all_prompt_logprobs
        use_reward_model = self.args.use_reward_model
        diverse_mode = self.args.diverse_mode
        is_token_healing = self.args.token_healing_mode
        is_first_token_constraint_mode = self.args.first_token_constraint_mode

        is_outlines_constraint_mode = self.args.output_constraint_mode == "outlines"
        is_xgrammar_constraint_mode = self.args.output_constraint_mode == "xgrammar"
        assert not (is_outlines_constraint_mode and is_xgrammar_constraint_mode), "only one constraint mode can be true"
        is_prefill_node = self.args.run_mode == "prefill"
        is_decode_node = self.args.run_mode == "decode"
        is_nixl_prefill_node = self.args.run_mode == "nixl_prefill"
        is_nixl_decode_node = self.args.run_mode == "nixl_decode"

        if is_prefill_node:
            if self.args.dp > 1:
                self.backend = DPChunkedForPrefillNode(self.info_queue)
            else:
                self.backend = ChunckedPrefillForPrefillNode(self.info_queue)
        elif is_nixl_prefill_node:
            if self.args.dp > 1:
                self.backend = NIXLDPChunkedForPrefillNode(self.info_queue)
            else:
                self.backend = NIXLChunckedPrefillForPrefillNode(self.info_queue)

        elif is_decode_node:
            if self.args.dp > 1:
                self.backend = DPForDecodeNode(self.info_queue)
            else:
                self.backend = DecodeNode(self.info_queue)

        elif is_nixl_decode_node:
            if self.args.dp > 1:
                self.backend = NIXLDPForDecodeNode(self.info_queue)
            else:
                self.backend = NIXLDecodeNode(self.info_queue)

        elif self.args.dp > 1:
            self.backend = DPChunkedPrefillBackend()
        elif use_reward_model:
            self.backend = RewardModelBackend()
        elif return_all_prompt_logprobs:
            self.backend = ReturnPromptLogProbBackend()
        elif diverse_mode:
            self.backend = DiversehBackend()
        elif is_token_healing:
            self.backend = TokenHealingBackend()
        elif is_outlines_constraint_mode:
            self.backend = OutlinesConstraintBackend()
        elif is_xgrammar_constraint_mode:
            self.backend = XgrammarBackend()
        elif is_first_token_constraint_mode:
            self.backend = FirstTokenConstraintBackend()
        else:
            self.backend = ChunkedPrefillBackend()

        logger.info(f"use {self.backend.__class__.__name__}")
        self.backend.init_model(kvargs)

        # only deepseekv3 can support auto_update_redundancy_expert
        if self.args.auto_update_redundancy_expert:
            self.redundancy_expert_manager = RedundancyExpertManager(self.backend.model)
            logger.info("init redundancy_expert_manager")
        else:
            self.redundancy_expert_manager = None
        return

    def get_max_total_token_num(self):
        return self.backend.get_max_total_token_num()

    def get_colora_stats(self):
        """Aggregate cumulative colora stats from all transformer layers."""
        backend_model = getattr(self.backend, "model", None)
        if backend_model is None:
            return {}
        layers_infer = getattr(backend_model, "layers_infer", None)
        if layers_infer is None:
            return {}

        result: Dict[str, float] = {}
        for layer in layers_infer:
            get_fn = getattr(layer, "get_cumulative_colora_stats", None)
            if callable(get_fn):
                stats = get_fn()
                for key, value in stats.items():
                    if key in result:
                        result[key] += float(value)
                    else:
                        result[key] = float(value)

        # Derive hit/miss rates from accumulated tokens
        hit = result.get("colora_hit_tokens", 0)
        miss = result.get("colora_miss_tokens", 0)
        total = hit + miss
        if total > 0:
            result["cache_hit_rate"] = hit / total
            result["cache_miss_rate"] = miss / total

        # Reset cumulative stats after reading so the next phase starts clean
        for layer in layers_infer:
            reset_fn = getattr(layer, "reset_cumulative_colora_stats", None)
            if callable(reset_fn):
                reset_fn()

        return result

    def set_colora_config(self, config):
        """Propagate COLoRA runtime config to all transformer layers."""
        backend_model = getattr(self.backend, "model", None)
        if backend_model is None:
            return {"error": "backend model not initialized"}
        layers_infer = getattr(backend_model, "layers_infer", None)
        if layers_infer is None:
            return {"error": "layers_infer not initialized"}

        for layer in layers_infer:
            dispatcher = getattr(layer, "lora_dispatcher_", None)
            if dispatcher is not None and callable(getattr(dispatcher, "set_colora_config", None)):
                dispatcher.set_colora_config(config)

        return {"status": "ok", "config": config}

    def promote_adapters(self, adapter_ids: List[Union[str, int]]):
        """Blocking promotion of all expert projections for given adapter IDs (strings or ints)."""
        backend_model = getattr(self.backend, "model", None)
        if backend_model is None:
            return {"error": "backend model not initialized"}
        layers_infer = getattr(backend_model, "layers_infer", None)
        if layers_infer is None:
            return {"error": "layers_infer not initialized"}

        # Map adapter names to bins using the first layer's dispatcher
        adapter_bins = []
        for layer in layers_infer:
            dispatcher = getattr(layer, "lora_dispatcher_", None)
            if dispatcher is not None:
                pool = getattr(dispatcher, "lora_mem_pool", None)
                if pool is not None:
                    for aid in adapter_ids:
                        # First, try parsing as a direct integer
                        try:
                            idx = int(aid)
                            adapter_bins.append(idx)
                            continue
                        except (ValueError, TypeError):
                            pass

                        # Then, try extracting the numeric suffix from things like "lora_dummy_41"
                        import re
                        match = re.search(r'(\d+)$', str(aid))
                        if match:
                            idx = int(match.group(1))
                            adapter_bins.append(idx)
                            continue

                        # Finally, try looking up the full string in pool.idx_map
                        idx = pool.idx_map.get(str(aid))
                        if idx is not None:
                            adapter_bins.append(int(idx))
                    break

        if not adapter_bins:
            return {"status": "ok", "total_promoted": 0, "total_transferred_bytes": 0, "note": "no_matching_adapters"}

        # Remove duplicates while preserving order
        seen = set()
        unique_adapter_bins = []
        for bin_idx in adapter_bins:
            if bin_idx not in seen:
                seen.add(bin_idx)
                unique_adapter_bins.append(bin_idx)
        adapter_bins = unique_adapter_bins

        total_promoted = 0
        total_bytes = 0
        for layer in layers_infer:
            dispatcher = getattr(layer, "lora_dispatcher_", None)
            if dispatcher is not None and callable(getattr(dispatcher, "promote_adapters_blocking", None)):
                result = dispatcher.promote_adapters_blocking(adapter_bins)
                total_promoted += result.get("promoted_count", 0)
                total_bytes += result.get("transferred_bytes", 0)

        return {
            "status": "ok",
            "total_promoted": total_promoted,
            "total_transferred_bytes": total_bytes,
        }

    def cleanup_shared_memory(self):
        if hasattr(self, "backend") and self.backend is not None:
            backend_model = getattr(self.backend, "model", None)
            if backend_model is not None and getattr(backend_model, "mem_manager", None) is not None:
                backend_model.mem_manager.cleanup_shared_memory()
            if getattr(self.backend, "multi_level_cache_module", None) is not None:
                self.backend.multi_level_cache_module.cpu_cache_client.cleanup_shared_memory()
                self.backend.multi_level_cache_module = None
            if getattr(self.backend, "shm_req_manager", None) is not None:
                self.backend.shm_req_manager.destroy()
                self.backend.shm_req_manager = None
            if getattr(self.backend, "shm_nixl_trans_io_buffer", None) is not None:
                self.backend.shm_nixl_trans_io_buffer.destroy()
                self.backend.shm_nixl_trans_io_buffer = None
            if getattr(self.backend, "shm_reqs_io_buffer", None) is not None:
                self.backend.shm_reqs_io_buffer.destroy()
                self.backend.shm_reqs_io_buffer = None
            if getattr(self.backend, "radix_cache", None) is not None:
                self.backend.radix_cache.cleanup_shared_memory()
                self.backend.radix_cache = None
            if getattr(self.backend, "dp_kv_shared_module", None) is not None:
                self.backend.dp_kv_shared_module.cleanup_shared_memory()
                self.backend.dp_kv_shared_module = None
            try:
                from lightllm.server.router.model_infer.infer_batch import g_infer_context
                g_infer_context.cleanup_shared_memory()
            except Exception:
                pass
            try:
                from lightllm.common.basemodel.infer_lock import g_infer_state_lock

                if getattr(g_infer_state_lock, "obj", None) is not None:
                    g_infer_state_lock.obj.cleanup_shared_memory()
                    g_infer_state_lock.obj = None
            except Exception:
                pass
        if hasattr(self, "rpc_shm_sync_status") and self.rpc_shm_sync_status is not None:
            self.rpc_shm_sync_status.destroy()
            self.rpc_shm_sync_status = None
        if hasattr(self, "rpc_shm_results") and self.rpc_shm_results is not None:
            self.rpc_shm_results.destroy()
            self.rpc_shm_results = None
        if hasattr(self, "rpc_shm_params") and self.rpc_shm_params is not None:
            self.rpc_shm_params.destroy()
            self.rpc_shm_params = None
        return


class ModelRpcClient:
    def __init__(self, rpc_event, rpc_finished_event):
        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()

        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event
        return

    async def init_model(self, kvargs):
        self.rpc_shm_params.write_func_params("init_model", (kvargs,))
        self.rpc_event.set()

        self.rpc_finished_event.wait()
        self.rpc_finished_event.clear()
        return

    async def get_max_total_token_num(self):
        self.rpc_shm_params.write_func_params("get_max_total_token_num", ())
        self.rpc_event.set()

        self.rpc_finished_event.wait()
        self.rpc_finished_event.clear()
        func_name, ret = self.rpc_shm_results.read_func_result()
        assert func_name == "get_max_total_token_num"
        return ret

    async def get_colora_stats(self):
        self.rpc_shm_params.write_func_params("get_colora_stats", ())
        self.rpc_event.set()

        self.rpc_finished_event.wait()
        self.rpc_finished_event.clear()
        func_name, ret = self.rpc_shm_results.read_func_result()
        assert func_name == "get_colora_stats"
        return ret

    async def set_colora_config(self, config):
        self.rpc_shm_params.write_func_params("set_colora_config", (config,))
        self.rpc_event.set()

        self.rpc_finished_event.wait()
        self.rpc_finished_event.clear()
        func_name, ret = self.rpc_shm_results.read_func_result()
        assert func_name == "set_colora_config"
        return ret

    async def promote_adapters(self, adapter_ids: List[str]):
        self.rpc_shm_params.write_func_params("promote_adapters", (adapter_ids,))
        self.rpc_event.set()

        self.rpc_finished_event.wait()
        self.rpc_finished_event.clear()
        func_name, ret = self.rpc_shm_results.read_func_result()
        assert func_name == "promote_adapters"
        return ret

    def cleanup_shared_memory(self):
        if hasattr(self, "rpc_shm_results") and self.rpc_shm_results is not None:
            self.rpc_shm_results.destroy()
            self.rpc_shm_results = None
        if hasattr(self, "rpc_shm_params") and self.rpc_shm_params is not None:
            self.rpc_shm_params.destroy()
            self.rpc_shm_params = None
        return


def _init_env(
    args,
    rank,
    rank_in_node,
    node_world_size,
    info_queue,
    router_lock,
    rpc_event: mp.Event,
    rpc_finished_event: mp.Event,
    success_event: mp.Event,
):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::model_infer:RANK{rank}")
    start_parent_check_thread()

    # 将调度锁注册到全局的共享变量中
    from lightllm.common.basemodel.infer_lock import g_router_lock

    g_router_lock.obj = router_lock

    model_rpc_server = ModelRpcServer(
        args, rank, rank_in_node, node_world_size, rpc_event, rpc_finished_event, info_queue
    )
    success_event.set()

    model_rpc_server.rpc_loop_thread.join()
    return


async def start_model_process(
    args,
    rank,
    rank_in_node,
    node_world_size,
    rpc_event,
    rpc_finished_event,
    info_queue: mp.Queue,
    router_lock: mp.Queue,
):
    import lightllm.utils.rpyc_fix_utils as _

    success_event = mp.Event()
    proc = mp.Process(
        target=_init_env,
        args=(
            args,
            rank,
            rank_in_node,
            node_world_size,
            info_queue,
            router_lock,
            rpc_event,
            rpc_finished_event,
            success_event,
        ),
    )
    proc.start()

    # Use asyncio.to_thread to make the blocking wait non-blocking
    await asyncio.to_thread(success_event.wait, timeout=40)
    assert proc.is_alive()

    return None
