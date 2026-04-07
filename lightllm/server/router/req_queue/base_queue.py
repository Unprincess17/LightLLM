from typing import List, Dict
import json
import time
from lightllm.utils.infer_utils import calculate_time
from ..batch import Batch, Req
from lightllm.server.core.objs import FinishStatus
from lightllm.common.basemodel.infer_lock import g_router_lock
from lightllm.utils.config_utils import get_fixed_kv_len
from lightllm.server.core.objs import StartArgs

_AGENT_DEBUG_LOG_PATH = "/home/shufan/LightLLM-integrate-to-SLoRA/.cursor/debug-93213c.log"
_AGENT_DEBUG_SESSION_ID = "93213c"


def _agent_debug_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


class BaseQueue:
    def __init__(self, args: StartArgs, router, dp_index, dp_size_in_node) -> None:
        self.args = args
        self.dp_index = dp_index
        self.dp_size_in_node = dp_size_in_node
        from lightllm.server.router.manager import RouterManager

        self.router: RouterManager = router
        # max_total_token_num - get_fixed_kv_len() 是为了减去被特定
        # 推理模式预先占用了部分token kv 资源，这会导致整体可用的kv 资源
        # 在极端情况下减少，在非特定模式下，get_fixed_kv_len() 返回的都是
        # 0， 不会有任何影响。
        self.max_total_tokens = args.max_total_token_num - get_fixed_kv_len()
        assert args.batch_max_tokens is not None
        self.batch_max_tokens = args.batch_max_tokens
        self.running_max_req_size = args.running_max_req_size  # Maximum number of concurrent requests
        self.waiting_req_list: List[Req] = []  # List of queued requests
        self.router_token_ratio = args.router_token_ratio  # ratio to determine whether the router is busy
        self.router_max_new_token_len = args.router_max_new_token_len

    def free_aborted_req_cpu_cache_pages(self, req: Req):
        if self.args.enable_cpu_cache:
            self.router.cpu_cache_client.lock.acquire_sleep1ms()
            self.router.cpu_cache_client.deref_pages(req.cpu_cache_match_page_indexes.get_all())
            req.cpu_cache_match_page_indexes.clear()
            self.router.cpu_cache_client.lock.release()

    def extend(self, req_group: List[Req]):
        for req in req_group:
            req.sample_params.suggested_dp_index = self.dp_index
        self.waiting_req_list.extend(req_group)
        return

    def get_wait_req_num(self):
        return len(self.waiting_req_list)

    def is_busy(self):
        # 计算当前所有的token使用量, 如果使用了dynamic prompt cache, 使用的token量中不包含，cache tree 中未被引用的数据。
        cur_all_used_tokens = self.router.get_used_tokens(self.dp_index)
        # 判断当前服务是否处于token使用率过高的状态，过高的情况下，调度要偏向保守
        cur_token_ratio = (
            cur_all_used_tokens + self.router.shared_token_load.get_frozened_token_count(self.dp_index)
        ) / self.max_total_tokens
        is_busy = cur_token_ratio >= self.router_token_ratio
        return is_busy

    def get_batch_dp_req_size(self, current_batch: Batch):
        if current_batch is None:
            return 0
        if self.dp_size_in_node == 1:
            return len(current_batch.reqs)

        return len([req for req in current_batch.reqs if req.sample_params.suggested_dp_index == self.dp_index])

    def generate_new_batch(self, current_batch: Batch):
        """
        args:
            current_batch: current batch
        return:
            new batch
        """
        raise NotImplementedError()

    def calcu_batch_token_load(self, current_batch: Batch):
        if current_batch is None:
            return 0, self.router.shared_token_load.get_frozened_token_count(self.dp_index) / self.max_total_tokens
        else:
            return self._calcu_batch_token_load_batch_not_none(current_batch)

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):
        raise NotImplementedError()

    def update_token_load(self, current_batch: Batch, force_update=False):
        # #region agent log
        if self.router.shared_token_load is None:
            _agent_debug_log(
                location="server/router/req_queue/base_queue.py:update_token_load",
                message="shared_token_load is None before update",
                data={
                    "force_update": bool(force_update),
                    "has_running_batch": bool(current_batch is not None),
                    "dp_index": int(self.dp_index),
                },
                hypothesis_id="H26",
            )
            # Router is shutting down and shared memory already cleaned.
            # Skip token-load update to avoid crashing during teardown race.
            return
        # #endregion
        if self.router.shared_token_load.need_update_dynamic_max_load() or force_update:
            estimated_peak_token_count, dynamic_max_load = self.calcu_batch_token_load(current_batch)
            token_ratio1 = self.router.get_used_tokens(self.dp_index) / self.router.max_total_token_num
            # #region agent log
            _agent_debug_log(
                location="server/router/req_queue/base_queue.py:update_token_load",
                message="updating token load",
                data={
                    "force_update": bool(force_update),
                    "dp_index": int(self.dp_index),
                    "token_ratio": float(token_ratio1),
                    "estimated_peak_token_count": int(estimated_peak_token_count),
                    "dynamic_max_load": float(dynamic_max_load),
                },
                hypothesis_id="H29",
            )
            # #endregion
            with g_router_lock.obj:
                self.router.shared_token_load.set_current_load(token_ratio1, self.dp_index)
                self.router.shared_token_load.set_estimated_peak_token_count(estimated_peak_token_count, self.dp_index)
                self.router.shared_token_load.set_dynamic_max_load(dynamic_max_load, self.dp_index)
        return
