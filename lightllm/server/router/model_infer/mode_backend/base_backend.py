import os
import json
import numpy as np
import torch
import time
import threading
from dataclasses import replace
import torch.distributed as dist
from typing import List, Tuple, Callable, Optional, Dict, Set
from transformers.configuration_utils import PretrainedConfig
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.log_utils import init_logger
from lightllm.models import get_model
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.model_infer.infer_batch import (
    InferReq,
    InferReqUpdatePack,
    get_req_adapter_bin,
    normalize_req_adapter_id,
)
from lightllm.server.router.token_load import TokenLoad
from lightllm.common.basemodel.infer_lock import g_infer_state_lock, InferStateLock
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.batch_objs import ModelOutput, ModelInput
from lightllm.common.basemodel.triton_kernel.mtp_utils import mtp_verify
from lightllm.utils.dist_utils import init_distributed_env
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs import ShmReqManager, StartArgs
from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig
from lightllm.server.core.objs.io_objs import AbortedReqCmd, StopStrMatchedReqCmd
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager
from lightllm.utils.dist_utils import get_global_rank, get_global_world_size, get_dp_size
from lightllm.utils.dist_utils import get_dp_world_size, get_global_dp_rank, get_current_rank_in_dp
from lightllm.utils.dist_utils import get_current_device_id, get_current_rank_in_node, get_node_world_size
from lightllm.utils.dist_utils import get_dp_rank_in_node, create_new_group_for_current_node
from lightllm.utils.envs_utils import (
    get_env_start_args,
    enable_radix_tree_timer_merge,
    get_radix_tree_merge_update_delta,
)
from lightllm.distributed import dist_group_manager
from lightllm.server.core.objs.shm_objs_io_buffer import ShmObjsIOBuffer
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager, OverlapEventPack
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.common.basemodel.triton_kernel.gather_token_id import scatter_token
from lightllm.server.pd_io_struct import NIXLChunckedTransTaskRet
from .multi_level_kv_cache import MultiLevelKvCacheModule
from lightllm.server.embed_cache.embed_cache_client import CpuEmbedCacheClient

class ModeBackend:
    def __init__(self) -> None:
        self.shm_req_manager = ShmReqManager()

        self.overlap_event_manager = OverlapEventManager()
        # 标识是否支持 overlap 功能，很多子类模式如 xgrammar 和 outlines 当前不支持 overlap 高性能模式
        self.support_overlap = True

        # prefill_mask_func 和 decode_mask_func 用于控制在采样输出前，通过对logics的调整，改变输出的选择空间，
        # 主要是为约束输出模式进行定制的操作
        self.prefill_mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None
        self.decode_mask_func: Optional[Callable[[List[InferReq], torch.Tensor], None]] = None
        # extra_post_req_handle_func 用于添加请求InferReq的状态变化中添加额外的后处理信息，主要是状态机相关的调整等。
        self.extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None

        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
        self.enable_prefill_microbatch_overlap = get_env_start_args().enable_prefill_microbatch_overlap

        # 控制 _get_classed_reqs 分类的参数变量，不同的 backend 具有可能需要不同的分类运行条件。
        self.classed_req_no_decode = False
        self.classed_req_strict_prefill = True

        # nixl pd mode callback func
        self.nixl_prefill_chuncked_handle_func: Optional[Callable[[InferReq, int, float, int], None]] = None

        # counter
        self._radix_tree_merge_counter: int = 0
        self._enable_radix_tree_timer_merge: bool = enable_radix_tree_timer_merge()
        self._radix_tree_merge_update_delta: int = get_radix_tree_merge_update_delta()
        self._decode_step_id: int = 0
        pass

    def _alloc_decode_step_id(self) -> int:
        decode_step_id = self._decode_step_id
        self._decode_step_id += 1
        return decode_step_id

    @staticmethod
    def _split_lora_dirs(lora_dir_arg: Optional[str]) -> List[str]:
        if not lora_dir_arg:
            return []
        return [item.strip() for item in lora_dir_arg.split(",") if item.strip()]

    def _build_lora_adapter_dirs(self, lora_dir_arg: Optional[str]) -> Dict[int, str]:
        lora_dirs = self._split_lora_dirs(lora_dir_arg)
        adapter_dirs: Dict[int, str] = {}
        for adapter_id, adapter_dir in enumerate(lora_dirs, start=1):
            abs_dir = os.path.abspath(adapter_dir)
            if not os.path.isdir(abs_dir):
                raise FileNotFoundError(f"LoRA directory not found: {abs_dir}")
            adapter_dirs[adapter_id] = abs_dir
        return adapter_dirs

    def _apply_colora_speculation_env(self) -> None:
        spec_enabled = bool(getattr(self.args, "colora_speculative_dispatch", False))
        raw_whitelist = getattr(self.args, "colora_spec_layer_whitelist", "")
        whitelist_tokens = []
        for token in str(raw_whitelist or "").split(","):
            token = token.strip()
            if token:
                whitelist_tokens.append(token)
        whitelist = ",".join(whitelist_tokens)
        os.environ["COLORA_SPEC_SUBMIT_ENABLE"] = "1" if spec_enabled else "0"
        os.environ["COLORA_SPEC_SUBMIT_LAYER_WHITELIST"] = whitelist
        temporal_enabled = bool(getattr(self.args, "colora_temporal_prefetch", False))
        raw_temporal_whitelist = getattr(self.args, "colora_temporal_prefetch_layer_whitelist", "")
        temporal_whitelist_tokens = []
        for token in str(raw_temporal_whitelist or "").split(","):
            token = token.strip()
            if token:
                temporal_whitelist_tokens.append(token)
        temporal_whitelist = ",".join(temporal_whitelist_tokens)
        os.environ["COLORA_TEMPORAL_PREFETCH_ENABLE"] = "1" if temporal_enabled else "0"
        os.environ["COLORA_TEMPORAL_PREFETCH_LAYER_WHITELIST"] = temporal_whitelist

    def init_model(self, kvargs):
        self.args: StartArgs = kvargs.get("args", None)
        assert self.args is not None
        # p d 分离模式下会有特殊的一些初始化, 所以需要传递
        # 模式参数到模型的初始化过程中进行控制
        self.run_mode = self.args.run_mode
        self.is_multimodal = False
        self.nnodes = self.args.nnodes
        self.node_rank = self.args.node_rank
        self.world_size = kvargs["world_size"]
        self.dp_size = self.args.dp
        # dp_size_in_node 计算兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
        self.dp_size_in_node = max(1, self.dp_size // self.nnodes)
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.disable_chunked_prefill = self.args.disable_chunked_prefill
        self.chunked_prefill_size = self.args.chunked_prefill_size
        self.return_all_prompt_logprobs = self.args.return_all_prompt_logprobs
        self.use_dynamic_prompt_cache = not self.args.disable_dynamic_prompt_cache
        self.batch_max_tokens = self.args.batch_max_tokens
        self.eos_id: List[int] = kvargs.get("eos_id", [2])
        self.disable_cudagraph = self.args.disable_cudagraph
        self.is_multinode_tp = self.args.nnodes > 1 and self.args.dp == 1
        self.is_nixl_pd_mode = self.run_mode in ["nixl_prefill", "nixl_decode"]
        self.is_nixl_decode_mode = self.run_mode == "nixl_decode"

        self.logger = init_logger(__name__)

        self.weight_dir = kvargs["weight_dir"]
        # p d 分离模式，decode节点才会使用的参数
        self.pd_rpyc_ports = kvargs.get("pd_rpyc_ports", None)
        max_total_token_num = kvargs["max_total_token_num"]

        init_distributed_env(kvargs)
        self.init_rank_infos()
        group_size = (
            2 if (self.args.enable_decode_microbatch_overlap or self.args.enable_prefill_microbatch_overlap) else 1
        )
        dist_group_manager.create_groups(group_size=group_size)  # set the default group

        self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", self.dp_size_in_node)

        # 为 p d 分离模式添加的全局锁管理，用于做一些同步操作。 一定需要在
        # init_process_group 之后调用
        g_infer_state_lock.obj = (
            InferStateLock(
                name=get_unique_server_name(),
                rank_in_dp=self.rank_in_dp,
                dp_rank_in_node=self.dp_rank_in_node,
                dp_world_size=self.dp_world_size,
            )
            if self.run_mode in ["prefill", "decode"]
            else None
        )
        g_infer_state_lock.dp_world_size = self.dp_world_size
        self.infer_state_lock = g_infer_state_lock
        # 防止InferStateLock 中的全局共享信息被重复异常初始化,导致同步异常的问题。
        # 所以做一次barrier等待
        dist.barrier()

        wait_events = []
        if self.args.enable_cpu_cache:
            self.multi_level_cache_module = MultiLevelKvCacheModule(self)
            wait_events.append(self.multi_level_cache_module)

        if self.args.enable_multimodal:
            g_infer_context.init_cpu_embed_cache_client()

        # Initialize node-local comm primitives before LoRA preload.
        # Batched LoRA startup may use rank0 broadcast fast path, which needs this group.
        if not hasattr(self, "node_nccl_group"):
            self.node_broadcast_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.node_nccl_group = create_new_group_for_current_node("nccl")

        model_cfg, _ = PretrainedConfig.get_config_dict(self.weight_dir)
        self._apply_colora_speculation_env()

        model_kvargs = {
            "weight_dir": self.weight_dir,
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "is_token_healing": kvargs.get("is_token_healing", False),
            "return_all_prompt_logics": self.return_all_prompt_logprobs,
            "disable_chunked_prefill": self.disable_chunked_prefill,
            "data_type": kvargs.get("data_type", "float16"),
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
            "mem_fraction": kvargs.get("mem_fraction", 0.9),
            "batch_max_tokens": kvargs.get("batch_max_tokens", None),
            "quant_type": kvargs.get("quant_type", None),
            "quant_cfg": kvargs.get("quant_cfg", None),
            "run_mode": self.run_mode,
            "wait_events": wait_events,
            # LoRA configuration for detached serving
            "lora_dir": kvargs.get("lora_dir", None),
            "lora_max_size": kvargs.get("lora_max_size", 1024),
            "lora_adapter_id": kvargs.get("lora_adapter_id", "default"),
            "lora_port": kvargs.get("lora_port", None),
        }
        self.model, self.is_multimodal = get_model(model_cfg, model_kvargs)
        self.model: TpPartBaseModel = self.model  # for easy typing
        set_random_seed(2147483647)

        # Initialize LoRA adapters for S-LoRA batched mode
        lora_dir = kvargs.get("lora_dir")
        if lora_dir:
            # Use batched mode for S-LoRA
            self.use_batched_lora_mode = True
            lora_adapter_dirs = self._build_lora_adapter_dirs(lora_dir)
            lora_compute_config = LoRAComputeConfig.from_string(self.args.compute_device)
            self._import_lora_modules(lora_compute_config=lora_compute_config)
            self.logger.info(
                "[LoRA Backend] Startup adapter map: "
                + ", ".join(f"{adapter_id}:{adapter_dir}" for adapter_id, adapter_dir in lora_adapter_dirs.items())
            )
            self.init_batched_lora_adapters(lora_adapter_dirs)

        self.radix_cache = (
            RadixCache(
                get_unique_server_name(),
                self.model.mem_manager.size,
                self.rank_in_node,
                mem_manager=self.model.mem_manager,
            )
            if self.use_dynamic_prompt_cache
            else None
        )

        if "prompt_cache_kv_buffer" in model_cfg:
            assert self.use_dynamic_prompt_cache
            self.preload_prompt_cache_kv_buffer(model_cfg)

        self.logger.info(f"loaded model class {self.model.__class__}")

        g_infer_context.register(
            backend=self,
            req_manager=self.model.req_manager,
            radix_cache=self.radix_cache,
            shm_req_manager=self.shm_req_manager,
            vocab_size=self.model.vocab_size,
        )

        # 初始化 dp 模式使用的通信 tensor, 对于非dp模式，不会使用到
        if self.dp_size > 1:
            self.dp_reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_gather_item_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.dp_all_gather_tensor = torch.tensor(
                [0 for _ in range(self.global_world_size)], dtype=torch.int32, device="cuda", requires_grad=False
            )

        # 用于协同读取 ShmObjsIOBuffer 中的请求信息的通信tensor和通信组对象。
        # May already be initialized before LoRA adapter preload.
        if not hasattr(self, "node_nccl_group"):
            self.node_broadcast_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
            self.node_nccl_group = create_new_group_for_current_node("nccl")

        # 用于在多节点tp模式下协同读取 ShmObjsIOBuffer 中的请求信息的通信tensor和通信组对象。
        if self.is_multinode_tp:
            self.multinode_tp_gather_item_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.multinode_tp_all_gather_tensor = torch.tensor(
                [0 for _ in range(self.global_world_size)], dtype=torch.int32, device="cuda", requires_grad=False
            )
            self.multinode_tp_nccl_group = dist.new_group(
                [rank for rank in range(self.global_world_size)], backend="nccl"
            )

        if (
            self.args.run_mode in ["nixl_prefill", "nixl_decode", "prefill", "decode"]
            or self.args.enable_dp_prompt_cache_fetch
        ):
            # 如果存在需要跨进程使用mem manger的特性，则将mem manager写入到 shm中，方便
            # 读取
            self.model.mem_manager.write_to_shm(req_manager=self.model.req_manager)
            dist.barrier(group=self.node_nccl_group)

        self.init_custom()

        if self.args.enable_dp_prompt_cache_fetch:
            self.init_dp_kv_shared()

        self.shm_reqs_io_buffer = ShmObjsIOBuffer()
        # 只会在 nixl pd 模式下才会使用，用于上传分块传输任务是否成功。
        self.shm_nixl_trans_io_buffer = ShmObjsIOBuffer(tail_str="nixl")

        # 开启 mtp 模式，需要完成mtp model的初始化
        if self.args.mtp_mode:
            self.init_mtp_draft_model(kvargs)

        # 启动infer_loop_thread, 启动两个线程进行推理，对于具备双batch推理折叠得场景
        # 可以降低 cpu overhead，大幅提升gpu得使用率。
        self.infer_loop_thread = threading.Thread(target=self.infer_loop, daemon=True)
        self.infer_loop_thread.start()
        self.infer_loop_thread1 = threading.Thread(target=self.infer_loop, daemon=True)
        self.infer_loop_thread1.start()
        return

    def init_custom(self):
        pass

    def init_dp_kv_shared(self):
        from lightllm.server.router.model_infer.mode_backend.dp_backend.dp_shared_kv_trans import DPKVSharedMoudle
        from lightllm.common.kv_cache_mem_manager import MemoryManager

        torch.cuda.set_device(get_current_device_id())

        self.dp_kv_shared_module = DPKVSharedMoudle(
            max_req_num=self.args.running_max_req_size,
            max_req_seq_len=self.args.max_req_total_len + 8,
            dp_size_in_node=self.dp_size_in_node,
            backend=self,
        )

        # Collect mem_managers from all ranks
        self.mem_managers = []
        for rank_idx in range(self.node_world_size):
            if rank_idx != self.rank_in_node:
                self.mem_managers.append(MemoryManager.loads_from_shm(rank_idx))
            else:
                self.mem_managers.append(self.model.mem_manager)
        return

    def get_max_total_token_num(self):
        return self.model.mem_manager.size

    def infer_loop(self):
        raise NotImplementedError()

    def prefill(self, event_pack: OverlapEventPack, prefill_reqs: List[InferReq]):
        raise NotImplementedError()

    def decode(self, event_pack: OverlapEventPack, decode_reqs: List[InferReq]):
        raise NotImplementedError()

    def init_mtp_draft_model(self, main_kvargs: dict):
        # 当前只支持 deepseekv3 模式的 mtp
        self.mtp_step = self.args.mtp_step
        self.draft_models: List[Deepseek3MTPModel] = []

        os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"

        if self.args.mtp_mode == "deepseekv3_vanilla":
            num_mtp_modules = self.args.mtp_step
        elif self.args.mtp_mode == "deepseekv3_eagle":
            num_mtp_modules = 1
        else:
            assert False, f"error mtp mode {self.args.mtp_mode}"

        for i in range(num_mtp_modules):
            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(self.args.mtp_draft_model_dir)
            mtp_model_kvargs = {
                "weight_dir": self.args.mtp_draft_model_dir,
                "max_total_token_num": self.model.mem_manager.size,
                "load_way": main_kvargs["load_way"],
                "mode": main_kvargs["mode"],
                "max_req_num": main_kvargs.get("max_req_num", 1000),
                "max_seq_length": main_kvargs.get("max_seq_length", 1024 * 5),
                "is_token_healing": False,
                "return_all_prompt_logics": False,
                "disable_chunked_prefill": self.disable_chunked_prefill,
                "data_type": main_kvargs.get("data_type", "float16"),
                "graph_max_batch_size": main_kvargs.get("graph_max_batch_size", 16),
                "graph_max_len_in_batch": main_kvargs.get("graph_max_len_in_batch", 8196),
                "disable_cudagraph": main_kvargs.get("disable_cudagraph", False),
                "mem_fraction": main_kvargs["mem_fraction"],
                "batch_max_tokens": main_kvargs.get("batch_max_tokens", None),
                "quant_type": main_kvargs.get("quant_type", None),
                "quant_cfg": main_kvargs.get("quant_cfg", None),
                "run_mode": "normal",
                "main_model": self.model,
                "mem_layer_start": self.model.config["num_hidden_layers"] + i * mtp_model_cfg["num_hidden_layers"],
            }

            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(self.args.mtp_draft_model_dir)
            assert mtp_model_cfg["model_type"] == "deepseek_v3"
            assert mtp_model_cfg["architectures"][0] == "DeepseekV3ForCausalLMNextN"
            self.draft_models.append(Deepseek3MTPModel(mtp_model_kvargs))

            self.logger.info(f"loaded mtp model class {self.draft_models[i].__class__}")
        return

    def _async_copy_next_token_infos_to_pin_mem(self, next_token_ids: torch.Tensor, next_token_logprobs: torch.Tensor):
        """
        这个函数会把next token id和logprobs保存到pinned memory中
        这样可以保障post_handle 函数可以读取到正常的输出结果。
        """
        next_token_ids_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="next_token_ids",
            gpu_tensor=next_token_ids,
        )
        next_token_logprobs_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor(
            key="next_token_logprobs",
            gpu_tensor=next_token_logprobs,
        )
        return next_token_ids_cpu, next_token_logprobs_cpu

    def _try_read_new_reqs(self):
        if self.is_multinode_tp:
            self._try_read_new_reqs_multinode_tp()
        else:
            self._try_read_new_reqs_normal()
        return

    def _try_read_new_reqs_normal(self):
        try:
            torch.cuda.synchronize()
        except Exception:
            raise
        if self.is_master_in_node:
            if self.shm_reqs_io_buffer.is_ready():
                self.node_broadcast_tensor.fill_(1)
            else:
                self.node_broadcast_tensor.fill_(0)

        src_rank_id = self.args.node_rank * self.node_world_size
        dist.broadcast(self.node_broadcast_tensor, src=src_rank_id, group=self.node_nccl_group, async_op=False)
        try:
            torch.cuda.synchronize()
        except Exception:
            raise
        new_buffer_is_ready = self.node_broadcast_tensor.detach().item()
        if new_buffer_is_ready:
            self._read_reqs_buffer_and_init_reqs()

        # nixl pd mode 从 shm_nixl_trans_io_buffer 读取分块传输的完成进度。
        if self.is_nixl_pd_mode:
            if self.is_master_in_node:
                if self.shm_nixl_trans_io_buffer.is_ready():
                    self.node_broadcast_tensor.fill_(1)
                else:
                    self.node_broadcast_tensor.fill_(0)

            src_rank_id = self.args.node_rank * self.node_world_size
            dist.broadcast(self.node_broadcast_tensor, src=src_rank_id, group=self.node_nccl_group, async_op=False)
            new_buffer_is_ready = self.node_broadcast_tensor.detach().item()
            if new_buffer_is_ready:
                self._read_nixl_trans_io_buffer_and_update_req_status()
        return

    def _try_read_new_reqs_multinode_tp(self):
        """
        多节点tp模式下,需要协调所有rank的行为同步。
        """
        if self.shm_reqs_io_buffer.is_ready():
            self.multinode_tp_gather_item_tensor.fill_(1)
        else:
            self.multinode_tp_gather_item_tensor.fill_(0)
        dist.all_gather_into_tensor(
            self.multinode_tp_all_gather_tensor,
            self.multinode_tp_gather_item_tensor,
            group=self.multinode_tp_nccl_group,
            async_op=False,
        )
        new_buffer_is_readys = self.multinode_tp_all_gather_tensor.detach().cpu().numpy()
        new_buffer_is_ready = np.all(new_buffer_is_readys == 1)

        if new_buffer_is_ready:
            self._read_reqs_buffer_and_init_reqs()

        assert self.is_nixl_pd_mode is False
        return

    def _read_reqs_buffer_and_init_reqs(self):
        cmds: List = self.shm_reqs_io_buffer.read_obj()
        self.shm_reqs_io_buffer.sub_state()
        if cmds:
            init_reqs = []
            for obj in cmds:
                if isinstance(obj, tuple):
                    init_reqs.append(obj)
                elif isinstance(obj, (AbortedReqCmd, StopStrMatchedReqCmd)):
                    if obj.req_id in g_infer_context.requests_mapping:
                        req: InferReq = g_infer_context.requests_mapping[obj.req_id]
                        req.infer_aborted = True
                else:
                    assert False, f"error type {type(obj)}"
            if init_reqs:
                req_ids = self._init_reqs(reqs=init_reqs)
                if self.args.enable_cpu_cache and req_ids:
                    self._load_cpu_cache_to_reqs(req_ids=req_ids)
        return

    def _read_nixl_trans_io_buffer_and_update_req_status(self):
        cmds: List[NIXLChunckedTransTaskRet] = self.shm_nixl_trans_io_buffer.read_obj()
        self.shm_nixl_trans_io_buffer.sub_state()
        if cmds:
            for obj in cmds:
                if obj.request_id in g_infer_context.requests_mapping:
                    req: InferReq = g_infer_context.requests_mapping[obj.request_id]
                    if obj.has_error:
                        req.nixl_pd_task_failed_num += 1
                    else:
                        req.nixl_pd_task_sunccess_num += 1
                        # nixl decode 节点需要预填充 prefill 节点发送过来的产生的首token信息，以使
                        # 推理过程可以继续。
                        if self.is_nixl_decode_mode:
                            if obj.first_gen_token_id is not None:
                                assert req.cur_output_len == 0
                                req.cur_output_len += 1
                                req_to_next_token_ids = (
                                    self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids
                                )
                                # to do 这个地方是否需要加流同步
                                req_to_next_token_ids[req.req_idx, 0:1].fill_(obj.first_gen_token_id)
                                torch.cuda.current_stream().synchronize()
                                InferReqUpdatePack(req_obj=req, output_len=req.cur_output_len).handle(
                                    next_token_id=obj.first_gen_token_id,
                                    next_token_logprob=obj.first_gen_token_logprob,
                                    eos_ids=self.eos_id,
                                    extra_post_req_handle_func=None,
                                    is_master_in_dp=self.is_master_in_dp,
                                    nixl_prefill_chuncked_handle_func=None,
                                )
        return

    # 一些可以复用的通用功能函数
    def _init_reqs(self, reqs: List[Tuple]):
        """
        init_req_obj 参数用于控制是否对请求对象的进行全量初始化，如果设置为True
        在 g_infer_context.add_reqs 函数中，会进行全量初始化，包括其 kv 信息等，
        如果设置为 False，则请求对象只是创建了基础信息，需要延迟到合适的时机调用
        请求对象的完整初始化，设计这个接口的用途是用于某些追求高性能场景的cpu gpu
        折叠，降低cpu 的overhead。
        """
        if self.dp_size_in_node != 1:
            dp_rank_in_node = self.dp_rank_in_node
            reqs = [req for req in reqs if req[3] == dp_rank_in_node]

        g_infer_state_lock.acquire()
        g_infer_context.add_reqs(reqs)
        g_infer_state_lock.release()
        req_ids = [e[0] for e in reqs]
        return req_ids

    def _load_cpu_cache_to_reqs(self, req_ids):
        req_objs: List[InferReq] = [g_infer_context.requests_mapping[req_id] for req_id in req_ids]
        g_infer_state_lock.acquire()
        self.multi_level_cache_module.load_cpu_cache_to_reqs(reqs=req_objs)
        g_infer_state_lock.release()
        return

    def _filter_not_ready_reqs(self, req_ids: List[int]) -> List[InferReq]:
        """
        将错误请求从 req_ids 中过滤出来, 然后让 _get_classed_reqs 进行处理。 该函数
        主要用于在 nixl pd 分离模式下, 由子类继承重载, prefill 和 decode 节点过滤 kv 传输错误，或者 kv
        传输没有完成的请求。
        """
        return [g_infer_context.requests_mapping[request_id] for request_id in req_ids]

    def _timer_merge_radix_tree(self):
        self._radix_tree_merge_counter += 1
        if (
            self._enable_radix_tree_timer_merge
            and (self._radix_tree_merge_counter % self._radix_tree_merge_update_delta == 0)
            and self.radix_cache is not None
        ):
            g_infer_state_lock.acquire()
            start = time.time()
            self.radix_cache.merge_unreferenced_nodes()
            self.logger.info(
                f"radix tree merge_unreferenced_nodes cost time {time.time() - start} s in rank {self.global_rank}"
            )
            g_infer_state_lock.release()
        return

    # 一些可以复用的通用功能函数
    def _get_classed_reqs(
        self,
        req_ids: List[int] = None,
        no_decode: bool = False,
        strict_prefill: bool = False,
        recover_paused: bool = False,
    ):
        """
        当将参数 no_decode 设置为True后，返回的 decode_reqs 永远为空list，主要是
        PD 分离的某些backend需要用这个参数进行控制，因为P节点永远只进行Prefill,
        避免一些特殊情况，如 radix cache 命中后，只有1token需要prefill，这个判断
        条件和decode请求的分类条件相同。所以添加一个参数进行区分。

        strict_prefill参数用于控制当 cur_kv_len + 1 == input_len 时，是否将请求
        分为 prefill,当 strict_prefill 设置为True时，表示需要将这个请求分为 prefill,
        为 False 时，将这个请求分为decode。 strict_prefill 主要是用于diverse mode
        使用时，其他模式目前不使用。

        将请求分类返回:
        1. wait_pause_reqs 因为推理资源不够，等待被暂停的请求。
        2. paused_reqs 已经被暂停的请求，可能会被恢复。
        3. finished_reqs 需要释放的请求, 包含正常结束和aborted退出的请求。
        4. prefill_reqs 需要进行prefill操作的请求
        5. decode_reqs 需要进行decode操作的请求
        """
        # 定期对 radix cache 进行 merge，防止查询插入的操作效率下降
        self._timer_merge_radix_tree()

        if self.args.enable_cpu_cache and len(g_infer_context.infer_req_ids) > 0:
            self.multi_level_cache_module.update_cpu_cache_task_states()

        if req_ids is None:
            req_ids = g_infer_context.infer_req_ids

        if len(req_ids) == 0:
            return [], []

        ready_reqs = self._filter_not_ready_reqs(req_ids)
        support_overlap = self.support_overlap

        wait_pause_reqs = []
        paused_reqs = []
        finished_reqs = []
        prefill_reqs = []
        decode_reqs = []

        # 一次性最多暂停请求的数量, 防止盲目暂停大量请求
        # 因为部分请求释放占用的token容量后，就会使推理可以正常进行。
        # 如果因为一次推理容量不足，就以当前token容量的判断暂停了大量
        # 请求，其逻辑是不适合的。
        pause_max_req_num = 2
        wait_pause_count = 0
        prefill_tokens = 0

        # 因为会使用到 radix cache 和 mem_manager 的计数信息
        # 所以需要加锁保护。
        g_infer_state_lock.acquire()
        can_alloc_token_num = g_infer_context.get_can_alloc_token_num()

        for req_obj in ready_reqs:

            if req_obj.filter_mark:
                finished_reqs.append(req_obj)
                continue

            if req_obj.wait_pause:
                wait_pause_reqs.append(req_obj)
                continue

            if req_obj.paused:
                paused_reqs.append(req_obj)
                continue

            if req_obj.colora_paused:
                # Request is paused waiting for COLoRA CPU completion
                continue

            if req_obj.infer_aborted or req_obj.finish_status.is_finished():
                if support_overlap:
                    # 延迟处理
                    req_obj.filter_mark = True
                    continue
                else:
                    finished_reqs.append(req_obj)
                    continue

            if no_decode:
                is_decode = False
            else:
                is_decode = req_obj.cur_kv_len + 1 == req_obj.get_cur_total_len()
                if is_decode and strict_prefill and req_obj.cur_kv_len + 1 == req_obj.shm_req.input_len:
                    is_decode = False

            if is_decode:
                token_num = req_obj.decode_need_token_num()
                if token_num <= can_alloc_token_num:
                    decode_reqs.append(req_obj)
                    can_alloc_token_num -= token_num
                else:
                    if wait_pause_count < pause_max_req_num:
                        req_obj.wait_pause = True
                        wait_pause_count += 1
            else:
                # 在 diverse mode 模式下，prefill 只会使用 master 状态的请求，slave 请求依靠后续
                # 的推理代码中将master请求的状态复制到slave请求中去， 所以这里 slave 状态的请求，不
                # 放入到 prefill reqs 队列中，在其他模式下，所有请求都是 master状态，所以也不受影响
                if req_obj.is_slave_req():
                    continue

                token_num = req_obj.prefill_need_token_num(is_chuncked_prefill=not self.disable_chunked_prefill)
                if prefill_tokens + token_num > self.batch_max_tokens:
                    continue
                if token_num <= can_alloc_token_num:
                    prefill_tokens += token_num
                    prefill_reqs.append(req_obj)
                    can_alloc_token_num -= token_num
                else:
                    if wait_pause_count < pause_max_req_num:
                        req_obj.wait_pause = True
                        wait_pause_count += 1

        g_infer_state_lock.release()

        self._pre_handle_finished_reqs(finished_reqs=finished_reqs)
        # 如果使能了 cpu cache 功能，对于已经完成的请求，进行 gpu kv 卸载到 cpu cache的操作。
        if self.args.enable_cpu_cache:
            true_finished_reqs = self.multi_level_cache_module.offload_finished_reqs_to_cpu_cache(
                finished_reqs=finished_reqs
            )
        else:
            true_finished_reqs = finished_reqs

        g_infer_context.filter_reqs(finished_reqs=true_finished_reqs)
        g_infer_context.pause_reqs(wait_pause_reqs, is_master_in_dp=self.is_master_in_dp)

        if recover_paused:
            g_infer_context.recover_paused_reqs(
                paused_reqs=paused_reqs, is_master_in_dp=self.is_master_in_dp, can_alloc_token_num=can_alloc_token_num
            )

        return prefill_reqs, decode_reqs

    def _pre_handle_finished_reqs(self, finished_reqs: List[InferReq]):
        """
        给 PD 分离模式下，prefill node 使用的继承钩子函数，用于发起 kv 传输任务。
        """
        pass

    # 一些可以复用的通用功能函数
    def _pre_post_handle(self, run_reqs: List[InferReq], is_chuncked_mode: bool) -> List[InferReqUpdatePack]:
        update_func_objs: List[InferReqUpdatePack] = []
        # 通用状态预先填充
        is_master_in_dp = self.is_master_in_dp
        for req_obj in run_reqs:
            req_obj: InferReq = req_obj
            if is_chuncked_mode:
                new_kv_len = req_obj.get_chuncked_input_token_len()
            else:
                new_kv_len = req_obj.get_cur_total_len()
            req_obj.cur_kv_len = new_kv_len
            if is_master_in_dp:
                req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len

            # 对于没有到达需要输出 token 阶段的请求，直接略过, 说明还
            # 处于chuncked prefill kv 填充的阶段。
            if req_obj.cur_kv_len < req_obj.get_cur_total_len():
                pack = InferReqUpdatePack(req_obj=req_obj, output_len=0)
                update_func_objs.append(pack)
                continue

            # 将生成的下一个token的信息写入到管理对象中。
            req_obj.cur_output_len += 1
            pack = InferReqUpdatePack(req_obj=req_obj, output_len=req_obj.cur_output_len)
            update_func_objs.append(pack)
        return update_func_objs

    # 一些可以复用的通用功能函数
    def _post_handle(
        self,
        run_reqs: List[InferReq],
        next_token_ids: List[int],
        next_token_logprobs: List[float],
        run_reqs_update_packs: List[InferReqUpdatePack],
        extra_post_req_handle_func: Optional[Callable[[InferReq, int, float], None]] = None,
        nixl_prefill_chuncked_handle_func: Optional[Callable[[InferReq, int, float, int], None]] = None,
    ):
        """
        extra_post_req_handle_func 用于提供在一个请求确定输出的时候，给出额外的后处理操作，主要是用于
        约束输出等模式，设置自己请求内部的状态机的状态，并添加额外的停止判定条件等。
        """
        for req_obj, next_token_id, next_token_logprob, pack in zip(
            run_reqs, next_token_ids, next_token_logprobs, run_reqs_update_packs
        ):
            req_obj: InferReq = req_obj
            pack: InferReqUpdatePack = pack
            pack.handle(
                next_token_id=next_token_id,
                next_token_logprob=next_token_logprob,
                eos_ids=self.eos_id,
                extra_post_req_handle_func=extra_post_req_handle_func,
                is_master_in_dp=self.is_master_in_dp,
                nixl_prefill_chuncked_handle_func=nixl_prefill_chuncked_handle_func,
            )

        g_infer_context.req_manager.req_sampling_params_manager.update_reqs_token_counter(
            req_objs=run_reqs, next_token_ids=next_token_ids
        )
        return

    # 一些可以复用的通用功能函数
    def _filter_reqs(self, reqs: List[InferReq]):
        if reqs:
            g_infer_state_lock.acquire()
            g_infer_context.filter_reqs(reqs)
            g_infer_state_lock.release()
        return

    # 一些可以复用的通用功能函数
    def _trans_req_ids_to_req_objs(self, req_ids: List[int]) -> List[InferReq]:
        return [g_infer_context.requests_mapping[req_id] for req_id in req_ids]

    def _verify_mtp_v2(
        self, new_next_token_ids: torch.Tensor, b_req_idx: torch.Tensor, b_req_mtp_start_loc: torch.Tensor
    ):
        mtp_accept_len, accepted_index = mtp_verify(
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=new_next_token_ids,
            b_req_idx=b_req_idx,
        )
        return mtp_accept_len, accepted_index

    def _update_mtp_accept_ratio(
        self,
        decode_reqs: List[InferReq],
        mtp_accept_len_cpu: torch.Tensor,
    ):
        if self.is_master_in_dp:
            for req, accept_len in zip(decode_reqs, mtp_accept_len_cpu):
                req.update_mtp_accepted_token_num(accept_token_num=accept_len - 1)
        return

    def _gen_argmax_token_ids(self, model_output: ModelOutput):
        logits = model_output.logits
        probs = torch.softmax(logits, dim=-1)
        draft_next_token_ids_gpu = torch.argmax(probs, dim=-1)
        return draft_next_token_ids_gpu

    def _sample_and_scatter_token(
        self,
        logits: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_mtp_index: torch.Tensor,
        run_reqs: List[InferReq],
        is_prefill: bool,
        b_prefill_has_output_cpu: torch.Tensor = None,
        mask_func: Optional[Callable] = None,
    ):
        if mask_func is not None:
            assert len(run_reqs) == logits.shape[0]
            mask_func(run_reqs, logits)

        next_token_ids, next_token_logprobs = sample(logits, run_reqs, self.eos_id)
        b_has_out = None
        if is_prefill:
            b_has_out = g_pin_mem_manager.gen_from_list(
                key="b_has_out", data=b_prefill_has_output_cpu, dtype=torch.bool
            ).cuda(non_blocking=True)

        scatter_token(
            next_token_ids=next_token_ids,
            req_to_next_token_ids=self.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            b_has_out=b_has_out,
        )
        g_infer_context.req_sampling_manager.update_reqs_out_token_counter_gpu(
            b_req_idx=b_req_idx,
            next_token_ids=next_token_ids,
            mask=b_has_out,
        )
        next_token_ids_cpu, next_token_logprobs_cpu = self._async_copy_next_token_infos_to_pin_mem(
            next_token_ids, next_token_logprobs
        )
        return next_token_ids, next_token_ids_cpu, next_token_logprobs_cpu

    def _dp_all_gather_prefill_and_decode_req_num(
        self, prefill_reqs: List[InferReq], decode_reqs: List[InferReq]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather the number of prefill requests across all DP ranks.
        """
        current_dp_prefill_num = len(prefill_reqs)
        self.dp_gather_item_tensor.fill_(current_dp_prefill_num)
        dist.all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        dp_prefill_req_nums = self.dp_all_gather_tensor.cpu().numpy()

        current_dp_decode_num = len(decode_reqs)
        self.dp_gather_item_tensor.fill_(current_dp_decode_num)
        dist.all_gather_into_tensor(self.dp_all_gather_tensor, self.dp_gather_item_tensor, group=None, async_op=False)
        dp_decode_req_nums = self.dp_all_gather_tensor.cpu().numpy()

        return dp_prefill_req_nums, dp_decode_req_nums

    def _dp_all_reduce_decode_req_num(self, decode_reqs: List[InferReq]) -> int:
        """
        Reduce the number of decode requests across all DP ranks.
        """
        current_dp_decode_num = len(decode_reqs)
        self.dp_reduce_tensor.fill_(current_dp_decode_num)
        dist.all_reduce(self.dp_reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_decode_num = self.dp_reduce_tensor.item()
        return max_decode_num

    def preload_prompt_cache_kv_buffer(self, model_cfg):
        self.logger.info("Preload prompt cache kv buffer.")
        cur_rank = dist.get_rank()
        prompt_cache_kv_buffer_path = os.path.join(
            self.weight_dir, model_cfg["prompt_cache_kv_buffer"][f"rank_{cur_rank}"]
        )
        prompt_cache_kv_buffer = torch.load(prompt_cache_kv_buffer_path, weights_only=True, map_location="cpu")
        intact_kv_len = len(model_cfg["prompt_cache_token_ids"])
        intact_kv_index = self.radix_cache.mem_manager.alloc(intact_kv_len)
        self.radix_cache.mem_manager.load_index_kv_buffer(intact_kv_index, prompt_cache_kv_buffer)
        self.radix_cache.insert(
            torch.tensor(model_cfg["prompt_cache_token_ids"], dtype=torch.int64, device="cpu"),
            intact_kv_index,
        )
        self.radix_cache.match_prefix(
            torch.tensor(model_cfg["prompt_cache_token_ids"], dtype=torch.int64, device="cpu"), update_refs=True
        )

    def init_rank_infos(self):
        self.node_world_size = get_node_world_size()
        self.rank_in_node = get_current_rank_in_node()
        self.current_device_id = get_current_device_id()
        self.rank_in_dp = get_current_rank_in_dp()
        self.global_dp_rank = get_global_dp_rank()
        self.dp_rank_in_node = get_dp_rank_in_node()
        self.dp_world_size = get_dp_world_size()
        self.global_rank = get_global_rank()
        self.global_world_size = get_global_world_size()
        self.dp_size = get_dp_size()

        if self.nnodes > 1 and self.dp_size == 1:
            if self.rank_in_node == 0:
                self.is_master_in_dp = True
            else:
                self.is_master_in_dp = False
        else:
            if self.rank_in_dp == 0:
                self.is_master_in_dp = True
            else:
                self.is_master_in_dp = False

        if self.rank_in_node == 0:
            self.is_master_in_node = True
        else:
            self.is_master_in_node = False
        return

    # =========================================================================
    # S-LoRA Batched LoRA Mode Support
    # =========================================================================
    # _import_lora_modules is called from init_model() to set up LoRA functions
    # for S-LoRA batched mode

    def _import_lora_modules(self, lora_compute_config: LoRAComputeConfig):
        """Import LoRA modules based on model type."""
        self.lora_support = False
        self._lora_compute_config = lora_compute_config

        model_module = getattr(self.model, '__module__', '')

        # Define module import paths for different model types
        lora_imports = [
            ('qwen3_vl_moe',
             'lightllm.models.qwen3_vl_moe.lora_dispatch',
             'load_lora_adapter', 'create_vl_moe_lora_dispatcher'),
            ('qwen3_vl',
             'lightllm.models.qwen3_vl.lora_dispatch',
             'load_lora_adapter', 'create_lora_dispatcher'),
        ]

        for model_pattern, module_path, load_fn_name, create_fn_name in lora_imports:
            if model_pattern in model_module:
                try:
                    module = __import__(module_path, fromlist=[load_fn_name, create_fn_name])
                    setattr(self, '_load_lora_adapter_fn', getattr(module, load_fn_name))
                    setattr(self, '_create_lora_dispatcher_fn', getattr(module, create_fn_name))
                    self.lora_support = True
                    self.logger.info(f"Using {model_pattern} LoRA modules")
                    return
                except (ImportError, AttributeError) as e:
                    self.logger.warning(f"Failed to import {model_pattern} LoRA modules: {e}")
                    continue

        # Fallback to default qwen3_vl modules
        try:
            from lightllm.models.qwen3_vl.layer_weights.lora_layer_weight import load_lora_adapter
            from lightllm.models.qwen3_vl.lora_dispatch import create_lora_dispatcher
            self._load_lora_adapter_fn = load_lora_adapter
            self._create_lora_dispatcher_fn = create_lora_dispatcher
            self.lora_support = True
            self.logger.info("Using default qwen3_vl LoRA modules (fallback)")
        except ImportError:
            self.lora_support = False
            self.logger.warning("LoRA modules not available, detached LoRA serving disabled")  # 0 means no adapter

    # =====================================================================
    # S-LoRA Batched LoRA Mode Support
    # =====================================================================

    def init_batched_lora_adapters(self, lora_adapter_dirs: Dict[str, str]):
        """
        Initialize LoRA adapters for S-LoRA batched mode.

        In batched mode, all adapters are pre-loaded into a memory pool,
        and req_bins tracks which adapter each request uses.

        Args:
            lora_adapter_dirs: Dict mapping adapter_id -> adapter_dir
        """
        if not lora_adapter_dirs:
            return

        self.lora_support = True
        self.lora_adapter_dirs = lora_adapter_dirs
        self.use_batched_lora_mode = True
        self.moe_expert_cache_manager = None

        self.logger.info(f"[LoRA Backend] Initializing batched LoRA mode with {len(lora_adapter_dirs)} adapters")

        # Create LoRA memory pool
        try:
            from lightllm.server.lora import (
                create_lora_mem_pool,
                MoEExpertCacheConfig,
                MoEExpertCacheManager,
            )

            config = self.model.config
            num_layers = config["num_hidden_layers"]
            num_heads = config.get("num_attention_heads", 32)
            num_kv_heads = config.get("num_key_value_heads", num_heads)
            head_dim = config.get("head_dim", 128)
            intermediate_dim = config.get("intermediate_size", 512)
            # For MoE models, use moe_intermediate_size for LoRA
            moe_intermediate_dim = config.get("moe_intermediate_size", intermediate_dim)
            # For multimodal models, get hidden_size from text_config
            if "hidden_size" in config:
                hidden_size = config["hidden_size"]
            elif hasattr(config, "get_text_config"):
                text_config = config.get_text_config()
                hidden_size = getattr(text_config, "hidden_size", 2048)
            else:
                hidden_size = 2048
            vocab_size = config.get("vocab_size", 151936)
            max_rank = 16  # Can be configured

            self.logger.info(f"[LoRA Backend] Config values: hidden_size={hidden_size}, intermediate_dim={intermediate_dim}, moe_intermediate_dim={moe_intermediate_dim}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

            # Extract vision config for multimodal models
            vision_config = config.get("vision_config", None)
            vl_hidden_size = vision_config.get("hidden_size") if vision_config else None
            vl_intermediate_size = vision_config.get("intermediate_size") if vision_config else None
            vl_out_hidden_size = vision_config.get("out_hidden_size") if vision_config else None
            vl_depth = vision_config.get("depth") if vision_config else None

            if vl_hidden_size:
                self.logger.info(f"[LoRA Backend] Vision config: vl_hidden_size={vl_hidden_size}, vl_intermediate_size={vl_intermediate_size}, vl_out_hidden_size={vl_out_hidden_size}, vl_depth={vl_depth}")

            effective_lora_compute_config = self._lora_compute_config
            use_colora_hybrid = (
                effective_lora_compute_config is not None
                and effective_lora_compute_config.should_compute_hybrid("moe")
            )
            if (
                use_colora_hybrid
                and effective_lora_compute_config is not None
                and effective_lora_compute_config.moe_storage != "cpu"
            ):
                effective_lora_compute_config = replace(effective_lora_compute_config, moe_storage="cpu")
                self.logger.info(
                    "[COLoRA] moe_compute=hybrid detected, force moe_storage=cpu "
                    "for expert-level asymmetric pool."
                )

            moe_compute_mode = (
                effective_lora_compute_config.get_compute_device("moe")
                if effective_lora_compute_config is not None
                else "gpu"
            )
            model_module = getattr(self.model, "__module__", "")
            if "qwen3_vl_moe" in model_module and moe_compute_mode in ("cpu", "hybrid"):
                try:
                    from lightllm.models.qwen3_vl_moe.lora_dispatch import is_moe_cpu_kernel_available
                except Exception as e:
                    raise RuntimeError(
                        "MoE-specific CPU kernel import failed while strict MoE CPU/hybrid mode "
                        f"is enabled (moe_compute={moe_compute_mode})."
                    ) from e
                if not bool(is_moe_cpu_kernel_available()):
                    raise RuntimeError(
                        "MoE-specific CPU kernel is required for qwen3_vl_moe when "
                        f"moe_compute={moe_compute_mode}, but it is unavailable."
                    )

            self.lora_mem_pool = create_lora_mem_pool(
                num_layers=num_layers,
                pool_size=1024,  # Can hold 1024 adapters
                max_rank=max_rank,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_kv_heads=num_kv_heads,
                dtype=torch.float16,
                lora_compute_config=effective_lora_compute_config,
                vl_hidden_size=vl_hidden_size,
                vl_intermediate_size=vl_intermediate_size,
                vl_out_hidden_size=vl_out_hidden_size,
                vl_depth=vl_depth,
                moe_intermediate_dim=moe_intermediate_dim,
                tp_world_size=get_global_world_size(),
            )

            if use_colora_hybrid:
                cache_cfg = MoEExpertCacheConfig(
                    cache_budget_mb=getattr(self.args, "colora_cache_budget_mb", 2048),
                    promote_min_hits=getattr(self.args, "colora_promote_min_hits", 2),
                    promote_window=getattr(self.args, "colora_promote_window", 128),
                    max_promote_per_step=getattr(self.args, "colora_max_promote_per_step", 8),
                    decay=getattr(self.args, "colora_decay", 0.9),
                    deferred_promotion_delta_steps=getattr(self.args, "colora_deferred_promotion_delta_steps", 4),
                    miss_policy=getattr(self.args, "colora_miss_policy", "cpu_first"),
                    queue_high_watermark=getattr(self.args, "colora_promote_window", 128),
                    promote_cooldown_steps=4,
                )
                self.moe_expert_cache_manager = MoEExpertCacheManager(cache_cfg)
                self.moe_expert_cache_manager.register_projection_pool("gate", self.lora_mem_pool.moe_gate_pool)
                self.moe_expert_cache_manager.register_projection_pool("up", self.lora_mem_pool.moe_up_pool)
                self.moe_expert_cache_manager.register_projection_pool("down", self.lora_mem_pool.moe_down_pool)
                self.logger.info(
                    "[COLoRA] Expert cache initialized: budget_mb=%s, promote_min_hits=%s, "
                    "window=%s, max_promote_per_step=%s, decay=%.4f, deferred_delta=%s, miss_policy=%s, queue_hwm=%s",
                    cache_cfg.cache_budget_mb,
                    cache_cfg.promote_min_hits,
                    cache_cfg.promote_window,
                    cache_cfg.max_promote_per_step,
                    cache_cfg.decay,
                    cache_cfg.deferred_promotion_delta_steps,
                    cache_cfg.miss_policy,
                    cache_cfg.queue_high_watermark,
                )

            # Set TP rank for sharded weight loading
            self.lora_mem_pool.tp_rank_ = self.rank_in_node

            self.logger.info(f"[LoRA Backend] Created LoRA memory pool for {num_layers} layers, max_rank={max_rank}")

        except ImportError as e:
            self.logger.error(f"[LoRA Backend] Failed to import LoRA memory pool: {e}")
            self.use_batched_lora_mode = False
            return

        # Load adapters into memory pool
        from lightllm.server.lora import LoRATargetType

        # Startup profiling for adapter preload bottlenecks.
        preload_prev_t = time.perf_counter()
        can_rank0_broadcast = (
            self.node_world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        node_src_rank = self.args.node_rank * self.node_world_size
        for adapter_id, adapter_dir in lora_adapter_dirs.items():
            try:
                preload_start_t = time.perf_counter()
                gap_ms = (preload_start_t - preload_prev_t) * 1000.0
                broadcast_ms = 0.0
                load_mode = "local_all_ranks"
                load_obj_ms = 0.0
                get_weights_ms = 0.0

                # Load adapter using the model's LoRA loading function.
                # In TP mode, try rank0-only disk load + broadcast to reduce duplicated slow I/O.
                if can_rank0_broadcast:
                    load_mode = "rank0_broadcast"
                    try:
                        payload = None
                        if self.rank_in_node == 0:
                            load_obj_t0 = time.perf_counter()
                            adapter = self._load_lora_adapter_fn(
                                adapter_dir=adapter_dir,
                                network_config=self.model.config,
                                data_type=self.model.data_type,
                                device="cpu",
                                swap=False
                            )
                            load_obj_t1 = time.perf_counter()
                            load_obj_ms = (load_obj_t1 - load_obj_t0) * 1000.0

                            rank = adapter.max_rank
                            scaling = adapter.lora_alpha / adapter.max_rank
                            get_weights_t0 = time.perf_counter()
                            layer_weights = adapter.get_all_weights()
                            get_weights_t1 = time.perf_counter()
                            get_weights_ms = (get_weights_t1 - get_weights_t0) * 1000.0
                            payload = {
                                "ok": True,
                                "rank": rank,
                                "scaling": scaling,
                                "layer_weights": layer_weights,
                            }

                        bcast_t0 = time.perf_counter()
                        object_list = [payload]
                        dist.broadcast_object_list(
                            object_list,
                            src=node_src_rank,
                            group=self.node_nccl_group,
                            device=torch.device("cuda", self.current_device_id),
                        )
                        bcast_t1 = time.perf_counter()
                        broadcast_ms = (bcast_t1 - bcast_t0) * 1000.0

                        received = object_list[0]
                        if not isinstance(received, dict) or not received.get("ok", False):
                            raise RuntimeError(
                                f"Invalid broadcast payload for adapter {adapter_id} at {adapter_dir}"
                            )

                        rank = received["rank"]
                        scaling = received["scaling"]
                        layer_weights = received["layer_weights"]
                    except Exception as broadcast_err:
                        # Guarded fallback to current behavior if broadcast fails.
                        self.logger.warning(
                            "[LoRA Backend][StartupTiming] rank0 broadcast failed for adapter %s (%s): %s; "
                            "falling back to per-rank local load",
                            adapter_id,
                            adapter_dir,
                            broadcast_err,
                        )
                        load_mode = "fallback_local"
                        load_obj_t0 = time.perf_counter()
                        adapter = self._load_lora_adapter_fn(
                            adapter_dir=adapter_dir,
                            network_config=self.model.config,
                            data_type=self.model.data_type,
                            device="cuda",
                            swap=False
                        )
                        load_obj_t1 = time.perf_counter()
                        load_obj_ms = (load_obj_t1 - load_obj_t0) * 1000.0

                        rank = adapter.max_rank
                        scaling = adapter.lora_alpha / adapter.max_rank
                        get_weights_t0 = time.perf_counter()
                        layer_weights = adapter.get_all_weights()
                        get_weights_t1 = time.perf_counter()
                        get_weights_ms = (get_weights_t1 - get_weights_t0) * 1000.0
                else:
                    load_obj_t0 = time.perf_counter()
                    adapter = self._load_lora_adapter_fn(
                        adapter_dir=adapter_dir,
                        network_config=self.model.config,
                        data_type=self.model.data_type,
                        device="cuda",
                        swap=False
                    )
                    load_obj_t1 = time.perf_counter()
                    load_obj_ms = (load_obj_t1 - load_obj_t0) * 1000.0

                    rank = adapter.max_rank
                    scaling = adapter.lora_alpha / adapter.max_rank
                    get_weights_t0 = time.perf_counter()
                    layer_weights = adapter.get_all_weights()
                    get_weights_t1 = time.perf_counter()
                    get_weights_ms = (get_weights_t1 - get_weights_t0) * 1000.0

                # Load into memory pool
                pool_load_t0 = time.perf_counter()
                loaded_ok = self.lora_mem_pool.load_adapter(
                    adapter_dir=adapter_dir,
                    rank=rank,
                    scaling=scaling,
                    layer_weights=layer_weights
                )
                if not loaded_ok:
                    raise RuntimeError(
                        f"Failed to load adapter_id={adapter_id} ({adapter_dir}) into memory pool"
                    )
                pool_load_t1 = time.perf_counter()

                total_ms = (pool_load_t1 - preload_start_t) * 1000.0
                pool_load_ms = (pool_load_t1 - pool_load_t0) * 1000.0

                self.logger.info(
                    "[LoRA Backend][StartupTiming] pid=%s rank=%s rank_in_node=%s adapter_id=%s "
                    "gap_before_ms=%.2f load_obj_ms=%.2f get_weights_ms=%.2f pool_load_ms=%.2f bcast_ms=%.2f "
                    "total_ms=%.2f mode=%s adapter_dir=%s",
                    os.getpid(),
                    self.global_rank,
                    self.rank_in_node,
                    adapter_id,
                    gap_ms,
                    load_obj_ms,
                    get_weights_ms,
                    pool_load_ms,
                    broadcast_ms,
                    total_ms,
                    load_mode,
                    adapter_dir,
                )
                self.logger.info(f"[LoRA Backend] Loaded adapter {adapter_id} from {adapter_dir}")
                preload_prev_t = pool_load_t1

            except Exception as e:
                self.logger.error(f"[LoRA Backend] Failed to load adapter {adapter_id} from {adapter_dir}: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Create dispatcher for each layer
        self.lora_dispatchers = []

        # Use max rank from registered adapters for buffer allocation
        # Actual weights come from memory pool per adapter, with each adapter
        # using its own rank (adapter.r) as tracked by a_len in the pool
        max_rank = 64  # default
        from lightllm.server.lora.manager import get_lora_manager
        lora_manager = get_lora_manager()
        adapters_list = lora_manager.list_adapters()
        for adapter_info in adapters_list:
            rank = adapter_info.get("lora_rank", 0)
            if rank and rank > max_rank:
                max_rank = rank

        num_layers = self.model.config.get("num_hidden_layers", self.model.layers_num)
        async_fallback_raw = getattr(self.args, "colora_async_fallback", 1)
        try:
            async_fallback_enabled = bool(int(async_fallback_raw))
        except (TypeError, ValueError):
            async_fallback_enabled = bool(async_fallback_raw)
        for layer_id in range(num_layers):
            dispatcher_kwargs = dict(
                num_layers=1,  # Single layer dispatcher
                lora_rank=max_rank,
                lora_alpha=1.0,  # scaling handled separately via a_scaling in pool
                lora_compute_config=effective_lora_compute_config,
                colora_async_fallback=async_fallback_enabled,
                colora_cpu_workers=int(getattr(self.args, "colora_cpu_workers", 4)),
                colora_cpu_queue_depth=int(getattr(self.args, "colora_cpu_queue_depth", 256)),
                colora_cpu_batch_timeout_us=int(getattr(self.args, "colora_cpu_batch_timeout_us", 50)),
                colora_deferred_promotion_delta_steps=int(
                    getattr(self.args, "colora_deferred_promotion_delta_steps", 4)
                ),
                colora_promotion_ema_alpha=float(getattr(self.args, "colora_promotion_ema_alpha", 0.5)),
                colora_temporal_prefetch=bool(getattr(self.args, "colora_temporal_prefetch", False)),
                colora_temporal_hot_cache_slots=int(getattr(self.args, "colora_temporal_hot_cache_slots", 64)),
                colora_request_skip=bool(getattr(self.args, "colora_request_skip", True)),
                colora_max_continuations=int(getattr(self.args, "colora_max_continuations", 8)),
                colora_hit_indexing=str(getattr(self.args, "colora_hit_indexing", "gpu")),
            )
            try:
                dispatcher = self._create_lora_dispatcher_fn(**dispatcher_kwargs)
            except TypeError:
                # Non-COLoRA dispatchers do not accept async fallback kwargs.
                dispatcher = self._create_lora_dispatcher_fn(
                    num_layers=1,
                    lora_rank=max_rank,
                    lora_alpha=1.0,
                    lora_compute_config=self._lora_compute_config,
                )
            self.lora_dispatchers.append(dispatcher)

        self.logger.info(f"[LoRA Backend] Created {len(self.lora_dispatchers)} LoRA dispatchers for batched mode")

        # Pass dispatchers to layer inference objects for batched S-LoRA mode
        # Each layer has its own dispatcher in self.lora_dispatchers
        if hasattr(self.model, 'set_lora_dispatcher'):
            for layer_id, dispatcher in enumerate(self.lora_dispatchers):
                if layer_id < len(self.model.layers_infer):
                    self.model.layers_infer[layer_id].set_lora_dispatcher(dispatcher, use_detached_lora=True)
            self.logger.info(f"[LoRA Backend] Set {len(self.lora_dispatchers)} LoRA dispatchers on layers")

    def _prepare_batched_lora_for_batch(self, batch) -> torch.Tensor:
        """
        Prepare LoRA for a batch of requests.

        Args:
            batch: Batch object containing requests

        Returns:
            req_bins tensor mapping request index -> adapter index

        Debug:
            Logs batch adapter distribution and req_bins configuration
        """
        if not self.use_batched_lora_mode:
            return None

        # Adapter IDs are 1-based in requests (0 means no adapter).
        # LoRA memory pool uses 0-based adapter indices.
        req_bins_list: List[int] = []
        active_adapter_ids = set()
        for req in batch.reqs:
            adapter_id = normalize_req_adapter_id(getattr(req, "adapter_id", 0))
            if adapter_id <= 0:
                adapter_bin = -1
            else:
                adapter_dir = self.lora_adapter_dirs.get(adapter_id) if hasattr(self, "lora_adapter_dirs") else None
                if adapter_dir is None:
                    raise RuntimeError(
                        f"Unknown adapter_id={adapter_id} in request; known ids={sorted(self.lora_adapter_dirs.keys())}"
                    )
                adapter_bin = int(self.lora_mem_pool.get_adapter_idx(adapter_dir))
                if adapter_bin < 0:
                    raise RuntimeError(
                        f"adapter_id={adapter_id} ({adapter_dir}) is not loaded in memory pool"
                    )
            req_bins_list.append(adapter_bin)
            if adapter_id > 0:
                active_adapter_ids.add(adapter_id)

        self.logger.debug(
            f"[LoRA Backend] Preparing batch: batch_size={len(batch.reqs)}, active_adapters={sorted(active_adapter_ids)}"
        )

        req_bins = torch.tensor(req_bins_list, dtype=torch.long, device="cuda")

        self.logger.debug(f"[LoRA Backend]   req_bins (per request)={req_bins.tolist()}")

        # EXPAND req_bins to per-token: each token gets its request's adapter index
        # This is required because input tensor has shape [total_tokens, hidden_dim]
        # and batch_lora_get_qkv expects bins to have one element per token
        token_counts = [req.get_cur_total_len() for req in batch.reqs]
        expanded_bins = []
        for req_idx, adapter_idx in enumerate(req_bins.tolist()):
            expanded_bins.extend([adapter_idx] * token_counts[req_idx])
        expanded_bins = torch.tensor(expanded_bins, dtype=torch.long, device="cuda")

        self.logger.debug(f"[LoRA Backend]   token_counts={token_counts}, total_tokens={sum(token_counts)}")
        # self.logger.debug(f"[LoRA Backend]   expanded_bins={expanded_bins.tolist()}")

        # Use expanded_bins for batched mode (per-token adapter indices)
        req_bins = expanded_bins
        if req_bins.numel() > 0:
            pos_bins = req_bins[req_bins >= 0]
            max_pos_bin = int(pos_bins.max().item()) if pos_bins.numel() > 0 else -1
            loaded_count = int(len(self.lora_mem_pool.adapter_dirs))
            if max_pos_bin >= loaded_count:
                raise RuntimeError(
                    f"req_bins out of range before dispatch: max_bin={max_pos_bin}, loaded_count={loaded_count}"
                )

        # Initialize batched mode for all dispatchers
        for dispatcher in self.lora_dispatchers:
            try:
                dispatcher.init_batched_mode(
                    self.lora_mem_pool,
                    req_bins,
                    expert_cache_manager=getattr(self, "moe_expert_cache_manager", None),
                )
            except TypeError:
                dispatcher.init_batched_mode(self.lora_mem_pool, req_bins)

        # Set req_bins on all layer inference objects
        for layer_infer in self.model.layers_infer:
            if hasattr(layer_infer, 'set_req_bins'):
                layer_infer.set_req_bins(req_bins)

        # self.logger.debug(f"[LoRA Backend]   Batched mode enabled for {len(self.lora_dispatchers)} dispatchers")
        return req_bins

    def _get_batch_adapter_status(self, reqs: list) -> Tuple[bool, bool]:
        """
        Analyze a batch of requests for adapter usage.

        Returns:
            Tuple of (has_any_adapter, has_mixed_adapters)
        """
        if not reqs:
            return False, False

        adapter_ids = set()
        has_adapter = False

        for req in reqs:
            if not hasattr(req, "adapter_id"):
                continue
            try:
                adapter_id = int(req.adapter_id)
            except (TypeError, ValueError):
                adapter_id = 0
            if adapter_id > 0:
                has_adapter = True
                adapter_ids.add(adapter_id)

        return has_adapter, len(adapter_ids) > 1

    def _run_batch_with_batched_lora(self, batch):
        """
        Run inference on a batch with S-LoRA batched LoRA mode.

        This is the main entry point for batched LoRA inference.
        """
        # Prepare batched LoRA
        req_bins = self._prepare_batched_lora_for_batch(batch)

        # Set LoRA enabled on all layers
        for layer_infer in self.model.layers_infer:
            layer_infer.use_detached_lora_ = True
            layer_infer.force_slow_lora_path = getattr(self, 'force_slow_lora_path', False)

        # Run the actual inference
        # The actual inference logic is in the subclass implementations
        # (prefill_batch / decode_batch methods)

        return req_bins

    def cleanup_batched_lora(self):
        """Clean up batched LoRA resources."""
        if hasattr(self, 'lora_mem_pool') and self.lora_mem_pool is not None:
            # Unload all adapters
            for adapter_dir in list(self.lora_mem_pool.adapter_dirs):
                self.lora_mem_pool.unload_adapter(adapter_dir)

        self.lora_mem_pool = None
        self.lora_dispatchers = []
        self.moe_expert_cache_manager = None

