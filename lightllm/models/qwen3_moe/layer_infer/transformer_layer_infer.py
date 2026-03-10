import os
import json
import logging
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple, Optional, Any, Dict, Callable
from contextlib import nullcontext
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from functools import partial
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from lightllm.utils.nvtx_utils import NvtxAnnotate

from lightllm.common.fused_moe.topk_select import select_experts

logger = init_logger(__name__)


class Qwen3MOETransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        self.n_routed_experts = network_config["num_experts"]
        self.is_moe = (
            network_config["num_experts"] > 0
            and layer_num not in network_config["mlp_only_layers"]
            and (layer_num + 1) % network_config["decoder_sparse_step"] == 0
        )
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        super().__init__(layer_num, network_config, mode)
        self.head_dim_ = network_config["head_dim"]
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)

        # LoRA dispatcher for MoE (set externally for detached mode)
        self.lora_dispatcher_: Optional[Any] = None
        self.use_detached_lora_: bool = False
        self.req_bins_: Optional[torch.Tensor] = None  # Per-request adapter indices for batched LoRA

        return

    def set_req_bins(self, req_bins: torch.Tensor):
        """Set the req_bins tensor for batched mode.

        Args:
            req_bins: Tensor of shape [batch_size] containing per-request adapter indices
        """
        self.req_bins_ = req_bins

    def _coalesce_lora_activations(
        self,
        activations: torch.Tensor,
        req_bins: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Pack expert activations into adapter-sorted contiguous blocks on GPU.

        The output metadata follows CSR-like semantics:
        - adapter_ids: sorted unique adapter IDs
        - adapter_offsets: prefix-sum offsets into packed activations
        - token_positions: positions in original activations for each packed row
        """
        if req_bins is None or activations.numel() == 0:
            return None

        bins = req_bins.to(device=activations.device, dtype=torch.long)
        valid_mask = bins >= 0
        if not torch.any(valid_mask):
            return None

        token_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        valid_bins = bins.index_select(0, token_positions)

        # Group rows by adapter to maximize contiguous D2H/H2D transfer efficiency.
        if valid_bins.numel() > 1:
            sorted_bins, sort_idx = torch.sort(valid_bins)
            token_positions = token_positions.index_select(0, sort_idx)
        else:
            sorted_bins = valid_bins

        packed_activations = activations.index_select(0, token_positions).contiguous()

        adapter_ids, adapter_counts = torch.unique_consecutive(sorted_bins, return_counts=True)
        adapter_offsets = torch.empty(adapter_counts.numel() + 1, dtype=torch.int32, device=activations.device)
        adapter_offsets[0] = 0
        adapter_offsets[1:] = torch.cumsum(adapter_counts.to(torch.int32), dim=0)

        return {
            "packed_activations": packed_activations,
            "packed_bins": sorted_bins.contiguous(),
            "token_positions": token_positions,
            "adapter_ids": adapter_ids,
            "adapter_offsets": adapter_offsets,
        }

    def _scatter_lora_from_packed(
        self,
        packed_output: torch.Tensor,
        token_positions: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        """Scatter packed LoRA output back to the original expert token order."""
        full_output = torch.zeros(
            (total_tokens, packed_output.shape[1]),
            dtype=packed_output.dtype,
            device=packed_output.device,
        )
        if packed_output.numel() > 0:
            full_output.index_copy_(0, token_positions, packed_output)
        return full_output

    def _dispatch_lora_with_optional_coalescing(
        self,
        dispatch_fn: Callable[..., torch.Tensor],
        input_tensor: torch.Tensor,
        layer_id: int,
        req_bins: Optional[torch.Tensor],
        expert_id: int,
        pack_meta: Optional[Dict[str, torch.Tensor]],
        reuse_packed_input: bool = False,
        phase_name: str = "",
        study2_prefix: Optional[str] = None,
    ) -> torch.Tensor:
        """Run LoRA with packed activations when metadata is available."""
        if pack_meta is None:
            return dispatch_fn(input_tensor, layer_id, req_bins, expert_id=expert_id)

        use_cpu_compute = self._should_use_moe_cpu_compute()
        if use_cpu_compute:
            return self._dispatch_lora_with_coalesced_cpu_roundtrip(
                dispatch_fn=dispatch_fn,
                input_tensor=input_tensor,
                layer_id=layer_id,
                expert_id=expert_id,
                pack_meta=pack_meta,
                reuse_packed_input=reuse_packed_input,
                phase_name=phase_name,
                study2_prefix=study2_prefix,
            )

        if reuse_packed_input:
            packed_input = pack_meta["packed_activations"]
        else:
            packed_input = input_tensor.index_select(0, pack_meta["token_positions"]).contiguous()
        packed_out = dispatch_fn(
            packed_input,
            layer_id,
            pack_meta["packed_bins"],
            expert_id=expert_id,
        )
        return self._scatter_lora_from_packed(
            packed_output=packed_out,
            token_positions=pack_meta["token_positions"],
            total_tokens=input_tensor.shape[0],
        )

    def _should_use_moe_cpu_compute(self) -> bool:
        """Best-effort detection for MoE CPU-compute mode across dispatcher variants."""
        dispatcher = self.lora_dispatcher_
        if dispatcher is None:
            return False

        should_cpu_fn = getattr(dispatcher, "_should_use_cpu_compute", None)
        if callable(should_cpu_fn):
            try:
                return bool(should_cpu_fn("moe"))
            except Exception:
                pass

        cfg = getattr(dispatcher, "lora_compute_config", None)
        if cfg is None:
            return False

        cfg_should_cpu = getattr(cfg, "should_compute_on_cpu", None)
        if callable(cfg_should_cpu):
            try:
                return bool(cfg_should_cpu("moe"))
            except Exception:
                pass

        moe_compute = getattr(cfg, "moe_compute", None)
        if isinstance(moe_compute, str):
            return moe_compute.lower() == "cpu"
        return False

    def _new_colora_stats(self) -> Dict[str, float]:
        return {
            "colora_hit_tokens": 0,
            "colora_miss_tokens": 0,
            "promotion_queue_depth": 0,
            "cache_hit_rate": 0.0,
            "cpu_compute_time": 0.0,
            "gpu_compute_time": 0.0,
            "cpu_queue_wait_time": 0.0,
            "d2h_bytes": 0.0,
            "h2d_bytes": 0.0,
            "overlap_ratio_sum": 0.0,
            "overlap_ratio_count": 0,
            "fallback_degrade_count": 0,
            "cpu_queue_depth": 0,
            "promotion_drop_total": 0,
            "promotion_drop_queue_high_watermark": 0,
            "promotion_drop_cooldown": 0,
            "moe_kernel_calls": 0,
            "moe_kernel_tokens": 0,
        }

    def _merge_colora_stats(self, agg_stats: Dict[str, float]) -> None:
        dispatcher = self.lora_dispatcher_
        if dispatcher is None:
            return

        pop_stats_fn = getattr(dispatcher, "pop_colora_stats", None)
        if not callable(pop_stats_fn):
            return

        stats = pop_stats_fn()
        if not isinstance(stats, dict):
            return

        agg_stats["colora_hit_tokens"] += int(stats.get("colora_hit_tokens", 0))
        agg_stats["colora_miss_tokens"] += int(stats.get("colora_miss_tokens", 0))
        agg_stats["cpu_compute_time"] += float(stats.get("cpu_compute_time", 0.0))
        agg_stats["gpu_compute_time"] += float(stats.get("gpu_compute_time", 0.0))
        agg_stats["cpu_queue_wait_time"] += float(stats.get("cpu_queue_wait_time", 0.0))
        agg_stats["d2h_bytes"] += float(stats.get("d2h_bytes", 0.0))
        agg_stats["h2d_bytes"] += float(stats.get("h2d_bytes", 0.0))
        agg_stats["fallback_degrade_count"] += int(stats.get("fallback_degrade_count", 0))
        agg_stats["cpu_queue_depth"] = int(stats.get("cpu_queue_depth", agg_stats["cpu_queue_depth"]))
        agg_stats["promotion_drop_total"] = int(stats.get("promotion_drop_total", agg_stats["promotion_drop_total"]))
        agg_stats["promotion_drop_queue_high_watermark"] = int(
            stats.get("promotion_drop_queue_high_watermark", agg_stats["promotion_drop_queue_high_watermark"])
        )
        agg_stats["promotion_drop_cooldown"] = int(
            stats.get("promotion_drop_cooldown", agg_stats["promotion_drop_cooldown"])
        )
        agg_stats["moe_kernel_calls"] += int(stats.get("moe_kernel_calls", 0))
        agg_stats["moe_kernel_tokens"] += int(stats.get("moe_kernel_tokens", 0))
        overlap_ratio = float(stats.get("overlap_ratio", 0.0))
        if overlap_ratio > 0.0:
            agg_stats["overlap_ratio_sum"] += overlap_ratio
            agg_stats["overlap_ratio_count"] += 1
        agg_stats["promotion_queue_depth"] = int(stats.get("promotion_queue_depth", agg_stats["promotion_queue_depth"]))
        agg_stats["cache_hit_rate"] = float(stats.get("cache_hit_rate", agg_stats["cache_hit_rate"]))

    def _get_study2_profile_prefix(self, expert_id: int, step_idx: int, token_count: int) -> Optional[str]:
        """Build Study2 NVTX prefix for real-model profiling when enabled via env."""
        if os.environ.get("MOE_STUDY2_PROFILE", "0") != "1":
            return None

        layer_filter = os.environ.get("MOE_STUDY2_LAYER", "").strip()
        if layer_filter:
            try:
                if int(layer_filter) != int(self.layer_num_):
                    return None
            except ValueError:
                logger.warning(f"[Study2] Invalid MOE_STUDY2_LAYER={layer_filter}, ignoring filter.")

        return (
            f"Study2/Layer={int(self.layer_num_)}"
            f"/Expert={int(expert_id)}"
            f"/Step={int(step_idx)}"
            f"/N={int(token_count)}"
        )

    def _dispatch_lora_with_coalesced_cpu_roundtrip(
        self,
        dispatch_fn: Callable[..., torch.Tensor],
        input_tensor: torch.Tensor,
        layer_id: int,
        expert_id: int,
        pack_meta: Dict[str, torch.Tensor],
        reuse_packed_input: bool,
        phase_name: str,
        study2_prefix: Optional[str],
    ) -> torch.Tensor:
        """Coalesced D2H->CPU compute->H2D path using packed activation order."""
        if reuse_packed_input:
            packed_input_gpu = pack_meta["packed_activations"]
        else:
            packed_input_gpu = input_tensor.index_select(0, pack_meta["token_positions"]).contiguous()

        if "packed_bins_cpu" not in pack_meta:
            pack_meta["packed_bins_cpu"] = pack_meta["packed_bins"].to(device="cpu", non_blocking=False)
        packed_bins_cpu = pack_meta["packed_bins_cpu"]

        if packed_input_gpu.device.type != "cuda":
            packed_out = dispatch_fn(
                packed_input_gpu,
                layer_id,
                packed_bins_cpu.to(device=packed_input_gpu.device),
                expert_id=expert_id,
            )
            return self._scatter_lora_from_packed(
                packed_output=packed_out,
                token_positions=pack_meta["token_positions"],
                total_tokens=input_tensor.shape[0],
            )

        phase_suffix = phase_name if phase_name else "LoRA"
        to_cpu_label = (
            f"{study2_prefix}/Transfer_ToCPU/{phase_suffix}"
            if study2_prefix is not None
            else "MoE_COLoRA_D2H_Activation"
        )
        cpu_compute_label = (
            f"{study2_prefix}/CPU_AVX_Compute/{phase_suffix}"
            if study2_prefix is not None
            else "MoE_COLoRA_CPU_AVX_Compute"
        )
        to_gpu_label = (
            f"{study2_prefix}/Transfer_ToGPU/{phase_suffix}"
            if study2_prefix is not None
            else "MoE_COLoRA_H2D_Activation"
        )

        with NvtxAnnotate(to_cpu_label):
            packed_input_cpu = torch.empty(
                packed_input_gpu.shape,
                dtype=packed_input_gpu.dtype,
                device="cpu",
                pin_memory=True,
            )
            packed_input_cpu.copy_(packed_input_gpu, non_blocking=True)
            # CPU kernel consumes host data directly; ensure D2H completion first.
            torch.cuda.current_stream().synchronize()

        with NvtxAnnotate(cpu_compute_label):
            packed_out_cpu = dispatch_fn(
                packed_input_cpu,
                layer_id,
                packed_bins_cpu,
                expert_id=expert_id,
            )

        if packed_out_cpu.device.type != "cpu":
            return self._scatter_lora_from_packed(
                packed_output=packed_out_cpu,
                token_positions=pack_meta["token_positions"],
                total_tokens=input_tensor.shape[0],
            )

        with NvtxAnnotate(to_gpu_label):
            # Keep H2D contiguous and pinned to maximize PCIe bandwidth.
            packed_out_cpu = packed_out_cpu.contiguous()
            packed_out_cpu_pinned = torch.empty(
                packed_out_cpu.shape,
                dtype=packed_out_cpu.dtype,
                device="cpu",
                pin_memory=True,
            )
            packed_out_cpu_pinned.copy_(packed_out_cpu, non_blocking=False)

            packed_out_gpu = torch.empty(
                packed_out_cpu_pinned.shape,
                dtype=packed_out_cpu_pinned.dtype,
                device=input_tensor.device,
            )
            packed_out_gpu.copy_(packed_out_cpu_pinned, non_blocking=True)

        return self._scatter_lora_from_packed(
            packed_output=packed_out_gpu,
            token_positions=pack_meta["token_positions"],
            total_tokens=input_tensor.shape[0],
        )

    def _bind_func(self):
        super()._bind_func()
        self._bind_ffn()
        return

    def _bind_ffn(self):
        if self.is_moe:
            moe_mode = os.environ.get("MOE_MODE", "TP")
            if moe_mode == "EP":
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn_edp, self)
                self._tpsp_ffn = self._tpsp_ffn_ep
            else:
                self._ffn = partial(Qwen3MOETransformerLayerInfer._moe_ffn, self)
                self._tpsp_ffn = self._tpsp_ffn_tp
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)
            self._tpsp_ffn = self._tpsp_ffn_tp

    def set_lora_dispatcher(self, dispatcher: Any, use_detached_lora: bool = True):
        """Set the LoRA dispatcher for MoE layers.

        Args:
            dispatcher: LoRA dispatcher instance (Qwen3MOELoRADispatcher)
            use_detached_lora: If True, LoRA runs in detached mode (parallel with base)
        """
        self.lora_dispatcher_ = dispatcher
        self.use_detached_lora_ = use_detached_lora

    def clear_lora_dispatcher(self):
        """Clear the LoRA dispatcher."""
        self.lora_dispatcher_ = None
        self.use_detached_lora_ = False

    def _log_adapter_expert_distribution(
        self,
        topk_ids: torch.Tensor,
        req_bins: Optional[torch.Tensor],
        num_experts: int,
        num_tokens: int,
        infer_state: LlamaInferStateInfo,
    ) -> None:
        """
        Log adapter x expert routing counts for one MoE layer call.

        Output format: one JSON object per line for easy downstream parsing.
        """
        if os.environ.get("MOE_ADAPTER_EXPERT_PROFILING", "0") != "1":
            return
        if req_bins is None or topk_ids is None:
            return

        if req_bins.numel() < num_tokens:
            logger.warning(
                f"[MoE Adapter Profile] Layer {self.layer_num_}: req_bins shorter than tokens "
                f"({req_bins.numel()} < {num_tokens}), skip profiling."
            )
            return

        token_bins = req_bins[:num_tokens].to(topk_ids.device)
        valid_token_mask = token_bins >= 0
        if not torch.any(valid_token_mask):
            return

        token_bins = token_bins[valid_token_mask].long()
        token_topk_ids = topk_ids[valid_token_mask].long()
        top_k = token_topk_ids.shape[1]

        adapter_ids = token_bins.unsqueeze(1).expand(-1, top_k).reshape(-1)
        expert_ids = token_topk_ids.reshape(-1)

        valid_pair_mask = (expert_ids >= 0) & (expert_ids < num_experts)
        if not torch.any(valid_pair_mask):
            return

        adapter_ids = adapter_ids[valid_pair_mask]
        expert_ids = expert_ids[valid_pair_mask]

        max_adapter = int(token_bins.max().item())

        def _matrix_to_dict(matrix: torch.Tensor) -> dict:
            result = {}
            non_zero = torch.nonzero(matrix, as_tuple=False)
            for pair in non_zero.tolist():
                adapter_idx, expert_idx = int(pair[0]), int(pair[1])
                count = int(matrix[adapter_idx, expert_idx].item())
                if count <= 0:
                    continue
                adapter_key = str(adapter_idx)
                if adapter_key not in result:
                    result[adapter_key] = {}
                result[adapter_key][str(expert_idx)] = count
            return result

        flat_ids = adapter_ids * num_experts + expert_ids
        pair_counts = torch.bincount(flat_ids, minlength=(max_adapter + 1) * num_experts)
        pair_counts = pair_counts.view(max_adapter + 1, num_experts).cpu()
        counts_dict = _matrix_to_dict(pair_counts)

        top1_expert_ids = token_topk_ids[:, 0]
        valid_top1_mask = (top1_expert_ids >= 0) & (top1_expert_ids < num_experts)
        top1_adapter_ids = token_bins[valid_top1_mask]
        top1_expert_ids = top1_expert_ids[valid_top1_mask]
        top1_flat_ids = top1_adapter_ids * num_experts + top1_expert_ids
        top1_pair_counts = torch.bincount(top1_flat_ids, minlength=(max_adapter + 1) * num_experts)
        top1_pair_counts = top1_pair_counts.view(max_adapter + 1, num_experts).cpu()
        top1_counts_dict = _matrix_to_dict(top1_pair_counts)


        record = {
            "event": "adapter_expert_routing",
            "layer": int(self.layer_num_),
            "mode": "prefill" if getattr(infer_state, "is_prefill", False) else "decode",
            "num_tokens": int(num_tokens),
            "top_k": int(top_k),
            "num_experts": int(num_experts),
            "counts_topk": counts_dict,
            "counts_top1": top1_counts_dict,
        }

        log_path = os.environ.get("MOE_ADAPTER_EXPERT_LOG_PATH", "/tmp/moe_adapter_expert_profile.log")
        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _get_local_expert_info(self, layer_weight):
        """Get information about local experts for EP mode.

        Returns:
            is_ep: Whether EP mode is enabled
            local_expert_ids: List of global expert IDs owned by this rank (empty if TP mode)
            local_to_global: Dict mapping local expert ID -> global expert ID
            global_to_local: Dict mapping global expert ID -> local expert ID (-1 if not local)
        """
        moe_mode = os.environ.get("MOE_MODE", "TP")
        if moe_mode != "EP":
            return False, [], {}, {}

        experts = layer_weight.experts
        ep_world_size = get_global_world_size()
        ep_rank = getattr(experts, 'global_rank_', 0)

        n_routed_experts = experts.n_routed_experts
        ep_n_routed_experts = n_routed_experts // ep_world_size

        # Compute global expert IDs owned by this rank
        start_expert = ep_rank * ep_n_routed_experts
        local_expert_ids = list(range(start_expert, start_expert + ep_n_routed_experts))

        # Build mapping dicts
        local_to_global = {i: g for i, g in enumerate(local_expert_ids)}
        global_to_local = {g: i for i, g in enumerate(local_expert_ids)}

        return True, local_expert_ids, local_to_global, global_to_local
        self.lora_dispatcher_ = None
        self.use_detached_lora_ = False

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _tpsp_get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input[0 : len(infer_state.position_cos), :]

        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )

        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)

        return q, cache_kv

    @NvtxAnnotate("MoE_FFN")
    def _moe_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:
        """Per-Expert LoRA Baseline implementation."""
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        # Check if we need per-expert LoRA
        # 确保 dispatcher 存在且开启了 detached lora 模式
        force_slow = getattr(self, 'force_slow_lora_path', False)
        use_per_expert_lora = (self.use_detached_lora_ and self.lora_dispatcher_ is not None) or force_slow
        colora_stats = self._new_colora_stats()

        # ----------------------------------------------------------------
        # Fast Path: 使用 Fused Kernel (无 LoRA)
        # ----------------------------------------------------------------
        if not use_per_expert_lora:
            router_logits = layer_weight.moe_gate.mm(hidden_states)
            # Profiling-only: explicitly compute top-k assignments to build
            # adapter x expert routing counts in fast path.
            if (
                os.environ.get("MOE_ADAPTER_EXPERT_PROFILING", "0") == "1"
                and self.req_bins_ is not None
            ):
                _, fast_topk_ids = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    correction_bias=getattr(layer_weight.experts, "e_score_correction_bias", None),
                    top_k=self.num_experts_per_tok,
                    renormalize=self.norm_topk_prob,
                    use_grouped_topk=False,
                    topk_group=None,
                    num_expert_group=None,
                    scoring_func=getattr(layer_weight.experts, "scoring_func", "softmax"),
                )
                self._log_adapter_expert_distribution(
                    topk_ids=fast_topk_ids,
                    req_bins=self.req_bins_,
                    num_experts=layer_weight.experts.n_routed_experts,
                    num_tokens=num_tokens,
                    infer_state=infer_state,
                )
            layer_weight.experts.experts(
                hidden_states,
                router_logits=router_logits,
                top_k=self.num_experts_per_tok,
                renormalize=self.norm_topk_prob,
                use_grouped_topk=False,
                topk_group=None,
                num_expert_group=None,
            )
            return hidden_states.view(num_tokens, hidden_dim)

        # ----------------------------------------------------------------
        # Slow Path: Per-Expert LoRA Baseline (Explicit Loop, Research purpose)
        # ----------------------------------------------------------------

        # 1. Router computation
        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # 2. Explicit routing
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=getattr(layer_weight.experts, "e_score_correction_bias", None),
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            scoring_func=getattr(layer_weight.experts, "scoring_func", "softmax"),
        )

        if hasattr(layer_weight.experts, "routed_scaling_factor"):
            # topk_weights = topk_weights * layer_weight.experts.routed_scaling_factor
            topk_weights.mul_(layer_weight.experts.routed_scaling_factor)

        self._log_adapter_expert_distribution(
            topk_ids=topk_ids,
            req_bins=self.req_bins_,
            num_experts=layer_weight.experts.n_routed_experts,
            num_tokens=num_tokens,
            infer_state=infer_state,
        )

        # 3. Check weights availability
        experts = layer_weight.experts
        # 必须确保使用了 keep_expert_lists=True
        assert hasattr(experts, "experts_gate_projs") and experts.experts_gate_projs[0] is not None, \
            "Per-Expert Baseline requires 'keep_expert_lists=True' in FusedMoeWeightTP."

        final_output = torch.zeros_like(hidden_states)
        total_experts = experts.n_routed_experts

        # EP mode: get local expert info
        is_ep, local_expert_ids, local_to_global, global_to_local = self._get_local_expert_info(layer_weight)

        # 4. Expert Loop with Compute-Transfer Pipelining
        # In TP mode: iterate over all experts
        # In EP mode: iterate only over local experts
        expert_iter_range = local_expert_ids if is_ep else range(total_experts)

        # Log token distribution per expert before entering the loop
        if logger.isEnabledFor(logging.DEBUG):
            token_counts = {}
            for global_eid in (local_to_global.values() if is_ep else range(total_experts)):
                count = int((topk_ids == global_eid).sum())
                if count > 0:
                    token_counts[global_eid] = count
            logger.debug(f"[MoE] Layer {self.layer_num_}: {num_tokens} tokens -> {token_counts}")

        with NvtxAnnotate("MoE_ActiveExpertExtraction_Optimized"):
            # 1. 扁平化 topk_ids [num_tokens * top_k]
            flat_topk_ids = topk_ids.flatten()

            # 2. 关键：全局仅触发 1 次 D2H 同步，获取所有专家的 token 分布
            # 这只占用不到 0.1ms 的时间
            expert_counts = torch.bincount(flat_topk_ids, minlength=total_experts).cpu().tolist()

            # Profile: 记录每层激活的experts (使用环境变量 MOE_PROFILING=1 开启)
            if os.environ.get("MOE_PROFILING", "0") == "1":
                layer_id = layer_weight.layer_num_
                active_experts = [i for i, count in enumerate(expert_counts) if count > 0]
                with open("/tmp/moe_profiling.log", "a") as f:
                    f.write(f"Layer {layer_id}: activated_experts={active_experts}, "
                            f"expert_counts={expert_counts}, total_tokens={sum(expert_counts)}\n")

            # 3. 纯 GPU 排序，瞬间将 Token 按 Expert 聚类
            sorted_token_indices = torch.argsort(flat_topk_ids)

            # 4. 在 CPU 端瞬间计算出全局内存块的偏移量
            global_offsets = [0] * (total_experts + 1)
            for i in range(total_experts):
                global_offsets[i+1] = global_offsets[i] + expert_counts[i]

            active_experts_data = []
            for local_expert_idx in expert_iter_range:
                global_expert_id = local_to_global.get(local_expert_idx, local_expert_idx) if is_ep else local_expert_idx
                count = expert_counts[global_expert_id]

                if count > 0:
                    start_idx = global_offsets[global_expert_id]
                    end_idx = start_idx + count

                    # 5. 纯 GPU View 截取，零同步！
                    token_idx = sorted_token_indices[start_idx:end_idx]
                    batch_indices = token_idx // self.num_experts_per_tok
                    k_indices = token_idx % self.num_experts_per_tok
                    
                    active_experts_data.append((local_expert_idx, batch_indices, k_indices))

        # Early exit if no active experts
        if not active_experts_data:
            return final_output.view(num_tokens, hidden_dim)

        # 4.2 Stream Setup for pipelining
        with NvtxAnnotate("MoE_StreamSetup"):
            transfer_stream = torch.cuda.Stream()
            compute_stream = torch.cuda.current_stream()

            def prefetch_weights(local_idx):
                """Prefetch weights for a specific expert using non-blocking transfer."""
                with NvtxAnnotate("LoRA_PCIe_HtoD_Weight"):
                    w1 = experts.experts_gate_projs[local_idx].cuda(non_blocking=True)
                    w3 = experts.experts_up_projs[local_idx].cuda(non_blocking=True)
                    w2 = experts.w2_list[local_idx].cuda(non_blocking=True)
                return w1, w3, w2

        # 4.3 Prime the Pipeline - prefetch first expert's weights
        with NvtxAnnotate("MoE_PipelinePrime"):
            first_expert_idx, _, _ = active_experts_data[0]
            with torch.cuda.stream(transfer_stream):
                next_weights = prefetch_weights(first_expert_idx)

        # 4.4 Pipelined Expert Loop
        for i, (local_expert_idx, batch_indices, k_indices) in enumerate(active_experts_data):
            with NvtxAnnotate(f"MoE_Expert_{local_expert_idx}"):
                expert_input = hidden_states[batch_indices]
                expert_req_bins = self.req_bins_[batch_indices] if self.req_bins_ is not None else None

                study2_prefix = self._get_study2_profile_prefix(
                    expert_id=local_expert_idx,
                    step_idx=i,
                    token_count=int(expert_input.shape[0]),
                )
                gpu_stream_label = f"{study2_prefix}/GPU_Stream" if study2_prefix is not None else None

                enable_coalescing = os.environ.get("MOE_COALESCING_PACKER", "1") == "1"
                pack_meta = None
                if enable_coalescing:
                    pack_label = f"{study2_prefix}/Gather" if study2_prefix is not None else "MoE_COLoRA_CoalescePack"
                    with NvtxAnnotate(pack_label):
                        pack_meta = self._coalesce_lora_activations(
                            activations=expert_input,
                            req_bins=expert_req_bins,
                        )

                # 4.4.1 Synchronize: ensure current expert's weights have arrived
                with NvtxAnnotate("MoE_WaitTransfer"):
                    compute_stream.wait_stream(transfer_stream)
                current_weights = next_weights

                # 4.4.2 Prefetch next expert's weights concurrently
                if i + 1 < len(active_experts_data):
                    with NvtxAnnotate("MoE_PrefetchNext"):
                        next_expert_idx, _, _ = active_experts_data[i + 1]
                        with torch.cuda.stream(transfer_stream):
                            next_weights = prefetch_weights(next_expert_idx)

                w1, w3, w2 = current_weights

                # 4.4.3 Compute Base GEMM
                with (NvtxAnnotate(gpu_stream_label) if gpu_stream_label is not None else nullcontext()):
                    with NvtxAnnotate("MoE_GateGEMM"):
                        gate_out = torch.mm(expert_input, w1.T)
                    with NvtxAnnotate("MoE_UpGEMM"):
                        up_out = torch.mm(expert_input, w3.T)

                # 4.4.4 Apply Per-Expert LoRA (Gate/Up)
                with NvtxAnnotate("MoE_GateLoRA"):
                    gate_lora = self._dispatch_lora_with_optional_coalescing(
                        dispatch_fn=self.lora_dispatcher_.batch_apply_gate_lora,
                        input_tensor=expert_input,
                        layer_id=layer_weight.layer_num_,
                        req_bins=expert_req_bins,
                        expert_id=local_expert_idx,
                        pack_meta=pack_meta,
                        reuse_packed_input=True,
                        phase_name="Gate",
                        study2_prefix=study2_prefix,
                    )
                    self._merge_colora_stats(colora_stats)
                with NvtxAnnotate("MoE_UpLoRA"):
                    up_lora = self._dispatch_lora_with_optional_coalescing(
                        dispatch_fn=self.lora_dispatcher_.batch_apply_up_lora,
                        input_tensor=expert_input,
                        layer_id=layer_weight.layer_num_,
                        req_bins=expert_req_bins,
                        expert_id=local_expert_idx,
                        pack_meta=pack_meta,
                        reuse_packed_input=True,
                        phase_name="Up",
                        study2_prefix=study2_prefix,
                    )
                    self._merge_colora_stats(colora_stats)

                gate_out += gate_lora
                up_out += up_lora

                # 4.4.5 Activation
                with (NvtxAnnotate(gpu_stream_label) if gpu_stream_label is not None else nullcontext()):
                    with NvtxAnnotate("MoE_Activation"):
                        current_hidden = torch.nn.functional.silu(gate_out) * up_out

                    # 4.4.6 Down Projection Base
                    with NvtxAnnotate("MoE_DownGEMM"):
                        down_out = torch.mm(current_hidden, w2.T)

                # 4.4.7 Apply Per-Expert LoRA (Down)
                with NvtxAnnotate("MoE_DownLoRA"):
                    down_lora = self._dispatch_lora_with_optional_coalescing(
                        dispatch_fn=self.lora_dispatcher_.batch_apply_down_lora,
                        input_tensor=current_hidden,
                        layer_id=layer_weight.layer_num_,
                        req_bins=expert_req_bins,
                        expert_id=local_expert_idx,
                        pack_meta=pack_meta,
                        phase_name="Down",
                        study2_prefix=study2_prefix,
                    )
                    self._merge_colora_stats(colora_stats)
                down_out += down_lora

                # 4.4.8 Weighted Aggregation (Corrected)
                with NvtxAnnotate("MoE_Aggregation"):
                    # routing_weights: [num_selected, 1]
                    routing_weights = topk_weights[batch_indices, k_indices].view(-1, 1)
                    weighted_output = (down_out * routing_weights).to(hidden_states.dtype)

                    # 使用 index_add_ 在 final_output 上原地累加
                    final_output.index_add_(0, batch_indices, weighted_output)

                # 4.4.9 Cleanup - eagerly free GPU tensor references
                with NvtxAnnotate("MoE_Cleanup"):
                    del current_weights, w1, w3, w2

        if (colora_stats["colora_hit_tokens"] + colora_stats["colora_miss_tokens"]) > 0:
            overlap_ratio_avg = 0.0
            if colora_stats["overlap_ratio_count"] > 0:
                overlap_ratio_avg = colora_stats["overlap_ratio_sum"] / float(colora_stats["overlap_ratio_count"])
            logger.debug(
                "[COLoRA] layer=%s hit_tokens=%s miss_tokens=%s queue_depth=%s hit_rate=%.4f "
                "cpu_compute_time=%.6f gpu_compute_time=%.6f cpu_queue_wait=%.6f "
                "d2h_bytes=%.0f h2d_bytes=%.0f overlap_ratio=%.4f fallback_degrade_count=%s cpu_queue_depth=%s "
                "promotion_drop_total=%s promotion_drop_queue=%s promotion_drop_cooldown=%s "
                "moe_kernel_calls=%s moe_kernel_tokens=%s",
                self.layer_num_,
                colora_stats["colora_hit_tokens"],
                colora_stats["colora_miss_tokens"],
                colora_stats["promotion_queue_depth"],
                colora_stats["cache_hit_rate"],
                colora_stats["cpu_compute_time"],
                colora_stats["gpu_compute_time"],
                colora_stats["cpu_queue_wait_time"],
                colora_stats["d2h_bytes"],
                colora_stats["h2d_bytes"],
                overlap_ratio_avg,
                colora_stats["fallback_degrade_count"],
                colora_stats["cpu_queue_depth"],
                colora_stats["promotion_drop_total"],
                colora_stats["promotion_drop_queue_high_watermark"],
                colora_stats["promotion_drop_cooldown"],
                colora_stats["moe_kernel_calls"],
                colora_stats["moe_kernel_tokens"],
            )

        return final_output.view(num_tokens, hidden_dim)

    def _moe_ffn_edp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input
        token_num, hidden_dim = hidden_states.shape

        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # Apply moe_gate LoRA using batched S-LoRA API
        if self.use_detached_lora_ and self.lora_dispatcher_ is not None:
            gate_lora = self.lora_dispatcher_.batch_apply_gate_lora(
                hidden_states, layer_weight.layer_num_, self.req_bins_
            )
            router_logits = router_logits + gate_lora

        if (
            os.environ.get("MOE_ADAPTER_EXPERT_PROFILING", "0") == "1"
            and self.req_bins_ is not None
        ):
            _, edp_topk_ids = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                correction_bias=getattr(layer_weight.experts, "e_score_correction_bias", None),
                top_k=self.num_experts_per_tok,
                renormalize=self.norm_topk_prob,
                use_grouped_topk=False,
                topk_group=None,
                num_expert_group=None,
                scoring_func=getattr(layer_weight.experts, "scoring_func", "softmax"),
            )
            self._log_adapter_expert_distribution(
                topk_ids=edp_topk_ids,
                req_bins=self.req_bins_,
                num_experts=layer_weight.experts.n_routed_experts,
                num_tokens=token_num,
                infer_state=infer_state,
            )

        ep_output = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            is_prefill=infer_state.is_prefill,
        )

        # Apply w2 (down_proj) LoRA using batched S-LoRA API
        if self.use_detached_lora_ and self.lora_dispatcher_ is not None:
            down_lora = self.lora_dispatcher_.batch_apply_down_lora(
                ep_output, layer_weight.layer_num_, self.req_bins_
            )
            ep_output = ep_output + down_lora

        ep_output = ep_output.view(token_num, hidden_dim)
        return ep_output

    def _tpsp_ffn(
        self, input: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ):
        raise Exception("need bind to real impl")

    def _tpsp_ffn_tp(
        self, input: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        if self.tp_world_size_ > 1:
            sp_token_num, hidden_dim = input.shape
            gather_input = self.alloc_tensor(
                (sp_token_num * self.tp_world_size_, hidden_dim), dtype=input.dtype, device=input.device
            )
            all_gather_into_tensor(gather_input, input, group=infer_state.dist_group, async_op=False)
            input = gather_input

        ffn2_out = self._ffn(input=input, infer_state=infer_state, layer_weight=layer_weight)

        if self.tp_world_size_ > 1:
            sp_token_num = ffn2_out.shape[0] // self.tp_world_size_
            reduce_o_tensor = self.alloc_tensor(
                (sp_token_num, self.embed_dim_), dtype=ffn2_out.dtype, device=ffn2_out.device
            )
            reduce_scatter_tensor(
                reduce_o_tensor, ffn2_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False
            )
            ffn2_out = reduce_o_tensor
        return ffn2_out

    def _tpsp_ffn_ep(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Qwen3MOETransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)

        ffn2_out = self._ffn(input=input, infer_state=infer_state, layer_weight=layer_weight)

        return ffn2_out

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_token_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._token_attention_kernel(_0_q, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)
        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 dispatch
        (
            _0_recv_x,
            _0_masked_m,
            _0_topk_idx,
            _0_topk_weight,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.low_latency_dispatch(_0_input1, _0_router_logits)
        infer_state.hook = _0_hook

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._token_attention_kernel(_1_q, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)
        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        # 1 dispatch
        (
            _1_recv_x,
            _1_masked_m,
            _1_topk_idx,
            _1_topk_weight,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.low_latency_dispatch(_1_input1, _1_router_logits)
        infer_state1.hook = _1_hook

        # moe calu
        expected_m = triton.cdiv(
            input_embdings.shape[0] * get_global_world_size() * self.num_experts_per_tok, self.n_routed_experts
        )
        _0_moe_out = layer_weight.experts.masked_group_gemm(_0_recv_x, _0_masked_m, input_embdings.dtype, expected_m)

        # 1 hook
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        # 0 combine
        _0_ffn_out, _0_hook = layer_weight.experts.low_latency_combine(
            _0_moe_out, _0_topk_idx, _0_topk_weight, _0_handle
        )

        infer_state.hook = _0_hook

        # to do moe caclue
        _1_moe_out = layer_weight.experts.masked_group_gemm(_1_recv_x, _1_masked_m, input_embdings1.dtype, expected_m)

        # 0 hook
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))
            infer_state.hook = None

        # 1 combine
        _1_ffn_out, _1_hook = layer_weight.experts.low_latency_combine(
            _1_moe_out, _1_topk_idx, _1_topk_weight, _1_handle
        )

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1

    def overlap_tpsp_context_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ):
        if not self.is_moe:
            return super().overlap_tpsp_context_forward(
                input_embdings, input_embdings1, infer_state, infer_state1, layer_weight
            )
        # 0 attention
        _0_input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        _0_q, _0_cache_kv = self._tpsp_get_qkv(_0_input1, infer_state, layer_weight)
        _0_input1 = None
        self._post_cache_kv(_0_cache_kv, infer_state, layer_weight)
        _0_o = self._context_attention_kernel(_0_q, _0_cache_kv, infer_state, layer_weight)
        _0_q = None
        _0_o = self._tpsp_get_o(_0_o, infer_state, layer_weight)
        input_embdings.add_(_0_o.view(-1, self.embed_dim_))
        _0_o = None
        _0_input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        _0_router_logits = layer_weight.moe_gate.mm(_0_input1)

        # wait last 1 combine
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_topk_weight, _0_topk_idx, _0_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _0_input1, _0_router_logits
        )
        from deep_ep import Buffer

        _0_overlap_event = Buffer.capture()

        # 1 attention
        _1_input1 = self._att_norm(input_embdings1, infer_state1, layer_weight)
        _1_q, _1_cache_kv = self._tpsp_get_qkv(_1_input1, infer_state1, layer_weight)
        _1_input1 = None
        self._post_cache_kv(_1_cache_kv, infer_state1, layer_weight)
        _1_o = self._context_attention_kernel(_1_q, _1_cache_kv, infer_state1, layer_weight)
        _1_q = None
        _1_o = self._tpsp_get_o(_1_o, infer_state1, layer_weight)
        input_embdings1.add_(_1_o.view(-1, self.embed_dim_))
        _1_o = None
        _1_input1 = self._ffn_norm(input_embdings1, infer_state1, layer_weight)
        # to do gate and disptatch

        _1_router_logits = layer_weight.moe_gate.mm(_1_input1)

        # 0 dispatch execute
        (
            _0_recv_x,
            _0_recv_topk_idx,
            _0_recv_topk_weight,
            _0_num_recv_tokens_per_expert_list,
            _0_handle,
            _0_hook,
        ) = layer_weight.experts.dispatch(_0_qinput_tensor, _0_topk_idx, _0_topk_weight, overlap_event=_0_overlap_event)
        infer_state.hook = _0_hook

        # wait 0 dispatch
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_topk_weight, _1_topk_idx, _1_qinput_tensor = layer_weight.experts.select_experts_and_quant_input(
            _1_input1, _1_router_logits
        )

        _1_overlap_event = Buffer.capture()

        # 0 moe calu
        _0_moe_out = layer_weight.experts.prefilled_group_gemm(
            _0_num_recv_tokens_per_expert_list, _0_recv_x, _0_recv_topk_idx, _0_recv_topk_weight
        )

        # 1 dispatch execute
        (
            _1_recv_x,
            _1_recv_topk_idx,
            _1_recv_topk_weight,
            _1_num_recv_tokens_per_expert_list,
            _1_handle,
            _1_hook,
        ) = layer_weight.experts.dispatch(_1_qinput_tensor, _1_topk_idx, _1_topk_weight, overlap_event=_1_overlap_event)
        infer_state1.hook = _1_hook

        # wait 1 dispatch
        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        _0_combine_event = Buffer.capture()
        # 0 combine execute
        _0_ffn_out, _0_hook = layer_weight.experts.combine(_0_moe_out, _0_handle, _0_combine_event)
        infer_state.hook = _0_hook

        # 1 moe calc
        _1_moe_out = layer_weight.experts.prefilled_group_gemm(
            _1_num_recv_tokens_per_expert_list, _1_recv_x, _1_recv_topk_idx, _1_recv_topk_weight
        )

        # wait 0 combine
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        _1_combine_event = Buffer.capture()

        input_embdings.add_(_0_ffn_out.view(-1, self.embed_dim_))

        # 1 combine execute
        _1_ffn_out, _1_hook = layer_weight.experts.combine(_1_moe_out, _1_handle, _1_combine_event)

        def _1_hook_post():
            _1_hook()
            nonlocal _1_ffn_out
            input_embdings1.add_(_1_ffn_out.view(-1, self.embed_dim_))
            return

        infer_state1.hook = _1_hook_post

        return input_embdings, input_embdings1
