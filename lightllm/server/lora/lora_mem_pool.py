"""
S-LoRA Style Memory Pool for LoRA Adapter Weights

This module provides a centralized memory pool for storing LoRA adapter weights
in a format optimized for batched heterogeneous inference (S-LoRA style).

Supported LoRA Targets:
- Vision-Language Adapter: vl.q_proj, vl.k_proj, vl.v_proj, vl.o_proj, vl.linear_fc1, vl.linear_fc2
- Attention: self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
- MoE MLP: moe.gate_proj, moe.up_proj, moe.down_proj
- Language Model Head: moe.lm_head

Key Design:
- Each module type has its own memory pool (due to different hidden dimensions)
- req_bins tracks which adapter is used by each request
- dispatch_bgmv kernel applies LoRA per-request based on req_bins

Debugging:
- Set LIGHTLLM_LOGGING=DEBUG to enable verbose LoRA logging
"""
import attr
from networkx import attribute_assortativity_coefficient
import torch
import os
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from safetensors import safe_open
import glob

# Configure logging using global env var
_LOG_LEVEL = os.environ.get("LIGHTLLM_LOGGING", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL, logging.INFO)
logger = logging.getLogger("lightllm.lora")
logger.setLevel(_LOG_LEVEL)


# Module type enumeration for LoRA targets
class LoRATargetType:
    # LLM Attention
    ATTN_Q_PROJ = "attn_q"
    ATTN_K_PROJ = "attn_k"
    ATTN_V_PROJ = "attn_v"
    ATTN_O_PROJ = "attn_o"
    # LLM MoE
    MOE_EXPERT_GATE = "moe_expert_gate"
    MOE_EXPERT_UP = "moe_expert_up"
    MOE_EXPERT_DOWN = "moe_expert_down"
    # LLM Head
    LM_HEAD = "lm_head"
    # Vision
    VL_Q_PROJ = "vl_q"
    VL_K_PROJ = "vl_k"
    VL_V_PROJ = "vl_v"
    VL_O_PROJ = "vl_o"
    VL_FC1 = "vl_fc1"
    VL_FC2 = "vl_fc2"


@dataclass
class LoRAModulePool:
    """
    Memory pool for a specific module type.

    Each module type has its own pool due to different hidden dimensions.
    For attention (Q/K/V/O): hidden = num_heads * head_dim
    For MLP (gate/up/down): hidden = intermediate_dim
    For lm_head: hidden = vocab_size

    For GQA models (K/V projections), A and B can have different dimensions:
    - A (key_buffer): input dim = hidden_size
    - B (value_buffer): output dim = num_kv_heads * head_dim
    """
    # Shape: [pool_size, max_rank, a_hidden_dim] for A
    key_buffer: torch.Tensor  # LoRA A weights
    # Shape: [pool_size, max_rank, b_hidden_dim] for B
    value_buffer: torch.Tensor  # LoRA B weights

    # Metadata
    a_start: torch.Tensor  # [num_adapters] - start offset per adapter
    a_len: torch.Tensor  # [num_adapters] - length per adapter (rank)
    a_scaling: torch.Tensor  # [num_adapters] - scaling factor per adapter
    max_rank: int
    a_hidden_dim: int  # Input dimension for A matrix
    b_hidden_dim: int  # Output dimension for B matrix
    pool_size: int

    @property
    def a_buffer(self) -> torch.Tensor:
        """Get the A weight buffer."""
        return self.key_buffer

    @property
    def b_buffer(self) -> torch.Tensor:
        """Get the B weight buffer."""
        return self.value_buffer

    @classmethod
    def create(
        cls,
        pool_size: int,
        max_rank: int,
        input_dim: int,
        output_dim: int | None = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ) -> "LoRAModulePool":
        """Create a module pool.

        Args:
            pool_size: Maximum number of adapters in pool
            max_rank: Maximum LoRA rank
            input_dim: Input dimension for A matrix (and x tensor)
            output_dim: Output dimension for B matrix (and y tensor). If None, uses input_dim.
            dtype: Data type for weights
            device: Device for tensors
        """
        if output_dim is None:
            output_dim = input_dim
        return cls(
            key_buffer=torch.empty((pool_size, max_rank, input_dim), dtype=dtype, device=device),
            value_buffer=torch.empty((pool_size, max_rank, output_dim), dtype=dtype, device=device),
            a_start=torch.zeros(0, dtype=torch.long, device=device),
            a_len=torch.zeros(0, dtype=torch.long, device=device),
            a_scaling=torch.zeros(0, dtype=dtype, device=device),
            max_rank=max_rank,
            a_hidden_dim=input_dim,
            b_hidden_dim=output_dim,
            pool_size=pool_size
        )

    def can_fit(self, rank: int) -> bool:
        """Check if an adapter with given rank can fit."""
        used_slots = self.a_len.sum().item() if len(self.a_len) > 0 else 0
        return (used_slots + rank) <= self.pool_size

    def _compute_location(self) -> int:
        """Compute next available location."""
        if len(self.a_len) == 0:
            return 0
        return (self.a_start[-1] + self.a_len[-1]).item()

    def load_adapter(
        self,
        adapter_idx: int,
        rank: int,
        scaling: float,
        layer_weights: Dict[int, torch.Tensor]
    ) -> bool:
        """
        Load adapter weights for all layers.

        Args:
            adapter_idx: Index of this adapter in the global pool
            rank: LoRA rank
            scaling: Scaling factor
            layer_weights: Dict mapping layer_id -> {proj_A, proj_B} for this module type
        """
        if not self.can_fit(rank):
            return False

        loc_start = self._compute_location()

        # Extend metadata
        self.a_start = torch.cat([
            self.a_start,
            torch.tensor([loc_start], dtype=torch.long, device=self.a_start.device)
        ])
        self.a_len = torch.cat([
            self.a_len,
            torch.tensor([rank], dtype=torch.long, device=self.a_len.device)
        ])
        self.a_scaling = torch.cat([
            self.a_scaling,
            torch.tensor([scaling], dtype=self.a_scaling.dtype, device=self.a_scaling.device)
        ])

        # Store weights for each layer
        for layer_id, weights in layer_weights.items():
            if weights is None or not weights:
                continue

            # weights is a dict with "proj_A" and "proj_B" keys
            a_weight = weights.get("A")
            b_weight = weights.get("B")

            if a_weight is not None:
                # A matrix: [hidden, rank] -> [rank, hidden]
                self.a_buffer[loc_start + layer_id, :rank] = a_weight.T.to(self.a_buffer.dtype)
            if b_weight is not None:
                # B matrix: [rank, hidden] -> [hidden, rank] (stored as [rank, hidden])
                self.b_buffer[loc_start + layer_id, :rank] = b_weight.T.to(self.b_buffer.dtype)

        return True

    def unload_adapter(self, adapter_idx: int) -> bool:
        """Unload adapter and free memory."""
        if adapter_idx >= len(self.a_start):
            return False

        loc_start = self.a_start[adapter_idx].item()
        num_slots = self.a_len[adapter_idx].item()

        # Zero out the slots
        self.key_buffer[loc_start:loc_start + num_slots].zero_()
        self.value_buffer[loc_start:loc_start + num_slots].zero_()

        # Remove from metadata (in place for efficiency)
        if len(self.a_start) == 1:
            self.a_start = torch.zeros(0, dtype=torch.long, device=self.a_start.device)
            self.a_len = torch.zeros(0, dtype=torch.long, device=self.a_len.device)
            self.a_scaling = torch.zeros(0, dtype=self.a_scaling.dtype, device=self.a_scaling.device)
        else:
            mask = torch.ones(len(self.a_start), dtype=torch.bool, device=self.a_start.device)
            mask[adapter_idx] = False
            self.a_start = self.a_start[mask]
            self.a_len = self.a_len[mask]
            self.a_scaling = self.a_scaling[mask]

        return True


@dataclass
class LoRAMemPool:
    """
    Centralized memory pool for all LoRA adapter weights.

    This pool manages multiple module-type-specific pools:
    - Attention Q/K/V/O: share same hidden dimension (num_heads * head_dim)
    - Vision Q/K/V/O: share same hidden dimension
    - MLP (gate/up/down): share intermediate dimension
    - lm_head: vocabulary dimension

    The pool enables efficient batched LoRA computation where different requests
    in the same batch can use different adapters.
    """
    # Module-type specific pools
    vl_q_pool: Optional[LoRAModulePool] = None
    vl_k_pool: Optional[LoRAModulePool] = None
    vl_v_pool: Optional[LoRAModulePool] = None
    vl_o_pool: Optional[LoRAModulePool] = None
    vl_fc1_pool: Optional[LoRAModulePool] = None
    vl_fc2_pool: Optional[LoRAModulePool] = None

    attn_q_pool: Optional[LoRAModulePool] = None
    attn_k_pool: Optional[LoRAModulePool] = None
    attn_v_pool: Optional[LoRAModulePool] = None
    attn_o_pool: Optional[LoRAModulePool] = None

    moe_gate_pool: Optional[LoRAModulePool] = None
    moe_up_pool: Optional[LoRAModulePool] = None
    moe_down_pool: Optional[LoRAModulePool] = None
    lm_head_pool: Optional[LoRAModulePool] = None

    # Adapter tracking
    adapter_dirs: List[str] = field(default_factory=list)
    idx_map: Dict[str, int] = field(default_factory=dict)
    max_rank: int = 64
    pool_size: int = 1024

    # Model config for dimension inference
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_dim: int = 0
    hidden_size: int = 0
    vocab_size: int = 0

    @classmethod
    def create(
        cls,
        num_layers: int,
        pool_size: int,
        max_rank: int,
        num_heads: int,
        head_dim: int,
        intermediate_dim: int,
        hidden_size: int,
        vocab_size: int,
        num_kv_heads: int | None = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ) -> "LoRAMemPool":
        """Create a complete LoRA memory pool.

        Args:
            num_kv_heads: Number of key/value heads (for GQA models). If None, defaults to num_heads.
        """
        # Attention dimensions
        attn_hidden = hidden_size
        if num_kv_heads is None:
            num_kv_heads = num_heads
        # GQA: B matrix output dimension is smaller
        kv_hidden = num_kv_heads * head_dim
        mlp_inter = intermediate_dim

        # Vision dimensions (use LLM dims as default)
        vl_hidden = hidden_size
        vl_mlp_hidden = intermediate_dim

        logger.info(f"[LoRA Pool] Creating pool: attn_hidden={attn_hidden}, kv_hidden={kv_hidden}, mlp_hidden={mlp_inter}")

        pool = cls(
            # Vision-Language pools (Q/K/V/O preserve vl_hidden)
            vl_q_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device),
            vl_k_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device),
            vl_v_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device),
            vl_o_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device),
            # Vision MLP: FC1 expands, FC2 shrinks
            vl_fc1_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_mlp_hidden, dtype, device),
            vl_fc2_pool=LoRAModulePool.create(pool_size, max_rank, vl_mlp_hidden, vl_hidden, dtype, device),

            # Attention pools - Q/O use full hidden, K/V use GQA output dim
            attn_q_pool=LoRAModulePool.create(pool_size, max_rank, attn_hidden, attn_hidden, dtype, device),
            attn_k_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, kv_hidden, dtype, device),
            attn_v_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, kv_hidden, dtype, device),
            attn_o_pool=LoRAModulePool.create(pool_size, max_rank, attn_hidden, attn_hidden, dtype, device),

            # MoE MLP pools - Gate/Up expand, Down shrinks
            moe_gate_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, mlp_inter, dtype, device),
            moe_up_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, mlp_inter, dtype, device),
            moe_down_pool=LoRAModulePool.create(pool_size, max_rank, mlp_inter, hidden_size, dtype, device),
            lm_head_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, vocab_size, dtype, device),

            adapter_dirs=[],
            idx_map={},
            max_rank=max_rank,
            pool_size=pool_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_dim=intermediate_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size
        )
        return pool

    def get_pool(self, target_type: str) -> Optional[LoRAModulePool]:
        """Get the module pool for a target type."""
        pool_map = {
            LoRATargetType.VL_Q_PROJ: self.vl_q_pool,
            LoRATargetType.VL_K_PROJ: self.vl_k_pool,
            LoRATargetType.VL_V_PROJ: self.vl_v_pool,
            LoRATargetType.VL_O_PROJ: self.vl_o_pool,
            LoRATargetType.VL_FC1: self.vl_fc1_pool,
            LoRATargetType.VL_FC2: self.vl_fc2_pool,
            LoRATargetType.ATTN_Q_PROJ: self.attn_q_pool,
            LoRATargetType.ATTN_K_PROJ: self.attn_k_pool,
            LoRATargetType.ATTN_V_PROJ: self.attn_v_pool,
            LoRATargetType.ATTN_O_PROJ: self.attn_o_pool,
            LoRATargetType.MOE_EXPERT_GATE: self.moe_gate_pool,
            LoRATargetType.MOE_EXPERT_UP: self.moe_up_pool,
            LoRATargetType.MOE_EXPERT_DOWN: self.moe_down_pool,
            LoRATargetType.LM_HEAD: self.lm_head_pool,
        }
        return pool_map.get(target_type)

    def load_adapter(
        self,
        adapter_dir: str,
        rank: int,
        scaling: float,
        layer_weights: Dict[int, Dict[str, Dict[str, torch.Tensor]]]
    ) -> bool:
        """
        Load adapter weights into all relevant module pools.

        Args:
            adapter_dir: Path to adapter directory
            rank: LoRA rank
            scaling: Scaling factor (alpha / rank)
            layer_weights: Dict mapping:
                layer_id -> {target_type -> {module_name: {A: tensor, B: tensor}}}

        Debug:
            Logs adapter loading progress and statistics
        """
        if adapter_dir in self.idx_map:
            logger.debug(f"[LoRA] Adapter already loaded: {adapter_dir}")
            return True  # Already loaded

        logger.info(f"[LoRA] Loading adapter: {adapter_dir}")
        logger.debug(f"[LoRA]   rank={rank}, scaling={scaling}, layers={len(layer_weights)}")

        adapter_idx = len(self.adapter_dirs)

        # Load weights for each target type
        for layer_id, target_weights in layer_weights.items():
            # Handle vision layer offset (10000+) - strip offset for buffer indexing
            # Vision and text layers use different pools, so no conflict
            if layer_id >= 10000:
                buffer_layer_id = layer_id - 10000
            else:
                buffer_layer_id = layer_id

            if buffer_layer_id >= self.num_layers:
                continue

            for target_type, module_weights in target_weights.items():
                pool = self.get_pool(target_type)
                if pool is None:
                    continue

                # Strip module_name level - pool expects {layer_id: {"A": tensor, "B": tensor}}
                # module_weights is {module_name: {"A": tensor, "B": tensor}}, take first value
                weight_dict = next(iter(module_weights.values())) if module_weights else {}

                pool.load_adapter(
                    adapter_idx=adapter_idx,
                    rank=rank,
                    scaling=scaling,
                    layer_weights={buffer_layer_id: weight_dict}
                )
                logger.debug(f"[LoRA]   Loaded {target_type} for layer {layer_id}")

        self.adapter_dirs.append(adapter_dir)
        self.idx_map[adapter_dir] = adapter_idx
        logger.info(f"[LoRA] Adapter loaded: {adapter_dir} (idx={adapter_idx})")
        return True

    def unload_adapter(self, adapter_dir: str) -> bool:
        """Unload adapter from all pools.

        Debug:
            Logs adapter unloading
        """
        if adapter_dir not in self.idx_map:
            logger.warning(f"[LoRA] Adapter not found for unload: {adapter_dir}")
            return False

        adapter_idx = self.idx_map[adapter_dir]
        logger.info(f"[LoRA] Unloading adapter: {adapter_dir} (idx={adapter_idx})")

        # Unload from all pools
        for pool_attr in [
            'vl_q_pool', 'vl_k_pool', 'vl_v_pool', 'vl_o_pool',
            'vl_fc1_pool', 'vl_fc2_pool',
            'attn_q_pool', 'attn_k_pool', 'attn_v_pool', 'attn_o_pool',
            'moe_gate_pool', 'moe_up_pool', 'moe_down_pool', 'lm_head_pool'
        ]:
            pool = getattr(self, pool_attr)
            if pool is not None:
                pool.unload_adapter(adapter_idx)

        # Update tracking
        self.adapter_dirs.pop(adapter_idx)
        self.idx_map.pop(adapter_dir)
        logger.info(f"[LoRA] Adapter unloaded: {adapter_dir}")

        return True

    def get_adapter_idx(self, adapter_dir: str) -> int:
        """Get adapter index or -1 if not found."""
        return self.idx_map.get(adapter_dir, -1)

    def get_metadata(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "num_adapters": len(self.adapter_dirs),
            "max_rank": self.max_rank,
            "pool_size": self.pool_size,
            "adapter_dirs": self.adapter_dirs,
            "pools": {
                "vl_q": {"used": len(self.vl_q_pool.a_len) if self.vl_q_pool else 0},
                "attn_q": {"used": len(self.attn_q_pool.a_len) if self.attn_q_pool else 0},
                "moe_gate": {"used": len(self.moe_gate_pool.a_len) if self.moe_gate_pool else 0},
            }
        }


class LoRAAdapterLoader:
    """Helper class to load LoRA adapters from disk."""

    @staticmethod
    def load_from_dir(
        adapter_dir: str,
        network_config: Dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Load all weights from an adapter directory.

        Args:
            adapter_dir: Path to adapter directory
            network_config: (unused) kept for API compatibility
            dtype: Data type for weights
            device: Device for tensors

        Returns:
            Dict mapping layer_id -> target_type -> module_weights
        """
        safetensor_files = glob.glob(os.path.join(adapter_dir, "*.safetensors"))
        if not safetensor_files:
            raise ValueError(f"No safetensors found in {adapter_dir}")

        all_weights = {}
        for f in safetensor_files:
            with safe_open(f, "pt", "cpu") as sf:
                for k in sf.keys():
                    all_weights[k] = sf.get_tensor(k)

        result = {}

        # Pre-compiled regex patterns for speed
        # 1. Language Model Layers (e.g. model.language_model.layers.9...)
        re_llm_layer = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")

        # 2. LM Head (e.g. model.language_model.lm_head...)
        re_lm_head = re.compile(r"model\.language_model\.lm_head")

        # 3. Vision Blocks (e.g. model.visual.blocks.0...)
        re_vis_block = re.compile(r"model\.visual\.blocks\.(\d+)\.(.+)")

        # 4. Deepstack Merger (e.g. model.visual.deepstack_merger_list.0...)
        re_vis_deepstack = re.compile(r"model\.visual\.deepstack_merger_list\.(\d+)\.(.+)")

        # 5. Simple Merger (e.g. model.visual.merger...)
        re_vis_merger = re.compile(r"model\.visual\.merger\.(.+)")

        # Expert ID Pattern (nested inside layer suffix)
        re_expert_id = re.compile(r"experts\.(\d+)\.")

        for key, tensor in all_weights.items():
            if "lora_A" not in key and "lora_B" not in key:
                continue

            matrix_type = "A" if "lora_A" in key else "B"
            layer_id = None
            target_type = None
            expert_id = None

            # 1. Language Model Layers
            match = re_llm_layer.search(key)
            if match:
                layer_id = int(match.group(1))
                suffix = match.group(2)

                # Attention
                if "self_attn.q_proj" in suffix:
                    target_type = LoRATargetType.ATTN_Q_PROJ
                elif "self_attn.k_proj" in suffix:
                    target_type = LoRATargetType.ATTN_K_PROJ
                elif "self_attn.v_proj" in suffix:
                    target_type = LoRATargetType.ATTN_V_PROJ
                elif "self_attn.o_proj" in suffix:
                    target_type = LoRATargetType.ATTN_O_PROJ

                # MoE Experts (e.g. mlp.experts.0.down_proj)
                elif "mlp.experts" in suffix:
                    expert_match = re_expert_id.search(suffix)
                    if expert_match:
                        expert_id = int(expert_match.group(1))

                        if "gate_proj" in suffix:
                            target_type = LoRATargetType.MOE_EXPERT_GATE
                        elif "up_proj" in suffix:
                            target_type = LoRATargetType.MOE_EXPERT_UP
                        elif "down_proj" in suffix:
                            target_type = LoRATargetType.MOE_EXPERT_DOWN

            # 2. LM Head (Special Layer -1)
            elif re_lm_head.search(key):
                layer_id = -1
                target_type = LoRATargetType.LM_HEAD

            # 3. Vision Blocks (Offset +10000)
            elif (match := re_vis_block.search(key)):
                layer_id = 10000 + int(match.group(1))
                suffix = match.group(2)

                if "attn.q_proj" in suffix:
                    target_type = LoRATargetType.VL_Q_PROJ
                elif "attn.k_proj" in suffix:
                    target_type = LoRATargetType.VL_K_PROJ
                elif "attn.v_proj" in suffix:
                    target_type = LoRATargetType.VL_V_PROJ
                elif "attn.o_proj" in suffix:
                    target_type = LoRATargetType.VL_O_PROJ
                elif "mlp.linear_fc1" in suffix:
                    target_type = LoRATargetType.VL_FC1
                elif "mlp.linear_fc2" in suffix:
                    target_type = LoRATargetType.VL_FC2

            # 4. Deepstack Mergers (Offset +20000)
            elif (match := re_vis_deepstack.search(key)):
                layer_id = 20000 + int(match.group(1))
                suffix = match.group(2)

                if "linear_fc1" in suffix:
                    target_type = LoRATargetType.VL_FC1
                elif "linear_fc2" in suffix:
                    target_type = LoRATargetType.VL_FC2

            # 5. Simple Merger (Offset +29999)
            elif (match := re_vis_merger.search(key)):
                layer_id = 29999
                suffix = match.group(1)

                if "linear_fc1" in suffix:
                    target_type = LoRATargetType.VL_FC1
                elif "linear_fc2" in suffix:
                    target_type = LoRATargetType.VL_FC2

            # Storage Logic
            if layer_id is not None and target_type is not None:
                if layer_id not in result:
                    result[layer_id] = {}

                # Handle Experts separately if expert_id exists
                if expert_id is not None:
                    # Structure: result[layer][target_type][expert_id][A/B]
                    if target_type not in result[layer_id]:
                        result[layer_id][target_type] = {}

                    if expert_id not in result[layer_id][target_type]:
                        result[layer_id][target_type][expert_id] = {}

                    result[layer_id][target_type][expert_id][matrix_type] = tensor.to(dtype=dtype, device=device)
                else:
                    # Structure: result[layer][target_type][A/B]
                    if target_type not in result[layer_id]:
                        result[layer_id][target_type] = {}

                    result[layer_id][target_type][matrix_type] = tensor.to(dtype=dtype, device=device)

        return result


def create_lora_mem_pool(
    num_layers: int,
    pool_size: int = 1024,
    max_rank: int = 64,
    num_heads: int = 32,
    head_dim: int = 128,
    intermediate_dim: int = 512,
    hidden_size: int = 4096,
    vocab_size: int = 151936,
    num_kv_heads: int | None = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> LoRAMemPool:
    """Create a complete LoRA memory pool.

    Args:
        num_kv_heads: Number of key/value heads (for GQA models). If None, defaults to num_heads.
    """
    return LoRAMemPool.create(
        num_layers=num_layers,
        pool_size=pool_size,
        max_rank=max_rank,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        device=device
    )
