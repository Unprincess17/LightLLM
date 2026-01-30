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

    Memory Layout:
    - Each adapter needs num_layers slots (one per layer the adapter applies to)
    - Each slot stores [rank, hidden_dim] weights
    - a_len stores the number of slots (layers) the adapter occupies
    """
    # Shape: [pool_size, max_rank, a_hidden_dim] for A
    key_buffer: torch.Tensor  # LoRA A weights
    # Shape: [pool_size, max_rank, b_hidden_dim] for B
    value_buffer: torch.Tensor  # LoRA B weights

    # Metadata
    a_start: torch.Tensor  # [num_adapters] - start slot offset per adapter
    a_len: torch.Tensor  # [num_adapters] - number of slots (layers) per adapter
    a_scaling: torch.Tensor  # [num_adapters] - scaling factor per adapter
    max_rank: int
    a_hidden_dim: int  # Input dimension for A matrix
    b_hidden_dim: int  # Output dimension for B matrix
    pool_size: int
    num_layers: int = 1  # Number of layers this pool handles (for slot calculation)
    num_experts: int = 1  # Number of experts per layer (for MoE models)

    @property
    def a_buffer(self) -> torch.Tensor:
        """Get the A weight buffer."""
        return self.key_buffer

    @property
    def b_buffer(self) -> torch.Tensor:
        """Get the B weight buffer."""
        return self.value_buffer

    def __repr__(self) -> str:
        return (
            f"LoRAModulePool(a_hidden_dim={self.a_hidden_dim}, "
            f"b_hidden_dim={self.b_hidden_dim}, "
            f"key_buffer.shape={self.key_buffer.shape}, "
            f"value_buffer.shape={self.value_buffer.shape}, "
            f"num_layers={self.num_layers})"
        )

    @classmethod
    def create(
        cls,
        pool_size: int,
        max_rank: int,
        input_dim: int,
        output_dim: int | None = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        num_layers: int = 1,
        num_experts: int = 1,
    ) -> "LoRAModulePool":
        """Create a module pool.

        Args:
            pool_size: Maximum number of adapters in pool (in slots)
            max_rank: Maximum LoRA rank
            input_dim: Input dimension for A matrix (and x tensor)
            output_dim: Output dimension for B matrix (and y tensor). If None, uses input_dim.
            dtype: Data type for weights
            device: Device for tensors
            num_layers: Number of layers this pool handles (for slot calculation)
            num_experts: Number of experts per layer (for MoE models). num_experts=1 for non-MoE.
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
            pool_size=pool_size,
            num_layers=num_layers,
            num_experts=num_experts,
        )

    def can_fit(self, rank: int) -> bool:
        """Check if an adapter with given rank can fit.

        Each adapter needs num_layers * num_experts slots (one per layer/expert).
        The rank only affects the per-slot memory, not the slot count.
        """
        used_slots = self.a_len.sum().item() if len(self.a_len) > 0 else 0
        slots_needed = self.num_layers * self.num_experts  # Each adapter needs num_layers * num_experts slots
        return (used_slots + slots_needed) <= self.pool_size

    def _compute_location(self) -> int:
        """Compute next available slot location."""
        if len(self.a_len) == 0:
            return 0
        # a_len stores slots consumed per adapter, not rank
        return (self.a_start[-1] + self.a_len[-1]).item()

    def _write_weights(
        self,
        loc: int,
        rank: int,
        scaling: float,
        weights: dict,
        tp_rank: int = 0,
        tp_world_size: int = 1
    ) -> bool:
        """Write A and B weights to buffer at given location.

        Args:
            loc: Buffer location index
            rank: LoRA rank
            scaling: Scaling factor
            weights: Dict with "A" and "B" keys
            tp_rank: Tensor parallel rank
            tp_world_size: Tensor parallel world size
        """
        a_weight = weights.get("A")
        b_weight = weights.get("B")

        if a_weight is not None:
            # A weight matrix: [rank, hidden]
            # Case 1: Perfect match (TP=1 or pre-sharded weights)
            if self.a_buffer.shape[-1] == a_weight.shape[-1]:
                self.a_buffer[loc, :rank] = a_weight.to(self.a_buffer.dtype)
            # Case 2: TP sharding needed - validate math is consistent
            elif (self.a_buffer.shape[-1] < a_weight.shape[-1] and
                  a_weight.shape[-1] == self.a_buffer.shape[-1] * tp_world_size):
                split_size = self.a_buffer.shape[-1]
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                # Validate range
                if end_idx <= a_weight.shape[-1]:
                    a_weight_sharded = a_weight[:, start_idx:end_idx]
                    self.a_buffer[loc, :rank] = a_weight_sharded.to(self.a_buffer.dtype)
                else:
                    logger.error(f"TP slicing out of bounds! rank={tp_rank}, size={split_size}, w_shape={a_weight.shape}")
                    return False
            # Case 3: Invalid mismatch
            else:
                logger.error(
                    f"Shape Mismatch Error: Buffer {self.a_buffer.shape[-1]} vs Weight {a_weight.shape[-1]}. "
                    f"TP_Size={tp_world_size}. This is not a valid TP split."
                )
                return False

        if b_weight is not None:
            # B weight matrix: [rank, hidden]
            # Case 1: Perfect match (TP=1 or pre-sharded weights)
            if self.b_buffer.shape[-1] == b_weight.shape[-1]:
                self.b_buffer[loc, :rank] = b_weight.to(self.b_buffer.dtype)
            # Case 2: TP sharding needed - validate math is consistent
            elif (self.b_buffer.shape[-1] < b_weight.shape[-1] and
                  b_weight.shape[-1] == self.b_buffer.shape[-1] * tp_world_size):
                split_size = self.b_buffer.shape[-1]
                start_idx = tp_rank * split_size
                end_idx = (tp_rank + 1) * split_size
                # Validate range
                if end_idx <= b_weight.shape[-1]:
                    b_weight_sharded = b_weight[:, start_idx:end_idx]
                    self.b_buffer[loc, :rank] = b_weight_sharded.to(self.b_buffer.dtype)
                else:
                    logger.error(f"TP slicing out of bounds! rank={tp_rank}, size={split_size}, w_shape={b_weight.shape}")
                    return False
            # Case 3: Invalid mismatch
            else:
                logger.error(
                    f"Shape Mismatch Error: Buffer {self.b_buffer.shape[-1]} vs Weight {b_weight.shape[-1]}. "
                    f"TP_Size={tp_world_size}. This is not a valid TP split."
                )
                return False

        return True

    def load_adapter(
        self,
        adapter_idx: int,
        rank: int,
        scaling: float,
        layer_weights: Dict,
        tp_rank: int = 0,
        tp_world_size: int = 1
    ) -> bool:
        """
        Load adapter weights for all layers.

        Supports two structures:
        - Flat (Attention): {layer_id: {"A": tensor, "B": tensor}}
        - MoE Nested: {layer_id: {expert_id: {"A": tensor, "B": tensor}}}

        Args:
            adapter_idx: Index of this adapter in the global pool
            rank: LoRA rank
            scaling: Scaling factor
            layer_weights: Dict mapping layer_id -> weights
            tp_rank: Tensor parallel rank for sharded weight loading
            tp_world_size: Tensor parallel world size for sharded weight loading
        """
        if not self.can_fit(rank):
            return False

        loc_start = self._compute_location()

        # Detect structure: MoE nested or flat
        # Check first layer's first value to determine structure
        first_layer_id = next(iter(layer_weights.keys())) if layer_weights else None
        if first_layer_id is None:
            return False

        first_layer_content = layer_weights[first_layer_id]
        first_val = next(iter(first_layer_content.values())) if first_layer_content else None
        is_moe_structure = isinstance(first_val, dict) and "A" in first_val

        # Count valid layers for metadata
        valid_layers = 0
        for layer_id in layer_weights.keys():
            if layer_id >= 10000:
                buffer_layer_id = layer_id - 10000
            else:
                buffer_layer_id = layer_id
            if 0 <= buffer_layer_id < self.num_layers:
                valid_layers += 1

        if valid_layers == 0:
            logger.warning(f"[LoRA] No valid layers for adapter in pool (num_layers={self.num_layers})")
            return False

        # Extend metadata - a_len stores number of slots (layers * experts) this adapter occupies
        self.a_start = torch.cat([
            self.a_start,
            torch.tensor([loc_start], dtype=torch.long, device=self.a_start.device)
        ])
        self.a_len = torch.cat([
            self.a_len,
            torch.tensor([valid_layers], dtype=torch.long, device=self.a_len.device)
        ])
        self.a_scaling = torch.cat([
            self.a_scaling,
            torch.tensor([scaling], dtype=self.a_scaling.dtype, device=self.a_scaling.device)
        ])

        try:
            if is_moe_structure:
                # MoE nested structure: {layer_id: {expert_id: {"A": ..., "B": ...}}}
                for layer_id, expert_weights in layer_weights.items():
                    if expert_weights is None or not expert_weights:
                        continue

                    # Convert layer_id to buffer base index
                    if layer_id >= 10000:
                        buffer_layer_id = layer_id - 10000
                    else:
                        buffer_layer_id = layer_id

                    if buffer_layer_id < 0 or buffer_layer_id >= self.num_layers:
                        continue

                    # Process each expert
                    for expert_id, weights in expert_weights.items():
                        # Calculate flattened index: layer * num_experts + expert
                        loc = loc_start + buffer_layer_id * self.num_experts + expert_id
                        self._write_weights(loc, rank, scaling, weights, tp_rank, tp_world_size)
            else:
                # Flat structure: {layer_id: {"A": ..., "B": ...}}
                for layer_id, weights in layer_weights.items():
                    if weights is None or not weights:
                        continue

                    # Convert layer_id to buffer index
                    if layer_id >= 10000:
                        buffer_layer_id = layer_id - 10000
                    else:
                        buffer_layer_id = layer_id

                    if buffer_layer_id < 0 or buffer_layer_id >= self.num_layers:
                        continue

                    loc = loc_start + buffer_layer_id
                    self._write_weights(loc, rank, scaling, weights, tp_rank, tp_world_size)
        except Exception as e:
            logger.error(f"Error loading adapter weights: {e}")
            return False

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
    moe_intermediate_dim: int = 0  # For MoE models (gate/up/down projection size)
    hidden_size: int = 0
    vocab_size: int = 0
    num_experts: int = 1  # Number of experts per layer (for MoE models). num_experts=1 for non-MoE.

    # TP world info for sharded weight loading
    tp_rank_: int = 0
    tp_world_size_: int = 1

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
        device: str = "cuda",
        # Vision config parameters (optional)
        vl_hidden_size: int | None = None,
        vl_intermediate_size: int | None = None,
        vl_out_hidden_size: int | None = None,
        vl_depth: int | None = None,
        # MoE config parameter (optional)
        moe_intermediate_dim: int | None = None,
        # MoE number of experts
        num_experts: int = 1,
        # TP world size for sharded dimensions
        tp_world_size: int = 1,
    ) -> "LoRAMemPool":
        """Create a complete LoRA memory pool.

        Args:
            num_kv_heads: Number of key/value heads (for GQA models). If None, defaults to num_heads.
            vl_hidden_size: Vision model hidden size. If None, uses hidden_size.
            vl_intermediate_size: Vision MLP intermediate size. If None, uses intermediate_dim.
            vl_out_hidden_size: Vision output hidden size. If None, uses vl_hidden_size.
            vl_depth: Vision model depth for pool layer capacity. If None, uses num_layers.
            moe_intermediate_dim: MoE intermediate size for gate/up/down projections.
            num_experts: Number of experts per layer (for MoE models).
            tp_world_size: Tensor parallel world size for sharded dimensions (default: 1).
        """
        # Use vision dimensions if provided, otherwise fall back to text model dimensions
        if vl_hidden_size is None:
            vl_hidden_size = hidden_size
        if vl_intermediate_size is None:
            vl_intermediate_size = intermediate_dim
        if vl_out_hidden_size is None:
            vl_out_hidden_size = vl_hidden_size
        if vl_depth is None:
            vl_depth = num_layers

        # For MoE models, use moe_intermediate_dim if provided, otherwise fall back to intermediate_dim
        if moe_intermediate_dim is None:
            moe_intermediate_dim = intermediate_dim
        mlp_inter = moe_intermediate_dim  # Use MoE intermediate size for MoE pools

        # Attention dimensions
        
        # 1. Input Dimension (From Residual Stream)
        attn_in_hidden = hidden_size
        
        # 2. Internal Attention Dimension (Q output / O input)
        attn_internal_dim = num_heads * head_dim 

        if num_kv_heads is None:
            num_kv_heads = num_heads

        # 3. KV Dimension (K/V output)
        # GQA: B matrix output dimension
        kv_internal_dim = num_kv_heads * head_dim

        # Vision dimensions
        vl_hidden = vl_hidden_size
        vl_mlp_hidden = vl_intermediate_size
        vl_out_hidden = vl_out_hidden_size

        logger.info(f"[LoRA Pool] Creating pool: attn_in={attn_in_hidden}, attn_internal={attn_internal_dim}, kv_internal={kv_internal_dim}, mlp_hidden={mlp_inter}, vl_hidden={vl_hidden}, vl_mlp_hidden={vl_mlp_hidden}, vl_out_hidden={vl_out_hidden}, vl_depth={vl_depth}, tp_world_size={tp_world_size}")

        pool = cls(
            # Vision-Language pools (Q/K/V/O use vl_hidden, FC1/FC2 use vl_hidden/vl_mlp_hidden)
            # Vision pools use vl_depth for num_layers
            vl_q_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device, num_layers=vl_depth),
            vl_k_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device, num_layers=vl_depth),
            vl_v_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device, num_layers=vl_depth),
            vl_o_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_hidden, dtype, device, num_layers=vl_depth),
            # Vision MLP: FC1 expands from vl_hidden to vl_mlp_hidden, FC2 shrinks from vl_mlp_hidden to vl_out_hidden
            vl_fc1_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, vl_mlp_hidden, dtype, device, num_layers=vl_depth),
            vl_fc2_pool=LoRAModulePool.create(pool_size, max_rank, vl_mlp_hidden, vl_hidden, dtype, device, num_layers=vl_depth),

            # Q Pool: Column Parallel -> Output is SPLIT
            # b_hidden_dim must be 4096 // 2 = 2048
            attn_q_pool = LoRAModulePool.create(
                pool_size, max_rank, 
                attn_in_hidden, 
                attn_internal_dim // tp_world_size,  # <--- DIVIDE BY TP
                dtype, device, num_layers=num_layers
            ),

            # K Pool: Column Parallel -> Output is SPLIT
            # b_hidden_dim must be 512 // 2 = 256
            attn_k_pool = LoRAModulePool.create(
                pool_size, max_rank, 
                attn_in_hidden, 
                kv_internal_dim // tp_world_size,    # <--- DIVIDE BY TP
                dtype, device, num_layers=num_layers
            ),

            # V Pool: Column Parallel -> Output is SPLIT
            # b_hidden_dim must be 512 // 2 = 256
            attn_v_pool = LoRAModulePool.create(
                pool_size, max_rank, 
                attn_in_hidden, 
                kv_internal_dim // tp_world_size,    # <--- DIVIDE BY TP
                dtype, device, num_layers=num_layers
            ),

            # O Pool: Row Parallel -> Input is SPLIT (Correct in your code)
            # a_hidden_dim must be 2048 (which is 4096 // 2)
            attn_o_pool = LoRAModulePool.create(
                pool_size, max_rank,
                attn_internal_dim // tp_world_size,  # <--- DIVIDE BY TP
                attn_in_hidden,
                dtype, device, num_layers=num_layers
            ),

            # MoE MLP pools - Gate/Up expand, Down shrinks
            moe_gate_pool=LoRAModulePool.create(
                pool_size, max_rank, hidden_size, mlp_inter,
                dtype, device, num_layers=num_layers, num_experts=num_experts
            ),
            moe_up_pool=LoRAModulePool.create(
                pool_size, max_rank, hidden_size, mlp_inter,
                dtype, device, num_layers=num_layers, num_experts=num_experts
            ),
            moe_down_pool=LoRAModulePool.create(
                pool_size, max_rank, mlp_inter, hidden_size,
                dtype, device, num_layers=num_layers, num_experts=num_experts
            ),
            
            # LM Head pool
            lm_head_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, vocab_size, dtype, device, num_layers=1),

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
            moe_intermediate_dim=moe_intermediate_dim,
            vocab_size=vocab_size,
            num_experts=num_experts,
            tp_world_size_=tp_world_size
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

        # Collect weights per pool type: {target_type: {buffer_layer_id: weight_dict}}
        pool_weights: Dict[str, Dict[int, Dict]] = {}

        for layer_id, target_weights in layer_weights.items():
            # Handle vision layer offset (10000+) - strip offset for buffer indexing
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

                # Initialize pool_weights entry if needed
                if target_type not in pool_weights:
                    pool_weights[target_type] = {}

                # Strip module_name level - pool expects {layer_id: {"A": tensor, "B": tensor}}
                # module_weights is {module_name: {"A": tensor, "B": tensor}}, take first value
                weight_dict = next(iter(module_weights.values())) if module_weights else {}

                # Add to pool's weight collection
                pool_weights[target_type][buffer_layer_id] = weight_dict

        # Now load all weights for each pool in a single call
        for target_type, weights_by_layer in pool_weights.items():
            pool = self.get_pool(target_type)
            if pool is not None:
                pool.load_adapter(
                    adapter_idx=adapter_idx,
                    rank=rank,
                    scaling=scaling,
                    layer_weights=weights_by_layer,
                    tp_rank=self.tp_rank_,
                    tp_world_size=self.tp_world_size_
                )
                logger.debug(f"[LoRA]   Loaded {target_type} with {len(weights_by_layer)} layers")

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
    device: str = "cuda",
    # Vision config parameters (optional)
    vl_hidden_size: int | None = None,
    vl_intermediate_size: int | None = None,
    vl_out_hidden_size: int | None = None,
    vl_depth: int | None = None,
    # MoE config parameter (optional)
    moe_intermediate_dim: int | None = None,
    num_experts: int = 1,
    # TP world size for sharded dimensions
    tp_world_size: int = 1,
) -> LoRAMemPool:
    """Create a complete LoRA memory pool.

    Args:
        num_kv_heads: Number of key/value heads (for GQA models). If None, defaults to num_heads.
        vl_hidden_size: Vision model hidden size. If None, uses hidden_size.
        vl_intermediate_size: Vision MLP intermediate size. If None, uses intermediate_dim.
        vl_out_hidden_size: Vision output hidden size. If None, uses vl_hidden_size.
        vl_depth: Vision model depth for pool layer capacity. If None, uses num_layers.
        moe_intermediate_dim: MoE intermediate size. If None, uses intermediate_dim.
        num_experts: Number of experts per layer (for MoE models).
        tp_world_size: Tensor parallel world size for sharded dimensions (default: 1).
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
        device=device,
        vl_hidden_size=vl_hidden_size,
        vl_intermediate_size=vl_intermediate_size,
        vl_out_hidden_size=vl_out_hidden_size,
        vl_depth=vl_depth,
        moe_intermediate_dim=moe_intermediate_dim,
        num_experts=num_experts,
        tp_world_size=tp_world_size,
    )
