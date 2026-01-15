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
import torch
import os
import logging
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
    VL_Q_PROJ = "vl_q_proj"
    VL_K_PROJ = "vl_k_proj"
    VL_V_PROJ = "vl_v_proj"
    VL_O_PROJ = "vl_o_proj"
    VL_FC1 = "vl_linear_fc1"
    VL_FC2 = "vl_linear_fc2"
    ATTN_Q_PROJ = "self_attn_q_proj"
    ATTN_K_PROJ = "self_attn_k_proj"
    ATTN_V_PROJ = "self_attn_v_proj"
    ATTN_O_PROJ = "self_attn_o_proj"
    MOE_GATE_PROJ = "moe_gate_proj"
    MOE_UP_PROJ = "moe_up_proj"
    MOE_DOWN_PROJ = "moe_down_proj"
    LM_HEAD = "moe_lm_head"


@dataclass
class LoRAModulePool:
    """
    Memory pool for a specific module type.

    Each module type has its own pool due to different hidden dimensions.
    For attention (Q/K/V/O): hidden = num_heads * head_dim
    For MLP (gate/up/down): hidden = intermediate_dim
    For lm_head: hidden = vocab_size
    """
    # Shape: [pool_size, max_rank, hidden_dim]
    key_buffer: torch.Tensor  # LoRA A weights
    value_buffer: torch.Tensor  # LoRA B weights

    # Metadata
    a_start: torch.Tensor  # [num_adapters] - start offset per adapter
    a_len: torch.Tensor  # [num_adapters] - length per adapter (rank)
    a_scaling: torch.Tensor  # [num_adapters] - scaling factor per adapter
    max_rank: int
    hidden_dim: int
    pool_size: int

    @classmethod
    def create(
        cls,
        pool_size: int,
        max_rank: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ) -> "LoRAModulePool":
        """Create a module pool."""
        return cls(
            key_buffer=torch.empty((pool_size, max_rank, hidden_dim), dtype=dtype, device=device),
            value_buffer=torch.empty((pool_size, max_rank, hidden_dim), dtype=dtype, device=device),
            a_start=torch.zeros(0, dtype=torch.long, device=device),
            a_len=torch.zeros(0, dtype=torch.long, device=device),
            a_scaling=torch.zeros(0, dtype=dtype, device=device),
            max_rank=max_rank,
            hidden_dim=hidden_dim,
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
        layer_weights: Dict[str, torch.Tensor]
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
                self.key_buffer[loc_start + layer_id, :rank] = a_weight.T.to(self.key_buffer.dtype)
            if b_weight is not None:
                # B matrix: [rank, hidden] -> [hidden, rank] (stored as [rank, hidden])
                self.value_buffer[loc_start + layer_id, :rank] = b_weight.T.to(self.value_buffer.dtype)

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
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ) -> "LoRAMemPool":
        """Create a complete LoRA memory pool."""
        attn_hidden = num_heads * head_dim
        mlp_hidden = intermediate_dim
        vl_hidden = hidden_size  # Vision adapter typically uses hidden_size
        head_hidden = vocab_size

        logger.info(f"[LoRA Pool] Creating pool: attn_hidden={attn_hidden}, mlp_hidden={mlp_hidden}, vl_hidden={vl_hidden}, head_hidden={head_hidden}")

        pool = cls(
            # Vision-Language pools
            vl_q_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, dtype, device),
            vl_k_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, dtype, device),
            vl_v_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, dtype, device),
            vl_o_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, dtype, device),
            vl_fc1_pool=LoRAModulePool.create(pool_size, max_rank, vl_hidden, dtype, device),
            vl_fc2_pool=LoRAModulePool.create(pool_size, max_rank, mlp_hidden, dtype, device),

            # Attention pools
            attn_q_pool=LoRAModulePool.create(pool_size, max_rank, attn_hidden, dtype, device),
            attn_k_pool=LoRAModulePool.create(pool_size, max_rank, attn_hidden, dtype, device),
            attn_v_pool=LoRAModulePool.create(pool_size, max_rank, attn_hidden, dtype, device),
            attn_o_pool=LoRAModulePool.create(pool_size, max_rank, attn_hidden, dtype, device),

            # MoE MLP pools
            moe_gate_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, dtype, device),
            moe_up_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, dtype, device),
            moe_down_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, dtype, device),
            lm_head_pool=LoRAModulePool.create(pool_size, max_rank, hidden_size, dtype, device),

            adapter_dirs=[],
            idx_map={},
            max_rank=max_rank,
            pool_size=pool_size,
            num_layers=num_layers,
            num_heads=num_heads,
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
            LoRATargetType.MOE_GATE_PROJ: self.moe_gate_pool,
            LoRATargetType.MOE_UP_PROJ: self.moe_up_pool,
            LoRATargetType.MOE_DOWN_PROJ: self.moe_down_pool,
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
            if layer_id >= self.num_layers:
                continue

            for target_type, module_weights in target_weights.items():
                pool = self.get_pool(target_type)
                if pool is None:
                    continue

                pool.load_adapter(
                    adapter_idx=adapter_idx,
                    rank=rank,
                    scaling=scaling,
                    layer_weights={layer_id: module_weights}
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

        num_layers = network_config["num_hidden_layers"]
        result = {}

        # Mapping of safetensor keys to target types
        # Structure varies by adapter format, this is a flexible mapper
        for layer_id in range(num_layers):
            layer_result = {}

            # Try different key patterns
            patterns = {
                # Vision adapter
                "vision": f"base_model.model.model.vision_tower.layers.{layer_id}.",
                # Attention
                "self_attn": f"base_model.model.model.language_model.layers.{layer_id}.self_attn.",
                # MoE MLP
                "mlp": f"base_model.model.model.language_model.layers.{layer_id}.mlp.",
            }

            for proj_type, prefix in [("q_proj", "q_proj"), ("k_proj", "k_proj"),
                                       ("v_proj", "v_proj"), ("o_proj", "o_proj")]:
                for key_prefix, target_prefix in [
                    (f"{prefixes['self_attn']}", f"self_attn_{proj_type}"),
                    (f"{prefixes['vision']}", f"vl_{proj_type}")
                ]:
                    pass  # Placeholder

            # Generic key matching
            for key, tensor in all_weights.items():
                # Match layer
                if f".layers.{layer_id}." not in key:
                    continue

                # Identify target type
                if "vision_tower" in key:
                    if "q_proj" in key:
                        target_type = LoRATargetType.VL_Q_PROJ
                    elif "k_proj" in key:
                        target_type = LoRATargetType.VL_K_PROJ
                    elif "v_proj" in key:
                        target_type = LoRATargetType.VL_V_PROJ
                    elif "o_proj" in key:
                        target_type = LoRATargetType.VL_O_PROJ
                    elif "fc1" in key or "linear_fc1" in key:
                        target_type = LoRATargetType.VL_FC1
                    elif "fc2" in key or "linear_fc2" in key:
                        target_type = LoRATargetType.VL_FC2
                    else:
                        continue
                elif "self_attn" in key:
                    if "q_proj" in key:
                        target_type = LoRATargetType.ATTN_Q_PROJ
                    elif "k_proj" in key:
                        target_type = LoRATargetType.ATTN_K_PROJ
                    elif "v_proj" in key:
                        target_type = LoRATargetType.ATTN_V_PROJ
                    elif "o_proj" in key:
                        target_type = LoRATargetType.ATTN_O_PROJ
                    else:
                        continue
                elif "mlp" in key or "moe" in key:
                    if "gate_proj" in key:
                        target_type = LoRATargetType.MOE_GATE_PROJ
                    elif "up_proj" in key:
                        target_type = LoRATargetType.MOE_UP_PROJ
                    elif "down_proj" in key:
                        target_type = LoRATargetType.MOE_DOWN_PROJ
                    elif "lm_head" in key:
                        target_type = LoRATargetType.LM_HEAD
                    else:
                        continue
                else:
                    continue

                # Extract A or B matrix
                if "lora_A" in key or "lora_B" in key:
                    if target_type not in layer_result:
                        layer_result[target_type] = {}

                    if "lora_A" in key:
                        layer_result[target_type]["A"] = tensor.to(dtype=dtype, device=device)
                    elif "lora_B" in key:
                        layer_result[target_type]["B"] = tensor.to(dtype=dtype, device=device)

            if layer_result:
                result[layer_id] = layer_result

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
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> LoRAMemPool:
    """Create a complete LoRA memory pool."""
    return LoRAMemPool.create(
        num_layers=num_layers,
        pool_size=pool_size,
        max_rank=max_rank,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dtype=dtype,
        device=device
    )
