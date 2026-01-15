"""
Qwen3-VL-MoE Detached LoRA Layer Weight

This module implements detached LoRA weight storage for Qwen3-VL-MoE.
LoRA weights are stored separately from base model weights and can be
loaded/unloaded dynamically for detached serving.

LoRA is applied to:
- MLP layers: gate_proj, up_proj, down_proj (6 matrices)
- Attention layers: q_proj, k_proj, v_proj, o_proj (8 matrices)

Each attention LoRA is optional and can be enabled independently:
- q_lora_rank: LoRA rank for q_proj (0 = disabled)
- k_lora_rank: LoRA rank for k_proj (0 = disabled)
- v_lora_rank: LoRA rank for v_proj (0 = disabled)
- o_lora_rank: LoRA rank for o_proj (0 = disabled)

Total: 14 LoRA matrices per layer (6 MLP + 8 Attention)
"""
import torch
import os
from typing import Optional, Dict, Any
from safetensors import safe_open


class Qwen3VLMoELoRALayerWeight:
    """LoRA layer weights for Qwen3-VL-MoE MLP and Attention projections.

    Stores LoRA A and B matrices for:
    - MLP: gate_proj, up_proj, down_proj (6 matrices)
    - Attention: q_proj, k_proj, v_proj, o_proj (8 matrices, each optional)

    Each attention LoRA can be enabled independently by setting its rank > 0.
    Supports CPU/GPU swapping for memory efficiency.
    """

    def __init__(
        self,
        layer_num: int,
        network_config: Dict[str, Any],
        data_type: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        self.layer_num_ = layer_num
        self.network_config = network_config
        self.data_type_ = data_type
        self.device_ = device

        # LoRA ranks from config (each optional, 0 = disabled)
        self.q_lora_rank = network_config.get("q_lora_rank", 0)
        self.k_lora_rank = network_config.get("k_lora_rank", 0)
        self.v_lora_rank = network_config.get("v_lora_rank", 0)
        self.o_lora_rank = network_config.get("o_lora_rank", 0)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)

        # Scaling factors (each optional)
        self.q_scaling = self.lora_alpha / self.q_lora_rank if self.q_lora_rank > 0 else 1.0
        self.k_scaling = self.lora_alpha / self.k_lora_rank if self.k_lora_rank > 0 else 1.0
        self.v_scaling = self.lora_alpha / self.v_lora_rank if self.v_lora_rank > 0 else 1.0
        self.o_scaling = self.lora_alpha / self.o_lora_rank if self.o_lora_rank > 0 else 1.0

        # Check if any attention LoRA is enabled
        self.has_attn_lora = (
            self.q_lora_rank > 0 or self.k_lora_rank > 0 or
            self.v_lora_rank > 0 or self.o_lora_rank > 0
        )

        # Hidden size per head for TP
        self.hidden_size = network_config["hidden_size"]
        self.tp_size = 1  # Single GPU for now
        self.split_hidden_size = self.hidden_size // self.tp_size

        # MLP LoRA matrices (lazy loading)
        self.gate_proj_A: Optional[torch.Tensor] = None
        self.gate_proj_B: Optional[torch.Tensor] = None
        self.up_proj_A: Optional[torch.Tensor] = None
        self.up_proj_B: Optional[torch.Tensor] = None
        self.down_proj_A: Optional[torch.Tensor] = None
        self.down_proj_B: Optional[torch.Tensor] = None

        # Attention LoRA matrices (q, k, v, o for MoE - each optional)
        self.q_proj_A: Optional[torch.Tensor] = None
        self.q_proj_B: Optional[torch.Tensor] = None
        self.k_proj_A: Optional[torch.Tensor] = None
        self.k_proj_B: Optional[torch.Tensor] = None
        self.v_proj_A: Optional[torch.Tensor] = None
        self.v_proj_B: Optional[torch.Tensor] = None
        self.o_proj_A: Optional[torch.Tensor] = None
        self.o_proj_B: Optional[torch.Tensor] = None

        # CPU storage for swap mode - MLP
        self.gate_proj_A_home: Optional[torch.Tensor] = None
        self.gate_proj_B_home: Optional[torch.Tensor] = None
        self.up_proj_A_home: Optional[torch.Tensor] = None
        self.up_proj_B_home: Optional[torch.Tensor] = None
        self.down_proj_A_home: Optional[torch.Tensor] = None
        self.down_proj_B_home: Optional[torch.Tensor] = None

        # CPU storage for swap mode - Attention (separate k, v, o)
        self.q_proj_A_home: Optional[torch.Tensor] = None
        self.q_proj_B_home: Optional[torch.Tensor] = None
        self.k_proj_A_home: Optional[torch.Tensor] = None
        self.k_proj_B_home: Optional[torch.Tensor] = None
        self.v_proj_A_home: Optional[torch.Tensor] = None
        self.v_proj_B_home: Optional[torch.Tensor] = None
        self.o_proj_A_home: Optional[torch.Tensor] = None
        self.o_proj_B_home: Optional[torch.Tensor] = None

        # State tracking
        self.is_loaded_ = False
        self.is_on_gpu_ = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor], swap: bool = False):
        """Load LoRA weights from HuggingFace format weights dict.

        Args:
            weights: Dictionary of weight tensors from safetensors
            swap: If True, keep weights on CPU (pinned memory)
        """
        if not self.has_attn_lora:
            return  # No attention LoRA to load

        mlp_prefix = f"base_model.model.model.language_model.layers.{self.layer_num_}.mlp"
        attn_prefix = f"base_model.model.model.language_model.layers.{self.layer_num_}.self_attn"

        # Get TP slice indices
        tp_idx_start = 0
        tp_idx_end = self.split_hidden_size

        # ========== Load MLP LoRA weights ==========
        gate_a_key = f"{mlp_prefix}.gate_proj.lora_A.weight"
        gate_b_key = f"{mlp_prefix}.gate_proj.lora_B.weight"
        if gate_a_key in weights:
            self._load_weight(gate_a_key, weights, "gate_proj_A", swap, tp_idx_start, tp_idx_end)
        if gate_b_key in weights:
            self._load_weight(gate_b_key, weights, "gate_proj_B", swap, tp_idx_start, tp_idx_end)

        up_a_key = f"{mlp_prefix}.up_proj.lora_A.weight"
        up_b_key = f"{mlp_prefix}.up_proj.lora_B.weight"
        if up_a_key in weights:
            self._load_weight(up_a_key, weights, "up_proj_A", swap, tp_idx_start, tp_idx_end)
        if up_b_key in weights:
            self._load_weight(up_b_key, weights, "up_proj_B", swap, tp_idx_start, tp_idx_end)

        down_a_key = f"{mlp_prefix}.down_proj.lora_A.weight"
        down_b_key = f"{mlp_prefix}.down_proj.lora_B.weight"
        if down_a_key in weights:
            self._load_weight(down_a_key, weights, "down_proj_A", swap, tp_idx_start, tp_idx_end)
        if down_b_key in weights:
            self._load_weight(down_b_key, weights, "down_proj_B", swap, tp_idx_start, tp_idx_end)

        # ========== Load Attention LoRA weights (MoE: q, k, v, o) ==========
        if self.q_lora_rank > 0:
            q_a_key = f"{attn_prefix}.q_proj.lora_A.weight"
            q_b_key = f"{attn_prefix}.q_proj.lora_B.weight"
            if q_a_key in weights:
                self._load_weight(q_a_key, weights, "q_proj_A", swap, tp_idx_start, tp_idx_end)
            if q_b_key in weights:
                self._load_weight(q_b_key, weights, "q_proj_B", swap, tp_idx_start, tp_idx_end)

        if self.k_lora_rank > 0:
            k_a_key = f"{attn_prefix}.k_proj.lora_A.weight"
            k_b_key = f"{attn_prefix}.k_proj.lora_B.weight"
            if k_a_key in weights:
                self._load_weight(k_a_key, weights, "k_proj_A", swap, tp_idx_start, tp_idx_end)
            if k_b_key in weights:
                self._load_weight(k_b_key, weights, "k_proj_B", swap, tp_idx_start, tp_idx_end)

        if self.v_lora_rank > 0:
            v_a_key = f"{attn_prefix}.v_proj.lora_A.weight"
            v_b_key = f"{attn_prefix}.v_proj.lora_B.weight"
            if v_a_key in weights:
                self._load_weight(v_a_key, weights, "v_proj_A", swap, tp_idx_start, tp_idx_end)
            if v_b_key in weights:
                self._load_weight(v_b_key, weights, "v_proj_B", swap, tp_idx_start, tp_idx_end)

        if self.o_lora_rank > 0:
            o_a_key = f"{attn_prefix}.o_proj.lora_A.weight"
            o_b_key = f"{attn_prefix}.o_proj.lora_B.weight"
            if o_a_key in weights:
                self._load_weight(o_a_key, weights, "o_proj_A", swap, tp_idx_start, tp_idx_end)
            if o_b_key in weights:
                self._load_weight(o_b_key, weights, "o_proj_B", swap, tp_idx_start, tp_idx_end)

        self.is_loaded_ = True
        self.is_on_gpu_ = not swap

    def _load_weight(
        self,
        key: str,
        weights: Dict[str, torch.Tensor],
        attr_name: str,
        swap: bool,
        tp_start: int,
        tp_end: int
    ):
        """Load a single LoRA weight matrix."""
        if key not in weights:
            return

        weight = weights[key]

        if weight.dim() == 2:
            pass
        elif weight.dim() == 1:
            return  # Bias vector - skip

        weight = weight.to(dtype=self.data_type_)

        # A matrices need to be transposed to [hidden, rank] for input @ A
        if attr_name.endswith("_A"):
            weight = weight.transpose(0, 1).contiguous()

        if swap:
            setattr(self, f"{attr_name}_home", weight.pin_memory())
            setattr(self, attr_name, None)
        else:
            setattr(self, attr_name, weight.to(self.device_))

    def load_to_gpu(self, non_blocking: bool = True):
        """Load weights from CPU to GPU."""
        if self.is_on_gpu_ or not self.has_attn_lora:
            return

        # MLP LoRA weights
        for attr in ["gate_proj_A", "gate_proj_B", "up_proj_A", "up_proj_B", "down_proj_A", "down_proj_B"]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)

        # Attention LoRA weights (q, k, v, o)
        for attr in ["q_proj_A", "q_proj_B", "k_proj_A", "k_proj_B", "v_proj_A", "v_proj_B", "o_proj_A", "o_proj_B"]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)

        self.is_on_gpu_ = True

    def offload_from_gpu(self):
        """Offload weights from GPU to CPU."""
        if not self.is_on_gpu_ or not self.has_attn_lora:
            return

        # MLP LoRA weights
        for attr in ["gate_proj_A", "gate_proj_B", "up_proj_A", "up_proj_B", "down_proj_A", "down_proj_B"]:
            gpu_attr = getattr(self, attr, None)
            if gpu_attr is not None:
                setattr(self, f"{attr}_home", gpu_attr.cpu().pin_memory())
                setattr(self, attr, None)

        # Attention LoRA weights (q, k, v, o)
        for attr in ["q_proj_A", "q_proj_B", "k_proj_A", "k_proj_B", "v_proj_A", "v_proj_B", "o_proj_A", "o_proj_B"]:
            gpu_attr = getattr(self, attr, None)
            if gpu_attr is not None:
                setattr(self, f"{attr}_home", gpu_attr.cpu().pin_memory())
                setattr(self, attr, None)

        self.is_on_gpu_ = False

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get all LoRA weights on GPU (both MLP and Attention)."""
        result = {}
        # MLP LoRA weights
        for attr in ["gate_proj_A", "gate_proj_B", "up_proj_A", "up_proj_B", "down_proj_A", "down_proj_B"]:
            w = getattr(self, attr, None)
            if w is not None:
                result[attr] = w
        # Attention LoRA weights (q, k, v, o)
        for attr in ["q_proj_A", "q_proj_B", "k_proj_A", "k_proj_B", "v_proj_A", "v_proj_B", "o_proj_A", "o_proj_B"]:
            w = getattr(self, attr, None)
            if w is not None:
                result[attr] = w
        return result

    def verify(self) -> bool:
        """Verify all weights are loaded."""
        if not self.has_attn_lora:
            return True

        # MLP LoRA weights
        for attr in ["gate_proj_A", "gate_proj_B", "up_proj_A", "up_proj_B", "down_proj_A", "down_proj_B"]:
            w_gpu = getattr(self, attr, None)
            w_cpu = getattr(self, f"{attr}_home", None)
            if w_gpu is None and w_cpu is None:
                print(f"Warning: {attr} not loaded for layer {self.layer_num_}")
                return False

        # Attention LoRA weights (q, k, v, o)
        for attr in ["q_proj_A", "q_proj_B", "k_proj_A", "k_proj_B", "v_proj_A", "v_proj_B", "o_proj_A", "o_proj_B"]:
            w_gpu = getattr(self, attr, None)
            w_cpu = getattr(self, f"{attr}_home", None)
            if w_gpu is None and w_cpu is None:
                print(f"Warning: {attr} not loaded for layer {self.layer_num_}")
                return False
        return True


class Qwen3VLMoELoRAAdapter:
    """Manages a complete LoRA adapter across all transformer layers for MoE."""

    def __init__(
        self,
        adapter_dir: str,
        network_config: Dict[str, Any],
        data_type: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        swap: bool = False
    ):
        self.adapter_dir = adapter_dir
        self.network_config = network_config
        self.data_type = data_type
        self.device = device
        self.swap = swap

        # Extract LoRA config from adapter
        self.q_lora_rank = network_config.get("q_lora_rank", 0)
        self.k_lora_rank = network_config.get("k_lora_rank", 0)
        self.v_lora_rank = network_config.get("v_lora_rank", 0)
        self.o_lora_rank = network_config.get("o_lora_rank", 0)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)

        # Check if any attention LoRA is enabled
        self.has_attn_lora = (
            self.q_lora_rank > 0 or self.k_lora_rank > 0 or
            self.v_lora_rank > 0 or self.o_lora_rank > 0
        )

        # Create layer weights
        self.num_layers = network_config["num_hidden_layers"]
        self.layers = [
            Qwen3VLMoELoRALayerWeight(i, network_config, data_type, device)
            for i in range(self.num_layers)
        ]

        # Load weights
        self._load_weights()

    def _load_weights(self):
        """Load LoRA weights from adapter directory."""
        if not self.has_attn_lora:
            return

        # Find safetensor files
        import glob
        safetensor_files = glob.glob(os.path.join(self.adapter_dir, "*.safetensors"))

        # Load all weights
        all_weights = {}
        for f in safetensor_files:
            with safe_open(f, "pt", "cpu") as sf:
                for k in sf.keys():
                    all_weights[k] = sf.get_tensor(k)

        # Distribute to layers
        for layer in self.layers:
            layer.load_hf_weights(all_weights, swap=self.swap)

        # Move to GPU if not swapping
        if not self.swap:
            self.load_to_gpu()

    def load_to_gpu(self, non_blocking: bool = True):
        """Load all layer weights to GPU."""
        for layer in self.layers:
            layer.load_to_gpu(non_blocking)

    def offload_from_gpu(self):
        """Offload all layer weights from GPU."""
        for layer in self.layers:
            layer.offload_from_gpu()

    def is_on_gpu(self) -> bool:
        """Check if adapter is on GPU."""
        return self.layers[0].is_on_gpu_ if self.layers else False

    def get_layer_weights(self, layer_id: int) -> Dict[str, torch.Tensor]:
        """Get LoRA weights for a specific layer."""
        if 0 <= layer_id < len(self.layers):
            return self.layers[layer_id].get_weights()
        return {}

    def verify(self) -> bool:
        """Verify all weights are loaded correctly."""
        for layer in self.layers:
            if not layer.verify():
                return False
        return True


def load_moe_lora_adapter(
    adapter_dir: str,
    network_config: Dict[str, Any],
    data_type: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    swap: bool = False
) -> Qwen3VLMoELoRAAdapter:
    """Factory function to load a MoE LoRA adapter.

    Args:
        adapter_dir: Path to LoRA adapter directory
        network_config: Model network configuration
        data_type: Data type for weights
        device: Device to load weights on
        swap: If True, keep weights on CPU (for memory efficiency)

    Returns:
        Qwen3VLMoELoRAAdapter instance
    """
    return Qwen3VLMoELoRAAdapter(adapter_dir, network_config, data_type, device, swap)
