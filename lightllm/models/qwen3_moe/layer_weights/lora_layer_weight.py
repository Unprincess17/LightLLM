"""
Qwen3-MOE Detached LoRA Layer Weight

This module implements detached LoRA weight storage for Qwen3-MOE.
LoRA weights are stored separately from base model weights and can be
loaded/unloaded dynamically for detached serving.

LoRA is applied to three points in MoE:
1. moe_gate - Router/logits modification (1 pair: gate_lora_A/B)
2. w1 - Fused Gate + Up projection (1 pair: w1_lora_A/B)
3. w2 - Down projection (1 pair: w2_lora_A/B)

Total: 6 LoRA matrices per MoE layer
"""
import torch
import os
from typing import Optional, Dict, Any
from safetensors import safe_open


class Qwen3MOELoRALayerWeight:
    """LoRA layer weights for Qwen3-MOE MoE projections.

    Stores LoRA A and B matrices for:
    - moe_gate: Router/logits modification (gate_lora_A, gate_lora_B)
    - w1: Fused gate+up projection (w1_lora_A, w1_lora_B) - SINGLE LoRA
    - w2: Down projection (w2_lora_A, w2_lora_B)

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

        # LoRA config
        self.lora_rank = network_config.get("lora_rank", 64)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.scaling = self.lora_alpha / self.lora_rank if self.lora_rank > 0 else 1.0

        # Hidden size per head for TP
        self.hidden_size = network_config["hidden_size"]
        self.tp_size = 1  # Single GPU for now
        self.split_hidden_size = self.hidden_size // self.tp_size

        # MoE intermediate size
        self.moe_intermediate_size = network_config.get("moe_intermediate_size", 0)
        self.num_experts = network_config.get("num_experts", 0)

        # MoE Gate LoRA matrices (router logits modification)
        self.gate_lora_A: Optional[torch.Tensor] = None
        self.gate_lora_B: Optional[torch.Tensor] = None

        # w1 LoRA (fused gate+up projection) - SINGLE LoRA
        self.w1_lora_A: Optional[torch.Tensor] = None
        self.w1_lora_B: Optional[torch.Tensor] = None

        # w2 LoRA (down projection)
        self.w2_lora_A: Optional[torch.Tensor] = None
        self.w2_lora_B: Optional[torch.Tensor] = None

        # CPU storage for swap mode
        self.gate_lora_A_home: Optional[torch.Tensor] = None
        self.gate_lora_B_home: Optional[torch.Tensor] = None
        self.w1_lora_A_home: Optional[torch.Tensor] = None
        self.w1_lora_B_home: Optional[torch.Tensor] = None
        self.w2_lora_A_home: Optional[torch.Tensor] = None
        self.w2_lora_B_home: Optional[torch.Tensor] = None

        # State tracking
        self.is_loaded_ = False
        self.is_on_gpu_ = False

    @property
    def has_moe_gate_lora(self) -> bool:
        """Check if moe_gate LoRA is enabled."""
        return self.lora_rank > 0

    @property
    def has_w1_lora(self) -> bool:
        """Check if w1 LoRA is enabled."""
        return self.lora_rank > 0

    @property
    def has_w2_lora(self) -> bool:
        """Check if w2 LoRA is enabled."""
        return self.lora_rank > 0

    @property
    def has_any_lora(self) -> bool:
        """Check if any LoRA is enabled."""
        return self.lora_rank > 0

    def load_hf_weights(self, weights: Dict[str, torch.Tensor], swap: bool = False):
        """Load LoRA weights from HuggingFace format weights dict.

        Args:
            weights: Dictionary of weight tensors from safetensors
            swap: If True, keep weights on CPU (pinned memory)
        """
        if not self.has_any_lora:
            return  # No LoRA to load

        layer_prefix = f"model.layers.{self.layer_num_}"

        # ========== Load moe_gate LoRA weights ==========
        gate_a_key = f"{layer_prefix}.mlp.gate.lora_A.weight"
        gate_b_key = f"{layer_prefix}.mlp.gate.lora_B.weight"
        if gate_a_key in weights:
            self._load_weight(gate_a_key, weights, "gate_lora_A", swap)
        if gate_b_key in weights:
            self._load_weight(gate_b_key, weights, "gate_lora_B", swap)

        # ========== Load w1 LoRA (fused gate+up) ==========
        w1_a_key = f"{layer_prefix}.mlp.experts.w1.lora_A.weight"
        w1_b_key = f"{layer_prefix}.mlp.experts.w1.lora_B.weight"
        if w1_a_key in weights:
            self._load_weight(w1_a_key, weights, "w1_lora_A", swap)
        if w1_b_key in weights:
            self._load_weight(w1_b_key, weights, "w1_lora_B", swap)

        # ========== Load w2 LoRA (down projection) ==========
        w2_a_key = f"{layer_prefix}.mlp.experts.w2.lora_A.weight"
        w2_b_key = f"{layer_prefix}.mlp.experts.w2.lora_B.weight"
        if w2_a_key in weights:
            self._load_weight(w2_a_key, weights, "w2_lora_A", swap)
        if w2_b_key in weights:
            self._load_weight(w2_b_key, weights, "w2_lora_B", swap)

        self.is_loaded_ = True
        self.is_on_gpu_ = not swap

    def _load_weight(
        self,
        key: str,
        weights: Dict[str, torch.Tensor],
        attr_name: str,
        swap: bool
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
        if self.is_on_gpu_ or not self.has_any_lora:
            return

        # MoE Gate LoRA
        for attr in ["gate_lora_A", "gate_lora_B"]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)

        # w1 LoRA
        for attr in ["w1_lora_A", "w1_lora_B"]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)

        # w2 LoRA
        for attr in ["w2_lora_A", "w2_lora_B"]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)

        self.is_on_gpu_ = True

    def offload_from_gpu(self):
        """Offload weights from GPU to CPU."""
        if not self.is_on_gpu_ or not self.has_any_lora:
            return

        # MoE Gate LoRA
        for attr in ["gate_lora_A", "gate_lora_B"]:
            gpu_attr = getattr(self, attr, None)
            if gpu_attr is not None:
                setattr(self, f"{attr}_home", gpu_attr.cpu().pin_memory())
                setattr(self, attr, None)

        # w1 LoRA
        for attr in ["w1_lora_A", "w1_lora_B"]:
            gpu_attr = getattr(self, attr, None)
            if gpu_attr is not None:
                setattr(self, f"{attr}_home", gpu_attr.cpu().pin_memory())
                setattr(self, attr, None)

        # w2 LoRA
        for attr in ["w2_lora_A", "w2_lora_B"]:
            gpu_attr = getattr(self, attr, None)
            if gpu_attr is not None:
                setattr(self, f"{attr}_home", gpu_attr.cpu().pin_memory())
                setattr(self, attr, None)

        self.is_on_gpu_ = False

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get all LoRA weights on GPU."""
        result = {}

        # MoE Gate LoRA
        for attr in ["gate_lora_A", "gate_lora_B"]:
            w = getattr(self, attr, None)
            if w is not None:
                result[attr] = w

        # w1 LoRA
        for attr in ["w1_lora_A", "w1_lora_B"]:
            w = getattr(self, attr, None)
            if w is not None:
                result[attr] = w

        # w2 LoRA
        for attr in ["w2_lora_A", "w2_lora_B"]:
            w = getattr(self, attr, None)
            if w is not None:
                result[attr] = w

        return result

    def verify(self) -> bool:
        """Verify all weights are loaded."""
        if not self.has_any_lora:
            return True

        all_attrs = [
            "gate_lora_A", "gate_lora_B",
            "w1_lora_A", "w1_lora_B",
            "w2_lora_A", "w2_lora_B"
        ]

        for attr in all_attrs:
            w_gpu = getattr(self, attr, None)
            w_cpu = getattr(self, f"{attr}_home", None)
            if w_gpu is None and w_cpu is None:
                print(f"Warning: {attr} not loaded for layer {self.layer_num_}")
                return False
        return True


class Qwen3MOELoRAAdapter:
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
        self.lora_rank = network_config.get("lora_rank", 64)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.has_lora = self.lora_rank > 0

        # Create layer weights
        self.num_layers = network_config["num_hidden_layers"]
        self.layers = [
            Qwen3MOELoRALayerWeight(i, network_config, data_type, device)
            for i in range(self.num_layers)
        ]

        # Load weights
        self._load_weights()

    def _load_weights(self):
        """Load LoRA weights from adapter directory."""
        if not self.has_lora:
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
) -> Qwen3MOELoRAAdapter:
    """Factory function to load a MoE LoRA adapter.

    Args:
        adapter_dir: Path to LoRA adapter directory
        network_config: Model network configuration
        data_type: Data type for weights
        device: Device to load weights on
        swap: If True, keep weights on CPU (for memory efficiency)

    Returns:
        Qwen3MOELoRAAdapter instance
    """
    return Qwen3MOELoRAAdapter(adapter_dir, network_config, data_type, device, swap)
