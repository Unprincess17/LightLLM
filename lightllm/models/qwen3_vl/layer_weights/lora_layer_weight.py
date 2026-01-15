"""
Qwen3-VL Detached LoRA Layer Weight

This module implements detached LoRA weight storage for Qwen3-VL.
LoRA weights are stored separately from base model weights and can be
loaded/unloaded dynamically for detached serving.

LoRA is applied to:
- MLP layers: gate_proj, up_proj, down_proj (6 matrices)
- Attention layers: q_proj, k_proj, v_proj, o_proj (8 matrices)

Total: 14 LoRA matrices per layer
"""
import torch
import os
from typing import Optional, Dict, Any
from safetensors import safe_open


class Qwen3VLLoRALayerWeight:
    """LoRA layer weights for Qwen3-VL MLP and Attention projections.

    Stores LoRA A and B matrices for:
    - MLP: gate_proj, up_proj, down_proj (6 matrices)
    - Attention: q_proj, k_proj, v_proj, o_proj (8 matrices)

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

        # LoRA rank from config
        self.lora_rank = network_config.get("lora_rank", 0)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.scaling = self.lora_alpha / self.lora_rank if self.lora_rank > 0 else 1.0

        # Hidden size per head for TP
        self.hidden_size = network_config["hidden_size"]
        self.tp_size = 1  # Single GPU for now
        self.split_hidden_size = self.hidden_size // self.tp_size

        # Initialize MLP LoRA matrices as None (lazy loading)
        self.gate_proj_A: Optional[torch.Tensor] = None
        self.gate_proj_B: Optional[torch.Tensor] = None
        self.up_proj_A: Optional[torch.Tensor] = None
        self.up_proj_B: Optional[torch.Tensor] = None
        self.down_proj_A: Optional[torch.Tensor] = None
        self.down_proj_B: Optional[torch.Tensor] = None

        # Initialize Attention LoRA matrices as None (lazy loading)
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

        # CPU storage for swap mode - Attention
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
        if self.lora_rank == 0:
            return  # No LoRA to load

        mlp_prefix = f"base_model.model.model.language_model.layers.{self.layer_num_}.mlp"
        attn_prefix = f"base_model.model.model.language_model.layers.{self.layer_num_}.self_attn"

        # Get TP slice indices
        tp_idx_start = 0
        tp_idx_end = self.split_hidden_size

        # ========== Load MLP LoRA weights ==========
        # Gate proj A, B
        gate_a_key = f"{mlp_prefix}.gate_proj.lora_A.weight"
        gate_b_key = f"{mlp_prefix}.gate_proj.lora_B.weight"
        if gate_a_key in weights:
            self._load_weight(gate_a_key, weights, "gate_proj_A", swap, tp_idx_start, tp_idx_end)
        if gate_b_key in weights:
            self._load_weight(gate_b_key, weights, "gate_proj_B", swap, tp_idx_start, tp_idx_end)

        # Up proj A, B
        up_a_key = f"{mlp_prefix}.up_proj.lora_A.weight"
        up_b_key = f"{mlp_prefix}.up_proj.lora_B.weight"
        if up_a_key in weights:
            self._load_weight(up_a_key, weights, "up_proj_A", swap, tp_idx_start, tp_idx_end)
        if up_b_key in weights:
            self._load_weight(up_b_key, weights, "up_proj_B", swap, tp_idx_start, tp_idx_end)

        # Down proj A, B
        down_a_key = f"{mlp_prefix}.down_proj.lora_A.weight"
        down_b_key = f"{mlp_prefix}.down_proj.lora_B.weight"
        if down_a_key in weights:
            self._load_weight(down_a_key, weights, "down_proj_A", swap, tp_idx_start, tp_idx_end)
        if down_b_key in weights:
            self._load_weight(down_b_key, weights, "down_proj_B", swap, tp_idx_start, tp_idx_end)

        # ========== Load Attention LoRA weights ==========
        # Q proj A, B
        q_a_key = f"{attn_prefix}.q_proj.lora_A.weight"
        q_b_key = f"{attn_prefix}.q_proj.lora_B.weight"
        if q_a_key in weights:
            self._load_weight(q_a_key, weights, "q_proj_A", swap, tp_idx_start, tp_idx_end)
        if q_b_key in weights:
            self._load_weight(q_b_key, weights, "q_proj_B", swap, tp_idx_start, tp_idx_end)

        # K proj A, B
        k_a_key = f"{attn_prefix}.k_proj.lora_A.weight"
        k_b_key = f"{attn_prefix}.k_proj.lora_B.weight"
        if k_a_key in weights:
            self._load_weight(k_a_key, weights, "k_proj_A", swap, tp_idx_start, tp_idx_end)
        if k_b_key in weights:
            self._load_weight(k_b_key, weights, "k_proj_B", swap, tp_idx_start, tp_idx_end)

        # V proj A, B
        v_a_key = f"{attn_prefix}.v_proj.lora_A.weight"
        v_b_key = f"{attn_prefix}.v_proj.lora_B.weight"
        if v_a_key in weights:
            self._load_weight(v_a_key, weights, "v_proj_A", swap, tp_idx_start, tp_idx_end)
        if v_b_key in weights:
            self._load_weight(v_b_key, weights, "v_proj_B", swap, tp_idx_start, tp_idx_end)

        # O proj A, B
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
        """Load a single LoRA weight matrix.

        Args:
            key: Weight key in the weights dict
            weights: Weights dictionary
            attr_name: Attribute name to store in self (ends with _A or _B)
            swap: If True, store on CPU
            tp_start, tp_end: TP slice indices
        """
        if key not in weights:
            return

        weight = weights[key]

        # Handle different weight shapes
        # A matrices: [rank, hidden] from safetensors
        # B matrices: [output_dim, rank] from safetensors
        if weight.dim() == 2:
            # No slicing needed - weights are already in final shape
            pass
        elif weight.dim() == 1:
            # Bias vector - skip
            return

        weight = weight.to(dtype=self.data_type_)

        # A matrices need to be transposed to [hidden, rank] for input @ A
        # Shape: [rank, hidden] -> [hidden, rank]
        if attr_name.endswith("_A"):
            weight = weight.transpose(0, 1).contiguous()
        # B matrices need to be transposed to [output_dim, rank] -> [rank, output_dim]
        # But we need them as [output_dim, rank] for B.t() in dispatcher
        # Actually, B is stored as [output_dim, rank], we want B.t() = [rank, output_dim]
        # So we keep B as-is and transpose in dispatcher

        if swap:
            # Store on CPU (pinned memory for fast GPU transfer)
            setattr(self, f"{attr_name}_home", weight.pin_memory())
            setattr(self, attr_name, None)
        else:
            # Store on GPU
            setattr(self, attr_name, weight.to(self.device_))

    def load_to_gpu(self, non_blocking: bool = True):
        """Load weights from CPU to GPU."""
        if self.is_on_gpu_ or self.lora_rank == 0:
            return

        transfer = (torch.tensor([]).to(self.device_, non_blocking=non_blocking).get_device() != -1)

        # MLP LoRA weights
        for attr in ["gate_proj_A", "gate_proj_B", "up_proj_A", "up_proj_B", "down_proj_A", "down_proj_B"]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)

        # Attention LoRA weights
        for attr in ["q_proj_A", "q_proj_B", "k_proj_A", "k_proj_B", "v_proj_A", "v_proj_B", "o_proj_A", "o_proj_B"]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)

        self.is_on_gpu_ = True

    def offload_from_gpu(self):
        """Offload weights from GPU to CPU."""
        if not self.is_on_gpu_ or self.lora_rank == 0:
            return

        # MLP LoRA weights
        for attr in ["gate_proj_A", "gate_proj_B", "up_proj_A", "up_proj_B", "down_proj_A", "down_proj_B"]:
            gpu_attr = getattr(self, attr, None)
            if gpu_attr is not None:
                setattr(self, f"{attr}_home", gpu_attr.cpu().pin_memory())
                setattr(self, attr, None)

        # Attention LoRA weights
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
        # Attention LoRA weights
        for attr in ["q_proj_A", "q_proj_B", "k_proj_A", "k_proj_B", "v_proj_A", "v_proj_B", "o_proj_A", "o_proj_B"]:
            w = getattr(self, attr, None)
            if w is not None:
                result[attr] = w
        return result

    def verify(self) -> bool:
        """Verify all weights are loaded."""
        if self.lora_rank == 0:
            return True

        # MLP LoRA weights
        for attr in ["gate_proj_A", "gate_proj_B", "up_proj_A", "up_proj_B", "down_proj_A", "down_proj_B"]:
            w_gpu = getattr(self, attr, None)
            w_cpu = getattr(self, f"{attr}_home", None)
            if w_gpu is None and w_cpu is None:
                print(f"Warning: {attr} not loaded for layer {self.layer_num_}")
                return False

        # Attention LoRA weights (optional - may not be present)
        for attr in ["q_proj_A", "q_proj_B", "k_proj_A", "k_proj_B", "v_proj_A", "v_proj_B", "o_proj_A", "o_proj_B"]:
            w_gpu = getattr(self, attr, None)
            w_cpu = getattr(self, f"{attr}_home", None)
            if w_gpu is None and w_cpu is None:
                print(f"Warning: {attr} not loaded for layer {self.layer_num_}")
                return False
        return True


class Qwen3VLLoRAAdapter:
    """Manages a complete LoRA adapter across all transformer layers.

    This class wraps all LoRA layer weights for an adapter and provides
    methods for loading/unloading to/from GPU.
    """

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
        self.lora_rank = network_config.get("lora_rank", 0)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.scaling = self.lora_alpha / self.lora_rank if self.lora_rank > 0 else 1.0

        # Create layer weights
        self.num_layers = network_config["num_hidden_layers"]
        self.layers = [
            Qwen3VLLoRALayerWeight(i, network_config, data_type, device)
            for i in range(self.num_layers)
        ]

        # Load weights
        self._load_weights()

    def _load_weights(self):
        """Load LoRA weights from adapter directory."""
        if self.lora_rank == 0:
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


def load_lora_adapter(
    adapter_dir: str,
    network_config: Dict[str, Any],
    data_type: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    swap: bool = False
) -> Qwen3VLLoRAAdapter:
    """Factory function to load a LoRA adapter.

    Args:
        adapter_dir: Path to LoRA adapter directory
        network_config: Model network configuration
        data_type: Data type for weights
        device: Device to load weights on
        swap: If True, keep weights on CPU (for memory efficiency)

    Returns:
        Qwen3VLLoRAAdapter instance
    """
    return Qwen3VLLoRAAdapter(adapter_dir, network_config, data_type, device, swap)
