"""
Qwen3-VL-MoE Detached LoRA Layer Weight

This module implements detached LoRA weight storage for Qwen3-VL-MoE.
LoRA weights are stored separately from base model weights and can be
loaded/unloaded dynamically for detached serving.

LoRA is applied to:
- MLP layers: gate_proj, up_proj, down_proj (MoE experts)
- Attention layers: q_proj, k_proj, v_proj, o_proj
- Vision adapter: vl.q_proj, vl.k_proj, vl.v_proj, vl.o_proj, vl.linear_fc1, vl.linear_fc2
"""
import json
import torch
import os
import re
from typing import Optional, Dict, Any
from safetensors import safe_open

# Define target type strings directly (matching lora_mem_pool.py)
TARGET_TYPE = {
    # LLM Attention
    "ATTN_Q_PROJ": "attn_q",
    "ATTN_K_PROJ": "attn_k",
    "ATTN_V_PROJ": "attn_v",
    "ATTN_O_PROJ": "attn_o",
    # LLM MoE
    "MOE_EXPERT_GATE": "moe_expert_gate",
    "MOE_EXPERT_UP": "moe_expert_up",
    "MOE_EXPERT_DOWN": "moe_expert_down",
    # LLM Head
    "LM_HEAD": "lm_head",
    # Vision
    "VL_Q_PROJ": "vl_q",
    "VL_K_PROJ": "vl_k",
    "VL_V_PROJ": "vl_v",
    "VL_O_PROJ": "vl_o",
    "VL_FC1": "vl_fc1",
    "VL_FC2": "vl_fc2",
}


class Qwen3VLMoELoRALayerWeight:
    """LoRA layer weights for Qwen3-VL-MoE.

    Stores LoRA A and B matrices for:
    - MLP: gate_proj, up_proj, down_proj (MoE experts)
    - Attention: q_proj, k_proj, v_proj, o_proj
    - Vision: vl.q_proj, vl.k_proj, vl.v_proj, vl.o_proj, vl.linear_fc1, vl.linear_fc2

    Uses regex-based key matching for flexible safetensor parsing.
    """

    def __init__(
        self,
        layer_num: int,
        network_config: Dict[str, Any],
        data_type: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        lora_alpha: float = 1.0,
        q_lora_rank: int = 0,
        k_lora_rank: int = 0,
        v_lora_rank: int = 0,
        o_lora_rank: int = 0,
        vl_lora_rank: int = 0,
    ):
        self.layer_num_ = layer_num
        self.network_config = network_config
        self.data_type_ = data_type
        self.device_ = device

        # LoRA config (passed from adapter, not parsed from network_config)
        self.lora_alpha = lora_alpha
        self.q_lora_rank = q_lora_rank
        self.k_lora_rank = k_lora_rank
        self.v_lora_rank = v_lora_rank
        self.o_lora_rank = o_lora_rank
        self.vl_lora_rank = vl_lora_rank

        # Hidden size
        self.hidden_size = network_config.get("hidden_size", 4096)
        self.tp_size = 1  # Single GPU for now
        self.split_hidden_size = self.hidden_size // self.tp_size

        # Storage for weights
        # Format: {target_type: {module_name: {"A": tensor, "B": tensor}}}
        self.weights: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]] = {}

        # State tracking
        self.is_loaded_ = False
        self.is_on_gpu_ = False

    def load_hf_weights(self, weights: Dict[str, torch.Tensor], swap: bool = False):
        """Load LoRA weights from HuggingFace format weights dict.

        Args:
            weights: Dictionary of weight tensors from safetensors
            swap: If True, keep weights on CPU (pinned memory)
        """
        # Pre-compiled regex patterns
        re_llm_layer = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")
        re_vis_block = re.compile(r"model\.visual\.blocks\.(\d+)\.(.+)")
        re_lm_head = re.compile(r"model\.language_model\.lm_head")
        re_expert = re.compile(r"experts\.(\d+)\.")

        # Get this layer number
        current_layer = self.layer_num_

        for key, tensor in weights.items():
            if "lora_A" not in key and "lora_B" not in key:
                continue

            matrix_type = "A" if "lora_A" in key else "B"
            target_type: Optional[str] = None
            module_name: Optional[str] = None

            # Check LLM layers
            match = re_llm_layer.search(key)
            if match and int(match.group(1)) == current_layer:
                suffix = match.group(2)

                # Attention: self_attn.q_proj, self_attn.k_proj, etc.
                if "self_attn.q_proj" in suffix:
                    if self.q_lora_rank <= 0:
                        continue
                    target_type = TARGET_TYPE["ATTN_Q_PROJ"]
                    module_name = "q_proj"
                elif "self_attn.k_proj" in suffix:
                    if self.k_lora_rank <= 0:
                        continue
                    target_type = TARGET_TYPE["ATTN_K_PROJ"]
                    module_name = "k_proj"
                elif "self_attn.v_proj" in suffix:
                    if self.v_lora_rank <= 0:
                        continue
                    target_type = TARGET_TYPE["ATTN_V_PROJ"]
                    module_name = "v_proj"
                elif "self_attn.o_proj" in suffix:
                    if self.o_lora_rank <= 0:
                        continue
                    target_type = TARGET_TYPE["ATTN_O_PROJ"]
                    module_name = "o_proj"

                # MoE MLP: mlp.gate_proj, mlp.experts.0.gate_proj, etc.
                elif "mlp.gate_proj" in suffix:
                    target_type = TARGET_TYPE["MOE_EXPERT_GATE"]
                    module_name = "gate_proj"
                elif "mlp.experts" in suffix:
                    # mlp.experts.0.gate_proj, mlp.experts.0.up_proj, etc.
                    expert_match = re_expert.search(suffix)
                    if expert_match:
                        # For MoE, we use shared weights across experts
                        if "gate_proj" in suffix:
                            target_type = TARGET_TYPE["MOE_EXPERT_GATE"]
                            module_name = "gate_proj"
                        elif "up_proj" in suffix:
                            target_type = TARGET_TYPE["MOE_EXPERT_UP"]
                            module_name = "up_proj"
                        elif "down_proj" in suffix:
                            target_type = TARGET_TYPE["MOE_EXPERT_DOWN"]
                            module_name = "down_proj"

                elif "mlp.up_proj" in suffix:
                    target_type = TARGET_TYPE["MOE_EXPERT_UP"]
                    module_name = "up_proj"
                elif "mlp.down_proj" in suffix:
                    target_type = TARGET_TYPE["MOE_EXPERT_DOWN"]
                    module_name = "down_proj"

            # Check Vision blocks (offset by 10000)
            vis_layer = current_layer - 10000
            if vis_layer >= 0:
                match = re_vis_block.search(key)
                if match and int(match.group(1)) == vis_layer:
                    suffix = match.group(2)
                    if self.vl_lora_rank <= 0:
                        continue

                    if "attn.q_proj" in suffix:
                        target_type = TARGET_TYPE["VL_Q_PROJ"]
                        module_name = "vl_q"
                    elif "attn.k_proj" in suffix:
                        target_type = TARGET_TYPE["VL_K_PROJ"]
                        module_name = "vl_k"
                    elif "attn.v_proj" in suffix:
                        target_type = TARGET_TYPE["VL_V_PROJ"]
                        module_name = "vl_v"
                    elif "attn.o_proj" in suffix:
                        target_type = TARGET_TYPE["VL_O_PROJ"]
                        module_name = "vl_o"
                    elif "mlp.linear_fc1" in suffix:
                        target_type = TARGET_TYPE["VL_FC1"]
                        module_name = "vl_fc1"
                    elif "mlp.linear_fc2" in suffix:
                        target_type = TARGET_TYPE["VL_FC2"]
                        module_name = "vl_fc2"

            # LM Head (layer -1)
            elif re_lm_head.search(key):
                target_type = TARGET_TYPE["LM_HEAD"]
                module_name = "lm_head"

            if target_type is None or module_name is None:
                continue

            # Convert tensor
            tensor = tensor.to(dtype=self.data_type_)

            # A matrices need transpose to [hidden, rank]
            if matrix_type == "A":
                if tensor.dim() == 2:
                    tensor = tensor.transpose(0, 1).contiguous()

            # TP slicing (if needed)
            tp_start = 0
            # TODO(FIX): the split hidden size is 2048 here. Is it correct?
            tp_end = self.split_hidden_size
            if tensor.shape[0] > tp_end:
                tensor = tensor[tp_start:tp_end]

            # Handle CPU swap
            if swap:
                tensor = tensor.pin_memory()

            # Store weight
            if target_type not in self.weights:
                self.weights[target_type] = {}
            if module_name not in self.weights[target_type]:
                self.weights[target_type][module_name] = {"A": None, "B": None}

            self.weights[target_type][module_name][matrix_type] = tensor

        self.is_loaded_ = True
        self.is_on_gpu_ = not swap

    def get_weights(self) -> Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """Get all LoRA weights.

        Returns:
            Dict mapping target_type -> module_name -> {"A": tensor, "B": tensor}
        """
        return self.weights

    def load_to_gpu(self, non_blocking: bool = True):
        """Load weights from CPU to GPU."""
        if self.is_on_gpu_:
            return

        for target_type in self.weights:
            for module_name in self.weights[target_type]:
                for matrix_type in ["A", "B"]:
                    tensor = self.weights[target_type][module_name].get(matrix_type)
                    if tensor is not None and tensor.device.type == "cpu":
                        self.weights[target_type][module_name][matrix_type] = tensor.to(
                            self.device_, non_blocking=non_blocking
                        )

        self.is_on_gpu_ = True

    def offload_from_gpu(self):
        """Offload weights from GPU to CPU."""
        if not self.is_on_gpu_:
            return

        for target_type in self.weights:
            for module_name in self.weights[target_type]:
                for matrix_type in ["A", "B"]:
                    tensor = self.weights[target_type][module_name].get(matrix_type)
                    if tensor is not None and tensor.device.type == "cuda":
                        self.weights[target_type][module_name][matrix_type] = tensor.cpu().pin_memory()

        self.is_on_gpu_ = False

    def verify(self) -> bool:
        """Verify all weights are loaded."""
        for target_type in self.weights:
            for module_name in self.weights[target_type]:
                if self.weights[target_type][module_name]["A"] is None:
                    return False
                if self.weights[target_type][module_name]["B"] is None:
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

        # Read adapter_config.json for LoRA configuration
        adapter_config = {}
        config_path = os.path.join(self.adapter_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                adapter_config = json.load(f)

        lora_r = adapter_config.get("r", 64)  # Default to 64
        assert lora_r > 0, "LoRA rank must be positive"

        self.lora_alpha = adapter_config.get("lora_alpha", 1.0)
        self.q_lora_rank = lora_r
        self.k_lora_rank = lora_r
        self.v_lora_rank = lora_r
        self.o_lora_rank = lora_r
        self.vl_lora_rank = lora_r
        self.max_rank = lora_r


        # Count layers (including vision layers at offset)
        num_llm_layers = network_config.get("num_hidden_layers", 0)
        vision_config = network_config.get("vision_config", {})
        num_vision_layers = vision_config.get("depth", network_config.get("vision_num_layers", 0))
        self.num_layers = num_llm_layers + num_vision_layers

        # Create layer weights for LLM layers
        self.llm_layers = [
            Qwen3VLMoELoRALayerWeight(
                i, network_config, data_type, device,
                lora_alpha=self.lora_alpha,
                q_lora_rank=self.q_lora_rank,
                k_lora_rank=self.k_lora_rank,
                v_lora_rank=self.v_lora_rank,
                o_lora_rank=self.o_lora_rank,
                vl_lora_rank=self.vl_lora_rank,
            )
            for i in range(num_llm_layers)
        ]

        # Create layer weights for Vision layers (offset by 10000)
        self.vision_layers = [
            Qwen3VLMoELoRALayerWeight(
                10000 + i, network_config, data_type, device,
                lora_alpha=self.lora_alpha,
                q_lora_rank=self.q_lora_rank,
                k_lora_rank=self.k_lora_rank,
                v_lora_rank=self.v_lora_rank,
                o_lora_rank=self.o_lora_rank,
                vl_lora_rank=self.vl_lora_rank,
            )
            for i in range(num_vision_layers)
        ]

        # All layers combined
        self.layers = self.llm_layers + self.vision_layers

        # Load weights
        self._load_weights()

    def _load_weights(self):
        """Load LoRA weights from adapter directory.

        Optimized to filter weights once and distribute to layers efficiently.
        """
        import glob

        safetensor_files = glob.glob(os.path.join(self.adapter_dir, "*.safetensors"))
        if not safetensor_files:
            raise ValueError(f"No safetensors found in {self.adapter_dir}")

        # Pre-compiled regex patterns for layer matching
        re_llm_layer = re.compile(r"model\.language_model\.layers\.(\d+)\.")
        re_vis_block = re.compile(r"model\.visual\.blocks\.(\d+)\.")
        re_lm_head = re.compile(r"model\.language_model\.lm_head")

        # Build layer index sets for O(1) lookup
        llm_layer_nums = {layer.layer_num_ for layer in self.llm_layers}
        vis_layer_nums = {layer.layer_num_ - 10000 for layer in self.vision_layers}

        # Pre-filter weights per layer to avoid repeated iteration
        # Dict[layer_num] -> Dict[key, tensor]
        llm_layer_weights: dict[int, dict[str, torch.Tensor]] = {i: {} for i in llm_layer_nums}
        vis_layer_weights: dict[int, dict[str, torch.Tensor]] = {i: {} for i in vis_layer_nums}
        lm_head_weights: dict[str, torch.Tensor] = {}

        for f in safetensor_files:
            with safe_open(f, "pt", "cpu") as sf:
                for k in sf.keys():
                    if "lora_A" not in k and "lora_B" not in k:
                        continue

                    tensor = sf.get_tensor(k)

                    # Check LM head first (no layer number)
                    if re_lm_head.search(k):
                        lm_head_weights[k] = tensor
                        continue

                    # Check LLM layers
                    llm_match = re_llm_layer.search(k)
                    if llm_match:
                        layer_num = int(llm_match.group(1))
                        if layer_num in llm_layer_nums:
                            llm_layer_weights[layer_num][k] = tensor
                            continue

                    # Check Vision layers
                    vis_match = re_vis_block.search(k)
                    if vis_match:
                        vis_layer = int(vis_match.group(1))
                        if vis_layer in vis_layer_nums:
                            vis_layer_weights[vis_layer][k] = tensor
                            continue

        # Distribute to LLM layers
        for layer in self.llm_layers:
            layer.load_hf_weights(llm_layer_weights[layer.layer_num_], swap=self.swap)

        # Distribute to Vision layers
        for layer in self.vision_layers:
            layer.load_hf_weights(vis_layer_weights[layer.layer_num_ - 10000], swap=self.swap)

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

    def get_layer_weights(self, layer_id: int) -> Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """Get LoRA weights for a specific layer.

        Args:
            layer_id: Layer index (0-N for LLM, 10000+ for vision)

        Returns:
            Dict mapping target_type -> module_name -> {"A": tensor, "B": tensor}
        """
        for layer in self.layers:
            if layer.layer_num_ == layer_id:
                return layer.get_weights()
        return {}

    def get_all_weights(self) -> Dict[int, Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]]]:
        """Get all weights across all layers.

        Returns:
            Dict mapping layer_id -> target_type -> module_name -> {"A": tensor, "B": tensor}
        """
        result = {}
        for layer in self.layers:
            weights = layer.get_weights()
            if weights:
                result[layer.layer_num_] = weights
        return result

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
