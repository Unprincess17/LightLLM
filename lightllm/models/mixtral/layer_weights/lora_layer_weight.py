"""
Mixtral Detached LoRA Layer Weight.

LoRA is applied to three points in MoE:
1. moe_gate - Router/logits modification
2. w1 - Gate projection (Mixtral naming: w1=gate_proj)
3. w3 - Up projection (Mixtral naming: w3=up_proj)
4. w2 - Down projection (Mixtral naming: w2=down_proj)

Mixtral weight naming convention:
  model.layers.{N}.block_sparse_moe.gate.*
  model.layers.{N}.block_sparse_moe.experts.{E}.w1.*
  model.layers.{N}.block_sparse_moe.experts.{E}.w2.*
  model.layers.{N}.block_sparse_moe.experts.{E}.w3.*
"""

import torch
import os
import glob
from typing import Optional, Dict, Any
from safetensors import safe_open


class MixtralLoRALayerWeight:
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

        self.lora_rank = network_config.get("lora_rank", 64)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.scaling = self.lora_alpha / self.lora_rank if self.lora_rank > 0 else 1.0

        self.hidden_size = network_config["hidden_size"]
        self.moe_intermediate_size = network_config.get("moe_intermediate_size", 0)
        self.num_experts = network_config.get("num_experts", 0)

        self.gate_lora_A: Optional[torch.Tensor] = None
        self.gate_lora_B: Optional[torch.Tensor] = None
        self.w1_lora_A: Optional[torch.Tensor] = None
        self.w1_lora_B: Optional[torch.Tensor] = None
        self.w3_lora_A: Optional[torch.Tensor] = None
        self.w3_lora_B: Optional[torch.Tensor] = None
        self.w2_lora_A: Optional[torch.Tensor] = None
        self.w2_lora_B: Optional[torch.Tensor] = None

        self.gate_lora_A_home: Optional[torch.Tensor] = None
        self.gate_lora_B_home: Optional[torch.Tensor] = None
        self.w1_lora_A_home: Optional[torch.Tensor] = None
        self.w1_lora_B_home: Optional[torch.Tensor] = None
        self.w3_lora_A_home: Optional[torch.Tensor] = None
        self.w3_lora_B_home: Optional[torch.Tensor] = None
        self.w2_lora_A_home: Optional[torch.Tensor] = None
        self.w2_lora_B_home: Optional[torch.Tensor] = None

        self.is_loaded_ = False
        self.is_on_gpu_ = False

    @property
    def has_moe_gate_lora(self) -> bool:
        return self.lora_rank > 0

    @property
    def has_w1_lora(self) -> bool:
        return self.lora_rank > 0

    @property
    def has_w3_lora(self) -> bool:
        return self.lora_rank > 0

    @property
    def has_w2_lora(self) -> bool:
        return self.lora_rank > 0

    @property
    def has_any_lora(self) -> bool:
        return self.lora_rank > 0

    def load_hf_weights(self, weights: Dict[str, torch.Tensor], swap: bool = False):
        if not self.has_any_lora:
            return

        layer_prefix = f"model.layers.{self.layer_num_}"

        gate_a_key = f"{layer_prefix}.block_sparse_moe.gate.lora_A.weight"
        gate_b_key = f"{layer_prefix}.block_sparse_moe.gate.lora_B.weight"
        if gate_a_key in weights:
            self._load_weight(gate_a_key, weights, "gate_lora_A", swap)
        if gate_b_key in weights:
            self._load_weight(gate_b_key, weights, "gate_lora_B", swap)

        w1_a_key = f"{layer_prefix}.block_sparse_moe.experts.w1.lora_A.weight"
        w1_b_key = f"{layer_prefix}.block_sparse_moe.experts.w1.lora_B.weight"
        if w1_a_key in weights:
            self._load_weight(w1_a_key, weights, "w1_lora_A", swap)
        if w1_b_key in weights:
            self._load_weight(w1_b_key, weights, "w1_lora_B", swap)

        w3_a_key = f"{layer_prefix}.block_sparse_moe.experts.w3.lora_A.weight"
        w3_b_key = f"{layer_prefix}.block_sparse_moe.experts.w3.lora_B.weight"
        if w3_a_key in weights:
            self._load_weight(w3_a_key, weights, "w3_lora_A", swap)
        if w3_b_key in weights:
            self._load_weight(w3_b_key, weights, "w3_lora_B", swap)

        w2_a_key = f"{layer_prefix}.block_sparse_moe.experts.w2.lora_A.weight"
        w2_b_key = f"{layer_prefix}.block_sparse_moe.experts.w2.lora_B.weight"
        if w2_a_key in weights:
            self._load_weight(w2_a_key, weights, "w2_lora_A", swap)
        if w2_b_key in weights:
            self._load_weight(w2_b_key, weights, "w2_lora_B", swap)

        self.is_loaded_ = True
        self.is_on_gpu_ = not swap

    def _load_weight(self, key, weights, attr_name, swap):
        if key not in weights:
            return
        weight = weights[key]
        if weight.dim() == 1:
            return
        weight = weight.to(dtype=self.data_type_)
        if attr_name.endswith("_A"):
            weight = weight.transpose(0, 1).contiguous()
        if swap:
            setattr(self, f"{attr_name}_home", weight.pin_memory())
            setattr(self, attr_name, None)
        else:
            setattr(self, attr_name, weight.to(self.device_))

    def load_to_gpu(self, non_blocking: bool = True):
        if self.is_on_gpu_ or not self.has_any_lora:
            return
        for attr in [
            "gate_lora_A", "gate_lora_B",
            "w1_lora_A", "w1_lora_B",
            "w3_lora_A", "w3_lora_B",
            "w2_lora_A", "w2_lora_B",
        ]:
            home_attr = f"{attr}_home"
            home = getattr(self, home_attr, None)
            if home is not None:
                setattr(self, attr, home.to(self.device_, non_blocking=non_blocking))
                setattr(self, home_attr, None)
        self.is_on_gpu_ = True

    def offload_from_gpu(self):
        if not self.is_on_gpu_ or not self.has_any_lora:
            return
        for attr in [
            "gate_lora_A", "gate_lora_B",
            "w1_lora_A", "w1_lora_B",
            "w3_lora_A", "w3_lora_B",
            "w2_lora_A", "w2_lora_B",
        ]:
            gpu_attr = getattr(self, attr, None)
            if gpu_attr is not None:
                setattr(self, f"{attr}_home", gpu_attr.cpu().pin_memory())
                setattr(self, attr, None)
        self.is_on_gpu_ = False

    def get_weights(self) -> Dict[str, torch.Tensor]:
        result = {}
        for attr in [
            "gate_lora_A", "gate_lora_B",
            "w1_lora_A", "w1_lora_B",
            "w3_lora_A", "w3_lora_B",
            "w2_lora_A", "w2_lora_B",
        ]:
            w = getattr(self, attr, None)
            if w is not None:
                result[attr] = w
        return result

    def verify(self) -> bool:
        if not self.has_any_lora:
            return True
        all_attrs = [
            "gate_lora_A", "gate_lora_B",
            "w1_lora_A", "w1_lora_B",
            "w3_lora_A", "w3_lora_B",
            "w2_lora_A", "w2_lora_B",
        ]
        for attr in all_attrs:
            w_gpu = getattr(self, attr, None)
            w_cpu = getattr(self, f"{attr}_home", None)
            if w_gpu is None and w_cpu is None:
                print(f"Warning: {attr} not loaded for layer {self.layer_num_}")
                return False
        return True


class MixtralLoRAAdapter:
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

        self.lora_rank = network_config.get("lora_rank", 64)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.has_lora = self.lora_rank > 0

        self.num_layers = network_config["num_hidden_layers"]
        self.layers = [
            MixtralLoRALayerWeight(i, network_config, data_type, device)
            for i in range(self.num_layers)
        ]

        self._load_weights()

    def _load_weights(self):
        if not self.has_lora:
            return
        safetensor_files = glob.glob(os.path.join(self.adapter_dir, "*.safetensors"))
        all_weights = {}
        for f in safetensor_files:
            with safe_open(f, "pt", "cpu") as sf:
                for k in sf.keys():
                    all_weights[k] = sf.get_tensor(k)
        for layer in self.layers:
            layer.load_hf_weights(all_weights, swap=self.swap)
        if not self.swap:
            self.load_to_gpu()

    def load_to_gpu(self, non_blocking: bool = True):
        for layer in self.layers:
            layer.load_to_gpu(non_blocking)

    def offload_from_gpu(self):
        for layer in self.layers:
            layer.offload_from_gpu()

    def is_on_gpu(self) -> bool:
        return self.layers[0].is_on_gpu_ if self.layers else False

    def get_layer_weights(self, layer_id: int) -> Dict[str, torch.Tensor]:
        if 0 <= layer_id < len(self.layers):
            return self.layers[layer_id].get_weights()
        return {}

    def verify(self) -> bool:
        for layer in self.layers:
            if not layer.verify():
                return False
        return True


def load_mixtral_lora_adapter(
    adapter_dir: str,
    network_config: Dict[str, Any],
    data_type: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    swap: bool = False
) -> MixtralLoRAAdapter:
    return MixtralLoRAAdapter(adapter_dir, network_config, data_type, device, swap)