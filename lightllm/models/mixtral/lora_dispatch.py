"""
LoRA Dispatch for Mixtral MoE Layers.

Reuses the model-agnostic Qwen3VLMoELoRADispatcher which operates on the
shared LoRA memory pool via BGMV kernels. The dispatch logic is independent
of model architecture; only weight naming and layer-infer integration differ.
"""

from lightllm.models.qwen3_vl_moe.lora_dispatch import (
    Qwen3VLMoELoRADispatcher,
    create_vl_moe_lora_dispatcher,
)


MixtralLoRADispatcher = Qwen3VLMoELoRADispatcher


def create_mixtral_lora_dispatcher(**kwargs):
    return create_vl_moe_lora_dispatcher(**kwargs)


def load_lora_adapter(adapter_dir, network_config, dtype=None, device=None):
    from lightllm.models.mixtral.layer_weights.lora_layer_weight import load_mixtral_lora_adapter
    return load_mixtral_lora_adapter(adapter_dir, network_config, dtype, device)