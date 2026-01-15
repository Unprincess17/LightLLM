#!/usr/bin/env python3
"""
Generate a dummy LoRA for Qwen3-VL-30B-A3B model.
Creates random LoRA weights without training - for testing/infrastructure validation.

LoRA Coverage:
- Vision: MLP (linear_fc1, linear_fc2), Attention (q, k, v, o), Merger
- LM (MoE): MLP (gate, up, down), Attention (q, k, v, o), LM Head
"""

import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Configuration
OUTPUT_DIR = "/home/shufan/Qwen-VL-FT/work/lora_dummy"
LORA_RANK = 16
LORA_ALPHA = 32.0

# Qwen3-VL-30B-A3B architecture parameters
# Vision encoder
VISION_HIDDEN_SIZE = 1152
VISION_INTERMEDIATE_SIZE = 4304
VISION_NUM_HEADS = 16
VISION_DEPTH = 27

# Language model (MoE)
LM_HIDDEN_SIZE = 4096
LM_NUM_ATTENTION_HEADS = 32
LM_ATTENTION_HEAD_DIM = 128
LM_INTERMEDIATE_SIZE = 53248
LM_NUM_EXPERTS = 64
LM_NUM_ACTIVE_EXPERTS = 6
LM_DEPTH = 48
LM_NUM_KEY_VALUE_HEADS = 8  # GQA: 8 key-value heads, 32 query heads

def generate_lora_weight_dict():
    """Generate dummy LoRA weights for both vision and language model."""
    weight_dict = {}

    # ========== Vision Encoder LoRA ==========
    # Attention projection dimensions
    VISION_ATTN_QKV = VISION_HIDDEN_SIZE  # Q, K, V are concatenated then split
    VISION_ATTN_OUT = VISION_HIDDEN_SIZE

    for layer_idx in range(VISION_DEPTH):
        # ========== Attention LoRA (q, k, v, o) ==========
        # In Qwen-VL, attention uses qkv (combined Q, K, V) projection
        # qkv: hidden -> hidden * 3, then split into q, k, v
        lora_A_key = f"model.visual.blocks.{layer_idx}.attn.q_proj.lora_A.weight"
        lora_B_key = f"model.visual.blocks.{layer_idx}.attn.q_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(VISION_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, VISION_HIDDEN_SIZE) * 0.01

        lora_A_key = f"model.visual.blocks.{layer_idx}.attn.k_proj.lora_A.weight"
        lora_B_key = f"model.visual.blocks.{layer_idx}.attn.k_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(VISION_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, VISION_HIDDEN_SIZE) * 0.01

        lora_A_key = f"model.visual.blocks.{layer_idx}.attn.v_proj.lora_A.weight"
        lora_B_key = f"model.visual.blocks.{layer_idx}.attn.v_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(VISION_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, VISION_HIDDEN_SIZE) * 0.01

        lora_A_key = f"model.visual.blocks.{layer_idx}.attn.o_proj.lora_A.weight"
        lora_B_key = f"model.visual.blocks.{layer_idx}.attn.o_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(VISION_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, VISION_HIDDEN_SIZE) * 0.01

        # ========== MLP LoRA (linear_fc1, linear_fc2) ==========
        # linear_fc1: hidden -> intermediate
        lora_A_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc1.lora_A.weight"
        lora_B_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc1.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(VISION_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, VISION_INTERMEDIATE_SIZE) * 0.01

        # linear_fc2: intermediate -> hidden
        lora_A_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc2.lora_A.weight"
        lora_B_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc2.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(VISION_INTERMEDIATE_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, VISION_HIDDEN_SIZE) * 0.01

    # ========== Vision Merger LoRA ==========
    merger_hidden = VISION_HIDDEN_SIZE * 4  # spatial_merge_size=2, so 4
    for merger_type in ["merger", "deepstack_merger_list.0", "deepstack_merger_list.1", "deepstack_merger_list.2"]:
        lora_A_key = f"model.visual.{merger_type}.linear_fc1.lora_A.weight"
        lora_B_key = f"model.visual.{merger_type}.linear_fc1.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(merger_hidden, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, merger_hidden) * 0.01

        lora_A_key = f"model.visual.{merger_type}.linear_fc2.lora_A.weight"
        lora_B_key = f"model.visual.{merger_type}.linear_fc2.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(merger_hidden, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, merger_hidden) * 0.01

    # ========== Language Model LoRA (MoE) ==========
    # Attention dimensions for GQA
    LM_QHidden = LM_NUM_ATTENTION_HEADS * LM_ATTENTION_HEAD_DIM  # 32 * 128 = 4096
    LM_KVHidden = LM_NUM_KEY_VALUE_HEADS * LM_ATTENTION_HEAD_DIM  # 8 * 128 = 1024
    LM_OHidden = LM_HIDDEN_SIZE  # 4096

    for layer_idx in range(LM_DEPTH):
        # ========== Attention LoRA (q, k, v, o) ==========
        # Shared across all experts in MoE
        lora_A_key = f"model.language_model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"
        lora_B_key = f"model.language_model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(LM_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, LM_QHidden) * 0.01

        lora_A_key = f"model.language_model.layers.{layer_idx}.self_attn.k_proj.lora_A.weight"
        lora_B_key = f"model.language_model.layers.{layer_idx}.self_attn.k_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(LM_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, LM_KVHidden) * 0.01

        lora_A_key = f"model.language_model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight"
        lora_B_key = f"model.language_model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(LM_HIDDEN_SIZE, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, LM_KVHidden) * 0.01

        lora_A_key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.lora_A.weight"
        lora_B_key = f"model.language_model.layers.{layer_idx}.self_attn.o_proj.lora_B.weight"
        weight_dict[lora_A_key] = torch.randn(LM_QHidden, LORA_RANK) * 0.01
        weight_dict[lora_B_key] = torch.randn(LORA_RANK, LM_OHidden) * 0.01

        # ========== MoE MLP LoRA (gate, up, down for each expert) ==========
        for expert_idx in range(LM_NUM_EXPERTS):
            # gate_proj: hidden -> intermediate
            lora_A_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_A.weight"
            lora_B_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_B.weight"
            weight_dict[lora_A_key] = torch.randn(LM_HIDDEN_SIZE, LORA_RANK) * 0.01
            weight_dict[lora_B_key] = torch.randn(LORA_RANK, LM_INTERMEDIATE_SIZE) * 0.01

            # up_proj: hidden -> intermediate
            lora_A_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_A.weight"
            lora_B_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_B.weight"
            weight_dict[lora_A_key] = torch.randn(LM_HIDDEN_SIZE, LORA_RANK) * 0.01
            weight_dict[lora_B_key] = torch.randn(LORA_RANK, LM_INTERMEDIATE_SIZE) * 0.01

            # down_proj: intermediate -> hidden
            lora_A_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_A.weight"
            lora_B_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_B.weight"
            weight_dict[lora_A_key] = torch.randn(LM_INTERMEDIATE_SIZE, LORA_RANK) * 0.01
            weight_dict[lora_B_key] = torch.randn(LORA_RANK, LM_HIDDEN_SIZE) * 0.01

    # ========== LM Head LoRA ==========
    lora_A_key = "model.language_model.lm_head.lora_A.weight"
    lora_B_key = "model.language_model.lm_head.lora_B.weight"
    weight_dict[lora_A_key] = torch.randn(LM_HIDDEN_SIZE, LORA_RANK) * 0.01
    weight_dict[lora_B_key] = torch.randn(LORA_RANK, 151936) * 0.01  # vocab size

    return weight_dict


def generate_adapter_config():
    """Generate adapter_config.json for PEFT/transformers compatibility."""
    config = {
        "base_model_name_or_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "bias": "none",
        "fan_in_fan_out": False,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "modules_to_save": [],
        "r": LORA_RANK,
        "target_modules": [
            # Vision Attention
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # Vision MLP
            "linear_fc1",
            "linear_fc2",
            # Language Attention
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            # Language MoE MLP
            "gate_proj",
            "up_proj",
            "down_proj",
            # Output
            "lm_head",
        ],
        "task_type": " Multimodal",
        "use_dora": False,
        "use_raven": False,
    }
    return config


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate and save LoRA weights
    print("Generating dummy LoRA weights...")
    weight_dict = generate_lora_weight_dict()

    output_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    save_file(weight_dict, output_path)
    print(f"Saved LoRA weights to: {output_path}")
    print(f"Total parameters: {sum(v.numel() for v in weight_dict.values()):,}")

    # Save adapter config
    config = generate_adapter_config()
    config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved adapter config to: {config_path}")

    # Print summary
    print("\n=== Dummy LoRA Summary ===")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"\nVision Encoder ({VISION_DEPTH} layers):")
    print(f"  - Attention: q_proj, k_proj, v_proj, o_proj")
    print(f"  - MLP: linear_fc1, linear_fc2")
    print(f"  - Merger: merger, deepstack_merger_list (3x)")
    print(f"\nLanguage Model ({LM_DEPTH} layers, MoE with {LM_NUM_EXPERTS} experts):")
    print(f"  - Attention (shared): q_proj, k_proj, v_proj, o_proj")
    print(f"  - MLP per expert: gate_proj, up_proj, down_proj")
    print(f"  - LM Head: lm_head")
    print("\nNote: This is a dummy LoRA with random weights!")


if __name__ == "__main__":
    main()
