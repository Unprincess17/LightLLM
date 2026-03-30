#!/usr/bin/env python3
"""
Generate dummy LoRA adapters for Qwen3-VL-30B-A3B.

The generator can emit one adapter or a numbered family of adapters using
different random seeds. This is intended for infrastructure validation and
cache-behavior experiments, not model quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import save_file


DEFAULT_OUTPUT_DIR = "/home/shufan/Qwen-VL-FT/work/lora_dummy"
DEFAULT_LORA_RANK = 16

# Qwen3-VL-30B-A3B architecture parameters.
VISION_HIDDEN_SIZE = 1152
VISION_INTERMEDIATE_SIZE = 4304
VISION_DEPTH = 27

LM_HIDDEN_SIZE = 2048
LM_NUM_ATTENTION_HEADS = 32
LM_ATTENTION_HEAD_DIM = 128
LM_INTERMEDIATE_SIZE = 768
LM_NUM_EXPERTS = 128
LM_DEPTH = 48
LM_NUM_KEY_VALUE_HEADS = 4
LM_VOCAB_SIZE = 151936


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy LoRA adapters for Qwen3-VL-30B-A3B")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for a single generated adapter.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Directory prefix for multi-adapter generation. Example: /path/lora_dummy_",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of adapters to generate.")
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting numeric suffix when --count > 1 and --output_prefix is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for single-adapter generation, or base seed when --count > 1.",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=1,
        help="Seed increment between generated adapters when --count > 1.",
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK, help="LoRA rank to generate.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha. Defaults to the chosen rank.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing adapter files instead of skipping them.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional JSON manifest path for all generated adapters.",
    )
    return parser.parse_args()


def _randn(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape) * 0.01


def generate_lora_weight_dict(lora_rank: int) -> Dict[str, torch.Tensor]:
    """Generate dummy LoRA weights for both vision and language model."""
    weight_dict: Dict[str, torch.Tensor] = {}

    for layer_idx in range(VISION_DEPTH):
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            a_key = f"model.visual.blocks.{layer_idx}.attn.{proj_name}.lora_A.weight"
            b_key = f"model.visual.blocks.{layer_idx}.attn.{proj_name}.lora_B.weight"
            weight_dict[a_key] = _randn((VISION_HIDDEN_SIZE, lora_rank))
            weight_dict[b_key] = _randn((lora_rank, VISION_HIDDEN_SIZE))

        a_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc1.lora_A.weight"
        b_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc1.lora_B.weight"
        weight_dict[a_key] = _randn((VISION_HIDDEN_SIZE, lora_rank))
        weight_dict[b_key] = _randn((lora_rank, VISION_INTERMEDIATE_SIZE))

        a_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc2.lora_A.weight"
        b_key = f"model.visual.blocks.{layer_idx}.mlp.linear_fc2.lora_B.weight"
        weight_dict[a_key] = _randn((VISION_INTERMEDIATE_SIZE, lora_rank))
        weight_dict[b_key] = _randn((lora_rank, VISION_HIDDEN_SIZE))

    merger_hidden = VISION_HIDDEN_SIZE * 4
    for merger_type in ("merger", "deepstack_merger_list.0", "deepstack_merger_list.1", "deepstack_merger_list.2"):
        a_key = f"model.visual.{merger_type}.linear_fc1.lora_A.weight"
        b_key = f"model.visual.{merger_type}.linear_fc1.lora_B.weight"
        weight_dict[a_key] = _randn((merger_hidden, lora_rank))
        weight_dict[b_key] = _randn((lora_rank, merger_hidden))

        a_key = f"model.visual.{merger_type}.linear_fc2.lora_A.weight"
        b_key = f"model.visual.{merger_type}.linear_fc2.lora_B.weight"
        weight_dict[a_key] = _randn((merger_hidden, lora_rank))
        weight_dict[b_key] = _randn((lora_rank, merger_hidden))

    lm_q_hidden = LM_NUM_ATTENTION_HEADS * LM_ATTENTION_HEAD_DIM
    lm_kv_hidden = LM_NUM_KEY_VALUE_HEADS * LM_ATTENTION_HEAD_DIM

    for layer_idx in range(LM_DEPTH):
        for proj_name, out_dim in (
            ("q_proj", lm_q_hidden),
            ("k_proj", lm_kv_hidden),
            ("v_proj", lm_kv_hidden),
            ("o_proj", LM_HIDDEN_SIZE),
        ):
            a_key = f"model.language_model.layers.{layer_idx}.self_attn.{proj_name}.lora_A.weight"
            b_key = f"model.language_model.layers.{layer_idx}.self_attn.{proj_name}.lora_B.weight"
            in_dim = LM_HIDDEN_SIZE if proj_name != "o_proj" else lm_q_hidden
            weight_dict[a_key] = _randn((in_dim, lora_rank))
            weight_dict[b_key] = _randn((lora_rank, out_dim))

        for expert_idx in range(LM_NUM_EXPERTS):
            a_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_A.weight"
            b_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.lora_B.weight"
            weight_dict[a_key] = _randn((LM_HIDDEN_SIZE, lora_rank))
            weight_dict[b_key] = _randn((lora_rank, LM_INTERMEDIATE_SIZE))

            a_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_A.weight"
            b_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.lora_B.weight"
            weight_dict[a_key] = _randn((LM_HIDDEN_SIZE, lora_rank))
            weight_dict[b_key] = _randn((lora_rank, LM_INTERMEDIATE_SIZE))

            a_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_A.weight"
            b_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.lora_B.weight"
            weight_dict[a_key] = _randn((LM_INTERMEDIATE_SIZE, lora_rank))
            weight_dict[b_key] = _randn((lora_rank, LM_HIDDEN_SIZE))

    weight_dict["model.language_model.lm_head.lora_A.weight"] = _randn((LM_HIDDEN_SIZE, lora_rank))
    weight_dict["model.language_model.lm_head.lora_B.weight"] = _randn((lora_rank, LM_VOCAB_SIZE))
    return weight_dict


def generate_adapter_config(lora_rank: int, lora_alpha: float) -> dict:
    return {
        "base_model_name_or_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "bias": "none",
        "fan_in_fan_out": False,
        "lora_alpha": float(lora_alpha),
        "lora_dropout": 0.0,
        "modules_to_save": [],
        "r": int(lora_rank),
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "linear_fc1",
            "linear_fc2",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        "task_type": "Multimodal",
        "use_dora": False,
        "use_raven": False,
    }


def generate_single_adapter(
    output_dir: Path,
    lora_rank: int,
    lora_alpha: float,
    seed: Optional[int],
    overwrite: bool,
) -> dict:
    model_path = output_dir / "adapter_model.safetensors"
    config_path = output_dir / "adapter_config.json"
    metadata_path = output_dir / "dummy_lora_generation_metadata.json"

    if model_path.exists() and config_path.exists() and not overwrite:
        print(f"Skipping existing adapter: {output_dir}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        return {
            "output_dir": str(output_dir),
            "seed": seed,
            "rank": int(existing_config.get("r", lora_rank)),
            "alpha": float(existing_config.get("lora_alpha", lora_alpha)),
            "skipped": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        torch.manual_seed(int(seed))

    print(f"Generating adapter at {output_dir} (rank={lora_rank}, alpha={lora_alpha}, seed={seed})")
    weight_dict = generate_lora_weight_dict(lora_rank)
    save_file(weight_dict, str(model_path))

    config = generate_adapter_config(lora_rank, lora_alpha)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    total_parameters = int(sum(tensor.numel() for tensor in weight_dict.values()))
    metadata = {
        "output_dir": str(output_dir),
        "seed": seed,
        "rank": int(lora_rank),
        "alpha": float(lora_alpha),
        "total_parameters": total_parameters,
        "script": str(Path(__file__).resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved LoRA weights to: {model_path}")
    print(f"Saved adapter config to: {config_path}")
    print(f"Saved generation metadata to: {metadata_path}")
    print(f"Total parameters: {total_parameters:,}")
    return {**metadata, "skipped": False}


def build_output_dirs(args: argparse.Namespace) -> List[Path]:
    if args.count <= 1:
        return [Path(args.output_dir)]

    if args.output_prefix is None:
        raise ValueError("--output_prefix is required when --count > 1")

    return [Path(f"{args.output_prefix}{index}") for index in range(args.start_index, args.start_index + args.count)]


def build_seeds(args: argparse.Namespace) -> List[Optional[int]]:
    if args.seed is None:
        return [None] * args.count
    return [int(args.seed) + offset * int(args.seed_step) for offset in range(args.count)]


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.rank <= 0:
        raise ValueError("--rank must be positive")

    lora_alpha = float(args.alpha) if args.alpha is not None else float(args.rank)
    output_dirs = build_output_dirs(args)
    seeds = build_seeds(args)

    records = []
    for output_dir, seed in zip(output_dirs, seeds):
        record = generate_single_adapter(
            output_dir=output_dir,
            lora_rank=int(args.rank),
            lora_alpha=lora_alpha,
            seed=seed,
            overwrite=bool(args.overwrite),
        )
        records.append(record)

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")

    created = sum(1 for record in records if not record.get("skipped"))
    skipped = sum(1 for record in records if record.get("skipped"))
    print("\n=== Generation Summary ===")
    print(f"Requested adapters: {len(records)}")
    print(f"Created adapters: {created}")
    print(f"Skipped adapters: {skipped}")
    print(f"Rank: {args.rank}")
    print(f"Alpha: {lora_alpha}")
    if args.seed is not None:
        print(f"Seed base: {args.seed}")
        print(f"Seed step: {args.seed_step}")


if __name__ == "__main__":
    main()
