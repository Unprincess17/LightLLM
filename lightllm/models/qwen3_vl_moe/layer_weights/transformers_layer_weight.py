import os
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight


class Qwen3VLMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    """Qwen3-VL-MoE transformer layer weight.

    This class extends Qwen3MOETransformerLayerWeight to handle the HuggingFace
    weight format for Qwen3-VL-MoE, which saves expert weights as fused tensors.

    Specifically, it converts:
        model.layers.X.mlp.experts.gate_up_proj  [E, H, 2I]
        model.layers.X.mlp.experts.down_proj     [E, I, H]
    Into per-expert weights:
        model.layers.X.mlp.experts.{e}.gate_proj.weight
        model.layers.X.mlp.experts.{e}.up_proj.weight
        model.layers.X.mlp.experts.{e}.down_proj.weight

    Note: LoRA adapters are managed separately via LoRAManager/LoRAMemPool,
    not embedded in layer weights. See lightllm/server/lora/ for details.
    """

    def load_hf_weights(self, weights):
        """Load HuggingFace weights, splitting fused expert tensors."""
        moe_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        gate_up_name = f"{moe_prefix}.gate_up_proj"
        down_name = f"{moe_prefix}.down_proj"

        if gate_up_name in weights:
            gate_up = weights[gate_up_name]  # [E, H, 2I]
            E, H, twoI = gate_up.shape
            assert twoI % 2 == 0, f"gate_up_proj last dim must be even, got {twoI}"
            I_dim = twoI // 2

            down = weights.get(down_name)  # [E, I, H] or None

            for e in range(E):
                gate_up_e = gate_up[e]
                gate_e = gate_up_e[:, :I_dim].transpose(0, 1).contiguous()
                up_e = gate_up_e[:, I_dim:].transpose(0, 1).contiguous()

                weights[f"{moe_prefix}.{e}.gate_proj.weight"] = gate_e
                weights[f"{moe_prefix}.{e}.up_proj.weight"] = up_e

                if down is not None:
                    weights[f"{moe_prefix}.{e}.down_proj.weight"] = down[e].transpose(0, 1).contiguous()

            del weights[gate_up_name]
            if down_name in weights:
                del weights[down_name]

        super().load_hf_weights(weights)
