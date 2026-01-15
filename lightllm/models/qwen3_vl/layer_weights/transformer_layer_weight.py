import torch
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    NormWeight,
    ROWMMWeight,
)


class Qwen3VLTransformerLayerWeight(Qwen3TransformerLayerWeight):
    """Qwen3-VL transformer layer weight with LoRA support.

    This class extends the base Qwen3TransformerLayerWeight to support
    loading LoRA adapter weights for the text decoder.

    LoRA is applied to both MLP and Attention layers:
    - MLP: gate_proj, up_proj, down_proj
    - Attention: q_proj, k_proj, v_proj, o_proj
    """

    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        # Store LoRA configuration
        self.lora_rank = network_config.get("lora_rank", 0)
        self.lora_alpha = network_config.get("lora_alpha", 1.0)
        self.lora_dropout = network_config.get("lora_dropout", 0.0)

        # LoRA projection weights (for MLP)
        self.has_lora_mlp = self.lora_rank > 0
        if self.has_lora_mlp:
            self.gate_proj_a = None
            self.gate_proj_b = None
            self.up_proj_a = None
            self.up_proj_b = None
            self.down_proj_a = None
            self.down_proj_b = None

        # LoRA projection weights (for Attention)
        self.has_lora_attn = self.lora_rank > 0
        if self.has_lora_attn:
            self.q_proj_a = None
            self.q_proj_b = None
            self.k_proj_a = None
            self.k_proj_b = None
            self.v_proj_a = None
            self.v_proj_b = None
            self.o_proj_a = None
            self.o_proj_b = None

        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

        # Initialize LoRA weights after base weights are set up
        if self.has_lora_mlp or self.has_lora_attn:
            self._init_lora_weights()

    def _init_lora_weights(self):
        """Initialize LoRA projection weights for MLP and Attention layers."""
        layer_prefix = f"model.layers.{self.layer_num_}"

        # ========== MLP LoRA weights ==========
        if self.has_lora_mlp:
            # Gate proj LoRA
            self.gate_proj_a = ROWMMWeight(
                weight_names=f"{layer_prefix}.mlp.gate_proj.lora_A.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="gate_proj_lora_A",
            )
            self.gate_proj_b = ROWMMWeight(
                weight_names=f"{layer_prefix}.mlp.gate_proj.lora_B.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="gate_proj_lora_B",
            )

            # Up proj LoRA
            self.up_proj_a = ROWMMWeight(
                weight_names=f"{layer_prefix}.mlp.up_proj.lora_A.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="up_proj_lora_A",
            )
            self.up_proj_b = ROWMMWeight(
                weight_names=f"{layer_prefix}.mlp.up_proj.lora_B.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="up_proj_lora_B",
            )

            # Down proj LoRA
            self.down_proj_a = ROWMMWeight(
                weight_names=f"{layer_prefix}.mlp.down_proj.lora_A.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="down_proj_lora_A",
            )
            self.down_proj_b = ROWMMWeight(
                weight_names=f"{layer_prefix}.mlp.down_proj.lora_B.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="down_proj_lora_B",
            )

        # ========== Attention LoRA weights ==========
        if self.has_lora_attn:
            attn_prefix = f"{layer_prefix}.self_attn"

            # Q proj LoRA
            self.q_proj_a = ROWMMWeight(
                weight_names=f"{attn_prefix}.q_proj.lora_A.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="q_proj_lora_A",
            )
            self.q_proj_b = ROWMMWeight(
                weight_names=f"{attn_prefix}.q_proj.lora_B.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="q_proj_lora_B",
            )

            # K proj LoRA
            self.k_proj_a = ROWMMWeight(
                weight_names=f"{attn_prefix}.k_proj.lora_A.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="k_proj_lora_A",
            )
            self.k_proj_b = ROWMMWeight(
                weight_names=f"{attn_prefix}.k_proj.lora_B.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="k_proj_lora_B",
            )

            # V proj LoRA
            self.v_proj_a = ROWMMWeight(
                weight_names=f"{attn_prefix}.v_proj.lora_A.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="v_proj_lora_A",
            )
            self.v_proj_b = ROWMMWeight(
                weight_names=f"{attn_prefix}.v_proj.lora_B.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="v_proj_lora_B",
            )

            # O proj LoRA
            self.o_proj_a = ROWMMWeight(
                weight_names=f"{attn_prefix}.o_proj.lora_A.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="o_proj_lora_A",
            )
            self.o_proj_b = ROWMMWeight(
                weight_names=f"{attn_prefix}.o_proj.lora_B.weight",
                data_type=self.data_type_,
                layer_num=self.layer_num_,
                name="o_proj_lora_B",
            )

    def load_hf_weights(self, weights):
        """Load base model weights and LoRA weights.

        Args:
            weights: Dictionary of weight tensors from safetensors
        """
        # First, load base model weights
        super().load_hf_weights(weights)

        # Then load LoRA weights if present
        if not self.has_lora_mlp and not self.has_lora_attn:
            return

        # The LoRA weights are loaded by the ROWMMWeight objects
        # when their load_hf_weights method is called
        # But we need to ensure they are properly initialized
        lora_weights = {}
        for key in list(weights.keys()):
            if "lora_A" in key or "lora_B" in key:
                lora_weights[key] = weights.pop(key)

        if lora_weights:
            # Load MLP LoRA weights through the weight objects
            if self.has_lora_mlp:
                if self.gate_proj_a is not None:
                    self.gate_proj_a.load_hf_weights(lora_weights)
                if self.gate_proj_b is not None:
                    self.gate_proj_b.load_hf_weights(lora_weights)
                if self.up_proj_a is not None:
                    self.up_proj_a.load_hf_weights(lora_weights)
                if self.up_proj_b is not None:
                    self.up_proj_b.load_hf_weights(lora_weights)
                if self.down_proj_a is not None:
                    self.down_proj_a.load_hf_weights(lora_weights)
                if self.down_proj_b is not None:
                    self.down_proj_b.load_hf_weights(lora_weights)

            # Load Attention LoRA weights through the weight objects
            if self.has_lora_attn:
                if self.q_proj_a is not None:
                    self.q_proj_a.load_hf_weights(lora_weights)
                if self.q_proj_b is not None:
                    self.q_proj_b.load_hf_weights(lora_weights)
                if self.k_proj_a is not None:
                    self.k_proj_a.load_hf_weights(lora_weights)
                if self.k_proj_b is not None:
                    self.k_proj_b.load_hf_weights(lora_weights)
                if self.v_proj_a is not None:
                    self.v_proj_a.load_hf_weights(lora_weights)
                if self.v_proj_b is not None:
                    self.v_proj_b.load_hf_weights(lora_weights)
                if self.o_proj_a is not None:
                    self.o_proj_a.load_hf_weights(lora_weights)
                if self.o_proj_b is not None:
                    self.o_proj_b.load_hf_weights(lora_weights)

    def verify_load(self):
        """Verify all weights are loaded correctly."""
        super().verify_load()

        # Verify MLP LoRA weights
        if self.has_lora_mlp:
            assert self.gate_proj_a.weight is not None, f"gate_proj_a not loaded for layer {self.layer_num_}"
            assert self.gate_proj_b.weight is not None, f"gate_proj_b not loaded for layer {self.layer_num_}"
            assert self.up_proj_a.weight is not None, f"up_proj_a not loaded for layer {self.layer_num_}"
            assert self.up_proj_b.weight is not None, f"up_proj_b not loaded for layer {self.layer_num_}"
            assert self.down_proj_a.weight is not None, f"down_proj_a not loaded for layer {self.layer_num_}"
            assert self.down_proj_b.weight is not None, f"down_proj_b not loaded for layer {self.layer_num_}"

        # Verify Attention LoRA weights
        if self.has_lora_attn:
            assert self.q_proj_a.weight is not None, f"q_proj_a not loaded for layer {self.layer_num_}"
            assert self.q_proj_b.weight is not None, f"q_proj_b not loaded for layer {self.layer_num_}"
            assert self.k_proj_a.weight is not None, f"k_proj_a not loaded for layer {self.layer_num_}"
            assert self.k_proj_b.weight is not None, f"k_proj_b not loaded for layer {self.layer_num_}"
            assert self.v_proj_a.weight is not None, f"v_proj_a not loaded for layer {self.layer_num_}"
            assert self.v_proj_b.weight is not None, f"v_proj_b not loaded for layer {self.layer_num_}"
            assert self.o_proj_a.weight is not None, f"o_proj_a not loaded for layer {self.layer_num_}"
            assert self.o_proj_b.weight is not None, f"o_proj_b not loaded for layer {self.layer_num_}"
