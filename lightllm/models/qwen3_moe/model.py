import torch
from typing import final, Optional, Any
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager


logger = init_logger(__name__)


@ModelRegistry("qwen3_moe")
class Qwen3MOEModel(Qwen3TpPartModel):
    # weight class
    transformer_weight_class = Qwen3MOETransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3MOETransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        # LoRA dispatcher for MoE layers (set externally for detached mode)
        self.lora_dispatcher_: Optional[Any] = None
        self.use_detached_lora_: bool = False
        return

    def _init_custom(self):
        super()._init_custom()
        dist_group_manager.new_deepep_group(self.config["num_experts"], self.config["hidden_size"])

    def set_lora_dispatcher(self, dispatcher: Any, use_detached_lora: bool = True):
        """Set the LoRA dispatcher for MoE layers.

        Args:
            dispatcher: LoRA dispatcher instance (Qwen3MOELoRADispatcher)
            use_detached_lora: If True, LoRA runs in detached mode (parallel with base)
        """
        self.lora_dispatcher_ = dispatcher
        self.use_detached_lora_ = use_detached_lora

        # Propagate to all layer infers
        for layer_infer in self.layer_infers:
            if hasattr(layer_infer, 'set_lora_dispatcher'):
                layer_infer.set_lora_dispatcher(dispatcher, use_detached_lora)

    def clear_lora_dispatcher(self):
        """Clear the LoRA dispatcher."""
        self.lora_dispatcher_ = None
        self.use_detached_lora_ = False

        # Clear in all layer infers
        for layer_infer in self.layer_infers:
            if hasattr(layer_infer, 'clear_lora_dispatcher'):
                layer_infer.clear_lora_dispatcher()
