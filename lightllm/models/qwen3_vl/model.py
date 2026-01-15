import os
import json
from lightllm.common.build_utils import repair_config
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen3_vl.layer_weights.transformer_layer_weight import Qwen3VLTransformerLayerWeight
from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
from lightllm.models.qwen3.model import Qwen3TpPartModel


class QWen3VLTokenizer(QWen2VLTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, **kwargs):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.min_pixel = self.image_processor.size["shortest_edge"]
        self.max_pixel = self.image_processor.size["longest_edge"]
        self.patch_size = self.image_processor.patch_size
        self.merge_size = self.image_processor.merge_size
        self.image_start_id = kwargs["model_cfg"]["vision_start_token_id"]
        self.image_end_id = kwargs["model_cfg"]["vision_end_token_id"]
        self.image_token_id = kwargs["model_cfg"]["image_token_id"]


@ModelRegistry(["qwen3_vl"], is_multimodal=True)
class Qwen3VLTpPartModel(Qwen3TpPartModel):

    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen3VLTransformerLayerInfer

    pre_and_post_weight_class = Qwen3VLPreAndPostLayerWeight
    transformer_weight_class = Qwen3VLTransformerLayerWeight

    infer_state_class = Qwen3VLInferStateInfo

    def __init__(self, kvargs):
        # LoRA support is handled in base_backend.py's init_batched_lora_adapters()
        # which creates dispatchers and passes them to layers via set_lora_dispatcher_to_layers()
        super().__init__(kvargs)
        return

    def _init_inferstate_cls(self):
        pass

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            all_config = json.load(json_file)
            self.config = all_config["text_config"]

        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

        # Add LoRA configuration from network args
        # These can be passed via network_config when using LoRA adapters
        if "lora_rank" in all_config:
            self.config["lora_rank"] = all_config["lora_rank"]
        if "lora_alpha" in all_config:
            self.config["lora_alpha"] = all_config["lora_alpha"]
        if "lora_dropout" in all_config:
            self.config["lora_dropout"] = all_config["lora_dropout"]

        return
