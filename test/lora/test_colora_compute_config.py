from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig


def test_colora_compute_config_parses_hybrid_moe_mode():
    cfg = LoRAComputeConfig.from_string(
        "attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:hybrid,vl:off"
    )

    assert cfg.should_compute_hybrid("moe")
    assert cfg.get_storage_device("moe") == "cpu"
    assert cfg.get_compute_device("moe") == "hybrid"
    assert cfg.get_compute_device("attn") == "gpu"
    assert cfg.get_compute_device("vl") == "off"
