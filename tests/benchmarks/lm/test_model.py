from benchmarks.lm.model import extract_text_config, remap_llava_language_key


def test_extract_text_config_preserves_qwen_architecture():
    config = {"model_type": "llava_onevision", "text_config": {"model_type": "qwen2", "hidden_size": 896}}
    assert extract_text_config(config) == {"model_type": "qwen2", "hidden_size": 896}


def test_remap_llava_language_keys_to_qwen_causal_lm():
    assert remap_llava_language_key("language_model.model.layers.0.self_attn.q_proj.weight") == "model.layers.0.self_attn.q_proj.weight"
    assert remap_llava_language_key("language_model.lm_head.weight") == "lm_head.weight"
    assert remap_llava_language_key("vision_tower.vision_model.embeddings.weight") is None
