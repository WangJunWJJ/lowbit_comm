import json
from pathlib import Path


def extract_text_config(config: dict) -> dict:
    if "text_config" not in config:
        raise ValueError("LLaVA config does not contain text_config")
    return dict(config["text_config"])


def remap_llava_language_key(key: str) -> str | None:
    prefix = "language_model."
    return key[len(prefix):] if key.startswith(prefix) else None


def load_qwen2_text_model(model_path: str | Path, dtype=None):
    import torch
    from safetensors import safe_open
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model_path = Path(model_path)
    root_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config = Qwen2Config.from_dict(extract_text_config(root_config))
    model = Qwen2ForCausalLM(config)
    state = {}
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors files in {model_path}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as source:
            for key in source.keys():
                mapped = remap_llava_language_key(key)
                if mapped is not None:
                    state[mapped] = source.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    remaining = set(missing) - allowed_missing
    if remaining or unexpected:
        raise RuntimeError(f"text weight mismatch: missing={sorted(remaining)}, unexpected={unexpected}")
    if dtype is not None:
        model.to(dtype=dtype)
    return model
