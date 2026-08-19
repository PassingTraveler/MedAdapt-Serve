from __future__ import annotations

from typing import Any


def load_processor(model_id_or_path: str, trust_remote_code: bool = False):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)


def load_model(
    model_id_or_path: str,
    *,
    qlora: bool = False,
    torch_dtype: str = "bfloat16",
    device_map: str | dict | None = None,
    trust_remote_code: bool = False,
    text_only: bool = False,
):
    import torch
    import transformers

    dtype = getattr(torch, torch_dtype)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    if qlora:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    if text_only:
        # 只做文本时跳过视觉塔：去掉 config 中的 vision/video 字段后按纯文本模型加载，
        # 避免多模态 Auto 类把整个视觉编码器实例化进显存（README 1.3 的要求）。
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
            for key in ("vision_config", "video_config", "audio_config"):
                if key in getattr(config, "to_dict", lambda: {})():
                    delattr(config, key)
            cls = getattr(transformers, "AutoModelForCausalLM", None)
            if cls is not None:
                return cls.from_pretrained(model_id_or_path, config=config, **kwargs)
        except (ValueError, ImportError, OSError, KeyError, RuntimeError, TypeError, AttributeError) as exc:
            print(f"[hf_loader] text-only load failed ({exc}); falling back to default candidate chain", flush=True)

    candidates = [
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
    ]
    last_error = None
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_id_or_path, **kwargs)
        except (ValueError, ImportError, OSError, KeyError, RuntimeError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not load model {model_id_or_path}: {last_error}")


def render_messages(processor, messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def tokenize_messages(processor, messages: list[dict[str, str]], *, max_length: int = 4096) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_dict=True,
    )

