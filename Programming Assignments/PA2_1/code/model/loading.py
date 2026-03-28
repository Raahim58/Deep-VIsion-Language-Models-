from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from model.lora import attach_lora, build_lora_config
from model.value_model import ValueModel
from utils.memory import get_device, get_torch_dtype

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


def _load_error_message(model_name: str, exc: Exception) -> RuntimeError:
    token_present = bool(os.environ.get("HF_TOKEN"))
    message = (
        f"Failed to load model '{model_name}'. "
        f"HF_TOKEN present={token_present}. "
        "If this is a gated Llama checkpoint, accept the model license on Hugging Face "
        "and export HF_TOKEN before running."
    )
    error = RuntimeError(message)
    error.__cause__ = exc
    return error


def _maybe_quant_config(load_in_8bit: bool):
    if not load_in_8bit or BitsAndBytesConfig is None or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(load_in_8bit=True)


def _resolve_source(source: str | None, fallback: str) -> tuple[str, str | None]:
    if not source:
        return fallback, None
    path = Path(source)
    if path.exists() and (path / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(str(path))
        return peft_config.base_model_name_or_path, str(path)
    return source, None


def load_policy_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_reward_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def _freeze_model(model) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_policy_model(config: dict[str, Any], checkpoint: str | None = None, trainable: bool = True):
    model_name = config["models"]["policy_name"]
    source = checkpoint or config["models"].get("policy_checkpoint") or model_name
    base_name, adapter_path = _resolve_source(source, model_name)
    dtype = get_torch_dtype(config["prefer_bf16"])
    device = get_device()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except Exception as exc:
        raise _load_error_message(base_name, exc)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif trainable:
        model = attach_lora(model, build_lora_config(config, TaskType.CAUSAL_LM))

    model.config.use_cache = False
    if trainable:
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        _freeze_model(model)

    if device.type == "cuda":
        model.to(device)
    return model


def load_reference_model(config: dict[str, Any], checkpoint: str | None = None):
    model_name = config["models"]["policy_name"]
    source = checkpoint or config["models"].get("sft_checkpoint") or model_name
    base_name, adapter_path = _resolve_source(source, model_name)
    dtype = get_torch_dtype(config["prefer_bf16"])
    quantization_config = _maybe_quant_config(config["use_8bit_frozen"])
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config is not None else None,
            attn_implementation="eager",
        )
    except Exception as exc:
        raise _load_error_message(base_name, exc)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.config.use_cache = False
    _freeze_model(model)
    if quantization_config is None and get_device().type == "cuda":
        model.to(get_device())
    return model


def _scalar_token_id(token_id):
    if isinstance(token_id, (list, tuple)):
        if not token_id:
            return None
        return int(token_id[0])
    return None if token_id is None else int(token_id)


def load_reward_model(config: dict[str, Any], checkpoint: str | None = None, trainable: bool = False):
    model_name = config["models"]["reward_name"]
    source = checkpoint or config["models"].get("rm_checkpoint") or model_name
    base_name, adapter_path = _resolve_source(source, model_name)
    dtype = get_torch_dtype(config["prefer_bf16"])
    quantization_config = _maybe_quant_config(config["use_8bit_frozen"] and not trainable)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=1,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config is not None else None,
            attn_implementation="eager",
        )
    except Exception as exc:
        raise _load_error_message(base_name, exc)

    pad_id = _scalar_token_id(model.config.pad_token_id)
    if pad_id is None:
        pad_id = _scalar_token_id(model.config.eos_token_id)
    if pad_id is None:
        raise ValueError(f"Could not resolve a scalar pad_token_id for reward model {base_name}")
    model.config.pad_token_id = pad_id

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    elif trainable and config["rm"].get("use_lora", True):
        model = attach_lora(
            model,
            build_lora_config(config, TaskType.SEQ_CLS, modules_to_save=["score"]),
        )
    if trainable:
        model.train()
    else:
        _freeze_model(model)
    if quantization_config is None and get_device().type == "cuda":
        model.to(get_device())
    return model


def load_value_model(config: dict[str, Any]):
    dtype = get_torch_dtype(config["prefer_bf16"])
    model = ValueModel(
        backbone_name=config["models"]["value_name"],
        torch_dtype=dtype,
        freeze_backbone=config["ppo"].get("freeze_value_backbone", True),
    )
    model.train()
    if get_device().type == "cuda":
        model.to(get_device())
    return model
