from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def build_lora_config(config: dict, task_type: TaskType, modules_to_save: list[str] | None = None) -> LoraConfig:
    return LoraConfig(
        task_type=task_type,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        modules_to_save=modules_to_save,
    )


def maybe_prepare_for_kbit(model):
    try:
        return prepare_model_for_kbit_training(model)
    except Exception:
        return model


def attach_lora(model, lora_config: LoraConfig, use_kbit_prep: bool = False):
    if use_kbit_prep:
        model = maybe_prepare_for_kbit(model)
    return get_peft_model(model, lora_config)
