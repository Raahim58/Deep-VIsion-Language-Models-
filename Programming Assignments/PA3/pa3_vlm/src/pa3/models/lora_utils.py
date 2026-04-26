from peft import LoraConfig, TaskType, get_peft_model


def apply_lora(model, r=16, alpha=32, dropout=0.05):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    wrapped = get_peft_model(model, cfg)
    found = sorted({name.split(".")[-1] for name, _ in wrapped.named_modules() if name.split(".")[-1] in cfg.target_modules})
    print("LoRA target modules found:", found)
    wrapped.print_trainable_parameters()
    return wrapped

