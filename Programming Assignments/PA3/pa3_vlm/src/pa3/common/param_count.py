def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def trainable_parameter_names(model, max_names=200):
    return [n for n, p in model.named_parameters() if p.requires_grad][:max_names]


def print_trainable_report(model, title="model"):
    total = count_parameters(model, False)
    trainable = count_parameters(model, True)
    pct = 100 * trainable / max(total, 1)
    names = trainable_parameter_names(model)
    print(f"{title}: total_params={total:,} trainable_params={trainable:,} trainable_pct={pct:.4f}%")
    print("Trainable parameter groups:")
    for n in names:
        print("  ", n)
    return {"total_params": total, "trainable_params": trainable, "trainable_pct": pct}

