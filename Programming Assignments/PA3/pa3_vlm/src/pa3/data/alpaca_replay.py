from datasets import load_dataset
import torch


def format_alpaca(ex):
    if ex.get("input"):
        return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"


def load_alpaca_texts(n=1000):
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{n}]")
    texts = [format_alpaca(ex) for ex in ds]
    print("Alpaca sample count:", len(texts))
    print("Alpaca sample:", texts[0][:300])
    return texts


def assert_no_visual_tokens(input_ids: torch.Tensor, visual_start_id: int, name="Alpaca batch"):
    bad = (input_ids >= visual_start_id).any().item()
    assert not bad, f"{name} contains visual token IDs >= {visual_start_id}"
    print(f"{name}: no visual tokens, shape={tuple(input_ids.shape)}")


def collate_alpaca(tokenizer, visual_start_id, max_length=256):
    def _collate(texts):
        enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        assert_no_visual_tokens(enc["input_ids"], visual_start_id)
        return enc
    return _collate

