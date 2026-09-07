"""Exact training serialization/loss definition, tested without Qwen training."""

import json
import random

from pilot_v1.protocol import ROOT, load_split
from prompt_v2.protocol import build_messages

CONFIG = ROOT / "training_protocol_v1/config.json"


def config():
    return json.loads(CONFIG.read_text())


def encode_example(tokenizer, prompt, row):
    messages = build_messages(prompt, row, "theorem_and_informal")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    completion = tokenizer.encode(row["proof_body"], add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prefix + completion
    if len(ids) > config()["training"]["max_sequence_length"]:
        raise ValueError("Sequence too long: truncation is forbidden")
    return {"id": row["id"], "input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + completion, "prompt_length": len(prefix),
            "target_length": len(completion), "sequence_length": len(ids)}


def epoch_order(epoch):
    if not 0 <= epoch < config()["training"]["epochs"]:
        raise ValueError("Epoch outside the frozen budget")
    ids = sorted(r["id"] for r in load_split("train"))
    random.Random(config()["training"]["seed"] + epoch).shuffle(ids)
    return ids


def completion_loss(model, encoded):
    """Single-example causal CE, computing only the supervised logit positions.

    Qwen2's pinned forward supports tensor `logits_to_keep`. Positions P-1..L-2
    predict tokens P..L-1 (proof plus EOS). Do not pass `labels` to the model as
    that would apply an additional shift. No optimizer or backward call here.
    """
    import torch
    import torch.nn.functional as functional
    p, length = encoded["prompt_length"], encoded["sequence_length"]
    labels = encoded["labels"]
    if p < 1 or length != len(labels) or labels[:p] != [-100] * p or labels[p:] != encoded["input_ids"][p:]:
        raise ValueError("Invalid completion-only mask")
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device="cpu")
    attention = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device="cpu")
    positions = torch.arange(p - 1, length - 1, dtype=torch.long, device="cpu")
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False, logits_to_keep=positions).logits
    targets = torch.tensor(labels[p:], dtype=torch.long, device="cpu")
    return functional.cross_entropy(logits[0].float(), targets, reduction="mean")
