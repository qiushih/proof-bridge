"""Actual deterministic update/epoch loop; the CLI only runs feasibility."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

from baseline.run import runtime_versions
from baseline.setup_model import MODEL_DIR, check_model
from pilot_v1.protocol import load_split
from prompt_v2.experiment import load_frozen
from training_protocol_v1.adaptation import LoRALinear, attach_for_future_training
from training_protocol_v1.data import config, completion_loss, encode_example, epoch_order


def seed_runtime():
    c = config()["training"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    random.seed(c["python_seed"])
    np.random.seed(c["numpy_seed"])
    torch.manual_seed(c["torch_seed"])
    torch.set_num_threads(c["torch_threads"])
    torch.set_num_interop_threads(c["torch_interop_threads"])
    torch.use_deterministic_algorithms(c["deterministic_algorithms"])


def tensor_digest(tensor):
    # CPU contiguous arrays are viewed, not copied. In particular, never clone
    # the large base embedding just to establish weight preservation.
    value = tensor.detach()
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("Hashing expects contiguous CPU tensors")
    h = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(memoryview(value.numpy()).cast("B"))
    return h.hexdigest()


def base_parameters(model):
    result = {}
    for name, parameter in model.named_parameters():
        if name.endswith((".A", ".B")):
            continue
        canonical = name.replace(".q_proj.base.", ".q_proj.").replace(".v_proj.base.", ".v_proj.")
        if canonical in result:
            raise ValueError("Duplicate canonical base parameter")
        result[canonical] = parameter
    return result


def base_hashes(model):
    return {name: tensor_digest(p) for name, p in base_parameters(model).items()}


def audit_trainable(model, strict=True):
    expected = {}
    wrappers = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            wrappers += 1
            expected[name + ".A"] = module.A
            expected[name + ".B"] = module.B
            if strict and not name.endswith(("self_attn.q_proj", "self_attn.v_proj")):
                raise ValueError("Adapter outside query/value projections")
    actual = {name:p for name,p in model.named_parameters() if p.requires_grad}
    if not expected or set(actual) != set(expected) or any(actual[k] is not p for k,p in expected.items()):
        raise ValueError("Only the explicit LoRA A/B tensors may be trainable")
    if any(p.requires_grad or p.grad is not None for p in base_parameters(model).values()):
        raise ValueError("Base model has a trainable parameter or a gradient")
    count = sum(p.numel() for p in actual.values())
    if strict and (wrappers != 48 or len(actual) != 96 or count != 540672):
        raise ValueError("Unexpected LoRA modules/parameter count")
    return actual, {"adapter_modules":wrappers, "trainable_tensors":len(actual), "trainable_parameters":count,
                    "base_trainable_parameters":0, "base_gradient_tensors":0}


def setup_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lock = check_model()
    versions = runtime_versions()
    seed_runtime()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.float32, attn_implementation="eager").to("cpu")
    load_seconds = time.perf_counter() - started
    before = base_hashes(model)
    # Model construction may consume RNG. Adapter initialization has its own
    # explicitly reset frozen seed, so it cannot depend on loading internals.
    torch.manual_seed(config()["training"]["torch_seed"])
    attach_for_future_training(model)
    model.train()
    _, trainable = audit_trainable(model)
    _, prompt = load_frozen()
    rows = load_split("train")
    if len(rows) != 24:
        raise ValueError("Expected the frozen 24 training rows")
    encoded = {row["id"]:encode_example(tokenizer,prompt,row) for row in rows}
    return model, tokenizer, encoded, before, {"model":lock, "packages":versions, "model_load_seconds":load_seconds,
                "trainable_audit":trainable, "training_rows":len(encoded),
                "gradient_checkpointing":model.is_gradient_checkpointing, "use_cache":model.config.use_cache,
                "device":str(model.device), "dtype":str(model.dtype)}


def make_optimizer(model, strict=True):
    named, audit = audit_trainable(model, strict)
    c = config()["training"]
    optimizer = torch.optim.AdamW(list(named.values()), lr=c["learning_rate"], betas=tuple(c["betas"]),
        eps=c["epsilon"], weight_decay=c["weight_decay"], foreach=c["foreach"], fused=c["fused"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _:1.0)
    optimizer.zero_grad(set_to_none=True)
    actual_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if actual_ids != {id(p) for p in named.values()}:
        raise ValueError("Optimizer contains a base weight or omits an adapter")
    return optimizer, scheduler


class RightPaddedForward(nn.Module):
    """Allocate the frozen maximum input length without altering text or targets.

    The unchanged loss function still indexes only the original proof positions.
    Added EOS padding has attention=0 and never reaches a supervised position.
    """
    def __init__(self, model, length, pad_id):
        super().__init__()
        self.model, self.length, self.pad_id = model, length, pad_id

    def forward(self, input_ids, attention_mask, **kwargs):
        padding = self.length - input_ids.shape[1]
        if padding < 0:
            raise ValueError("Input exceeds the padded length")
        ids = torch.nn.functional.pad(input_ids, (0,padding), value=self.pad_id)
        mask = torch.nn.functional.pad(attention_mask, (0,padding), value=0)
        return self.model(input_ids=ids, attention_mask=mask, **kwargs)


def optimizer_update(model, optimizer, scheduler, examples, emit, update, pad_to=None, pad_id=None,
                     strict=True, loss_fn=completion_loss):
    c = config()["training"]
    if len(examples) != c["gradient_accumulation_steps"]:
        raise ValueError("Each update needs exactly four batch-size-one microbatches")
    named, _ = audit_trainable(model, strict)
    old_adapters = {n:tensor_digest(p) for n,p in named.items()}
    start = time.perf_counter()
    losses, timings = [], []
    effective_model = RightPaddedForward(model,pad_to,pad_id) if pad_to else model
    for micro, example in enumerate(examples,1):
        emit("forward_started", update=update, microbatch=micro, example_id=example["id"],
             native_length=example["sequence_length"], tensor_length=pad_to or example["sequence_length"])
        t = time.perf_counter()
        loss = loss_fn(effective_model,example)
        forward_seconds = time.perf_counter()-t
        scalar = float(loss.detach())
        if not math.isfinite(scalar):
            raise FloatingPointError("Non-finite completion-only loss")
        emit("forward_finished", update=update, microbatch=micro, loss=scalar, forward_seconds=forward_seconds,
             finite_loss=True, target_tokens=example["target_length"])
        t = time.perf_counter()
        (loss / c["gradient_accumulation_steps"]).backward()
        backward_seconds = time.perf_counter()-t
        del loss
        audit_trainable(model, strict)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in named.values()):
            raise FloatingPointError("Missing or non-finite adapter gradient")
        emit("backward_finished", update=update, microbatch=micro, backward_seconds=backward_seconds,
             finite_gradients=True, base_gradient_tensors=0)
        losses.append(scalar)
        timings.append({"forward_seconds":forward_seconds,"backward_seconds":backward_seconds})
    grad_norm = float(torch.nn.utils.clip_grad_norm_(list(named.values()),c["max_grad_norm"],error_if_nonfinite=True))
    emit("optimizer_started",update=update,gradient_norm_before_clip=grad_norm)
    t = time.perf_counter()
    optimizer.step()
    scheduler.step()
    optimizer_seconds = time.perf_counter()-t
    optimizer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter()-start
    changed = sum(old_adapters[n] != tensor_digest(p) for n,p in named.items())
    if changed == 0:
        raise ValueError("Optimizer step did not change any adapter")
    _, audit = audit_trainable(model, strict)
    metrics = {"update":update,"microbatches":4,"example_ids":[e["id"] for e in examples],
               "native_lengths":[e["sequence_length"] for e in examples],"pad_to":pad_to,
               "mean_completion_loss":sum(losses)/4,"losses":losses,"microbatch_timings":timings,
               "gradient_norm_before_clip":grad_norm,"optimizer_seconds":optimizer_seconds,
               "update_wall_seconds":elapsed,"adapter_tensors_changed":changed,
               "learning_rate":optimizer.param_groups[0]["lr"],"trainable_audit":audit}
    emit("optimizer_finished",**metrics)
    return metrics


def save_checkpoint(model, optimizer, scheduler, output, step, epoch, order, fingerprints,
                    purpose, strict=True):
    if purpose != "full_pilot":
        raise ValueError("Feasibility adapter updates must never be saved as checkpoints")
    named, _ = audit_trainable(model, strict)
    path = Path(output) / f"step-{step:04d}"
    if path.exists():
        raise ValueError("Checkpoint exists; refusing replacement")
    path.mkdir(parents=True)
    save_file({name:p.detach().contiguous() for name,p in named.items()},str(path/"adapter.safetensors"))
    torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
                "optimizer_step":step,"epoch":epoch,"data_order":order,"fingerprints":fingerprints},path/"trainer_state.pt")
    (path/"checkpoint.json").write_text(json.dumps({"status":"COMPLETE","purpose":purpose,"step":step,
        "epoch":epoch,"adapter_parameters_only":True,"base_weights_saved":False,"fingerprints":fingerprints},indent=2)+"\n")
    return path


def train_pilot(output, on_checkpoint):
    """Full future loop; no CLI calls it in this feasibility task.

    The caller must supply the frozen six-dev-row assessment callback. Step zero
    reuses the saved baseline; trained checkpoint selection continues to use the
    unchanged training_protocol_v1.scoring.selection_score and bound reviews.
    """
    from lora_training_v1.preflight import check_inputs, fingerprints
    if not callable(on_checkpoint):
        raise ValueError("A frozen-protocol checkpoint evaluation callback is required")
    check_inputs()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    model, tokenizer, encoded, before, runtime = setup_model()
    optimizer,scheduler = make_optimizer(model)
    (output/"step-zero.json").write_text(json.dumps({"source":"results/pretraining-dev-v1", "optimizer_step":0})+"\n")
    update = 0
    with (output/"training_metrics.jsonl").open("x") as log:
        def emit(event,**values):
            log.write(json.dumps({"event":event,**values})+"\n"); log.flush()
        emit("training_started",runtime=runtime,fingerprints=fingerprints())
        for epoch in range(config()["training"]["epochs"]):
            order = epoch_order(epoch)
            for index in range(0,len(order),4):
                update += 1
                optimizer_update(model,optimizer,scheduler,[encoded[i] for i in order[index:index+4]],emit,update)
            if base_hashes(model) != before:
                raise ValueError("A base parameter changed")
            path = save_checkpoint(model,optimizer,scheduler,output,update,epoch,order,fingerprints(),"full_pilot")
            model.eval()
            on_checkpoint(model,tokenizer,path,update)
            model.train()
        if update != 60:
            raise ValueError("Frozen step budget differs")
        emit("training_completed",optimizer_steps=update,base_weights_unchanged=True)
    check_model()
    check_inputs()
    return output
