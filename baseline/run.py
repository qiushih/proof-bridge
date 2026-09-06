"""Run exactly one greedy completion for each of 12 examples in two conditions."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import time

from baseline.dataset import (
    DEV_MANIFEST, DEV_PAIRS, EVAL_REPORT, EVAL_SOURCE, ROOT,
    check_evaluation_evidence, digest, write_json,
)
from baseline.protocol import CONDITIONS, PROMPT_FILE, build_messages, evaluate_candidate
from baseline.setup_model import MODEL_DIR, MODEL_LOCK, check_model
from verifier import LOCK_FILE, sha256_file

CONFIG = ROOT / "baseline/config.json"
REQUIREMENTS = ROOT / "baseline/requirements.lock"


def runtime_versions():
    versions = {}
    for requirement in REQUIREMENTS.read_text().splitlines():
        if not requirement.strip() or requirement.startswith("#"):
            continue
        name, expected = requirement.split("==")
        installed = importlib.metadata.version(name)
        if installed != expected:
            raise ValueError(f"Inference dependency mismatch: {name} expected {expected}, found {installed}")
        versions[name] = installed
    return versions


def experiment_fingerprints():
    paths = [CONFIG, REQUIREMENTS, PROMPT_FILE, MODEL_LOCK, DEV_MANIFEST, DEV_PAIRS,
             EVAL_SOURCE, EVAL_REPORT, LOCK_FILE, ROOT / "verifier.py", ROOT / "baseline/run.py",
             ROOT / "baseline/protocol.py", ROOT / "baseline/dataset.py", ROOT / "scripts/seed_schema.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def run(output):
    if output.exists():
        raise ValueError("Output directory already exists; refusing to overwrite or repeat an attempt. Use a new directory for an explicit reproduction run.")
    examples = check_evaluation_evidence()
    config = json.loads(CONFIG.read_text())
    if (config["conditions"] != list(CONDITIONS) or config["attempts_per_example_condition"] != 1
            or config["repair_attempts"] != 0 or config["development_demonstrations"] != 0):
        raise ValueError("Baseline v1 requires exactly two zero-shot conditions, one attempt, and no repair")
    settings = config["generation"]
    if settings["do_sample"] or settings["num_beams"] != 1 or settings["device"] != "cpu":
        raise ValueError("Baseline v1 is greedy CPU inference")
    model_lock = check_model()
    versions = runtime_versions()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import torch

    torch.set_num_threads(settings["torch_threads"])
    torch.set_num_interop_threads(settings["torch_interop_threads"])
    torch.manual_seed(settings["seed"])
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation=settings["attention_implementation"],
    ).to("cpu").eval()
    model_load_seconds = time.perf_counter() - started
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=settings["max_new_tokens"], use_cache=settings["use_cache"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    manifest = {
        "schema_version": "baseline-run-0.1", "experiment_id": config["experiment_id"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "configuration": config,
        "fingerprints": experiment_fingerprints(), "model": model_lock,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "machine": platform.machine(), "packages": versions, "torch_build": torch.__config__.show(),
                    "model_load_seconds": model_load_seconds},
        "effective_generation_config": generation.to_dict(),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "model_device": str(model.device), "model_dtype": str(model.dtype),
        "evaluation_ids": [example["id"] for example in examples],
        "expected_attempts": 2 * len(examples), "training_performed": False,
        "argument_fidelity_policy": "Automatic structure proxy scored against the same reference in both conditions; optional assistant reviews are separate and hash-bound.",
    }
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    # One started event per call provides evidence of the generation budget.
    # No retries or resume path exists, including after an interrupted attempt.
    with (output / "results.jsonl").open("x", encoding="utf-8") as results, (output / "attempts.jsonl").open("x", encoding="utf-8") as journal:
        for example in examples:
            for condition in CONDITIONS:
                messages = build_messages(example, condition)
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                prompt_ids = inputs["input_ids"][0].tolist()
                key = f"{example['id']}:{condition}"
                journal.write(json.dumps({"key": key, "attempt": 1, "event": "started", "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                journal.flush()
                print(f"Generating {key} (one attempt)", flush=True)
                started = time.perf_counter()
                error = None
                new_ids = []
                try:
                    with torch.inference_mode():
                        generated = model.generate(**inputs, generation_config=generation)
                    new_ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
                    text = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                except Exception as exception:
                    # Account for an attempted generation failure; never retry.
                    error = {"type": type(exception).__name__, "message": str(exception)}
                    text = ""
                elapsed = time.perf_counter() - started
                record = {
                    "schema_version": "baseline-result-0.1", "key": key, "example_id": example["id"],
                    "condition": condition, "attempt": 1, "repair_attempts": 0,
                    "formal_statement": example["formal_statement"], "generation_family": example["generation_family"],
                    "prompt": {"messages": messages, "rendered_text": rendered, "sha256": digest(rendered),
                               "input_token_ids": prompt_ids},
                    "model": {"model_id": model_lock["model_id"], "revision": model_lock["revision"],
                              "lock_sha256": sha256_file(MODEL_LOCK)},
                    "generation_settings": settings,
                    "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(new_ids),
                              "total_tokens": len(prompt_ids) + len(new_ids)},
                    "generation_latency_seconds": elapsed, "completion_token_ids": new_ids,
                    "finish_reason": "generation_error" if error else ("eos" if new_ids and new_ids[-1] == tokenizer.eos_token_id else "max_new_tokens"),
                    "generation_error": error,
                } | evaluate_candidate(example, text, config["verifier_timeout_seconds"])
                results.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.flush()
                journal.write(json.dumps({"key": key, "attempt": 1, "event": "finished"}) + "\n")
                journal.flush()
                print(f"{key} {record['verification']['status']} {record['verification']['category']} ({len(new_ids)} tokens, {elapsed:.2f}s)", flush=True)
    write_json(output / "completion.json", {
        "status": "COMPLETE", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": len(examples) * 2,
        "files_sha256": {name: sha256_file(output / name) for name in ["manifest.json", "results.jsonl", "attempts.jsonl"]},
    })
    from baseline.report import build_report
    build_report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=lambda p: ROOT / p, default=ROOT / "results/baseline-v1")
    args = parser.parse_args()
    run(args.output)
