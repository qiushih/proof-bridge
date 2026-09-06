"""Run the preregistered development comparison, freeze its winner, then evaluate."""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import time

from baseline.dataset import ROOT, check_development, digest, write_json
from baseline.protocol import CONDITIONS
from baseline.run import runtime_versions
from baseline.setup_model import MODEL_DIR, MODEL_LOCK, check_model
from prompt_v2.protocol import (
    FROZEN, PLAN, SELECTED, SYSTEM, assess_output, build_messages, load_plan,
    make_prompt, metrics, rank_candidates, selection_partition,
)
from verifier import sha256_file

DEV_OUTPUT = ROOT / "results/prompt-v2-development"
EVAL_OUTPUT = ROOT / "results/baseline-v2"


def core_fingerprints():
    # Deliberately no evaluation records or v1 evaluation results here.
    paths = [PLAN, SYSTEM, ROOT / "prompt_v2/protocol.py", ROOT / "prompt_v2/experiment.py",
             ROOT / "baseline/config.json", MODEL_LOCK, ROOT / "baseline/requirements.lock",
             ROOT / "baseline/protocol.py", ROOT / "baseline/run.py", ROOT / "baseline/dataset.py",
             ROOT / "baseline/setup_model.py", ROOT / "scripts/seed_schema.py", ROOT / "verifier.py",
             ROOT / "environment.lock.json", ROOT / "data/development/pairs.jsonl", ROOT / "data/development/manifest.json"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


class Inference:
    def __init__(self, settings):
        self.lock = check_model()
        versions = runtime_versions()
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch = torch
        torch.set_num_threads(settings["torch_threads"])
        torch.set_num_interop_threads(settings["torch_interop_threads"])
        torch.manual_seed(settings["seed"])
        torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation=settings["attention_implementation"],
        ).to("cpu").eval()
        self.generation = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=settings["max_new_tokens"], use_cache=settings["use_cache"],
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        self.runtime = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
                        "packages": versions, "model_load_seconds": time.perf_counter() - started,
                        "model_parameter_count": sum(p.numel() for p in self.model.parameters()),
                        "model_dtype": str(self.model.dtype), "model_device": str(self.model.device),
                        "torch_build": torch.__config__.show()}

    def generate(self, messages):
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        prompt_ids = inputs["input_ids"][0].tolist()
        started = time.perf_counter()
        error, new_ids, text = None, [], ""
        try:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, generation_config=self.generation)
            new_ids = output[0, len(prompt_ids):].tolist()
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception as exception:
            error = {"type": type(exception).__name__, "message": str(exception)}
        return text, {
            "prompt": {"messages": messages, "rendered_text": rendered, "sha256": digest(rendered), "input_token_ids": prompt_ids},
            "completion_token_ids": new_ids,
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(new_ids), "total_tokens": len(prompt_ids) + len(new_ids)},
            "generation_latency_seconds": time.perf_counter() - started,
            "generation_error": error,
            "finish_reason": "generation_error" if error else "eos" if new_ids and new_ids[-1] == self.tokenizer.eos_token_id else "max_new_tokens",
        }


def phase_summary(manifest, rows):
    summary = {"phase": manifest["phase"], "total_generation_attempts": len(rows), "repair_attempts": 0, "fine_tuning_performed": False, "prompts": {}}
    eligible = set((manifest.get("selection_partition") or {}).get("selection_ids", []))
    for prompt in manifest["prompts"]:
        subset = [r for r in rows if r["prompt_id"] == prompt["id"]]
        entry = {"all_examples": {condition: metrics([r for r in subset if r["condition"] == condition]) for condition in CONDITIONS}}
        if manifest["phase"] == "development":
            selection_rows = [r for r in subset if r["example_id"] in eligible]
            entry["selection_subset"] = metrics(selection_rows)
            entry["selection_by_condition"] = {condition: metrics([r for r in selection_rows if r["condition"] == condition]) for condition in CONDITIONS}
        summary["prompts"][prompt["id"]] = entry
    if manifest["phase"] == "development":
        summary["ranking"] = rank_candidates(manifest["plan"], rows, eligible)
        summary["selection_partition"] = manifest["selection_partition"]
    return summary


def run_phase(phase, output):
    if output.exists():
        raise ValueError("Output already exists; refusing to overwrite or retry any generation")
    plan = load_plan()
    development = check_development()
    freeze = None
    if phase == "development":
        if FROZEN.exists():
            raise ValueError("Prompt v2 is frozen; no further development runs are allowed")
        examples = development
        prompts = [make_prompt(candidate, development) for candidate in plan["candidates"]]
    elif phase == "evaluation":
        freeze, selected = load_frozen()
        # This is the only evaluation-data loader in the v2 pipeline, gated above.
        from baseline.dataset import check_evaluation_evidence
        examples = check_evaluation_evidence()
        prompts = [selected]
    else:
        raise ValueError("Unknown phase")
    settings = json.loads((ROOT / "baseline/config.json").read_text())
    engine = Inference(settings["generation"])
    partition = selection_partition(plan, development)
    manifest = {
        "schema_version": "prompt-v2-run-0.1", "phase": phase, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan, "prompts": prompts, "example_ids": [e["id"] for e in examples],
        "conditions": list(CONDITIONS), "selection_partition": partition if phase == "development" else None,
        "model": engine.lock, "runtime": engine.runtime, "generation_settings": settings["generation"],
        "effective_generation_config": engine.generation.to_dict(), "verifier_timeout_seconds": settings["verifier_timeout_seconds"],
        "expected_attempts": len(prompts) * len(examples) * 2, "fingerprints": core_fingerprints(),
        "evaluation_used_for_selection": False, "repair_attempts": 0, "fine_tuning_performed": False,
    }
    if freeze:
        manifest["prompt_freeze_sha256"] = sha256_file(FROZEN)
        manifest["prompt_frozen_at_utc"] = freeze["frozen_at_utc"]
        manifest["evaluation_files_sha256"] = {name: sha256_file(ROOT / name) for name in ["data/evaluation/curated.json", "data/evaluation/verification_report.json"]}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    rows = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as results, (output / "attempts.jsonl").open("x", encoding="utf-8") as journal:
        for prompt in prompts:
            for example in examples:
                for condition in CONDITIONS:
                    key = f"{prompt['id']}:{example['id']}:{condition}"
                    journal.write(json.dumps({"key": key, "event": "started", "attempt": 1, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                    journal.flush()
                    print(f"Generating {key}", flush=True)
                    text, generation_record = engine.generate(build_messages(prompt, example, condition))
                    row = {
                        "schema_version": "prompt-v2-result-0.1", "key": key, "phase": phase, "prompt_id": prompt["id"],
                        "example_id": example["id"], "condition": condition, "attempt": 1, "repair_attempts": 0,
                        "formal_statement": example["formal_statement"], "generation_family": example["generation_family"],
                        "model": {"model_id": engine.lock["model_id"], "revision": engine.lock["revision"], "lock_sha256": sha256_file(MODEL_LOCK)},
                        "generation_settings": settings["generation"],
                        "is_demonstration": example["id"] in prompt["demonstration_ids"],
                        "selection_eligible": phase == "development" and example["id"] in partition["selection_ids"],
                    } | generation_record | assess_output(example, text, settings["verifier_timeout_seconds"])
                    rows.append(row)
                    results.write(json.dumps(row, ensure_ascii=False) + "\n")
                    results.flush()
                    journal.write(json.dumps({"key": key, "event": "finished", "attempt": 1}) + "\n")
                    journal.flush()
                    print(f"{key}: format={row['format_compliance']['compliant']} {row['verification']['status']} {row['verification']['category']} fidelity={row['argument_fidelity']['status']} ({row['usage']['completion_tokens']} tokens)", flush=True)
    write_json(output / "summary.json", phase_summary(manifest, rows))
    write_json(output / "completion.json", {
        "status": "COMPLETE", "finished_at_utc": datetime.now(timezone.utc).isoformat(), "attempts": len(rows),
        "files_sha256": {name: sha256_file(output / name) for name in ["manifest.json", "results.jsonl", "attempts.jsonl", "summary.json"]},
    })
    print(f"COMPLETE: {len(rows)} one-attempt outputs recorded at {output}", flush=True)


def load_phase(output):
    from collections import Counter
    completion = json.loads((output / "completion.json").read_text())
    for name, expected_hash in completion["files_sha256"].items():
        if sha256_file(output / name) != expected_hash:
            raise ValueError(f"Run artifact changed: {name}")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["fingerprints"] != core_fingerprints():
        raise ValueError("Prompt-development inputs changed")
    if manifest["phase"] == "development":
        examples = check_development()
        expected_prompts = [make_prompt(c, examples) for c in load_plan()["candidates"]]
    else:
        freeze, selected = load_frozen()
        if (manifest["prompt_freeze_sha256"] != sha256_file(FROZEN)
                or datetime.fromisoformat(manifest["started_at_utc"]) <= datetime.fromisoformat(freeze["frozen_at_utc"])):
            raise ValueError("Evaluation did not use the frozen prompt")
        from baseline.dataset import check_evaluation_evidence
        examples = check_evaluation_evidence()
        expected_prompts = [selected]
        for name, expected_hash in manifest["evaluation_files_sha256"].items():
            if sha256_file(ROOT / name) != expected_hash:
                raise ValueError("Evaluation input changed")
    if manifest["prompts"] != expected_prompts or manifest["example_ids"] != [e["id"] for e in examples]:
        raise ValueError("Prompt or example set changed")
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    expected_keys = {f"{p['id']}:{e['id']}:{c}" for p in expected_prompts for e in examples for c in CONDITIONS}
    if (completion["status"] != "COMPLETE" or completion["attempts"] != len(expected_keys)
            or len(rows) != len(expected_keys) or {r["key"] for r in rows} != expected_keys):
        raise ValueError("Missing, duplicate, or unexpected attempts")
    events = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    for event in ["started", "finished"]:
        if Counter(e["key"] for e in events if e["event"] == event) != Counter({key: 1 for key in expected_keys}):
            raise ValueError("Attempt journal does not match the one-attempt budget")
    if any(e["attempt"] != 1 for e in events):
        raise ValueError("An attempt was retried")
    by_id = {e["id"]: e for e in examples}
    prompts_by_id = {p["id"]: p for p in expected_prompts}
    from baseline.protocol import assess_structure, extract_body
    from prompt_v2.protocol import format_compliance
    for row in rows:
        example = by_id[row["example_id"]]
        prompt = prompts_by_id[row["prompt_id"]]
        body, extraction = extract_body(row["generated_text"])
        if (row["formal_statement"] != example["formal_statement"] or row["attempt"] != 1 or row["repair_attempts"] != 0
                or row["key"] != f"{prompt['id']}:{example['id']}:{row['condition']}"
                or row["prompt"]["messages"] != build_messages(prompt, example, row["condition"])
                or row["prompt"]["sha256"] != digest(row["prompt"]["rendered_text"])
                or row["proof_body"] != body or row["proof_body_sha256"] != digest(body) or row["output_extraction"] != extraction
                or row["generation_settings"] != manifest["generation_settings"]):
            raise ValueError("Candidate, prompt, theorem, or generation budget was altered")
        if row["model"] != {"model_id": manifest["model"]["model_id"], "revision": manifest["model"]["revision"], "lock_sha256": sha256_file(MODEL_LOCK)}:
            raise ValueError("Model identity changed")
        prompt_tokens, completion_tokens = len(row["prompt"]["input_token_ids"]), len(row["completion_token_ids"])
        if (row["usage"] != {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
                or completion_tokens > manifest["generation_settings"]["max_new_tokens"]):
            raise ValueError("Token usage or generation budget changed")
        evidence = row["verification"]
        if row["failure_category"] != (None if evidence["status"] == "PASS" else evidence["category"]):
            raise ValueError("Failure category disagrees with verification")
        if evidence["status"] == "PASS":
            from verifier import render_source
            if (evidence["category"] != "VERIFIED" or evidence["kernel_checked"] is not True
                    or evidence["assumptions_checked"] is not True
                    or evidence["source_sha256"] != digest(render_source(example["formal_statement"], body))):
                raise ValueError("PASS evidence does not match the generated proof")
        if (row["format_compliance"] != format_compliance(example["formal_statement"], row["generated_text"])
                or row["argument_fidelity"] != assess_structure(example, body)):
            raise ValueError("Format/fidelity annotations are stale")
    if json.loads((output / "summary.json").read_text()) != phase_summary(manifest, rows):
        raise ValueError("Saved development/evaluation metrics disagree with records")
    return manifest, rows


def freeze_selection():
    if FROZEN.exists() or SELECTED.exists():
        raise ValueError("Refusing to replace a selected/frozen prompt")
    manifest, rows = load_phase(DEV_OUTPUT)
    if manifest["phase"] != "development" or any(r["generation_error"] for r in rows):
        raise ValueError("A complete development comparison without generation errors is required")
    plan = load_plan()
    partition = selection_partition(plan, check_development())
    ranking = rank_candidates(plan, rows, partition["selection_ids"])
    winner = next(p for p in manifest["prompts"] if p["id"] == ranking[0]["candidate_id"])
    write_json(SELECTED, winner)
    hashes = core_fingerprints() | {str(SELECTED.relative_to(ROOT)): sha256_file(SELECTED)}
    for name in ["manifest.json", "results.jsonl", "summary.json", "attempts.jsonl", "completion.json"]:
        path = DEV_OUTPUT / name
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(FROZEN, {
        "schema_version": "prompt-v2-freeze-0.1", "status": "FROZEN", "selected_candidate": winner["id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "development_completed_at_utc": json.loads((DEV_OUTPUT / "completion.json").read_text())["finished_at_utc"],
        "development_only_selection": True, "evaluation_records_loaded_for_selection": 0,
        "selection_rule": plan["selection_rule"], "selection_partition": partition, "ranking": ranking,
        "files_sha256": hashes,
    })
    print(f"FROZEN: {winner['id']}; selection used only development outputs")


def load_frozen():
    if not FROZEN.exists() or not SELECTED.exists():
        raise ValueError("Select and freeze Prompt v2 before loading any evaluation records")
    freeze = json.loads(FROZEN.read_text())
    if freeze["status"] != "FROZEN" or freeze["development_only_selection"] is not True:
        raise ValueError("Invalid prompt freeze")
    for name, expected_hash in freeze["files_sha256"].items():
        if sha256_file(ROOT / name) != expected_hash:
            raise ValueError(f"Frozen prompt input changed: {name}")
    selected = json.loads(SELECTED.read_text())
    if selected["id"] != freeze["selected_candidate"]:
        raise ValueError("Selected prompt does not match its freeze")
    return freeze, selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["develop", "freeze", "evaluate", "check"])
    args = parser.parse_args()
    if args.phase == "develop":
        run_phase("development", DEV_OUTPUT)
    elif args.phase == "freeze":
        freeze_selection()
    elif args.phase == "evaluate":
        run_phase("evaluation", EVAL_OUTPUT)
    else:
        load_phase(DEV_OUTPUT)
        load_frozen()
        if EVAL_OUTPUT.exists():
            load_phase(EVAL_OUTPUT)
        print("PASS: recorded phases, prompt freeze, and metrics are consistent")
