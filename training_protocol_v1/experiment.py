"""Freeze protocol, run exactly nine dev outputs, replay and freeze evidence."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import unittest

from baseline.dataset import digest, write_json
from baseline.protocol import extract_body
from baseline.setup_model import check_model
from constrained_v1.experiment import generate, tokenizer_only
from constrained_v1.tokens import Vocabulary
from pilot_v1.protocol import ROOT, load_split
from prompt_v2.experiment import Inference, load_frozen
from prompt_v2.protocol import build_messages, format_compliance
from training_protocol_v1.data import config, encode_example, epoch_order
from training_protocol_v1.scoring import CONDITIONS, mathematical_structure, step_adherence
from verifier import sha256_file, verify

PROTOCOL_DIR = ROOT / "training_protocol_v1"
FROZEN = PROTOCOL_DIR / "frozen.json"
PREPARATION = PROTOCOL_DIR / "preparation.json"
OUTPUT = ROOT / "results/pretraining-dev-v1"
RELEASE = PROTOCOL_DIR / "release.json"


def protected_check():
    record = json.loads((PROTOCOL_DIR / "protected_baselines.json").read_text())
    for name, expected in record["files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen historical file changed: {name}")
    return len(record["files_sha256"])


def fingerprints():
    paths = list(PROTOCOL_DIR.glob("*.py")) + [PROTOCOL_DIR / "config.json", PROTOCOL_DIR / "hardware.json", PROTOCOL_DIR / "protected_baselines.json",
        PROTOCOL_DIR / "TRAINING_PROTOCOL.md", ROOT / "tests/test_training_protocol_v1.py", ROOT / "tests/test_pretraining_v1_results.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def dev_groups():
    rows = load_split("dev")
    if [r["id"] for r in rows] != config()["data"]["dev_ids"]:
        raise ValueError("Dev rows changed")
    groups = []
    for theorem in config()["baseline"]["theorem_ids"]:
        by_argument = {r["argument_id"]: r for r in rows if r["theorem_id"] == theorem}
        if set(by_argument) != {"A", "B"} or len({r["formal_statement"] for r in by_argument.values()}) != 1:
            raise ValueError("Malformed dev A/B theorem")
        groups.append({"theorem_id": theorem, "arguments": by_argument})
    return groups


def messages(group, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown baseline condition")
    _, prompt = load_frozen()
    example = group["arguments"]["B" if condition == "argument_B" else "A"]
    return build_messages(prompt, example, "theorem_only" if condition == "theorem_only" else "theorem_and_informal")


def freeze_protocol():
    if FROZEN.exists() or PREPARATION.exists() or OUTPUT.exists():
        raise ValueError("Protocol/preparation/run already exists; no overwrite")
    protected_check()
    model = check_model()
    c = config()
    if model["model_id"] != c["model_id"] or model["revision"] != c["model_revision"]:
        raise ValueError("Unexpected base model")
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_training_protocol_v1.py"))
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError("Run preparation with pinned .venv; all preflight tests must pass\n" + log.getvalue())
    tokenizer = tokenizer_only()
    _, prompt = load_frozen()
    serializations = []
    for split in ("train", "dev"):
        for row in load_split(split):
            encoded = encode_example(tokenizer, prompt, row)
            p = encoded["prompt_length"]
            if encoded["labels"][:p] != [-100] * p or encoded["labels"][p:] != encoded["input_ids"][p:]:
                raise ValueError("Completion-only labels invalid")
            if encoded["labels"][-1] != tokenizer.eos_token_id:
                raise ValueError("Final target EOS not supervised")
            serializations.append({"id": row["id"], "split": split,
                "prompt_tokens_masked": p, "target_tokens_including_eos": encoded["target_length"],
                "total_sequence_tokens": encoded["sequence_length"],
                "serialized_ids_and_mask_sha256": digest(json.dumps(encoded, sort_keys=True))})
    references = []
    for group in dev_groups():
        for argument, example in group["arguments"].items():
            result = asdict(verify(example["formal_statement"], example["proof_body"]))
            if result["status"] != "PASS" or not step_adherence(example, example["proof_body"], "argument_" + argument)["matched"]:
                raise ValueError("Dev reference/scoring preflight failed")
            references.append({"id": example["id"], "verification": result})
    write_json(PREPARATION, {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests": tests.testsRun, "test_log": log.getvalue(), "serialization": serializations,
        "epoch_orders": {str(epoch): epoch_order(epoch) for epoch in range(c["training"]["epochs"])},
        "dev_references": references, "holdout_content_reads": 0,
        "qwen_training_forward_backward_calls": 0, "qwen_optimizer_steps": 0, "qwen_adapter_attachments": 0,
        "model_generations": 0, "model": model})
    write_json(FROZEN, {"status": "FROZEN_BEFORE_BASELINE_AND_TRAINING", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": c["protocol_id"], "input_fingerprints": fingerprints(),
        "preparation_sha256": sha256_file(PREPARATION), "expected_baseline_attempts": 9,
        "model_weights_changed": False, "training_runs": 0, "holdout_content_reads": 0})
    print(f"FROZEN: protocol, {tests.testsRun} tests, 30 completion-only masks, six Rocq references; zero generations")


def check_protocol():
    protected_check()
    frozen = json.loads(FROZEN.read_text())
    if (frozen["input_fingerprints"] != fingerprints() or frozen["preparation_sha256"] != sha256_file(PREPARATION)
            or frozen["status"] != "FROZEN_BEFORE_BASELINE_AND_TRAINING"):
        raise ValueError("Protocol or preparation changed after freezing")
    return frozen


def run():
    if OUTPUT.exists():
        raise ValueError("Baseline exists; refusing overwrite, retry, resume or extra attempts")
    frozen = check_protocol()
    groups = dev_groups()
    settings = json.loads((ROOT / "baseline/config.json").read_text())
    engine = Inference(settings["generation"])
    old_run = json.loads((ROOT / "results/constrained-v1/manifest.json").read_text())
    if engine.lock != old_run["model"] or engine.generation.to_dict() != old_run["effective_generation_config"]:
        raise ValueError("Base model/generation configuration differs from frozen constrained-v1")
    # Read-only architecture check. No adapters are attached to the actual model.
    modules = [m for name,m in engine.model.named_modules() if name.endswith(("self_attn.q_proj", "self_attn.v_proj"))]
    adapter_count = sum(8 * (m.in_features + m.out_features) for m in modules)
    if len(modules) != 48 or adapter_count != 540672:
        raise ValueError("Actual architecture does not match planned adapter configuration")
    vocabulary = Vocabulary(engine.tokenizer)
    manifest = {"experiment_id": "proofbridge-pretraining-dev-v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_frozen_at_utc": frozen["frozen_at_utc"], "protocol_freeze_sha256": sha256_file(FROZEN),
        "preparation_sha256": sha256_file(PREPARATION), "fingerprints": fingerprints(),
        "theorem_ids": [g["theorem_id"] for g in groups], "conditions": list(CONDITIONS),
        "expected_attempts": 9, "attempts_per_condition": 1, "repair_attempts": 0,
        "model": engine.lock, "runtime": engine.runtime, "generation_settings": settings["generation"],
        "effective_generation_config": engine.generation.to_dict(), "verifier_timeout_seconds": 10.0,
        "prompt": "unchanged prompt_v2.protocol.build_messages with frozen selected.json",
        "decoder": "unchanged constrained_v1.experiment.generate", "weights_updated": False,
        "qwen_adapter_attachments": 0, "qwen_optimizer_steps": 0, "holdout_content_reads": 0,
        "architecture_read_only_check": {"q_v_modules": len(modules), "planned_adapter_parameters": adapter_count}}
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "manifest.json", manifest)
    with (OUTPUT / "raw_generations.jsonl").open("x") as output, (OUTPUT / "attempts.jsonl").open("x") as journal:
        for group in groups:
            for condition in CONDITIONS:
                key = f"pretraining_v1:{group['theorem_id']}:{condition}"
                journal.write(json.dumps({"key": key, "event": "started", "attempt": 1, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                journal.flush()
                print(f"Generating {key}", flush=True)
                example = group["arguments"]["B" if condition == "argument_B" else "A"]
                generated_text, generation = generate(engine, vocabulary, example["formal_statement"], messages(group, condition))
                body, extraction = extract_body(generated_text)
                evidence = asdict(verify(example["formal_statement"], body))
                row = {"key": key, "theorem_id": group["theorem_id"], "dev_example_id": example["id"] if condition != "theorem_only" else None,
                    "condition": condition, "formal_statement": example["formal_statement"], "attempt": 1, "repair_attempts": 0,
                    "generated_text": generated_text, "proof_body": body, "proof_body_sha256": digest(body), "output_extraction": extraction,
                    "verification": evidence, "failure_category": None if evidence["status"] == "PASS" else evidence["category"],
                    "format_compliance": format_compliance(example["formal_statement"], generated_text),
                    "requested_proof_steps": step_adherence(example, body, condition),
                    "mathematical_structure_review_aid": mathematical_structure(example["formal_statement"], body)} | generation
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                journal.write(json.dumps({"key": key, "event": "finished", "attempt": 1}) + "\n")
                journal.flush()
                print(f"{key}: {evidence['status']} {evidence['category']}; steps {row['requested_proof_steps']['status']}", flush=True)
    # A second checksum validates the unchanged on-disk base, not a new inference.
    if check_model() != engine.lock:
        raise ValueError("Base model files changed")
    protected_check()
    write_json(OUTPUT / "completion.json", {"status": "COMPLETE", "attempts": 9,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "base_model_files_unchanged": True,
        "files_sha256": {name: sha256_file(OUTPUT / name) for name in ("manifest.json", "raw_generations.jsonl", "attempts.jsonl")}})
    print("COMPLETE: exactly nine one-attempt generations; zero weight updates", flush=True)


def check(output=OUTPUT, replay_tokens=False, reverify=False):
    check_protocol()
    completion = json.loads((output / "completion.json").read_text())
    for name, expected in completion["files_sha256"].items():
        if sha256_file(output / name) != expected:
            raise ValueError("Raw baseline artifact changed")
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["fingerprints"] != fingerprints() or manifest["protocol_freeze_sha256"] != sha256_file(FROZEN)
            or manifest["preparation_sha256"] != sha256_file(PREPARATION)
            or manifest["started_at_utc"] <= manifest["protocol_frozen_at_utc"]):
        raise ValueError("Protocol did not precede inference or changed")
    old_run = json.loads((ROOT / "results/constrained-v1/manifest.json").read_text())
    for field in ("model", "generation_settings", "effective_generation_config"):
        if manifest[field] != old_run[field]:
            raise ValueError("Frozen model/generation settings changed")
    rows = [json.loads(line) for line in (output / "raw_generations.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    groups = {g["theorem_id"]: g for g in dev_groups()}
    expected = {f"pretraining_v1:{t}:{c}" for t in groups for c in CONDITIONS}
    if completion["status"] != "COMPLETE" or completion["attempts"] != 9 or len(rows) != 9 or {r["key"] for r in rows} != expected:
        raise ValueError("Expected exactly nine unique outputs")
    if len(events) != 18 or any(e["attempt"] != 1 for e in events):
        raise ValueError("Invalid one-attempt journal")
    for event in ("started", "finished"):
        if Counter(e["key"] for e in events if e["event"] == event) != Counter({k:1 for k in expected}):
            raise ValueError("Missing/duplicate attempt event")
    tokenizer = tokenizer_only() if replay_tokens else None
    vocabulary = Vocabulary(tokenizer) if tokenizer else None
    for row in rows:
        group = groups[row["theorem_id"]]
        example = group["arguments"]["B" if row["condition"] == "argument_B" else "A"]
        if (row["key"] != f"pretraining_v1:{row['theorem_id']}:{row['condition']}"
                or row["condition"] not in CONDITIONS or row["formal_statement"] != example["formal_statement"]
                or row["dev_example_id"] != (example["id"] if row["condition"] != "theorem_only" else None)
                or row["prompt"]["messages"] != messages(group, row["condition"])
                or row["attempt"] != 1 or row["repair_attempts"] != 0
                or (row["proof_body"], row["output_extraction"]) != extract_body(row["generated_text"])
                or row["proof_body_sha256"] != digest(row["proof_body"])):
            raise ValueError("Saved candidate/input changed")
        p, n = len(row["prompt"]["input_token_ids"]), len(row["completion_token_ids"])
        if n > 256 or row["usage"] != {"prompt_tokens":p, "completion_tokens":n, "total_tokens":p+n}:
            raise ValueError("Token budget/usage changed")
        if (row["requested_proof_steps"] != step_adherence(example, row["proof_body"], row["condition"])
                or row["mathematical_structure_review_aid"] != mathematical_structure(example["formal_statement"], row["proof_body"])
                or row["format_compliance"] != format_compliance(example["formal_statement"], row["generated_text"])):
            raise ValueError("Assessment drift")
        if tokenizer:
            rendered = tokenizer.apply_chat_template(messages(group, row["condition"]), tokenize=False, add_generation_prompt=True)
            if row["prompt"]["rendered_text"] != rendered or row["prompt"]["sha256"] != digest(rendered) or row["prompt"]["input_token_ids"] != tokenizer.encode(rendered, add_special_tokens=False):
                raise ValueError("Prompt token replay mismatch")
            from constrained_v1.grammar import initial, feed
            state = initial(row["formal_statement"])
            for position, token in enumerate(row["completion_token_ids"]):
                if token not in vocabulary.allowed(state):
                    raise ValueError("Generated token violates frozen decoder")
                if token == vocabulary.eos_id:
                    if position != n - 1:
                        raise ValueError("Premature EOS")
                else:
                    state = feed(state, vocabulary.pieces[token])
            if tokenizer.decode(row["completion_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False) != row["generated_text"]:
                raise ValueError("Completion token replay mismatch")
        if reverify:
            actual = asdict(verify(row["formal_statement"], row["proof_body"]))
            for field in ("status", "category", "source_sha256", "kernel_checked", "assumptions_checked"):
                if actual[field] != row["verification"][field]:
                    raise ValueError("Rocq replay differs")
    if RELEASE.exists() and output == OUTPUT:
        for name, expected in json.loads(RELEASE.read_text())["files_sha256"].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f"Frozen result changed: {name}")
    return manifest, rows


def freeze_results():
    if RELEASE.exists():
        raise ValueError("Results already frozen")
    check()
    from training_protocol_v1.report import build_report
    report, assessed, summary = build_report()
    if (OUTPUT / "PRETRAINING_DEV_BASELINE.md").read_text() != report:
        raise ValueError("Report is stale")
    saved_rows = [json.loads(line) for line in (OUTPUT / "assessed_results.jsonl").read_text().splitlines()]
    if saved_rows != assessed or json.loads((OUTPUT / "summary.json").read_text()) != summary:
        raise ValueError("Assessed records/summary differ")
    validation = json.loads((OUTPUT / "validation.json").read_text())
    if validation["status"] != "PASS" or validation["token_replays"] != 9 or validation["rocq_replays"] != 9:
        raise ValueError("Final validation/replay evidence required")
    check_model()
    paths = [p for p in PROTOCOL_DIR.iterdir() if p.is_file()] + [p for p in OUTPUT.iterdir() if p.is_file()]
    paths += [ROOT / "tests/test_training_protocol_v1.py", ROOT / "tests/test_pretraining_v1_results.py"]
    write_json(RELEASE, {"status": "FROZEN", "release_id": "proofbridge-pretraining-dev-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "generations": 9, "training_runs": 0,
        "holdout_content_reads": 0, "files_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}})
    print("FROZEN: protocol and nine assessed baseline outputs; no training or holdout inspection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze-protocol", "run", "check", "freeze-results"])
    parser.add_argument("--replay-tokens", action="store_true")
    parser.add_argument("--reverify", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze-protocol":
        freeze_protocol()
    elif args.action == "run":
        run()
    elif args.action == "freeze-results":
        freeze_results()
    else:
        _, rows = check(replay_tokens=args.replay_tokens, reverify=args.reverify)
        print(f"PASS: {len(rows)} saved outputs checked; token replay={args.replay_tokens}; Rocq replay={args.reverify}; no new generations")
