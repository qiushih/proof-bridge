"""Validate, then run exactly 60 development attempts with frozen Prompt v2."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
import unittest

from baseline.dataset import ROOT, check_development, digest, write_json
from baseline.protocol import CONDITIONS
from baseline.setup_model import MODEL_DIR, MODEL_LOCK, check_model
from constrained_v1.grammar import accepts, can_end, feed, initial
from constrained_v1.tokens import Mask, Vocabulary
from prompt_v2.experiment import DEV_OUTPUT, Inference, core_fingerprints, load_frozen, load_phase
from prompt_v2.protocol import assess_output, build_messages
from verifier import sha256_file, verify

OUTPUT = ROOT / "results/constrained-v1"
PREFLIGHT = ROOT / "constrained_v1/preflight.json"
PLAN = ROOT / "constrained_v1/plan.json"


def fingerprints():
    paths = [PLAN, ROOT / "constrained_v1/grammar.py", ROOT / "constrained_v1/tokens.py",
             ROOT / "constrained_v1/experiment.py", ROOT / "tests/test_constrained_v1.py",
             ROOT / "prompt_v2/frozen.json", ROOT / "prompt_v2/selected.json",
             ROOT / "results/prompt-v2-development/results.jsonl",
             ROOT / "results/prompt-v2-development-audit/failure_analysis.json"]
    return core_fingerprints() | {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def tokenizer_only():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)


def validate():
    if PREFLIGHT.exists() or OUTPUT.exists():
        raise ValueError("Preflight/run exists; use check instead of replacing frozen validation")
    load_frozen()
    check_model()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_constrained_v1.py")
    tested = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise ValueError(log.getvalue())
    tokenizer = tokenizer_only()
    vocabulary = Vocabulary(tokenizer)
    outcomes = []
    for example in check_development():
        statement, body = example["formal_statement"], example["proof_body"]
        state = initial(statement)
        ids = tokenizer.encode(body, add_special_tokens=False)
        pieces = []
        for position, token_id in enumerate(ids):
            if token_id not in vocabulary.allowed(state):
                raise ValueError(f"Reference token blocked: {example['id']} token {position}")
            piece = vocabulary.pieces[token_id]
            pieces.append(piece)
            state = feed(state, piece)
        if "".join(pieces) != body or not can_end(state) or vocabulary.eos_id not in vocabulary.allowed(state):
            raise ValueError("Reference tokenization/EOS does not preserve the exact body")
        evidence = verify(statement, body)
        if evidence.status != "PASS":
            raise ValueError(f"Reference did not verify: {example['id']}: {evidence}")
        outcomes.append({"example_id": example["id"], "proof_body_sha256": digest(body),
                         "tokens": len(ids), "all_tokens_allowed": True, "EOS_allowed": True,
                         "verification": asdict(evidence)})
        print(f"Preflight {example['id']}: exact reference token path allowed; Rocq PASS", flush=True)
    write_json(PREFLIGHT, {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                          "fingerprints": fingerprints(), "unit_tests": tested.testsRun,
                          "unit_test_log": log.getvalue(), "reference_results": outcomes,
                          "allowed_vocabulary_pieces": len(vocabulary.pieces), "inference_attempts": 0})
    print(f"PREFLIGHT PASS: {tested.testsRun} tests; 30 reference token paths and Rocq proofs", flush=True)


def check_preflight():
    evidence = json.loads(PREFLIGHT.read_text())
    if evidence["status"] != "PASS" or evidence["fingerprints"] != fingerprints():
        raise ValueError("Preflight missing, failed, or stale")
    examples = check_development()
    if len(evidence["reference_results"]) != 30 or evidence["inference_attempts"] != 0:
        raise ValueError("Invalid preflight coverage")
    for e, r in zip(examples, evidence["reference_results"]):
        if (r["example_id"] != e["id"] or r["proof_body_sha256"] != digest(e["proof_body"])
                or not r["all_tokens_allowed"] or not r["EOS_allowed"] or r["verification"]["status"] != "PASS"):
            raise ValueError("Reference preflight evidence changed")
    return evidence


def generate(engine, vocabulary, statement, messages):
    rendered = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = engine.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    prompt_ids = inputs["input_ids"][0].tolist()
    mask = Mask(vocabulary, initial(statement), len(prompt_ids))
    new_ids, error = [], None
    started = time.perf_counter()
    try:
        with engine.torch.inference_mode():
            output = engine.model.generate(**inputs, generation_config=engine.generation, logits_processor=[mask])
        new_ids = output[0, len(prompt_ids):].tolist()
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
        new_ids = mask.seen_ids
    text = engine.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text, {
        "prompt": {"messages": messages, "rendered_text": rendered, "sha256": digest(rendered), "input_token_ids": prompt_ids},
        "completion_token_ids": new_ids,
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(new_ids), "total_tokens": len(prompt_ids) + len(new_ids)},
        "generation_latency_seconds": time.perf_counter() - started,
        "generation_error": error,
        "finish_reason": "generation_error" if error else "eos" if new_ids and new_ids[-1] == vocabulary.eos_id else "max_new_tokens",
        "constraint": {"method": "character_grammar_vocabulary_trie_logits_mask", "complete_body": accepts(statement, text),
                       "mask_seconds": mask.mask_seconds, "mask_calls": len(mask.trace),
                       "top_token_blocked_count": sum(step["top_token_blocked"] for step in mask.trace),
                       "trace": mask.trace},
    }


def run(output=OUTPUT):
    if output.exists():
        raise ValueError("Output exists; refusing overwrite, resume, or retry")
    preflight = check_preflight()
    freeze, selected = load_frozen()
    baseline_manifest, baseline_rows = load_phase(DEV_OUTPUT)
    settings = json.loads((ROOT / "baseline/config.json").read_text())
    examples = check_development()
    engine = Inference(settings["generation"])
    setup_started = time.perf_counter()
    vocabulary = Vocabulary(engine.tokenizer)
    vocabulary_seconds = time.perf_counter() - setup_started
    if engine.generation.to_dict() != baseline_manifest["effective_generation_config"]:
        raise ValueError("Effective generation configuration differs from saved v2")
    manifest = {"schema_version": "constrained-run-1.0", "experiment_id": "proofbridge-constrained-v1",
        "phase": "development", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": json.loads(PLAN.read_text()), "prompt": selected, "prompt_freeze_sha256": sha256_file(ROOT / "prompt_v2/frozen.json"),
        "preflight_sha256": sha256_file(PREFLIGHT), "preflight_completed_at_utc": preflight["completed_at_utc"],
        "fingerprints": fingerprints(), "model": engine.lock, "runtime": engine.runtime,
        "generation_settings": settings["generation"], "effective_generation_config": engine.generation.to_dict(),
        "verifier_timeout_seconds": settings["verifier_timeout_seconds"], "expected_attempts": 60,
        "conditions": list(CONDITIONS), "example_ids": [e["id"] for e in examples],
        "selection_partition": freeze["selection_partition"], "repair_attempts": 0, "fine_tuning_performed": False,
        "diagnostic_evaluation_records_loaded": 0, "vocabulary_setup_seconds": vocabulary_seconds,
        "vocabulary_cache_policy": "Fresh vocabulary per run; pure grammar mask cache shared across examples. No reference paths prewarm this run.",
    }
    by_key = {(r["example_id"], r["condition"]): r for r in baseline_rows if r["prompt_id"] == selected["id"]}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    rows = []
    with (output / "results.jsonl").open("x") as results, (output / "attempts.jsonl").open("x") as journal:
        for example in examples:
            for condition in CONDITIONS:
                key = f"constrained_v1:{example['id']}:{condition}"
                journal.write(json.dumps({"key": key, "event": "started", "attempt": 1, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                journal.flush()
                messages = build_messages(selected, example, condition)
                original = by_key[example["id"], condition]
                if messages != original["prompt"]["messages"]:
                    raise ValueError("Prompt changed")
                print(f"Generating {key}", flush=True)
                text, generated = generate(engine, vocabulary, example["formal_statement"], messages)
                row = {"schema_version": "constrained-result-1.0", "key": key, "phase": "development",
                    "prompt_id": selected["id"], "example_id": example["id"], "condition": condition,
                    "formal_statement": example["formal_statement"], "generation_family": example["generation_family"],
                    "attempt": 1, "repair_attempts": 0, "is_demonstration": example["id"] in selected["demonstration_ids"],
                    "selection_eligible": example["id"] in freeze["selection_partition"]["selection_ids"],
                    "generation_settings": settings["generation"], "model": original["model"], "baseline_key": original["key"],
                } | generated | assess_output(example, text, settings["verifier_timeout_seconds"])
                if row["prompt"]["input_token_ids"] != original["prompt"]["input_token_ids"]:
                    raise ValueError("Prompt token IDs changed")
                rows.append(row)
                results.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.flush()
                journal.write(json.dumps({"key": key, "event": "finished", "attempt": 1}) + "\n")
                journal.flush()
                print(f"{key}: {row['verification']['status']} {row['verification']['category']} ({row['usage']['completion_tokens']} tokens; {row['generation_latency_seconds']:.2f}s)", flush=True)
    write_json(output / "completion.json", {"status": "COMPLETE", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": len(rows), "files_sha256": {name: sha256_file(output / name) for name in ["manifest.json", "results.jsonl", "attempts.jsonl"]}})
    print("COMPLETE: 60 one-attempt development outputs", flush=True)


def check(output=OUTPUT, replay_tokens=False, reverify=False):
    check_preflight()
    load_frozen()
    completion = json.loads((output / "completion.json").read_text())
    for name, expected in completion["files_sha256"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"Run file changed: {name}")
    manifest = json.loads((output / "manifest.json").read_text())
    baseline_manifest, baseline_rows = load_phase(DEV_OUTPUT)
    if manifest["fingerprints"] != fingerprints() or manifest["preflight_sha256"] != sha256_file(PREFLIGHT):
        raise ValueError("Inputs changed after preflight/run")
    if datetime.fromisoformat(manifest["started_at_utc"]) <= datetime.fromisoformat(manifest["preflight_completed_at_utc"]):
        raise ValueError("Inference preceded preflight")
    for field in ("generation_settings", "effective_generation_config", "model"):
        if manifest[field] != baseline_manifest[field]:
            raise ValueError(f"Model/settings differ from baseline: {field}")
    examples = {e["id"]: e for e in check_development()}
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    expected_keys = {f"constrained_v1:{e}:{c}" for e in examples for c in CONDITIONS}
    if len(rows) != 60 or {r["key"] for r in rows} != expected_keys or completion["attempts"] != 60:
        raise ValueError("Expected exactly 60 unique attempts")
    journal = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    if len(journal) != 120 or any(e["attempt"] != 1 for e in journal):
        raise ValueError("Invalid one-attempt journal")
    for event in ("started", "finished"):
        if Counter(e["key"] for e in journal if e["event"] == event) != Counter({k: 1 for k in expected_keys}):
            raise ValueError("Missing/duplicate journal entries")
    by_key = {r["key"]: r for r in baseline_rows}
    tokenizer = tokenizer_only() if replay_tokens else None
    vocabulary = Vocabulary(tokenizer) if replay_tokens else None
    from baseline.protocol import assess_structure, extract_body
    from prompt_v2.protocol import format_compliance
    for row in rows:
        example, original = examples[row["example_id"]], by_key[row["baseline_key"]]
        if (row["prompt"] != original["prompt"] or row["model"] != original["model"]
                or row["generation_settings"] != original["generation_settings"]
                or row["attempt"] != 1 or row["repair_attempts"] != 0
                or row["formal_statement"] != example["formal_statement"]
                or row["proof_body_sha256"] != digest(row["proof_body"])
                or (row["proof_body"], row["output_extraction"]) != extract_body(row["generated_text"])):
            raise ValueError(f"Candidate/prompt/settings modified: {row['key']}")
        n, m = len(row["prompt"]["input_token_ids"]), len(row["completion_token_ids"])
        if row["usage"] != {"prompt_tokens": n, "completion_tokens": m, "total_tokens": n + m} or m > 256:
            raise ValueError("Token budget changed")
        if row["argument_fidelity"] != assess_structure(example, row["proof_body"]) or row["format_compliance"] != format_compliance(example["formal_statement"], row["generated_text"]):
            raise ValueError("Stale format/structure annotations")
        if row["constraint"]["complete_body"] != accepts(example["formal_statement"], row["generated_text"]):
            raise ValueError("Incorrect grammar completion claim")
        if vocabulary:
            state = initial(example["formal_statement"])
            ids = row["completion_token_ids"]
            for position, token_id in enumerate(ids):
                if token_id not in vocabulary.allowed(state):
                    raise ValueError(f"Disallowed generated token: {row['key']}:{position}")
                if token_id == vocabulary.eos_id:
                    if position != len(ids) - 1:
                        raise ValueError("EOS before final token")
                else:
                    state = feed(state, vocabulary.pieces[token_id])
            decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if decoded != row["generated_text"]:
                raise ValueError("Token decode differs from recorded text")
        if reverify:
            evidence = asdict(verify(example["formal_statement"], row["proof_body"]))
            for field in ("status", "category", "stage", "source_sha256", "kernel_checked", "assumptions_checked"):
                if evidence[field] != row["verification"][field]:
                    raise ValueError(f"Reverification differs: {row['key']}: {field}")
    return manifest, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate", "run", "check"])
    parser.add_argument("--replay-tokens", action="store_true")
    parser.add_argument("--reverify", action="store_true")
    args = parser.parse_args()
    if args.phase == "validate":
        validate()
    elif args.phase == "run":
        run()
    else:
        _, rows = check(replay_tokens=args.replay_tokens, reverify=args.reverify)
        print(f"PASS: {len(rows)} saved constrained outputs checked; no new generations")
