"""Preflight and exactly 18 fixed-condition generations; no repair or tuning."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import unittest

from argument_switch_v1.protocol import CONDITIONS, PLAN, VARIANTS, assess_strategy, load_variants, messages, summarize
from baseline.dataset import ROOT, digest, write_json
from baseline.setup_model import check_model
from checkpoints.freeze_constrained_v1 import CHECKPOINT, check as check_checkpoint
from constrained_v1.experiment import generate, tokenizer_only
from constrained_v1.grammar import can_end, feed, initial
from constrained_v1.tokens import Vocabulary
from prompt_v2.experiment import Inference, load_frozen
from prompt_v2.protocol import assess_output
from verifier import render_source, sha256_file, verify

OUTPUT = ROOT / "results/argument-switch-v1"
PREFLIGHT = ROOT / "argument_switch_v1/preflight.json"


def fingerprints():
    paths = [PLAN, VARIANTS, CHECKPOINT, ROOT / "argument_switch_v1/protocol.py",
             ROOT / "argument_switch_v1/experiment.py", ROOT / "tests/test_argument_switch_v1.py"]
    paths += sorted((ROOT / "argument_switch_v1/references").glob("*.v"))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def validate():
    if PREFLIGHT.exists() or OUTPUT.exists():
        raise ValueError("Preflight/run already exists; refusing to replace")
    check_model()
    variants = load_variants()
    test_log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_argument_switch_v1.py")
    tests = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise ValueError(test_log.getvalue())
    tokenizer = tokenizer_only()
    vocabulary = Vocabulary(tokenizer)
    outcomes = []
    for variant in variants:
        for label, reference in variant["arguments"].items():
            statement, body = variant["formal_statement"], reference["proof_body"]
            source = render_source(statement, body)
            if (ROOT / reference["reference_file"]).read_text() != source:
                raise ValueError("Reference .v file differs from formal body")
            state = initial(statement)
            ids = tokenizer.encode(body, add_special_tokens=False)
            pieces = []
            for position, token_id in enumerate(ids):
                if token_id not in vocabulary.allowed(state):
                    raise ValueError(f"Blocked reference token: {variant['example_id']} {label} {position}")
                piece = vocabulary.pieces[token_id]
                pieces.append(piece)
                state = feed(state, piece)
            if "".join(pieces) != body or not can_end(state) or vocabulary.eos_id not in vocabulary.allowed(state):
                raise ValueError("Reference token path/EOS invalid")
            evidence = verify(statement, body)
            if evidence.status != "PASS" or evidence.source_sha256 != digest(source):
                raise ValueError(f"Reference {variant['example_id']} {label} failed: {evidence}")
            outcomes.append({"example_id": variant["example_id"], "argument": label, "reference_file": reference["reference_file"],
                             "proof_body_sha256": digest(body), "all_tokens_and_EOS_allowed": True, "tokens": len(ids),
                             "verification": asdict(evidence)})
            print(f"Reference {variant['example_id']} {label}: Rocq PASS; frozen decoder allows all tokens", flush=True)
    write_json(PREFLIGHT, {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                          "fingerprints": fingerprints(), "unit_tests": tests.testsRun, "test_log": test_log.getvalue(),
                          "references": outcomes, "inference_attempts": 0, "diagnostic_examples_used": 0})
    print("PREFLIGHT PASS: 12 reference proofs; zero model generations", flush=True)


def check_preflight():
    check_checkpoint()
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight["status"] != "PASS" or preflight["fingerprints"] != fingerprints() or len(preflight["references"]) != 12:
        raise ValueError("Reference/protocol preflight is stale or invalid")
    if preflight["inference_attempts"] != 0 or any(r["verification"]["status"] != "PASS" or not r["all_tokens_and_EOS_allowed"] for r in preflight["references"]):
        raise ValueError("Reference preflight did not pass")
    return preflight


def run(output=OUTPUT):
    if output.exists():
        raise ValueError("Run exists; no overwrite, retry, or resume")
    preflight = check_preflight()
    variants = load_variants()
    _, selected = load_frozen()
    settings = json.loads((ROOT / "baseline/config.json").read_text())
    frozen_run = json.loads((ROOT / "results/constrained-v1/manifest.json").read_text())
    engine = Inference(settings["generation"])
    vocabulary = Vocabulary(engine.tokenizer)
    if engine.lock != frozen_run["model"] or engine.generation.to_dict() != frozen_run["effective_generation_config"]:
        raise ValueError("Model or generation settings changed")
    manifest = {"experiment_id": "proofbridge-argument-switch-v1", "phase": "development_argument_switch",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "preflight_completed_at_utc": preflight["completed_at_utc"],
        "preflight_sha256": sha256_file(PREFLIGHT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "fingerprints": fingerprints(), "plan": json.loads(PLAN.read_text()), "prompt": selected,
        "example_ids": [v["example_id"] for v in variants], "conditions": list(CONDITIONS),
        "model": engine.lock, "runtime": engine.runtime, "generation_settings": settings["generation"],
        "effective_generation_config": engine.generation.to_dict(), "expected_attempts": 18, "repair_attempts": 0,
        "fine_tuning": False, "diagnostic_examples_used": 0, "verifier_timeout_seconds": settings["verifier_timeout_seconds"],
        "decoder": "Unmodified constrained_v1.experiment.generate", "cache_policy": "Fresh vocabulary/cache at inference; no reference paths prewarm it."}
    output.mkdir(parents=True)
    write_json(output / "manifest.json", manifest)
    rows = []
    with (output / "results.jsonl").open("x") as result_file, (output / "attempts.jsonl").open("x") as journal:
        for variant in variants:
            for condition in CONDITIONS:
                key = f"argument_switch_v1:{variant['example_id']}:{condition}"
                journal.write(json.dumps({"key": key, "event": "started", "attempt": 1, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                journal.flush()
                print(f"Generating {key}", flush=True)
                text, generated = generate(engine, vocabulary, variant["formal_statement"], messages(variant, condition))
                anchor = "B" if condition == "argument_B" else "A"
                scoring_example = {"formal_statement": variant["formal_statement"], "proof_body": variant["arguments"][anchor]["proof_body"]}
                row = {"schema_version": "argument-switch-result-1.0", "key": key, "example_id": variant["example_id"],
                       "condition": condition, "formal_statement": variant["formal_statement"], "attempt": 1, "repair_attempts": 0,
                       "generation_settings": settings["generation"], "structure_proxy_reference": anchor,
                       "model_id": engine.lock["model_id"], "model_revision": engine.lock["revision"]} | generated | assess_output(scoring_example, text, settings["verifier_timeout_seconds"])
                row["strategy"] = assess_strategy(variant, condition, row["proof_body"], row["verification"]["status"] == "PASS")
                rows.append(row)
                result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                result_file.flush()
                journal.write(json.dumps({"key": key, "event": "finished", "attempt": 1}) + "\n")
                journal.flush()
                print(f"{key}: {row['verification']['status']} {row['verification']['category']}; strategy {row['strategy']['observed_matching_arguments']}", flush=True)
    write_json(output / "summary.json", summarize(rows))
    write_json(output / "completion.json", {"status": "COMPLETE", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "attempts": 18, "files_sha256": {name: sha256_file(output / name) for name in ("manifest.json", "results.jsonl", "attempts.jsonl", "summary.json")}})
    print("COMPLETE: exactly 18 one-attempt generations", flush=True)


def check(output=OUTPUT, replay_tokens=False, reverify=False):
    check_preflight()
    completion = json.loads((output / "completion.json").read_text())
    for name, expected in completion["files_sha256"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["fingerprints"] != fingerprints() or manifest["preflight_sha256"] != sha256_file(PREFLIGHT):
        raise ValueError("Probe inputs changed")
    if manifest["started_at_utc"] <= manifest["preflight_completed_at_utc"]:
        raise ValueError("Inference preceded reference/protocol preflight")
    frozen_run = json.loads((ROOT / "results/constrained-v1/manifest.json").read_text())
    for field in ("model", "generation_settings", "effective_generation_config"):
        if manifest[field] != frozen_run[field]:
            raise ValueError("Frozen model/settings changed")
    variants = {v["example_id"]: v for v in load_variants()}
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    expected = {f"argument_switch_v1:{e}:{c}" for e in variants for c in CONDITIONS}
    if len(rows) != 18 or {r["key"] for r in rows} != expected or completion["attempts"] != 18:
        raise ValueError("Expected exactly 18 unique attempts")
    events = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    if len(events) != 36 or any(e["attempt"] != 1 for e in events):
        raise ValueError("Invalid one-attempt journal")
    for event in ("started", "finished"):
        if Counter(e["key"] for e in events if e["event"] == event) != Counter({k: 1 for k in expected}):
            raise ValueError("Missing/duplicate attempt event")
    tokenizer = tokenizer_only() if replay_tokens else None
    vocabulary = Vocabulary(tokenizer) if replay_tokens else None
    from baseline.protocol import extract_body
    from prompt_v2.protocol import format_compliance
    for row in rows:
        variant = variants[row["example_id"]]
        if (row["key"] != f"argument_switch_v1:{row['example_id']}:{row['condition']}"
                or row["prompt"]["messages"] != messages(variant, row["condition"])
                or row["formal_statement"] != variant["formal_statement"]
                or row["attempt"] != 1 or row["repair_attempts"] != 0
                or row["generation_settings"] != manifest["generation_settings"]
                or row["proof_body_sha256"] != digest(row["proof_body"])
                or (row["proof_body"], row["output_extraction"]) != extract_body(row["generated_text"])):
            raise ValueError(f"Changed candidate/prompt/budget: {row['key']}")
        n, m = len(row["prompt"]["input_token_ids"]), len(row["completion_token_ids"])
        if row["usage"] != {"prompt_tokens": n, "completion_tokens": m, "total_tokens": n + m} or m > 256:
            raise ValueError("Generation token budget changed")
        if row["strategy"] != assess_strategy(variant, row["condition"], row["proof_body"], row["verification"]["status"] == "PASS"):
            raise ValueError("Strategy score changed")
        if row["format_compliance"] != format_compliance(variant["formal_statement"], row["generated_text"]):
            raise ValueError("Format score changed")
        if vocabulary:
            rendered = tokenizer.apply_chat_template(messages(variant, row["condition"]), tokenize=False, add_generation_prompt=True)
            if row["prompt"]["rendered_text"] != rendered or row["prompt"]["sha256"] != digest(rendered) or row["prompt"]["input_token_ids"] != tokenizer.encode(rendered, add_special_tokens=False):
                raise ValueError("Frozen prompt tokenization changed")
            state = initial(variant["formal_statement"])
            ids = row["completion_token_ids"]
            for position, token_id in enumerate(ids):
                if token_id not in vocabulary.allowed(state):
                    raise ValueError("Saved token was not allowed by frozen decoder")
                if token_id == vocabulary.eos_id:
                    if position != len(ids) - 1:
                        raise ValueError("EOS before final token")
                else:
                    state = feed(state, vocabulary.pieces[token_id])
            if tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != row["generated_text"]:
                raise ValueError("Token IDs do not decode to saved output")
        if reverify:
            evidence = asdict(verify(variant["formal_statement"], row["proof_body"]))
            for field in ("status", "category", "stage", "source_sha256", "kernel_checked", "assumptions_checked"):
                if evidence[field] != row["verification"][field]:
                    raise ValueError(f"Reverification differs: {row['key']} {field}")
    if json.loads((output / "summary.json").read_text()) != summarize(rows):
        raise ValueError("Summary does not match outputs")
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
        print(f"PASS: {len(rows)} saved argument-switch outputs checked; no new generations")
