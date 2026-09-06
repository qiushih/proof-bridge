"""Validate immutable generation records, attach optional reviews, and summarize."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import statistics

from baseline.dataset import ROOT, check_evaluation_evidence, digest, write_json
from baseline.protocol import CONDITIONS, assess_structure, build_messages, extract_body
from baseline.run import experiment_fingerprints
from verifier import render_source, sha256_file, verify


def validate_records(records, journal, manifest, examples):
    expected = {f"{e['id']}:{c}" for e in examples for c in CONDITIONS}
    keys = [r["key"] for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate, or unexpected generation records")
    starts = [e["key"] for e in journal if e["event"] == "started"]
    finishes = [e["key"] for e in journal if e["event"] == "finished"]
    if Counter(starts) != Counter({k: 1 for k in expected}) or Counter(finishes) != Counter(starts):
        raise ValueError("Attempt journal must show exactly one complete attempt per condition")
    if any(e["attempt"] != 1 for e in journal):
        raise ValueError("Unexpected extra generation attempt")
    by_id = {e["id"]: e for e in examples}
    for record in records:
        example = by_id[record["example_id"]]
        if (record["key"] != f"{example['id']}:{record['condition']}" or record["attempt"] != 1
                or record["repair_attempts"] != 0 or record["formal_statement"] != example["formal_statement"]):
            raise ValueError("Changed theorem or generation budget")
        if record["prompt"]["messages"] != build_messages(example, record["condition"]):
            raise ValueError("Prompt differs from the frozen condition")
        if record["prompt"]["sha256"] != digest(record["prompt"]["rendered_text"]):
            raise ValueError("Rendered prompt hash mismatch")
        if (record["model"]["model_id"] != manifest["model"]["model_id"]
                or record["model"]["revision"] != manifest["model"]["revision"]
                or record["generation_settings"] != manifest["configuration"]["generation"]):
            raise ValueError("Model or generation configuration changed")
        body, extraction = extract_body(record["generated_text"])
        if body != record["proof_body"] or extraction != record["output_extraction"] or digest(body) != record["proof_body_sha256"]:
            raise ValueError("Generated proof was edited beyond the declared format extraction")
        if record["argument_fidelity"] != assess_structure(example, body):
            raise ValueError("Argument-structure evidence is stale")
        if record["usage"] != {
            "prompt_tokens": len(record["prompt"]["input_token_ids"]),
            "completion_tokens": len(record["completion_token_ids"]),
            "total_tokens": len(record["prompt"]["input_token_ids"]) + len(record["completion_token_ids"]),
        }:
            raise ValueError("Token usage disagrees with recorded token IDs")
        evidence = record["verification"]
        expected_failure = None if evidence["status"] == "PASS" else evidence["category"]
        if record["failure_category"] != expected_failure:
            raise ValueError("Failure category disagrees with verifier")
        if evidence["status"] == "PASS":
            if (evidence["category"] != "VERIFIED" or evidence["kernel_checked"] is not True
                    or evidence["assumptions_checked"] is not True
                    or evidence["source_sha256"] != digest(render_source(example["formal_statement"], body))):
                raise ValueError("PASS lacks kernel/type/assumption evidence for the fixed theorem")


def load_run(directory, reverify=False):
    completion = json.loads((directory / "completion.json").read_text())
    if completion["status"] != "COMPLETE" or completion["attempts"] != 24:
        raise ValueError("The baseline generation run is incomplete")
    for name, expected_hash in completion["files_sha256"].items():
        if sha256_file(directory / name) != expected_hash:
            raise ValueError(f"Recorded run artifact changed: {name}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["fingerprints"] != experiment_fingerprints():
        raise ValueError("Frozen experiment inputs changed")
    examples = check_evaluation_evidence()
    records = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    journal = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    validate_records(records, journal, manifest, examples)
    if reverify:
        for record in records:
            result = verify(record["formal_statement"], record["proof_body"], timeout=manifest["configuration"]["verifier_timeout_seconds"])
            saved = record["verification"]
            if (result.status, result.category, result.source_sha256) != (saved["status"], saved["category"], saved["source_sha256"]):
                raise ValueError(f"Reverification disagrees: {record['key']}")
        print(f"Reverified all {len(records)} generated candidates; every result agrees.")
    return manifest, records


def attach_reviews(directory, records):
    path = directory / "argument_reviews.json"
    if not path.exists():
        return deepcopy(records), False
    review_file = json.loads(path.read_text())
    if review_file["results_sha256"] != sha256_file(directory / "results.jsonl"):
        raise ValueError("Argument reviews refer to different generation results")
    reviews = {entry["key"]: entry for entry in review_file["reviews"]}
    if len(reviews) != len(review_file["reviews"]) or set(reviews) != {r["key"] for r in records}:
        raise ValueError("Argument reviews must cover each generation exactly once")
    assessed = deepcopy(records)
    for record in assessed:
        review = reviews[record["key"]]
        if (review["proof_body_sha256"] != record["proof_body_sha256"]
                or review["status"] not in {"FAITHFUL", "PARTIAL", "DIVERGENT", "UNASSESSABLE"}
                or not review["rationale"].strip()):
            raise ValueError("Invalid or stale argument review")
        record["argument_fidelity"]["assistant_review"] = review | {
            "reviewer": "assistant", "human_reviewed": False, "rubric": review_file["rubric"],
        }
    return assessed, True


def summarize(records):
    groups = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        passed = sum(r["verification"]["status"] == "PASS" for r in rows)
        groups[condition] = {
            "examples": len(rows), "verified": passed, "pass_at_1": passed / len(rows),
            "failures": dict(Counter(r["failure_category"] for r in rows if r["failure_category"])),
            "generation_errors": sum(r["generation_error"] is not None for r in rows),
            "verification_stages": dict(Counter(r["verification"]["stage"] for r in rows)),
            "structure_statuses": dict(Counter(r["argument_fidelity"]["status"] for r in rows)),
            "assistant_fidelity_statuses": dict(Counter(r["argument_fidelity"].get("assistant_review", {}).get("status", "NOT_REVIEWED") for r in rows)),
            "reviewed_output_kinds": dict(Counter(r["argument_fidelity"].get("assistant_review", {}).get("output_kind", "NOT_REVIEWED") for r in rows)),
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
            "median_generation_latency_seconds": statistics.median(r["generation_latency_seconds"] for r in rows),
            "total_generation_latency_seconds": sum(r["generation_latency_seconds"] for r in rows),
            "length_limited": sum(r["finish_reason"] == "max_new_tokens" for r in rows),
        }
    paired = Counter()
    by_key = {r["key"]: r for r in records}
    for example_id in sorted({r["example_id"] for r in records}):
        with_proof, alone = (by_key[f"{example_id}:{condition}"]["verification"]["status"] == "PASS" for condition in CONDITIONS)
        paired["both_pass" if with_proof and alone else "informal_only_pass" if with_proof else "theorem_only_pass" if alone else "neither_pass"] += 1
    return {"conditions": groups, "paired_verification_outcomes": dict(paired),
            "total_generation_attempts": len(records), "repair_attempts": 0, "fine_tuning_performed": False}


def build_report(directory, reverify=False):
    manifest, records = load_run(directory, reverify=reverify)
    assessed, reviewed = attach_reviews(directory, records)
    summary = summarize(assessed)
    summary["argument_reviews_complete"] = reviewed
    summary["source_results_sha256"] = sha256_file(directory / "results.jsonl")
    summary["development_count"] = 30
    summary["evaluation_count"] = 12
    summary["leakage_note"] = "No exact/alpha/definitional duplicates with development; six generation families overlap. This is a within-fragment diagnostic, not held-out-family generalization."
    summary["report_inputs_sha256"] = {"baseline/report.py": sha256_file(ROOT / "baseline/report.py")}
    if reviewed:
        summary["report_inputs_sha256"]["argument_reviews.json"] = sha256_file(directory / "argument_reviews.json")
    write_json(directory / "summary.json", summary)
    (directory / "assessed_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assessed), encoding="utf-8")
    lines = ["# ProofBridge baseline v1", "",
             f"Model: [{manifest['model']['model_id']}]({manifest['model']['model_card']}), revision `{manifest['model']['revision']}`.", "",
             "30 verified seeds are frozen as development data. The 12 new evaluation reference proofs all pass Rocq 9.2.0. No development examples or reference code were included in model prompts.", "",
             "Each example received one greedy completion per condition, with no repair: 24 attempts total. Inference used float32 on CPU, four threads, seed 1729, and a 256-new-token limit. Model/tokenizer artifacts, dependencies, prompts, and data are pinned and hashed. Only whitespace and one complete outer code fence may be removed before verification.", "",
             "| Input condition | Rocq PASS | Structure match | Assistant: faithful | Output tokens | Median generation time |",
             "| --- | --- | --- | --- | --- | --- |"]
    for condition, metrics in summary["conditions"].items():
        faithful = f"{metrics['assistant_fidelity_statuses'].get('FAITHFUL', 0)}/12" if reviewed else "Not reviewed"
        lines.append(f"| {condition} | {metrics['verified']}/12 ({metrics['pass_at_1']:.1%}) | {metrics['structure_statuses'].get('STRUCTURE_MATCH', 0)}/12 | {faithful} | {metrics['completion_tokens']} | {metrics['median_generation_latency_seconds']:.2f}s |")
    lines += ["", "Paired verification outcomes: " + ", ".join(f"{key}: {value}" for key, value in summary["paired_verification_outcomes"].items()) + ".", "",
              "| Failure category | Theorem + informal proof | Theorem only |", "| --- | --- | --- |"]
    categories = sorted({category for m in summary["conditions"].values() for category in m["failures"]})
    for category in categories:
        counts = [summary["conditions"][condition]["failures"].get(category, 0) for condition in CONDITIONS]
        lines.append(f"| {category} | {counts[0]} | {counts[1]} |")
    if all(r["verification"]["stage"] == "policy" for r in records):
        lines += ["", "Every candidate was rejected by the existing policy parser before Rocq compilation. These results measure a failure to produce an accepted proof body; they do not establish how often well-formed generated proofs would be mathematically correct."]
    if reviewed:
        kinds = Counter(r["argument_fidelity"]["assistant_review"].get("output_kind") for r in assessed)
        if kinds.get("informal_proof_echo"):
            unassessable = sum(r["argument_fidelity"]["assistant_review"]["status"] == "UNASSESSABLE" for r in assessed)
            lines += ["", f"Manual inspection found {kinds['informal_proof_echo']} outputs that exactly repeated the supplied English proof after `Proof:`, {kinds.get('prompt_contract_echo', 0)} that echoed the tactic instructions, and {kinds.get('informal_reasoning', 0)} prose-only reasoning outputs. Formal-fidelity reviews marked UNASSESSABLE: {unassessable}/24. Input copying is not credited as successful formalization."]
    length_counts = [summary["conditions"][condition]["length_limited"] for condition in CONDITIONS]
    lines += ["", f"Token-limit stops: theorem + informal proof {length_counts[0]}/12; theorem only {length_counts[1]}/12. This first fixed-prompt run is dominated by output-contract failures and does not establish the benefit of informal proofs for successful formalization."]
    lines += ["", "Argument fidelity is separate from correctness. The reproducible proxy compares induction/binder position, supplied-premise use, congruence, symmetry, rewrite direction, and branch closing steps. A structure match is not a proof of English fidelity. Assistant reviews assess whether the output follows the supplied reference argument, including for theorem-only outputs where that argument was hidden; they are not independent human judgments.", "",
              "This small diagnostic shares six argument families with development, although all 12 statements are new under binder renaming and definitional normalization. It cannot establish held-out-family generalization, exclude pretraining contamination, or show that a matching model output causally used the informal proof. Latencies are hardware-dependent; the fixed software/settings do not guarantee bit-identical outputs across platforms.", "",
              "Files: `results.jsonl` contains immutable raw prompts, tokens, completions, proof bodies, verifier evidence, and the fidelity proxy; `assessed_results.jsonl` adds any hash-bound assistant reviews; `summary.json` contains machine-readable aggregates. `manifest.json`, `completion.json`, and `attempts.jsonl` record provenance and the one-attempt budget."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["conditions"], indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=lambda p: ROOT / p, default=ROOT / "results/baseline-v1")
    parser.add_argument("--reverify", action="store_true", help="Rerun the existing verifier on all saved candidates, without model generation")
    args = parser.parse_args()
    build_report(args.run_dir, reverify=args.reverify)
