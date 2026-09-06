"""Read-only replay validation and reports; never selects a prompt from evaluation."""

import argparse
from collections import Counter
import json

from baseline.dataset import ROOT, write_json
from baseline.protocol import CONDITIONS
from prompt_v2.experiment import DEV_OUTPUT, EVAL_OUTPUT, load_frozen, load_phase
from prompt_v2.protocol import FROZEN, format_compliance, metrics
from verifier import sha256_file, verify


def reverify(manifest, rows):
    for row in rows:
        actual = verify(row["formal_statement"], row["proof_body"], timeout=manifest["verifier_timeout_seconds"])
        saved = row["verification"]
        fields = ["status", "category", "stage", "source_sha256", "kernel_checked", "assumptions_checked"]
        if any(getattr(actual, key) != saved[key] for key in fields):
            raise ValueError(f"Verifier replay disagrees: {row['key']}")
    print(f"PASS: {len(rows)} saved outputs reverified with identical outcomes; no model calls")


def report_development(recheck=False):
    manifest, rows = load_phase(DEV_OUTPUT)
    if recheck:
        reverify(manifest, rows)
    summary = json.loads((DEV_OUTPUT / "summary.json").read_text())
    lines = ["# Prompt v2 development comparison", "",
             "All prompt development used the frozen 30 development seeds. The three candidate prompts and selection rule were declared before generation. No evaluation record was loaded by development or prompt selection.", "",
             "Each candidate received one attempt on each seed in both conditions: 180 generations total, with no repair or fine-tuning. The same Qwen model revision, greedy decoding, 256-new-token cap, CPU/float32 settings, and verifier were retained.", "",
             "| Candidate | Target input | Raw format | Rocq verified | Verified structure match |",
             "| --- | --- | --- | --- | --- |"]
    for candidate, entry in summary["prompts"].items():
        for condition, m in entry["all_examples"].items():
            lines.append(f"| {candidate} | {condition} | {m['format_compliant']}/{m['outputs']} | {m['rocq_verified']}/{m['outputs']} | {m['verified_structure_matches']}/{m['outputs']} |")
    excluded = summary["selection_partition"]["excluded_demonstration_or_equivalent_ids"]
    lines += ["", "The shared selection subset excludes " + ", ".join(excluded) + ": the union of demonstrations and their definitional duplicates. This leaves 25 seeds, scored under both conditions (50 outputs per candidate). Full-set metrics above include demonstration targets and are not held-out accuracy.", "",
              "| Selection rank | Candidate | Raw format | Rocq verified | Verified structure match |", "| --- | --- | --- | --- | --- |"]
    for index, entry in enumerate(summary["ranking"], 1):
        m = entry["selection_metrics"]
        lines.append(f"| {index} | {entry['candidate_id']} | {m['format_compliant']}/50 | {m['rocq_verified']}/50 | {m['verified_structure_matches']}/50 |")
    lines += ["", f"Selected by the predeclared lexicographic rule: **{summary['ranking'][0]['candidate_id']}**. Raw format takes precedence, then verification, then verified structure agreement, then fewer demonstrations and candidate order.", "",
              "Raw format means the nonempty output passes the unchanged restricted parser after trimming edge whitespace only. Markdown fences fail that metric. Verification retains v1's narrowly defined outer-fence extraction, so a fenced valid proof can verify while failing format compliance. All raw and extracted texts are preserved.", "",
              "Argument fidelity here is a deterministic structure proxy, recorded separately from correctness. It compares induction/binder position, actual premise use, congruence/symmetry, rewrite references/directions, and branch closing steps. Verified structure match requires both kernel verification and a full proxy match; it is not an independent certification of English semantics.", "",
              "Fixed demonstrations are identical in both conditions and include their informal arguments. Only the final target's informal proof is ablated. The demonstrations are copied exactly from development and are not retrieved or adapted per target.", "",
              "Files: `manifest.json` preserves prompts, plan, source hashes, and settings; `results.jsonl` preserves all 180 outputs and scores; `attempts.jsonl` and `completion.json` audit the one-attempt budget; `summary.json` preserves the selection metrics. The selected prompt is separately frozen in `prompt_v2/selected.json` and `prompt_v2/frozen.json` before diagnostic evaluation."]
    (DEV_OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["ranking"], indent=2))


def report_evaluation(recheck=False):
    freeze, selected = load_frozen()
    manifest, rows = load_phase(EVAL_OUTPUT)
    if recheck:
        reverify(manifest, rows)
    # Historical evaluation data is used only here, after v2 has been frozen/run.
    from baseline.report import attach_reviews, load_run
    v1_manifest, v1_rows = load_run(ROOT / "results/baseline-v1")
    if (manifest["generation_settings"] != v1_manifest["configuration"]["generation"]
            or manifest["model"] != v1_manifest["model"]
            or manifest["effective_generation_config"] != v1_manifest["effective_generation_config"]):
        raise ValueError("V2 did not retain the v1 model and generation budget")
    assessed, reviewed = attach_reviews(EVAL_OUTPUT, rows)
    summary = {
        "selected_prompt": selected["id"], "demonstration_ids": selected["demonstration_ids"],
        "prompt_frozen_at_utc": freeze["frozen_at_utc"], "evaluation_started_at_utc": manifest["started_at_utc"],
        "prompt_freeze_sha256": sha256_file(FROZEN), "development_only_selection": True,
        "evaluation_generation_attempts": len(rows), "repair_attempts": 0, "fine_tuning_performed": False,
        "same_model_and_generation_settings_as_v1": True,
        "v2": {c: metrics([r for r in rows if r["condition"] == c]) for c in CONDITIONS},
        "v1": {c: metrics([r | {"format_compliance": format_compliance(r["formal_statement"], r["generated_text"])} for r in v1_rows if r["condition"] == c]) for c in CONDITIONS},
        "assistant_reviews_complete": reviewed,
        "assistant_fidelity": {c: dict(Counter(r["argument_fidelity"].get("assistant_review", {}).get("status", "NOT_REVIEWED") for r in assessed if r["condition"] == c)) for c in CONDITIONS},
        "files_sha256": {"results.jsonl": sha256_file(EVAL_OUTPUT / "results.jsonl"), "prompt_v2/report.py": sha256_file(ROOT / "prompt_v2/report.py")},
    }
    if reviewed:
        summary["files_sha256"]["argument_reviews.json"] = sha256_file(EVAL_OUTPUT / "argument_reviews.json")
    by_condition = {c: {r["example_id"] for r in rows if r["condition"] == c and r["verification"]["status"] == "PASS"} for c in CONDITIONS}
    full, alone = (by_condition[c] for c in CONDITIONS)
    summary["paired_outcomes"] = {"both_pass": len(full & alone), "informal_only_pass": len(full - alone), "theorem_only_pass": len(alone - full), "neither_pass": 12 - len(full | alone)}
    write_json(EVAL_OUTPUT / "report_summary.json", summary)
    (EVAL_OUTPUT / "assessed_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assessed))
    lines = ["# Prompt v2 diagnostic evaluation", "",
             f"Frozen prompt: **{selected['id']}**, using {len(selected['demonstration_ids'])} fixed development demonstrations ({', '.join(selected['demonstration_ids'])}). Selection used only development metrics; freeze time {freeze['frozen_at_utc']} precedes evaluation start {manifest['started_at_utc']}.", "",
             "The same Qwen2.5-Coder-0.5B-Instruct revision, existing verifier, greedy settings, and 256-new-token budget were used. Each of the unchanged 12 examples received one attempt per condition: 24 generations, no repair or fine-tuning.", "",
             "| Version | Target input | Raw format | Rocq verified | Verified structure match |", "| --- | --- | --- | --- | --- |"]
    for version in ["v1", "v2"]:
        for condition, m in summary[version].items():
            lines.append(f"| {version} | {condition} | {m['format_compliant']}/12 | {m['rocq_verified']}/12 | {m['verified_structure_matches']}/12 |")
    lines += ["", "V2 paired correctness: " + ", ".join(f"{k}={v}" for k, v in summary["paired_outcomes"].items()) + ".", "",
              "| Target input | Failure categories | Median generation time | Completion tokens |", "| --- | --- | --- | --- |"]
    for condition, m in summary["v2"].items():
        lines.append(f"| {condition} | {json.dumps(m['failure_categories'])} | {m['median_generation_latency_seconds']:.2f}s | {m['completion_tokens']} |")
    if reviewed:
        lines += ["", "Assistant semantic-fidelity reviews (not independent human review): " + "; ".join(f"{c}: {json.dumps(statuses)}" for c, statuses in summary["assistant_fidelity"].items()) + "."]
    lines += ["", "Format compliance tests raw code with whitespace trimming only; fences, explanations, and declarations fail. Rocq verification uses precisely v1's output extraction and unchanged policy/kernel checks. A format-compliant proof can still fail mathematically.", "",
              "Verified structure match requires both a verified proof and agreement with the fixed argument-structure proxy. It is stricter than correctness but does not certify English fidelity. A valid alternative can mismatch the proxy. The theorem-only condition is scored against the same hidden reference argument.", "",
              "Both conditions have identical development demonstrations, including their informal arguments; only the final target's informal proof differs. Evaluation shares arithmetic families with those demonstrations and was already used for the v1 diagnostic. This comparison does not establish unseen-family generalization, exclude pretraining contamination, or demonstrate causal use of every informal step.", "",
              "`results.jsonl` is the immutable record of every prompt, output, proof body, verifier result, format/fidelity score, token IDs, usage, settings, and latency. `assessed_results.jsonl` attaches available assistant reviews. `report_summary.json` records the comparison; v1 files and all datasets remain unchanged."]
    (EVAL_OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"v2": summary["v2"], "assistant_fidelity": summary["assistant_fidelity"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["development", "evaluation"])
    parser.add_argument("--reverify", action="store_true")
    args = parser.parse_args()
    (report_development if args.phase == "development" else report_evaluation)(args.reverify)
