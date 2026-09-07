"""Compare constrained and saved unconstrained development attempts only."""

from collections import Counter
import json
import statistics

from baseline.dataset import ROOT, digest, write_json
from constrained_v1.experiment import OUTPUT, check
from verifier import sha256_file


def failure_counts(row):
    if "primary_failure_category" in row:
        category = row["primary_failure_category"]
        return category == "invalid_hypothesis_reference", category == "missing_introductions"
    if row["verification"]["status"] == "PASS":
        return False, False
    message = row["verification"]["message"]
    missing = message == "Introduce all theorem binders before induction on a local."
    invalid = message == "Too many introduction names." or message.startswith("Reference is not an in-scope local:")
    return invalid, missing


def scores(rows):
    n = len(rows)
    passed = sum(r["verification"]["status"] == "PASS" for r in rows)
    faithful = sum(r["verification"]["status"] == "PASS" and r["manual_fidelity"]["status"] == "FAITHFUL" for r in rows)
    format_ok = sum(r["format_compliance"]["compliant"] for r in rows)
    invalid = sum(failure_counts(r)[0] for r in rows)
    missing = sum(failure_counts(r)[1] for r in rows)
    latencies = [r["generation_latency_seconds"] for r in rows]
    return {"outputs": n, "format_compliant": format_ok, "format_compliance_pct": round(100 * format_ok / n, 2),
            "invalid_reference_failures": invalid, "invalid_reference_failure_pct": round(100 * invalid / n, 2),
            "missing_introduction_failures": missing, "missing_introduction_failure_pct": round(100 * missing / n, 2),
            "rocq_verified": passed, "rocq_verification_pct": round(100 * passed / n, 2),
            "verified_faithful": faithful, "verified_faithful_pct": round(100 * faithful / n, 2),
            "verified_structure_matches": sum(r["verification"]["status"] == "PASS" and r["argument_fidelity"]["status"] == "STRUCTURE_MATCH" for r in rows),
            "median_generation_latency_seconds": statistics.median(latencies),
            "mean_generation_latency_seconds": statistics.mean(latencies), "total_generation_latency_seconds": sum(latencies),
            "generation_errors": sum(r.get("generation_error") is not None for r in rows),
            "length_limited": sum(r.get("finish_reason") == "max_new_tokens" for r in rows),
            "failure_categories": dict(Counter(r["verification"]["category"] for r in rows if r["verification"]["status"] != "PASS")),
            "manual_fidelity_statuses": dict(Counter(r["manual_fidelity"]["status"] for r in rows)),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows)}


def build_report():
    manifest, rows = check()
    audit = json.loads((ROOT / "results/prompt-v2-development-audit/failure_analysis.json").read_text())
    from prompt_v2.experiment import DEV_OUTPUT, load_phase
    _, baseline_raw = load_phase(DEV_OUTPUT)
    raw_by_key = {r["key"]: r for r in baseline_raw}
    baseline = []
    for reviewed in audit["records"]:
        if reviewed["prompt_id"] == "short_3shot":
            original = raw_by_key[reviewed["key"]]
            if reviewed["source_row_sha256"] != digest(json.dumps(original, sort_keys=True)):
                raise ValueError("Baseline audit/source mismatch")
            baseline.append(original | {"manual_fidelity": reviewed["manual_fidelity"],
                                        "primary_failure_category": reviewed["primary_failure_category"]})
    review_path = OUTPUT / "argument_reviews.json"
    reviews = json.loads(review_path.read_text())
    if reviews["results_sha256"] != sha256_file(OUTPUT / "results.jsonl"):
        raise ValueError("Manual reviews belong to different outputs")
    by_key = {r["key"]: r for r in reviews["reviews"]}
    if len(by_key) != 60 or len(reviews["reviews"]) != 60 or set(by_key) != {r["key"] for r in rows}:
        raise ValueError("Review all 60 outputs exactly once")
    assessed = []
    for row in rows:
        review = by_key[row["key"]]
        if review["proof_body_sha256"] != row["proof_body_sha256"] or not review["rationale"]:
            raise ValueError("Stale or empty manual review")
        if review["status"] not in ("FAITHFUL", "PARTIAL", "DIVERGENT", "UNASSESSABLE"):
            raise ValueError("Unknown fidelity status")
        if review["status"] == "FAITHFUL" and row["verification"]["status"] != "PASS":
            raise ValueError("A complete faithful proof must verify")
        assessed.append(row | {"manual_fidelity": review})
    comparison = {}
    for subset in ("all_30", "common_25"):
        grouped = {}
        for version, data in (("unconstrained_v2", baseline), ("constrained_v1", assessed)):
            selected_rows = data if subset == "all_30" else [r for r in data if r["selection_eligible"]]
            grouped[version] = {c: scores([r for r in selected_rows if r["condition"] == c]) for c in manifest["conditions"]}
            grouped[version]["pooled"] = scores(selected_rows)
        comparison[subset] = grouped
    b = comparison["common_25"]["unconstrained_v2"]["pooled"]["verified_faithful"]
    c = comparison["common_25"]["constrained_v1"]["pooled"]["verified_faithful"]
    paired = []
    base_map = {r["key"]: r for r in baseline}
    for row in assessed:
        original = base_map[row["baseline_key"]]
        paired.append({"key": row["key"], "baseline_key": original["key"], "selection_eligible": row["selection_eligible"],
                       "body_changed": row["proof_body"] != original["proof_body"],
                       "baseline_verified_faithful": original["verification"]["status"] == "PASS" and original["manual_fidelity"]["status"] == "FAITHFUL",
                       "constrained_verified_faithful": row["verification"]["status"] == "PASS" and row["manual_fidelity"]["status"] == "FAITHFUL"})
    summary = {"experiment_id": manifest["experiment_id"], "comparison": comparison,
               "success": {"criterion": "More verified faithful proofs on the common 25 seeds, pooled two conditions", "met": c > b,
                           "baseline": b, "constrained": c, "denominator": 50, "delta": c - b},
               "provenance": {"results_sha256": sha256_file(OUTPUT / "results.jsonl"), "manual_reviews_sha256": sha256_file(review_path),
                              "baseline_audit_sha256": sha256_file(ROOT / "results/prompt-v2-development-audit/failure_analysis.json"),
                              "report_script_sha256": sha256_file(ROOT / "constrained_v1/report.py"), "human_reviewed": False},
               "vocabulary_setup_seconds": manifest["vocabulary_setup_seconds"],
               "mask_seconds": sum(r["constraint"]["mask_seconds"] for r in rows),
               "paired": paired, "diagnostic_examples_used": 0, "repair_attempts": 0, "fine_tuning": False}
    return summary, assessed


def markdown(summary):
    success = summary["success"]
    lines = ["# Constrained decoding with frozen Prompt v2", "",
             f"**Success criterion {'met' if success['met'] else 'not met'}:** verified faithful proofs on the common 25-seed subset changed from **{success['baseline']}/50 to {success['constrained']}/50** (both conditions pooled).", "",
             "Exactly 60 new development generations: 30 seeds × two conditions, one attempt each, no repair. Prompt v2, model/tokenizer, verifier, data, greedy configuration, and 256-new-token budget were preserved. No diagnostic examples were used.", "",
             "Before inference, all 30 exact reference token paths including EOS passed the mask and all 30 reference proofs compiled in Rocq. Nine tests covered grammar, required introductions, invented references, IH/branch scope, malformed commands, and the command limit. `preflight.json` and the run manifest bind this evidence to the decoder sources before generation.", ""]
    for subset, title in (("all_30", "All 30 development seeds"), ("common_25", "Common 25 seeds; demonstrations/equivalents excluded")):
        lines += [f"## {title}", "", "| Version | Input | Format | Invalid refs | Missing intros | Rocq PASS | Verified + faithful | Median latency |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for version in ("unconstrained_v2", "constrained_v1"):
            for condition in ("theorem_and_informal", "theorem_only"):
                s = summary["comparison"][subset][version][condition]
                n = s["outputs"]
                lines.append(f"| {version} | {condition} | {s['format_compliant']}/{n} | {s['invalid_reference_failures']}/{n} | {s['missing_introduction_failures']}/{n} | {s['rocq_verified']}/{n} | {s['verified_faithful']}/{n} | {s['median_generation_latency_seconds']:.2f}s |")
        lines += [""]
    old = summary["comparison"]["all_30"]["unconstrained_v2"]
    new = summary["comparison"]["all_30"]["constrained_v1"]
    gains = sum(p["constrained_verified_faithful"] and not p["baseline_verified_faithful"] for p in summary["paired"])
    losses = sum(p["baseline_verified_faithful"] and not p["constrained_verified_faithful"] for p in summary["paired"])
    lines += ["Format is the unchanged verifier policy parser applied to raw output after edge whitespace trimming. Invalid-reference and missing-introduction counts use the development audit's specific root-cause definitions; invented excess H introductions count as invalid references. Rocq PASS and reviewed fidelity are distinct. Full counts, percentages, mean/median/total latency, token usage, failures, and paired outcomes are in `summary.json`.", "",
              f"There were **{gains} gains and {losses} losses** in verified faithful proofs. All 19 remaining failures reached Rocq: 18 proof errors and one incomplete proof. Three informal-condition attempts (027–029) repeated tactics to the 256-token limit. Their local names were in scope; induction/premise reasoning remained incorrect.", "",
              f"Mean generation latency with informal proofs increased from **{old['theorem_and_informal']['mean_generation_latency_seconds']:.2f}s to {new['theorem_and_informal']['mean_generation_latency_seconds']:.2f}s**; theorem-only changed from **{old['theorem_only']['mean_generation_latency_seconds']:.2f}s to {new['theorem_only']['mean_generation_latency_seconds']:.2f}s**. The long repeated outputs explain why the mean cost grows more than the median. Vocabulary construction took {summary['vocabulary_setup_seconds']:.2f}s separately; mask computation totaled {summary['mask_seconds']:.2f}s across all attempts.", "",
              "## Fidelity and limits", "",
              "All 60 new outputs were manually reviewed by the assistant against the same frozen informal arguments and reference code, including the theorem-only outputs. Equivalent local equality transport and induction-hypothesis congruence count as faithful; different strategies are flagged. These are not independent human reviews. Existing deterministic structure scores are retained separately in every record.", "",
              "The grammar observes only binder counts and generated code; it does not inspect reference proofs, informal text, goal states, or compiler feedback. It enforces proof-versus-nat reference kinds and branch scope, not mathematical applicability. An in-scope IH may still be unusable in the current goal. Correct syntax can still produce a wrong or unfaithful proof.", "",
              "Latency includes token-mask computation, but excludes model/vocabulary setup and verifier time. Vocabulary setup is reported separately; cache warm-up and output length affect timing. Unconstrained latency comes from the earlier saved run. These reused development results do not establish held-out generalization or causal use of every informal step.", "",
              "Reproduction and constraint details: `constrained_v1/README.md`. Raw outputs and mask traces: `results.jsonl`. Manual judgments: `argument_reviews.json`. Combined machine-readable records: `assessed_results.jsonl`. No raw output or previous artifact was repaired or overwritten.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    summary, assessed = build_report()
    write_json(OUTPUT / "summary.json", summary)
    (OUTPUT / "assessed_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assessed))
    (OUTPUT / "REPORT.md").write_text(markdown(summary))
    print(json.dumps(summary["success"], indent=2))
