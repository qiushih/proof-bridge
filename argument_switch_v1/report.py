"""Report a complete probe without inference or changes to frozen evidence."""

import json
import statistics

from argument_switch_v1.experiment import OUTPUT, check
from argument_switch_v1.protocol import CONDITIONS, load_variants
from baseline.dataset import write_json
from verifier import sha256_file


def build_report():
    manifest, rows = check()
    summary = json.loads((OUTPUT / "summary.json").read_text())
    manual = json.loads((OUTPUT / "manual_reviews.json").read_text())
    if manual["results_sha256"] != sha256_file(OUTPUT / "results.jsonl"):
        raise ValueError("Manual review does not match saved outputs")
    reviews = {r["key"]: r for r in manual["reviews"]}
    if len(manual["reviews"]) != 18 or set(reviews) != {r["key"] for r in rows}:
        raise ValueError("Review all 18 outputs exactly once")
    assessed = []
    for row in rows:
        review = reviews[row["key"]]
        if review["proof_body_sha256"] != row["proof_body_sha256"] or not review["rationale"]:
            raise ValueError("Stale/empty manual review")
        assessed.append(row | {"manual_review": review})
    by_key = {r["key"]: r for r in assessed}
    variants = {v["example_id"]: v for v in load_variants()}
    switch = summary["switching"]
    lines = ["# Development argument-switching probe", "",
             f"**Verified switching accuracy: {switch['successful_theorems']}/6 ({100*switch['switching_accuracy']:.1f}%).** Both A and B must verify and follow their respective fixed strategy requirements. A verified proof that ignores the supplied strategy is a switching failure.", "",
             f"**Verification accuracy: {summary['all_outputs']['verified']}/18 ({100*summary['all_outputs']['verification_accuracy']:.1f}%).** Exactly 18 new generations, one per theorem/condition, no repair or fine-tuning.", "",
             "Constrained-v1 was frozen first at Git tag `proofbridge-constrained-v1` (checkpoint commit `b20e388`). Its 50-file manifest protects the decoder, prompt/runtime pins, prior evidence, and tests. All 12 A/B references then passed Rocq and exact-token-path checks under that decoder before inference; nine probe tests passed. Reference preparation and scoring were fixed in the preflight manifest.", "",
             "## Accuracy by condition", "",
             "| Input | Rocq verified | Attempted requested strategy | Verified + requested strategy |", "| --- | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        m = summary["by_condition"][condition]
        attempted = "N/A" if m["requested_strategy_matched"] is None else f"{m['requested_strategy_matched']}/6"
        verified_strategy = "N/A" if m["verified_strategy_followed"] is None else f"{m['verified_strategy_followed']}/6"
        lines.append(f"| {condition} | {m['verified']}/6 | {attempted} | {verified_strategy} |")
    lines += ["", f"Attempted structural switching, independent of mathematical correctness, was **{switch['attempted_switches']}/6**. The primary denominator remains all six theorems, including failures. Theorem-only has no supplied strategy and is reported as a control.", "",
              "## Per-theorem results", "",
              "| Seed | Strategy A | Strategy B | Theorem-only | A output | B output | Verified switch |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for pair in summary["per_theorem"]:
        v = variants[pair["example_id"]]
        a, b, t = (by_key[pair[k]] for k in ("A_key", "B_key", "theorem_only_key"))
        def result(row):
            match = row["strategy"]["observed_matching_arguments"]
            return row["verification"]["status"] + "; " + ("/".join(match) if match else "neither")
        binder = "p" if v["example_id"] == "seed_011" else "n"
        lines.append(f"| {v['example_id']} | {v['arguments']['A']['strategy']} | induction on {binder} | {result(t)} | {result(a)} | {result(b)} | {'yes' if pair['verified_strategy_switch'] else 'no'} |")
    lines += ["", "A/B in the output columns identify which reference strategy the output matches. Scoring checks the induction variable and required branch-local uses of IH/premises, not just the presence of an induction keyword. Equivalent tactics are accepted; the saved structure proxy remains separate. All 18 outputs were also manually reviewed by the assistant, not an independent human reviewer.", "",
              "## Successful and failed switching examples", ""]
    successes = [p for p in summary["per_theorem"] if p["verified_strategy_switch"]]
    failures = [p for p in summary["per_theorem"] if not p["verified_strategy_switch"]]
    if not successes:
        lines += ["No verified successful A-to-B switch was observed. No success example is invented.", ""]
    examples = successes[:1]
    # Prefer a failure with a verified B output that uses A's strategy, if present.
    ignored = [p for p in failures if by_key[p["B_key"]]["verification"]["status"] == "PASS" and not by_key[p["B_key"]]["strategy"]["requested_strategy_matched"]]
    examples += (ignored[:1] or failures[:1])
    # Also show one failure that attempts B but cannot verify, if different.
    mathematical = [p for p in failures if by_key[p["B_key"]]["strategy"]["requested_strategy_matched"] and by_key[p["B_key"]]["verification"]["status"] == "FAIL" and p not in examples]
    examples += mathematical[:1]
    for pair in examples:
        lines += [f"### {pair['example_id']}: {'successful' if pair['verified_strategy_switch'] else 'failed'} switch", ""]
        for label in ("A", "B"):
            row = by_key[pair[label + "_key"]]
            lines += [f"Given argument {label}, the generated proof was {row['verification']['status']} and its requested-strategy match was {row['strategy']['requested_strategy_matched']}:", "", "```rocq", row["proof_body"], "```", "", row["manual_review"]["rationale"], ""]
    lines += ["## Interpretation and limits", "",
              "All six B outputs choose induction on the requested binder, including p in seed_011. Thus the result is not evidence that the model ignores the informal text entirely. In seeds 004–006 it changes from A's valid direct argument to the requested induction outline, but cannot complete a valid alternative proof. No verified successful switch was observed.", "",
              "The B outputs repeatedly use simplification/reflexivity for the base case and forward IH rewriting for the successor. This template ignores the reverse rewrites required by B in 004–006 and the branch-specific premise uses in 007 and 011. In 003 the model also uses induction when A requests only computation. These concrete failures support limited structural responsiveness with poor execution of the supplied argument.", "",
              "Verification alone is insufficient: an independently generated direct proof does not satisfy an induction request. Conversely, an output with the requested induction outline can still fail because an IH rewrite is inapplicable. The separate metrics distinguish these cases.", "",
              "These six theorems were already development examples. Four compare definitional computation with deliberately unnecessary induction; 007 compares direct premise use with induction; 011 compares rewriting with induction on the third binder. Several B references use IH redundantly. This is a small strategy-compliance diagnostic, not an unseen-family benchmark or proof of semantic understanding.", "",
              "Only the target informal paragraph differs between A and B. Frozen demonstrations, model, decoder, verifier, and greedy 256-token settings are unchanged. No diagnostic evaluation example was used, no existing theorem or seed was edited, and no model output was repaired. Manual reviews are recorded separately and do not change the predeclared scores.", "",
              "## Artifacts and reproduction", "",
              "- `argument_switch_v1/variants.json`: both informal paragraphs, formal bodies, strategy requirements, and provenance for each theorem.",
              "- `argument_switch_v1/references/seed_XXX_A.v` and `_B.v`: all 12 standalone reference proof files.",
              "- `argument_switch_v1/preflight.json`: Rocq and decoder acceptance evidence before inference.",
              "- `results.jsonl`, `manifest.json`, `attempts.jsonl`, `summary.json`, `completion.json`: immutable machine-readable generation and scoring evidence.",
              "- `manual_reviews.json` and `assessed_results.jsonl`: separate assistant review and combined records.",
              "- `argument_switch_v1/README.md`: exact reproduction/check commands.", ""]
    extra = {"summary": summary, "reviewer": "assistant", "human_reviewed": False,
             "files_sha256": {name: sha256_file(OUTPUT / name) for name in ("results.jsonl", "summary.json", "manual_reviews.json")},
             "latency_by_condition": {c: {"median_seconds": statistics.median(r["generation_latency_seconds"] for r in rows if r["condition"] == c),
                                           "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows if r["condition"] == c)} for c in CONDITIONS}}
    return "\n".join(lines), assessed, extra


if __name__ == "__main__":
    text, assessed, extra = build_report()
    (OUTPUT / "ARGUMENT_SWITCH_REPORT.md").write_text(text)
    (OUTPUT / "assessed_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assessed))
    write_json(OUTPUT / "report_summary.json", extra)
    print(json.dumps(extra["summary"]["switching"], indent=2))
