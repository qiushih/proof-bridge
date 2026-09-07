"""Predeclared step adherence and separate reviewed mathematical fidelity."""

from collections import Counter
import statistics

from pilot_v1.protocol import commands
from verifier import Rejection, render_source

CONDITIONS = ("theorem_only", "argument_A", "argument_B")


def normalized_trace(statement, body):
    render_source(statement, body)
    introduced, ih, trace = [], None, []
    for branch, command in commands(body):
        tactic, *args = command
        if tactic == "intros":
            introduced.extend(args)
            continue
        if tactic == "induction":
            ih = command[-2]
            trace.append([branch, "induction", introduced.index(args[0])])
        elif tactic in ("rewrite", "exact", "apply"):
            reference = "IH" if args[-1] == ih else f"local:{introduced.index(args[-1])}" if args[-1] in introduced else "other"
            trace.append([branch, "rewrite", reference, "reverse" if "<-" in args else "forward"] if tactic == "rewrite"
                         else [branch, "use", reference])
        else:
            trace.append([branch, tactic])
    return trace


def step_adherence(example, body, condition):
    if condition == "theorem_only":
        return {"status": "NOT_APPLICABLE", "matched": None, "reason": "No argument supplied"}
    try:
        actual = normalized_trace(example["formal_statement"], body)
    except Rejection as error:
        return {"status": "UNASSESSABLE", "matched": False, "reason": str(error)}
    expected = normalized_trace(example["formal_statement"], example["proof_body"])
    return {"status": "MATCH" if actual == expected else "MISMATCH", "matched": actual == expected,
            "actual_trace": actual, "reference_trace": expected,
            "method": "ordered branch/tactic trace; ignore local names and intros grouping; exact/apply same local equivalent; rewrite/ rewrite -> equivalent",
            "limitation": "A step mismatch does not imply mathematical unfaithfulness. Matching an unverified script is not success."}


def mathematical_structure(statement, body):
    try:
        trace = normalized_trace(statement, body)
    except Rejection as error:
        return {"status": "UNASSESSABLE", "reason": str(error)}
    induction = [r for r in trace if len(r) > 1 and r[1] == "induction"]
    ih_used = any(r[0] == 2 and len(r) > 2 and r[1] in ("rewrite", "use") and r[2] == "IH" for r in trace)
    return {"status": "COMPATIBLE" if induction == [[0, "induction", 0]] and ih_used else "DIFFERS",
            "induction_on_first_binder": induction == [[0, "induction", 0]], "IH_used_in_successor": ih_used,
            "limitation": "Review aid only: mathematical fidelity requires the separate reviewed judgment and Rocq verification."}


def attach_reviews(raw_rows, reviews):
    by_key = {r["key"]: r for r in reviews}
    if len(by_key) != len(reviews) or set(by_key) != {r["key"] for r in raw_rows}:
        raise ValueError("Every unique raw output needs a review")
    rows = []
    for raw in raw_rows:
        review = by_key[raw["key"]]
        if review["proof_body_sha256"] != raw["proof_body_sha256"] or not review["reason"].strip():
            raise ValueError("Unbound/empty mathematical review")
        status = review["status"]
        if status not in ("FAITHFUL", "UNFAITHFUL", "NOT_ESTABLISHED", "NOT_APPLICABLE"):
            raise ValueError("Invalid mathematical-fidelity judgment")
        if (raw["condition"] == "theorem_only") != (status == "NOT_APPLICABLE"):
            raise ValueError("Theorem-only has no requested argument")
        if status == "FAITHFUL" and raw["verification"]["status"] != "PASS":
            raise ValueError("Failed proof cannot establish mathematical faithfulness")
        rows.append(raw | {"mathematical_argument_fidelity": review})
    return rows


def selection_score(rows, optimizer_step):
    requested = [r for r in rows if r["condition"] != "theorem_only"]
    ids = [r["dev_example_id"] for r in requested]
    if sorted(ids) != [f"pd0{i}_{a}" for i in (1, 2, 3) for a in ("A", "B")]:
        raise ValueError("Selection requires exactly the six frozen dev argument rows")
    passed = lambda r: r["verification"]["status"] == "PASS"
    faithful = sum(passed(r) and r["mathematical_argument_fidelity"]["status"] == "FAITHFUL" for r in requested)
    steps = sum(passed(r) and r["requested_proof_steps"]["matched"] is True for r in requested)
    return [faithful, steps, sum(passed(r) for r in requested), -optimizer_step]


def summarize(rows):
    def metrics(group):
        requested = [r for r in group if r["condition"] != "theorem_only"]
        verified = sum(r["verification"]["status"] == "PASS" for r in group)
        faithful = sum(r["mathematical_argument_fidelity"]["status"] == "FAITHFUL" for r in requested)
        step_match = sum(r["requested_proof_steps"]["matched"] is True for r in requested)
        verified_steps = sum(r["requested_proof_steps"]["matched"] is True and r["verification"]["status"] == "PASS" for r in requested)
        return {"outputs": len(group), "verified": verified, "verification_accuracy": verified / len(group),
            "argument_outputs": len(requested), "verified_mathematically_faithful": faithful if requested else None,
            "mathematical_fidelity_accuracy": faithful / len(requested) if requested else None,
            "requested_steps_matched": step_match if requested else None,
            "verified_requested_steps": verified_steps if requested else None,
            "verified_requested_step_accuracy": verified_steps / len(requested) if requested else None,
            "failure_categories": dict(Counter(r["verification"]["category"] for r in group if r["verification"]["status"] != "PASS")),
            "mean_latency_seconds": statistics.mean(r["generation_latency_seconds"] for r in group),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in group)}
    if len(rows) != 9 or {(r["theorem_id"],r["condition"]) for r in rows} != {(f"pd0{i}",c) for i in (1,2,3) for c in CONDITIONS}:
        raise ValueError("Expected exactly nine unique baseline outputs")
    pairs = []
    for theorem in ("pd01", "pd02", "pd03"):
        a, b = (next(r for r in rows if r["theorem_id"] == theorem and r["condition"] == c) for c in ("argument_A", "argument_B"))
        pairs.append({"theorem_id": theorem, "body_changed": a["proof_body"] != b["proof_body"],
            "both_verified_and_step_adherent": all(r["verification"]["status"] == "PASS" and r["requested_proof_steps"]["matched"] for r in (a,b)),
            "both_verified_and_mathematically_faithful": all(r["mathematical_argument_fidelity"]["status"] == "FAITHFUL" for r in (a,b))})
    return {"all_outputs": metrics(rows), "by_condition": {c: metrics([r for r in rows if r["condition"] == c]) for c in CONDITIONS},
            "step_switch_pairs": pairs, "step_zero_selection_score": selection_score(rows, 0),
            "note": "Step-switching between equivalent IH tactics is not high-level mathematical strategy switching. Theorem-only is excluded from selection and argument-fidelity denominators."}
