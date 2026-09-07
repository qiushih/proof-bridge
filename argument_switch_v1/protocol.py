"""Predeclared strategy criteria and unchanged target-message construction."""

import json

from baseline.dataset import ROOT, check_development
from baseline.protocol import proof_signature
from checkpoints.freeze_constrained_v1 import check as check_checkpoint
from prompt_v2.experiment import load_frozen
from prompt_v2.protocol import build_messages
from verifier import Rejection

PLAN = ROOT / "argument_switch_v1/plan.json"
VARIANTS = ROOT / "argument_switch_v1/variants.json"
CONDITIONS = ("theorem_only", "argument_A", "argument_B")


def load_variants():
    check_checkpoint()
    freeze, _ = load_frozen()
    plan = json.loads(PLAN.read_text())
    variants = json.loads(VARIANTS.read_text())
    dev = {e["id"]: e for e in check_development()}
    if [v["example_id"] for v in variants] != plan["selected_ids"] or len(variants) != 6:
        raise ValueError("The six probe theorems must remain fixed")
    for variant in variants:
        example = dev[variant["example_id"]]
        if variant["example_id"] not in freeze["selection_partition"]["selection_ids"]:
            raise ValueError("Do not use demonstrations or their excluded equivalents")
        if variant["formal_statement"] != example["formal_statement"]:
            raise ValueError("Probe theorem differs from frozen seed")
        if set(variant["arguments"]) != {"A", "B"}:
            raise ValueError("Exactly two arguments are required")
        a, b = (variant["arguments"][key] for key in ("A", "B"))
        if a["informal_proof"] == b["informal_proof"] or a["requirements"]["uses_induction"] == b["requirements"]["uses_induction"]:
            raise ValueError("Need distinct informal arguments and mathematical strategies")
    return variants


def messages(variant, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown probe condition")
    _, selected = load_frozen()
    example = {"formal_statement": variant["formal_statement"]}
    if condition == "theorem_only":
        return build_messages(selected, example, "theorem_only")
    example["informal_proof"] = variant["arguments"][condition[-1]]["informal_proof"]
    return build_messages(selected, example, "theorem_and_informal")


def strategy_match(statement, body, requirements):
    try:
        signature = proof_signature(statement, body)
    except Rejection as error:
        return {"status": "UNASSESSABLE", "matches": False, "reason": str(error), "signature": None, "criteria": {}}
    trace = signature["rewrite_steps"] + signature["closing_steps"]

    def uses(kind, branch):
        return any(step["branch"] == branch and (step["action"] == "use:" + kind or step["action"].startswith("rewrite:" + kind + ":")) for step in trace)

    criteria = {"uses_induction": signature["uses_induction"] == requirements["uses_induction"]}
    if requirements["uses_induction"]:
        criteria["induction_binder_index"] = signature["induction_binder_index"] == requirements["induction_binder_index"]
        if requirements.get("requires_IH_in_successor"):
            criteria["IH_in_successor"] = uses("induction_hypothesis", 2)
        for branch in requirements.get("premise_branches", []):
            criteria[f"premise_in_branch_{branch}"] = uses("premise", branch)
    else:
        criteria["uses_premise"] = signature["uses_premise"] == requirements["uses_premise"]
    return {"status": "MATCH" if all(criteria.values()) else "MISMATCH", "matches": all(criteria.values()),
            "criteria": criteria, "signature": signature,
            "limitation": "Observable argument strategy, independently of Rocq verification; matching does not certify every English step."}


def assess_strategy(variant, condition, proof_body, verified):
    targets = {key: strategy_match(variant["formal_statement"], proof_body, reference["requirements"])
               for key, reference in variant["arguments"].items()}
    requested = condition[-1] if condition != "theorem_only" else None
    observed = [key for key, score in targets.items() if score["matches"]]
    return {"requested_argument": requested, "matches_reference": targets, "observed_matching_arguments": observed,
            "requested_strategy_matched": targets[requested]["matches"] if requested else None,
            "verified_requested_strategy": bool(verified and targets[requested]["matches"]) if requested else None}


def summarize(rows):
    def metrics(group):
        requested = [r for r in group if r["condition"] != "theorem_only"]
        n = len(group)
        verified = sum(r["verification"]["status"] == "PASS" for r in group)
        matching = sum(r["strategy"]["requested_strategy_matched"] is True for r in requested)
        both = sum(r["strategy"]["verified_requested_strategy"] is True for r in requested)
        return {"outputs": n, "verified": verified, "verification_accuracy": verified / n,
                "argument_conditioned_outputs": len(requested), "requested_strategy_matched": matching if requested else None,
                "attempted_strategy_accuracy": matching / len(requested) if requested else None,
                "verified_strategy_followed": both if requested else None,
                "verified_strategy_accuracy": both / len(requested) if requested else None}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["example_id"], {})[row["condition"]] = row
    if len(rows) != 18 or len(grouped) != 6 or any(set(g) != set(CONDITIONS) for g in grouped.values()):
        raise ValueError("Scoring requires all 18 unique outputs")
    pairs = []
    for example_id, group in grouped.items():
        a, b, t = (group[c] for c in ("argument_A", "argument_B", "theorem_only"))
        attempted = a["strategy"]["requested_strategy_matched"] and b["strategy"]["requested_strategy_matched"]
        success = a["strategy"]["verified_requested_strategy"] and b["strategy"]["verified_requested_strategy"]
        pairs.append({"example_id": example_id, "A_key": a["key"], "B_key": b["key"], "theorem_only_key": t["key"],
                      "attempted_strategy_switch": bool(attempted), "verified_strategy_switch": bool(success),
                      "generated_body_changed": a["proof_body"] != b["proof_body"],
                      "both_A_B_verified": all(r["verification"]["status"] == "PASS" for r in (a, b)),
                      "theorem_only_matching_arguments": t["strategy"]["observed_matching_arguments"]})
    success = sum(p["verified_strategy_switch"] for p in pairs)
    attempted = sum(p["attempted_strategy_switch"] for p in pairs)
    return {"all_outputs": metrics(rows), "by_condition": {c: metrics([r for r in rows if r["condition"] == c]) for c in CONDITIONS},
            "switching": {"successful_theorems": success, "theorems": 6, "switching_accuracy": success / 6,
                          "attempted_switches": attempted, "attempted_switching_accuracy": attempted / 6,
                          "criterion": "A and B must both verify and match their respective fixed strategy requirements; denominator is all six theorems"},
            "per_theorem": pairs}
