"""Prompt construction and scoring use frozen development data exclusively."""

from copy import deepcopy
import json

from baseline.dataset import ROOT, check_development, digest
from baseline.protocol import CONDITIONS, evaluate_candidate
from scripts.seed_schema import definitional_key
from verifier import Rejection, render_source

PLAN = ROOT / "prompt_v2/plan.json"
SYSTEM = ROOT / "prompt_v2/system.txt"
SELECTED = ROOT / "prompt_v2/selected.json"
FROZEN = ROOT / "prompt_v2/frozen.json"


def load_plan():
    plan = json.loads(PLAN.read_text())
    if (plan["conditions"] != list(CONDITIONS) or plan["attempts_per_candidate_example_condition"] != 1
            or plan["repair_attempts"] != 0 or plan["fine_tuning_performed"] is not False):
        raise ValueError("Prompt v2 requires one attempt, both fixed conditions, no repair or training")
    ids = [c["id"] for c in plan["candidates"]]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Candidate IDs must be nonempty and unique")
    for candidate in plan["candidates"]:
        demos = candidate["demonstration_ids"]
        if len(set(demos)) != len(demos) or not 2 <= len(demos) <= 4:
            raise ValueError("Each candidate needs 2–4 fixed, distinct demonstrations")
    return plan


def target_message(example, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    content = "Theorem: " + example["formal_statement"]
    if condition == "theorem_and_informal":
        content += "\nInformal proof: " + example["informal_proof"]
    return {"role": "user", "content": content + "\nOutput code only."}


def make_prompt(candidate, development=None):
    records = check_development() if development is None else development
    by_id = {record["id"]: record for record in records}
    messages = [{"role": "system", "content": SYSTEM.read_text().strip()}]
    for seed_id in candidate["demonstration_ids"]:
        if seed_id not in by_id:
            raise ValueError("Only frozen development seeds may be demonstrations")
        example = by_id[seed_id]
        messages += [target_message(example, "theorem_and_informal"),
                     {"role": "assistant", "content": example["proof_body"]}]
    return {"id": candidate["id"], "demonstration_ids": candidate["demonstration_ids"],
            "prefix_messages": messages}


def build_messages(prompt, example, condition):
    return deepcopy(prompt["prefix_messages"]) + [target_message(example, condition)]


def selection_partition(plan, development):
    demo_ids = {seed_id for c in plan["candidates"] for seed_id in c["demonstration_ids"]}
    demo_keys = {definitional_key(e["formal_statement"]) for e in development if e["id"] in demo_ids}
    excluded = [e["id"] for e in development if definitional_key(e["formal_statement"]) in demo_keys]
    included = [e["id"] for e in development if e["id"] not in excluded]
    if not included:
        raise ValueError("No development examples remain for prompt selection")
    return {"selection_ids": included, "excluded_demonstration_or_equivalent_ids": excluded}


def format_compliance(statement, text):
    body = text.strip()
    if not body:
        return {"compliant": False, "category": "EMPTY_OUTPUT", "reason": "No proof-body code"}
    try:
        render_source(statement, body)
    except Rejection as error:
        return {"compliant": False, "category": error.category, "reason": str(error)}
    return {"compliant": True, "category": "RESTRICTED_BODY", "reason": "Raw output accepted by the unchanged policy parser"}


def assess_output(example, text, timeout=10.0):
    return evaluate_candidate(example, text, timeout) | {
        "format_compliance": format_compliance(example["formal_statement"], text),
    }


def metrics(rows):
    from collections import Counter
    import statistics
    total = len(rows)
    formats = sum(r["format_compliance"]["compliant"] for r in rows)
    passed = sum(r["verification"]["status"] == "PASS" for r in rows)
    matches = sum(r["argument_fidelity"]["status"] == "STRUCTURE_MATCH" for r in rows)
    verified_matches = sum(r["verification"]["status"] == "PASS" and r["argument_fidelity"]["status"] == "STRUCTURE_MATCH" for r in rows)
    return {
        "outputs": total, "format_compliant": formats, "format_compliance_rate": formats / total,
        "rocq_verified": passed, "pass_at_1": passed / total,
        "structure_matches": matches, "verified_structure_matches": verified_matches,
        "verified_structure_match_rate": verified_matches / total,
        "fidelity_statuses": dict(Counter(r["argument_fidelity"]["status"] for r in rows)),
        "failure_categories": dict(Counter(r["failure_category"] for r in rows if r["failure_category"])),
        "verification_stages": dict(Counter(r["verification"]["stage"] for r in rows)),
        "generation_errors": sum(r["generation_error"] is not None for r in rows),
        "length_limited": sum(r["finish_reason"] == "max_new_tokens" for r in rows),
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
        "median_generation_latency_seconds": statistics.median(r["generation_latency_seconds"] for r in rows),
    }


def rank_candidates(plan, rows, selection_ids):
    ranking = []
    for index, candidate in enumerate(plan["candidates"]):
        selected_rows = [r for r in rows if r["prompt_id"] == candidate["id"] and r["example_id"] in selection_ids]
        if len(selected_rows) != 2 * len(selection_ids):
            raise ValueError("Prompt selection requires every planned development attempt")
        scores = metrics(selected_rows)
        rank = [scores["format_compliant"], scores["rocq_verified"], scores["verified_structure_matches"],
                -len(candidate["demonstration_ids"]), -index]
        ranking.append({"candidate_id": candidate["id"], "rank": rank, "selection_metrics": scores})
    return sorted(ranking, key=lambda entry: entry["rank"], reverse=True)
