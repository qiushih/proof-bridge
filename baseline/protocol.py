"""Fixed paired prompts, format-only extraction, and an explicit fidelity proxy."""

from dataclasses import asdict
import re

from baseline.dataset import ROOT, digest
from scripts.seed_schema import argument_features
from verifier import Rejection, StatementParser, render_source, tokenize, verify

CONDITIONS = ("theorem_and_informal", "theorem_only")
PROMPT_FILE = ROOT / "baseline/prompt.txt"


def build_messages(example, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown baseline condition")
    user_text = "Theorem:\n" + example["formal_statement"]
    if condition == "theorem_and_informal":
        user_text += "\n\nInformal proof:\n" + example["informal_proof"]
    # Explicit allowlist: no reference code, titles, features, or seed examples.
    return [{"role": "system", "content": PROMPT_FILE.read_text().strip()},
            {"role": "user", "content": user_text}]


def extract_body(text):
    body = text.strip()
    match = re.fullmatch(r"```(?:coq|rocq)?[ \t]*\n(.*?)\n```", body, flags=re.DOTALL | re.IGNORECASE)
    if match and "```" not in match[1]:
        return match[1].strip(), "outer_code_fence_removed"
    return body, "whitespace_only"


def proof_signature(statement, body):
    render_source(statement, body)  # Only parse code accepted by the existing policy.
    parser = StatementParser(tokenize(statement))
    _, binder_count = parser.parse()
    variables = len(parser.variables)
    tokens = tokenize(body)
    introduced = []
    hypothesis = None
    induction_position = None
    branch = 0
    trace = []
    command = []
    for token in tokens:
        if token == "-":
            branch += 1
            continue
        if token != ".":
            command.append(token)
            continue
        tactic, *args = command
        command = []
        if tactic == "intros":
            introduced.extend(args)
            continue
        if tactic == "induction":
            induction_position = introduced.index(args[0])
            hypothesis = args[5]
            continue
        action = tactic
        if tactic in {"rewrite", "exact", "apply"}:
            reference = args[-1]
            kind = "induction_hypothesis" if reference == hypothesis else (
                "premise" if reference in introduced[variables:binder_count] else "other_local")
            if tactic == "rewrite":
                action = f"rewrite:{kind}:{'reverse' if '<-' in args else 'forward'}"
            else:
                action = f"use:{kind}"
        trace.append({"branch": branch, "action": action})
    features = argument_features(statement, body)
    return {
        "uses_induction": features["uses_induction"],
        "induction_binder_index": induction_position,
        "uses_premise": features["uses_premise"],
        "f_equal_count": sum(t["action"] == "f_equal" for t in trace),
        "symmetry_count": sum(t["action"] == "symmetry" for t in trace),
        "rewrite_steps": [t for t in trace if t["action"].startswith("rewrite:")],
        "closing_steps": [t for t in trace if t["action"] == "reflexivity" or t["action"].startswith("use:")],
    }


def assess_structure(example, body):
    expected = proof_signature(example["formal_statement"], example["proof_body"])
    base = {"method": "deterministic_argument_structure_proxy_v1", "reference_signature": expected,
            "semantic_fidelity": "not_assessed", "human_reviewed": False,
            "limitation": "Matching these features does not certify English alignment or a complete proof. Alternate valid arguments may mismatch."}
    try:
        actual = proof_signature(example["formal_statement"], body)
    except Rejection as error:
        return base | {"status": "UNASSESSABLE", "score": None, "candidate_signature": None,
                       "reason": f"Candidate outside the restricted parser: {error.category}"}
    criteria = {key: actual[key] == value for key, value in expected.items()}
    return base | {"status": "STRUCTURE_MATCH" if all(criteria.values()) else "STRUCTURE_MISMATCH",
                   "score": sum(criteria.values()) / len(criteria), "criteria": criteria,
                   "candidate_signature": actual}


def evaluate_candidate(example, generated_text, timeout=10.0):
    body, extraction = extract_body(generated_text)
    result = verify(example["formal_statement"], body, timeout=timeout)
    return {
        "generated_text": generated_text, "proof_body": body, "proof_body_sha256": digest(body),
        "output_extraction": extraction, "verification": asdict(result),
        "failure_category": None if result.status == "PASS" else result.category,
        "argument_fidelity": assess_structure(example, body),
    }
