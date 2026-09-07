"""Audit saved development outputs only; never invoke inference or evaluation.

The annotation table below records an assistant's manual reading of all 180
outputs against the 30 frozen informal arguments, formal proofs, and diagnostics.
It is not a new automatic fidelity classifier. Re-running reproduces the audit.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from baseline.dataset import ROOT, check_development, digest
from prompt_v2.experiment import DEV_OUTPUT, load_frozen, load_phase
from verifier import sha256_file

OUTPUT = ROOT / "results/prompt-v2-development-audit"
CATEGORIES = {
    "formatting_parser_failure": "Malformed tactic arguments or branch syntax; excludes more specific binder/reference diagnoses.",
    "missing_introductions": "An omitted binder prevents the intended induction or use of a real theorem premise.",
    "invalid_hypothesis_reference": "An invented premise, undeclared induction hypothesis, or invalid local reference.",
    "mathematically_incorrect_proof_step": "A policy-accepted proof fails in Rocq: inapplicable rewrite, unequal reflexivity goal, wrong proof type, or an unfinished argument.",
    "forbidden_construct": "A command or tactic explicitly outside the existing allowlist; unknown locals are classified more specifically as references.",
    "verified_but_argument_unfaithful": "A verified proof uses a genuinely different mathematical strategy; equivalent local equality reasoning does not qualify.",
    "other": "A failure not covered above (none observed).",
}

# Each line is one seed in order 001..030: proof A / theorem only.
# F is a manually confirmed faithful proof, not an automatic consequence of PASS.
MANUAL_CODES = {
    "short_2shot": """
F/F F/F DI/DI F/DI DI/F F/F PA/DP PS/DP DP/DI DR/DP
DR/DI DI/DP DI/DP DI/DP DI/DI F/DO F/F PI/F PI/DO F/DO
PI/DO F/DO PI/F F/F PI/DO PI/DR PI/DO PI/PI PI/PO PI/DR
""".split(),
    "short_3shot": """
F/F F/F DH/DH PD/DI F/DH F/DH F/F PS/F F/F F/F
F/F F/F F/F PH/F F/F F/DO F/F PI/DH DH/DH F/DO
PI/DH F/DO PI/DH F/DO DH/DH DH/DO PI/DO PI/DO PI/DO DH/DH
""".split(),
    "short_4shot": """
F/F F/F DH/DH F/F DH/DH DH/DH DI/DI PS/DI DI/DI F/DI
F/DI F/DI F/DI DI/DI F/DI DH/DO DH/DH DH/DH DH/DH DH/DH
DH/DH DH/DO DH/DH DH/DH DH/DH DH/DH F/F F/F PI/F DH/DH
""".split(),
}

NOTES = {
    "DI": ("DIVERGENT", "different_strategy", "Attempts induction for a reference argument that uses only computation or the supplied equality. The induction is a different strategy and this attempt does not verify."),
    "DH": ("DIVERGENT", "invented_premise", "Introduces and uses an extra H although the theorem has no premise. This invented assumption changes the argument, even where part of the intended induction appears later."),
    "DR": ("DIVERGENT", "invented_induction_hypothesis", "Uses IH without any preceding induction. This is an invented justification, not an equivalent use of an available equality."),
    "DO": ("DIVERGENT", "omitted_induction", "Omits the reference's induction. Computation, a top-level rewrite, or repeated constructor rewrites do not implement the required base case and induction-hypothesis step."),
    "DP": ("DIVERGENT", "omitted_premise_reasoning", "Attempts simplification/reflexivity without using the assumed equality. The premise-based mathematical argument is absent."),
    "PI": ("PARTIAL", "same_induction_incomplete", "Attempts induction on the intended first natural-number binder, but leaves theorem binders unintroduced. The visible induction outline is relevant, but no complete faithful proof is established."),
    "PH": ("PARTIAL", "same_equality_transport_incomplete", "The rewrite H expresses the intended substitution, but m and the actual premise H were never introduced. The parser cannot accept this argument."),
    "PS": ("PARTIAL", "same_symmetry_argument_incomplete", "Uses symmetry (and sometimes rewriting) in the intended local equality argument but does not close the goal. This is partial alignment, not a faithful complete proof."),
    "PD": ("PARTIAL", "same_definition_argument_invalid", "Tries to rewrite the displayed arithmetic expression instead of reducing it. The relevant definitional equality is recognizable, but the rewrite has no local equality proof and is malformed."),
    "PA": ("PARTIAL", "same_premise_argument_invalid", "The attempted assumption tactic refers to the relevant premise, but assumption is outside the allowlist, its arguments are malformed, and the premise was not explicitly introduced."),
    "PO": ("PARTIAL", "same_induction_fragment_invalid", "Contains an induction-style successor rewrite with IH, but no induction command or well-formed pair of branches. It is only a fragment of the reference argument."),
}

EQUIVALENCE_NOTES = {
    7: "Rewriting H followed by reflexivity uses exactly the given equality, equivalently to exact H; no new mathematical strategy is introduced.",
    8: "Rewriting the given equality closes its symmetric instance. This is equivalent local equality transport to symmetry followed by exact H.",
    9: "Rewriting H under S implements the same successor congruence as f_equal followed by exact H.",
    12: "Forward rewriting replaces n by m, while the English reference replaces m by n. Both use the same equality-substitution argument; direction differs at step level, not at mathematical-strategy level.",
    17: "Induction is on the same variable with the same base case. Rewriting IH under S and reflexivity implement the reference's f_equal/exact IH congruence step.",
    22: "The same induction and definitional reduction of 1+k are used. Rewriting IH under S is equivalent to the reference's f_equal/exact IH step.",
    23: "The same induction establishes the same successor congruence. Rewriting IH under the outer S is equivalent to removing that S and applying IH.",
}


def manual_review(row, example):
    seed_number = int(example["id"].split("_")[1])
    condition_index = 0 if row["condition"] == "theorem_and_informal" else 1
    code = MANUAL_CODES[row["prompt_id"]][seed_number - 1].split("/")[condition_index]
    if code == "F":
        status, relation = "FAITHFUL", "same_mathematical_argument"
        if seed_number in EQUIVALENCE_NOTES and row["argument_fidelity"]["status"] != "STRUCTURE_MATCH":
            rationale = EQUIVALENCE_NOTES[seed_number]
        elif seed_number <= 6:
            rationale = "Uses the reference's definitional equality/reflexivity argument; simpl may be implicit in Rocq conversion or redundant. There is no added induction or assumed equality."
        elif seed_number <= 15:
            rationale = "Uses the supplied equality for the same substitution argument; any final computation is handled by reflexivity's definitional conversion."
        else:
            rationale = "Uses induction on the same first binder, the same base justification (including the premise when needed), and the same induction-hypothesis equality step."
    else:
        status, relation, rationale = NOTES[code]
    if row["key"] == "short_4shot:seed_029:theorem_and_informal":
        rationale += " In intros n m H, H actually names the third nat variable p; the equality premise is still missing, so exact H would also have the wrong type."
    if code == "PI" and seed_number in (27, 28, 29):
        rationale += " The base branch also uses reflexivity/another local in place of a correctly introduced equality premise; correcting binders alone is not a demonstrated repair."
    if code == "DI" and seed_number in (12, 13, 15) and "- simpl.\n  rewrite IH." in row["proof_body"]:
        rationale += " The base branch also refers to IH before any induction hypothesis exists in that branch."
    return {
        "status": status, "strategy_relation": relation, "rationale": rationale,
        "annotation_code": code, "reviewer": "assistant", "human_reviewed": False,
        "reference_for_theorem_only": "The same hidden development argument; a match does not prove use of English.",
    }


def primary_category(row, review):
    if row["verification"]["status"] == "PASS":
        return None if review["status"] == "FAITHFUL" else "verified_but_argument_unfaithful"
    message = row["verification"]["message"]
    if message == "Introduce all theorem binders before induction on a local.":
        return "missing_introductions"
    if message == "Reference is not an in-scope local: H" and row["key"] == "short_3shot:seed_014:theorem_and_informal":
        return "missing_introductions"
    if message == "Too many introduction names." or message.startswith("Reference is not an in-scope local:"):
        return "invalid_hypothesis_reference"
    if row["verification"]["category"].startswith("FORBIDDEN_"):
        return "forbidden_construct"
    if row["verification"]["stage"] == "policy" and row["verification"]["category"] == "SYNTAX_ERROR":
        return "formatting_parser_failure"
    if row["verification"]["category"] in ("PROOF_ERROR", "INCOMPLETE_PROOF"):
        return "mathematically_incorrect_proof_step"
    return "other"


def percent(numerator, denominator):
    return round(100 * numerator / denominator, 2) if denominator else None


def summarize(rows):
    failures = [r for r in rows if r["primary_failure_category"]]
    primary = Counter(r["primary_failure_category"] for r in failures)
    passed = sum(r["verification"]["status"] == "PASS" for r in rows)
    policy = sum(r["verification"]["status"] == "FAIL" and r["verification"]["stage"] == "policy" for r in rows)
    return {
        "outputs": len(rows), "rocq_verified": passed, "verification_failures": len(rows) - passed,
        "audit_failures": len(failures), "policy_rejections": policy,
        "policy_rejection_pct_of_verification_failures": percent(policy, len(rows) - passed),
        "raw_format_compliant": sum(r["format_compliance"]["compliant"] for r in rows),
        "primary_categories": {name: {
            "count": primary[name], "pct_all_outputs": percent(primary[name], len(rows)),
            "pct_audit_failures": percent(primary[name], len(failures)),
        } for name in CATEGORIES},
        "manual_fidelity": dict(Counter(r["manual_fidelity"]["status"] for r in rows)),
        "different_strategy_attempts": sum(r["manual_fidelity"]["strategy_relation"] == "different_strategy" for r in rows),
        "omitted_induction_attempts": sum(r["manual_fidelity"]["strategy_relation"] == "omitted_induction" for r in rows),
        "invented_premise_attempts": sum(r["manual_fidelity"]["strategy_relation"] == "invented_premise" for r in rows),
        "verified_proxy_mismatch_but_faithful": sum(r["verification"]["status"] == "PASS" and r["proxy_fidelity"]["status"] == "STRUCTURE_MISMATCH" and r["manual_fidelity"]["status"] == "FAITHFUL" for r in rows),
        "proxy_manual_status_cross_tab": dict(Counter(r["proxy_fidelity"]["status"] + "/" + r["manual_fidelity"]["status"] for r in rows)),
    }


def dependence(rows, examples):
    pairs = []
    by_key = {(r["example_id"], r["condition"]): r for r in rows}
    for example in examples:
        a = by_key.get((example["id"], "theorem_and_informal"))
        t = by_key.get((example["id"], "theorem_only"))
        if a is None or t is None:
            continue
        assert a["prompt_messages"][:-1] == t["prompt_messages"][:-1]
        for row in (a, t):
            target = row["prompt_messages"][-1]["content"]
            assert target.startswith("Theorem: " + example["formal_statement"] + "\n")
        expected_a = "Theorem: " + example["formal_statement"] + "\nInformal proof: " + example["informal_proof"] + "\nOutput code only."
        assert a["prompt_messages"][-1]["content"] == expected_a
        assert t["prompt_messages"][-1]["content"] == "Theorem: " + example["formal_statement"] + "\nOutput code only."
        a_induction = bool(re.search(r"\binduction\s", a["proof_body"]))
        t_induction = bool(re.search(r"\binduction\s", t["proof_body"]))
        a_pass, t_pass = (r["verification"]["status"] == "PASS" for r in (a, t))
        pairs.append({
            "example_id": example["id"], "theorem_only_key": t["key"], "proof_A_key": a["key"],
            "proof_B_key": None, "proof_B_status": "NOT_RUN_NO_SAVED_B",
            "selection_eligible": a["selection_eligible"], "is_demonstration": a["is_demonstration"],
            "body_changed": a["proof_body"] != t["proof_body"],
            "theorem_only_has_induction": t_induction, "proof_A_has_induction": a_induction,
            "induction_added_with_A": a_induction and not t_induction,
            "induction_removed_with_A": t_induction and not a_induction,
            "reference_uses_induction": example["argument_features"]["uses_induction"],
            "verification_transition": f"{t['verification']['status']}->{a['verification']['status']}",
            "manual_fidelity_transition": t["manual_fidelity"]["status"] + "->" + a["manual_fidelity"]["status"],
            "A_only_verified_and_faithful": a_pass and a["manual_fidelity"]["status"] == "FAITHFUL" and not t_pass,
            "theorem_only_verified_and_faithful": t_pass and t["manual_fidelity"]["status"] == "FAITHFUL" and not a_pass,
        })
    return {
        "available_comparison": "Saved theorem-only versus theorem + the single frozen informal proof A",
        "three_way_test": {"status": "NOT_RUN_NO_SAVED_B", "complete_triplets": 0,
            "reason": "No saved prompt contains an alternative informal proof B for the same theorem. The no-new-samples constraint precludes running the missing condition. Different few-shot prefixes are not different target proofs."},
        "pairs": len(pairs), "body_changed": sum(p["body_changed"] for p in pairs),
        "body_changed_pct": percent(sum(p["body_changed"] for p in pairs), len(pairs)),
        "induction_added_with_A": sum(p["induction_added_with_A"] for p in pairs),
        "induction_removed_with_A": sum(p["induction_removed_with_A"] for p in pairs),
        "verification_transitions": dict(Counter(p["verification_transition"] for p in pairs)),
        "fidelity_transitions": dict(Counter(p["manual_fidelity_transition"] for p in pairs)),
        "A_only_verified_and_faithful_ids": [p["example_id"] for p in pairs if p["A_only_verified_and_faithful"]],
        "theorem_only_verified_and_faithful_ids": [p["example_id"] for p in pairs if p["theorem_only_verified_and_faithful"]],
        "per_example": pairs,
        "structure_measure_limit": "Induction-presence comparisons inspect text even when policy rejects it; they show an attempted structure, not a valid or faithful proof.",
    }


def build_analysis():
    freeze, selected = load_frozen()
    manifest, saved_rows = load_phase(DEV_OUTPUT)  # Explicit development path only.
    if manifest["phase"] != "development":
        raise ValueError("Development data only")
    examples = check_development()
    by_id = {e["id"]: e for e in examples}
    assert all(len(codes) == 30 for codes in MANUAL_CODES.values())
    reviews = []
    for row in saved_rows:
        example = by_id[row["example_id"]]
        review = manual_review(row, example)
        if review["status"] == "FAITHFUL" and row["verification"]["status"] != "PASS":
            raise ValueError("A complete faithful proof must verify")
        primary = primary_category(row, review)
        if primary == "other":
            raise ValueError("Unexpected diagnostic needs manual review")
        observed = []
        if row["verification"]["stage"] == "policy":
            observed.append("rejected_before_rocq")
        if row["verification"]["message"] == "Too many introduction names.":
            observed.append("extra_H_without_a_theorem_premise")
        if row["verification"]["category"] == "INCOMPLETE_PROOF":
            observed.append("unfinished_goal")
        if review["annotation_code"] == "PI" and int(example["id"].split("_")[1]) in (27, 28, 29):
            observed.append("base_premise_step_also_missing_or_invalid")
        reviews.append({
            "key": row["key"], "prompt_id": row["prompt_id"], "example_id": row["example_id"],
            "condition": row["condition"], "is_demonstration": row["is_demonstration"],
            "selection_eligible": row["selection_eligible"], "formal_statement": example["formal_statement"],
            "informal_proof_A": example["informal_proof"], "reference_proof_body": example["proof_body"],
            "generated_text": row["generated_text"], "proof_body": row["proof_body"],
            "proof_body_sha256": row["proof_body_sha256"], "source_row_sha256": digest(json.dumps(row, sort_keys=True)),
            "prompt_messages": row["prompt"]["messages"], "prompt_sha256": row["prompt"]["sha256"],
            "verification": row["verification"], "format_compliance": row["format_compliance"],
            "primary_failure_category": primary, "additional_observations": observed,
            "proxy_fidelity": row["argument_fidelity"], "manual_fidelity": review,
        })
    chosen = [r for r in reviews if r["prompt_id"] == selected["id"]]
    common = [r for r in chosen if r["selection_eligible"]]
    candidates = {pid: [r for r in reviews if r["prompt_id"] == pid] for pid in MANUAL_CODES}
    # Inventory TARGET arguments, not demonstration arguments, across all saved candidates.
    variants = {e["formal_statement"]: set() for e in examples}
    for row in saved_rows:
        if row["condition"] == "theorem_and_informal":
            variants[row["formal_statement"]].add(row["prompt"]["messages"][-1]["content"])
    assert len(variants) == 30 and all(len(v) == 1 for v in variants.values())
    counts = summarize(chosen)
    decision = {
        "number_of_recommended_experiments": 1, "experiment": "binder_and_scope_aware_constrained_decoding",
        "status": "RECOMMENDED_NOT_RUN",
        "rationale": f"The frozen prompt has {counts['policy_rejections']}/{counts['verification_failures']} failures before Rocq ({counts['policy_rejection_pct_of_verification_failures']}%). Most involve invented or missing binders/references. All its verified proofs preserve the mathematical argument after manual review. This supports testing grammar/context constraints before a training intervention.",
        "design": "Compare the frozen Prompt v2's saved unconstrained outputs against one future constrained-decoding run on the same 30 development seeds and two existing conditions. Keep model, prompt, verifier, greedy decoding, 256-token cap, one attempt, and no repair unchanged. Report all 30 and the existing common 25-seed subset separately.",
        "constraints": "Mask invalid token continuations using the existing restricted grammar, theorem binder count, local variable/hypothesis types, and induction-branch scope. Reject invented H/IH and malformed commands during decoding; let the model choose the mathematical strategy. Do not hard-code reference proofs or repair completions.",
        "measurements": ["format acceptance", "binder/reference failure counts", "Rocq PASS", "manually reviewed fidelity", "latency and completion tokens"],
        "decision_rule": "Count success only if policy failures decrease AND more outputs verify and remain faithful on the common development subset. Moving failures into Rocq without improving verified faithful proofs is insufficient.",
        "limits": "A policy rejection reveals only the first barrier. No claim is made that constraining or fixing it would make the proof verify. Existing mathematical failures and omitted premise steps remain unresolved.",
        "evaluation_use": "None for this recommendation or proposed development experiment.",
    }
    protected = freeze["files_sha256"] | {"prompt_v2/frozen.json": sha256_file(ROOT / "prompt_v2/frozen.json")}
    return {
        "schema_version": "proofbridge-development-failure-audit-1.0",
        "scope": {"primary_prompt": selected["id"], "primary_outputs": 60, "all_saved_candidate_outputs_reviewed": 180,
            "unique_statement_body_pairs_reviewed": len({(r['formal_statement'], r['proof_body']) for r in reviews}),
            "development_examples": 30, "diagnostic_evaluation_records_read": 0, "new_generations": 0,
            "new_proof_pairs": 0, "repair_attempts": 0, "fine_tuning_performed": False,
            "prompt_dataset_verifier_changed": False, "verifier_evidence": "Reused saved results; no edited proofs or new compiler runs are needed for this audit."},
        "provenance": {"reviewer": "assistant", "review_method": "Manual reading of each saved candidate against the frozen English argument, reference code, and recorded diagnostic; shared notes for equivalent cases.",
            "human_reviewed": False, "independent_human_review": False,
            "inputs_sha256": protected, "audit_script_sha256": sha256_file(Path(__file__))},
        "taxonomy": {"categories": CATEGORIES,
            "counting": "One primary category per failed output; verified faithful outputs have null category. More specific binder/reference causes take precedence over generic parser failure. Additional observations are nonexclusive and are not added to category totals.",
            "extra_introduction_rule": "All observed excess introductions invent a nonexistent premise H: primary invalid_hypothesis_reference, although the original verifier reports SYNTAX_ERROR.",
            "missing_introduction_rule": "For induction or a real premise used before binding, missing_introductions is primary. If an accepted proof later fails on reflexivity/type checking, the primary cause remains mathematical even if a premise was not explicitly named.",
            "incomplete_rule": "An accepted but unfinished proof is a mathematical proof-step failure with unfinished_goal annotation, not a format failure.",
            "denominators": "pct_all_outputs includes successes; pct_audit_failures excludes verified faithful proofs. Primary counts are mutually exclusive."},
        "fidelity_rubric": {"FAITHFUL": "Verified and preserves the mathematical strategy, induction variable/cases, and essential premise use. Equivalent equality transport and definitional computation are accepted.",
            "PARTIAL": "Recognizable reference steps, but missing or invalid steps prevent a complete faithful proof.",
            "DIVERGENT": "Different/omitted main strategy, ignored premise, or invented justification. For failed outputs this describes the attempted argument, not a verified alternative proof.",
            "step_level_caveat": "seed_012 uses the opposite substitution direction; retain that deviation note while accepting the same mathematical equality-substitution argument."},
        "selected_prompt_summary": counts,
        "selected_prompt_by_condition": {c: summarize([r for r in chosen if r["condition"] == c]) for c in manifest["conditions"]},
        "selected_common_subset_summary": summarize(common),
        "all_candidates_summary": summarize(reviews),
        "candidate_summaries": {pid: summarize(rs) for pid, rs in candidates.items()},
        "fidelity_disagreement_notes": [{"key": r["key"], "proof_body_sha256": r["proof_body_sha256"],
            "proxy_status": r["proxy_fidelity"]["status"], "manual_status": r["manual_fidelity"]["status"],
            "note": r["manual_fidelity"]["rationale"]} for r in reviews if r["verification"]["status"] == "PASS" and r["proxy_fidelity"]["status"] != "STRUCTURE_MATCH"],
        "informal_proof_dependence": {
            "saved_target_argument_inventory": {"unique_theorems": len(variants), "theorems_with_one_saved_argument": len(variants), "theorems_with_A_and_B": 0},
            "selected_prompt": dependence(chosen, examples), "selected_common_subset": dependence(common, examples),
            "other_candidate_context": {pid: dependence(rs, examples) for pid, rs in candidates.items() if pid != selected["id"]},
            "interpretation": "Changing the target informal text changes some saved outputs, and some changes recover the reference's induction. This supports input sensitivity and coarse strategy cues, not dependable step-by-step argument following or the ability to follow two different valid arguments for a fixed theorem.",
            "limitations": ["No saved B condition; the requested three-way dependence test is not executable under the no-generation constraint.",
                "All outputs are development observations used during prompt selection, not held-out generalization evidence.",
                "The fixed demonstrations themselves contain informal proofs and the induction template. seed_016 is a demonstration, so its gain is not independent evidence.",
                "The three prompt candidates differ in demonstration context, not the target informal proof; comparing them cannot substitute for an A/B test.",
                "A changed induction keyword in a rejected proof is an attempted structure only."]},
        "next_experiment": decision,
        "records": reviews,
    }


def markdown(analysis):
    s = analysis["selected_prompt_summary"]
    common = analysis["selected_common_subset_summary"]
    dep = analysis["informal_proof_dependence"]["selected_prompt"]
    lines = ["# Prompt v2: development-only failure audit", "",
        "**Recommend one next experiment: binder- and scope-aware constrained decoding.** The frozen 3-shot prompt loses most failed outputs before Rocq can check them. This experiment is recommended only; no new samples were generated.", "",
        "Reviewed all **180 saved development outputs** (118 unique theorem/body pairs) from the 2-, 3-, and 4-shot candidates. Headline counts use only the frozen **short_3shot: 60 outputs on 30 seeds**, both conditions. The 12 diagnostic examples and their results were not read. Prompt, data, verifier, and saved outputs remain unchanged.", "",
        "## Failure counts", "",
        f"The frozen prompt has **{s['rocq_verified']}/60 verified outputs** and **{s['verification_failures']} failures**. All verified outputs are faithful under the reviewed mathematical-argument rubric. Percentages below use 32 failures or 60 outputs, as labeled.", "",
        "| Primary category | Count | % of failures | % of all outputs |", "| --- | ---: | ---: | ---: |"]
    for name, counts in s["primary_categories"].items():
        lines.append(f"| {name.replace('_', ' ')} | {counts['count']} | {counts['pct_audit_failures']:.2f}% | {counts['pct_all_outputs']:.2f}% |")
    lines += ["", "Each failure gets one primary cause. The 14 excess-introduction failures invent a nonexistent premise H and count as invalid references. The real-but-unintroduced H in seed_014 counts as missing introductions. Malformed `rewrite S n` counts as parser failure. The incomplete symmetry proof counts as an unfinished mathematical argument. These recategorizations do not alter the verifier's original labels.", "",
        f"**{s['policy_rejections']}/{s['verification_failures']} failures ({s['policy_rejection_pct_of_verification_failures']:.2f}%) occur before Rocq**, versus 9 failures in accepted proofs. The shared 25-seed subset has {common['policy_rejections']}/{common['verification_failures']} policy failures ({common['policy_rejection_pct_of_verification_failures']:.2f}%), so the conclusion survives excluding demonstrations and their equivalents. A policy failure can conceal further mathematical errors: for example, seed_027–029 with proof A also mishandle the base-case premise.", "",
        "| Saved candidate | Verified | Policy rejections | Rocq failures | Faithful despite proxy mismatch |", "| --- | ---: | ---: | ---: | ---: |"]
    for pid, c in analysis["candidate_summaries"].items():
        lines.append(f"| {pid} | {c['rocq_verified']}/60 | {c['policy_rejections']} | {c['verification_failures']-c['policy_rejections']} | {c['verified_proxy_mismatch_but_faithful']} |")
    lines += ["", "## Manual fidelity review", "",
        "These are **assistant reviews, not independent human reviews**. All 61 verified outputs across the three candidates preserve their reference's mathematical argument. The proxy rejects 15 of these, including 10 for the frozen prompt. No verified genuinely different strategy was found; failed attempts with different strategies, omitted induction, or invented premises are flagged separately in JSON.", "",
        "- Seeds 007–009: `rewrite H; reflexivity` implements the same premise use, symmetry, or successor congruence as `exact H`, `symmetry`, or `f_equal`.",
        "- Seed 012: substitution runs in the opposite direction. It is the same mathematical argument, with an explicit step-direction deviation note.",
        "- Seeds 017, 022, 023: `rewrite IH; reflexivity` and `f_equal; exact IH` implement the same induction-hypothesis congruence.", "",
        "Parser-unassessable attempts can still contain recognizable partial English alignment; that does not make them verified or faithful complete proofs. Every output has a hash-bound review, original diagnostic, and strategy label in `failure_analysis.json`.", "",
        "## Informal-proof dependence", "",
        "**The requested theorem-only / proof-A / proof-B test was not run.** The saved records contain one target informal proof per theorem and zero B outputs. Completing that test requires new generations, which this task forbids. Different few-shot candidates and definitionally related theorems are not substitutes for the missing condition.", "",
        "The available paired comparison holds the theorem, prompt prefix, model, and generation settings fixed:", "",
        "| Frozen prompt input | Format accepted | Verified / manually faithful |", "| --- | ---: | ---: |"]
    for condition in ("theorem_only", "theorem_and_informal"):
        c = analysis["selected_prompt_by_condition"][condition]
        lines.append(f"| {condition} | {c['raw_format_compliant']}/30 | {c['rocq_verified']}/30 |")
    lines += ["", f"- Proof A changes **{dep['body_changed']}/30 outputs**. It adds an attempted induction in **{dep['induction_added_with_A']}** pairs, but many such outputs still fail binder/reference checks.",
        "- It gains verified faithful proofs on **005, 006, 016, 020, 022, 024**, while losing them on **008 and 014**: 10 pass in both conditions, 12 in neither.",
        "- The successful induction gains are 016, 020, 022, 024. Seed 016 is itself a fixed demonstration. On the common 25-seed subset, proof A verifies 12/25 versus 9/25 theorem-only (five gains, two losses).",
        "- Proof A requests congruence and reverse rewriting in several seeds, yet the output repeats forward rewriting. Many longer induction arguments trigger the same short template with omitted binders or base-case assumptions.", "",
        "The saved pairs support sensitivity to the supplied text and some coarse induction cues. They do **not** establish faithful following of detailed steps or switching between two valid arguments for the same theorem. Manual faithful counts score the theorem-only output against the same hidden reference, not evidence that it read English.", "",
        "## Exactly one next experiment", "",
        "Test **binder- and scope-aware constrained decoding** with the frozen prompt and model on the same development examples. Constrain the existing grammar, binder count, local types/names, and induction-branch scope; leave mathematical strategy selection to the model. Keep the same greedy 256-token budget, one attempt, existing verifier, and two recorded input conditions. No prompt change, repair, data expansion, fine-tuning, or diagnostic-set decisions.", "",
        "Report all 30 seeds and the common 25-seed subset. Measure format acceptance, binder/reference failures, verification, reviewed fidelity, latency, and tokens. Success requires more **verified faithful proofs** on the common subset as well as fewer policy failures. Merely moving failures into Rocq is insufficient. No claim is made that the 23 rejected outputs would become correct after constraints.", "",
        "## Reproduce this audit without inference", "", "```sh", "python3 -m scripts.audit_prompt_v2_development --check", "python3 -m unittest discover -s tests -p test_prompt_v2_audit.py -v", "```", "",
        "`python3 -m scripts.audit_prompt_v2_development` rebuilds only this audit's two reports. Frozen input hashes and saved development evidence are checked; it never loads diagnostic examples or a model. The JSON includes per-candidate/condition totals, the common subset, all reviews, paired evidence, and the explicit missing-B status.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check reports without writing or generation")
    args = parser.parse_args()
    analysis = build_analysis()
    contents = {"failure_analysis.json": json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
                "FAILURE_ANALYSIS.md": markdown(analysis)}
    if args.check:
        for name, expected in contents.items():
            if (OUTPUT / name).read_text() != expected:
                raise ValueError(f"Stale audit report: {name}")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            (OUTPUT / name).write_text(content)
    print("PASS: 180 development outputs audited; 0 generations; no diagnostic records loaded; proof B unavailable")
    print(json.dumps(analysis["selected_prompt_summary"], indent=2))


if __name__ == "__main__":
    main()
