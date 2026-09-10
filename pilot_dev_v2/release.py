"""Prepare/check the eight-row v2 dev extension. No model generation or training."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from pilot_v1.dataset import ablations, token_evidence
from pilot_v1.protocol import (ROOT, audit_rows, digest, equivalence_key, features,
                               load_split, read_rows, validate_row)
from training_protocol_v1.scoring import step_adherence
from verifier import render_source, sha256_file, verify

DATA = ROOT / "data/pilot-v2"
STATEMENT = "forall n m p : nat, m + S p = S (m + p) -> (n + m) + S p = S ((n + m) + p)"
NEW_IDS = ["pd04_A", "pd04_B"]
FAMILY = "successor_reassociation_variants"
ARTIFACTS = ["additions.jsonl", "dev.jsonl", "references/pd04_A.v", "references/pd04_B.v",
             "verification_report.json", "family_review.json", "README.md", "VERIFICATION_REPORT.md"]


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def encoded_rows(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


def fingerprints():
    # Do not open holdout/diagnostic files, even for hashing. The old split
    # manifest commits their historical identities without revealing proofs.
    paths = [ROOT / p for p in (
        "verifier.py", "environment.lock.json", "data/pilot-v1/train.jsonl",
        "data/pilot-v1/dev.jsonl", "data/pilot-v1/split_manifest.json",
        "data/development/pairs.jsonl", "data/development/manifest.json",
        "scripts/seed_schema.py", "pilot_v1/protocol.py", "pilot_v1/dataset.py",
        "baseline/model.lock.json", "training_protocol_v1/data.py",
        "training_protocol_v1/config.json", "training_protocol_v1/scoring.py",
        "training_protocol_v1/TRAINING_PROTOCOL.md")]
    for name in ("prompt_v2", "constrained_v1"):
        paths.extend(p for p in (ROOT / name).rglob("*")
                     if p.is_file() and p.suffix in {".json", ".py", ".md"})
    result = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths))}
    lock = json.loads((ROOT / "baseline/model.lock.json").read_text())
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        path = ROOT / ".cache/baseline/model" / name
        require(sha256_file(path) == lock["files_sha256"][name], "Pinned tokenizer changed")
        result[str(path.relative_to(ROOT))] = sha256_file(path)
    return result


def authored_rows():
    common = [
        {"text": "Fix natural numbers n, m and p, and assume m + S p = S (m + p).",
         "code": "intros n m p H."},
        {"text": "Proceed by induction on n, keeping m, p and the premise fixed.",
         "code": "induction n as [| k IH]."},
        {"text": "At n = 0, simplify the goal to m + S p = S (m + p). This is precisely the premise, so use it to close the base case.",
         "code": "- simpl.\n  exact H."},
        {"text": "For the successor case, assume (k + m) + S p = S ((k + m) + p). At n = S k, simplification gives S ((k + m) + S p) = S (S ((k + m) + p)).",
         "code": "- simpl."},
    ]
    suffixes = {
        "A": {"text": "Use the induction hypothesis from left to right to replace (k + m) + S p inside the left successor by S ((k + m) + p). The two sides are then identical.",
              "code": "  rewrite IH.\n  reflexivity."},
        "B": {"text": "Both sides are successors. Apply successor congruence: it suffices to prove equality of the inner terms, which is the induction hypothesis. Use that hypothesis to finish.",
              "code": "  f_equal.\n  exact IH."},
    }
    rows = []
    for argument, last in suffixes.items():
        steps = [dict(s) for s in common] + [last]
        body = "\n".join(s["code"] for s in steps)
        rows.append({
            "schema_version": "pilot-1.0", "id": f"pd04_{argument}",
            "theorem_id": "pd04", "argument_id": argument, "split": "dev",
            "formal_statement": STATEMENT,
            "informal_statement": "If adding the successor of p to m gives the successor of m + p, the same identity holds with n + m in place of m, for every natural number n.",
            "informal_proof": " ".join(s["text"] for s in steps),
            "proof_body": body, "steps": steps, "argument_features": features(STATEMENT, body),
            "generation_family": FAMILY, "contrast_type": "equivalent_ih_realization",
            "argument_contract": {"induction_variable": "n", "base": "premise",
                                  "successor_ih_use": "rewrite" if argument == "A" else "congruence",
                                  "ih_direction": "forward" if argument == "A" else None},
            "review": {"reviewer": "assistant", "human_reviewed": False,
                       "primary_target_eligible": True, "alignment": "assistant-reviewed",
                       "unnecessary_induction": False, "redundant_ih_use": False,
                       "notes": "Natural propagation of the right-successor identity along the recursively defined left addition prefix. In the restricted proof vocabulary, the premise closes the base and the IH closes the successor. This is useful induction, not a claim of a globally shortest proof. A/B share the mathematical induction and differ in explicit successor steps."},
            "provenance": {"informal_statement_source": "assistant-curated",
                           "informal_proof_source": "assistant-curated",
                           "formal_proof_source": "assistant-curated",
                           "human_reviewed": False, "rocq_verified": False,
                           "model_generated": False, "historical_reference_copied": False,
                           "source_experiments": ["pilot-v2-coverage-audit"],
                           "historical_family_exposure": ["seed_018", "seed_028"],
                           "curation": "One newly authored theorem; right-successor generalization in the historically exposed dev family. No new theorem family or tactic."},
        })
    return rows


def family_review(additions, train, old_dev):
    historical = read_rows(ROOT / "data/development/pairs.jsonl")
    duplicates = {name: [r["id"] for r in group if equivalence_key(r["formal_statement"]) == equivalence_key(STATEMENT)]
                  for name, group in (("pilot_train", train), ("pilot_dev", old_dev), ("historical_development", historical))}
    require(not any(duplicates.values()), "New theorem duplicates an existing public theorem")
    return {
        "status": "PASS_WITH_DOCUMENTED_SCOPE", "reviewer": "assistant", "human_reviewed": False,
        "theorem_id": "pd04", "generation_family": FAMILY, "split": "dev",
        "public_duplicate_check": duplicates,
        "method": "Existing equivalence_key: definitional reduction, binder permutation, equality orientation; plus manual conservative family review.",
        "public_schema_and_family_audit": audit_rows(train + old_dev + additions),
        "family_reason": "The premise is the right-successor addition identity from seed_018. The conclusion propagates it under a common left prefix; seed_028 is its p=0 specialization up to definitional reduction. Both seeds are already assigned to successor_reassociation_variants. All A/B, renamed, reversed and definitional variants must remain in this dev family.",
        "training_family_exclusion": "There is no right-zero residual and the supplied equation is a composite right-successor identity, not a direct variable-substitution target. Existing train-family proof templates share induction mechanics, but the frozen family ledger separates right-zero and successor/reassociation identities.",
        "consumed_holdout_family_exclusion": "The declared conditional_permutation family swaps addends under a common prefix. This theorem transports a successor identity without permuting addends. Exclusion is a manual structural check against the existing family definition, not an exact comparison with hidden statements.",
        "holdout_content_read": False, "diagnostic_content_read": False,
        "limitations": ["No exact duplicate scan against holdout or diagnostic contents was performed; those files were not opened.",
                       "This is an exposed development-family extension, not a new independent holdout or unseen mathematics.",
                       "Family relationships are conservative manual annotations, not a proof of all semantic separation."],
    }


def prompt_lengths(rows):
    from constrained_v1.experiment import tokenizer_only
    from training_protocol_v1.data import encode_example
    tokenizer = tokenizer_only()
    prompt = json.loads((ROOT / "prompt_v2/selected.json").read_text())
    return {r["id"]: {k: v for k, v in encode_example(tokenizer, prompt, r).items()
                      if k in ("prompt_length", "target_length", "sequence_length")}
            for r in rows}


def validate_layout(directory):
    old = (ROOT / "data/pilot-v1/dev.jsonl").read_bytes()
    extra = (directory / "additions.jsonl").read_bytes()
    require((directory / "dev.jsonl").read_bytes() == old + extra, "Original six rows or additions changed")
    rows, additions = read_rows(directory / "dev.jsonl"), read_rows(directory / "additions.jsonl")
    require(len(rows) == 8 and [r["id"] for r in additions] == NEW_IDS, "Unexpected v2 membership")
    require(all(r["split"] == "dev" and r["generation_family"] == FAMILY for r in rows), "Dev family changed")
    require({r["formal_statement"] for r in additions} == {STATEMENT}, "Unexpected new theorem")
    audit_rows(load_split("train") + rows)
    return rows, additions


def check(directory=DATA, reverify=False):
    manifest = json.loads((directory / "dev_manifest.json").read_text())
    seal = json.loads((directory / "release.json").read_text())
    require(sha256_file(directory / "dev_manifest.json") == seal["dev_manifest_sha256"], "Manifest seal mismatch")
    require(manifest["status"] == "FROZEN_DEV_EXTENSION_ONLY", "Wrong release status")
    require(manifest["source_files_sha256"] == fingerprints(), "Protected source changed")
    require(manifest["builder_sha256"] == sha256_file(Path(__file__)), "Builder changed")
    require(set(manifest["files_sha256"]) == set(ARTIFACTS), "Unexpected artifact manifest")
    for name, expected in manifest["files_sha256"].items():
        require(sha256_file(directory / name) == expected, f"Frozen artifact changed: {name}")
    rows, additions = validate_layout(directory)
    require(manifest["dev_example_ids"] == [r["id"] for r in rows], "Manifest membership mismatch")
    for r in additions:
        require((directory / "references" / f"{r['id']}.v").read_text() == render_source(r["formal_statement"], r["proof_body"]), "Reference file mismatch")
    if reverify:
        token_evidence(rows)
        for r in rows:
            evidence = verify(r["formal_statement"], r["proof_body"])
            require(evidence.status == "PASS" and evidence.source_sha256 == r["verification"]["source_sha256"], f"Rocq failed: {r['id']}")
    return {"status": "PASS", "dev_rows": len(rows), "new_rows": len(additions),
            "original_six_byte_preserved": True, "reverified": 8 if reverify else 0,
            "model_generations": 0, "training_updates": 0}


def prepare(directory):
    require(not directory.exists(), "Release already exists; refusing overwrite")
    before = fingerprints()
    train, old_dev = load_split("train"), load_split("dev")
    additions = authored_rows()
    for row in additions:
        evidence = asdict(verify(STATEMENT, row["proof_body"]))
        require(evidence["status"] == "PASS", f"Rocq rejected {row['id']}: {evidence}")
        row["verification"] = evidence
        row["provenance"].update(rocq_verified=True, rocq_source_sha256=evidence["source_sha256"])
        row["quality_checks"] = {"reference_removal": ablations(row)}
        # Stronger check of the base premise than simply leaving a branch open.
        bad = row["proof_body"].replace("exact H.", "reflexivity.")
        bad_result = asdict(verify(STATEMENT, bad))
        require(bad_result["stage"] == "rocq" and bad_result["category"] == "PROOF_ERROR", "Premise-free base unexpectedly passes")
        row["quality_checks"]["base_reflexivity_replacement"] = {"mutated_body_sha256": digest(bad), "verification": bad_result}
        validate_row(row)
    combined = old_dev + additions
    tokens, lengths = token_evidence(combined), prompt_lengths(combined)
    for r in additions:
        r["quality_checks"]["decoder"] = tokens[r["id"]]
        r["quality_checks"]["frozen_prompt_length"] = lengths[r["id"]]
    family = family_review(additions, train, old_dev)
    reference_results = []
    for row in combined:
        v = asdict(verify(row["formal_statement"], row["proof_body"]))
        require(v["status"] == "PASS" and v["source_sha256"] == row["verification"]["source_sha256"], "Reference failed")
        reference_results.append({"id": row["id"], "verification": v, "decoder": tokens[row["id"]], "serialization": lengths[row["id"]]})
    contrasts = []
    for requested in additions:
        for supplied in additions:
            result = step_adherence(requested, supplied["proof_body"], "argument_" + requested["argument_id"])
            require(result["matched"] == (requested["id"] == supplied["id"]), "Step contracts do not separate A/B")
            contrasts.append({"requested": requested["id"], "reference_body": supplied["id"], "steps": result,
                              "mathematical_fidelity": "FAITHFUL", "reviewer": "assistant",
                              "reason": "Both verified bodies use the premise base and the same induction. Only the requested successor realization differs."})
    report = {"status": "PASS", "rocq_version": "9.2.0", "dev_references_verified": 8,
              "new_references_verified": 2, "token_paths_and_eos_allowed": 8,
              "new_negative_controls_failed": 6,
              "records": reference_results, "step_vs_fidelity_contract_checks": contrasts,
              "activity": {"model_generations": 0, "training_updates": 0,
                           "training_examples_added": 0, "dev_examples_added": 2,
                           "holdout_proof_files_opened": 0, "diagnostic_examples_opened": 0}}
    directory.mkdir(parents=True)
    (directory / "additions.jsonl").write_bytes(encoded_rows(additions))
    old_bytes = (ROOT / "data/pilot-v1/dev.jsonl").read_bytes()
    require(old_bytes.endswith(b"\n"), "Unexpected old JSONL boundary")
    (directory / "dev.jsonl").write_bytes(old_bytes + encoded_rows(additions))
    for row in additions:
        path = directory / "references" / f"{row['id']}.v"
        path.parent.mkdir(exist_ok=True)
        path.write_text(render_source(STATEMENT, row["proof_body"]))
    write_json(directory / "verification_report.json", report)
    write_json(directory / "family_review.json", family)
    (directory / "README.md").write_text(README)
    (directory / "VERIFICATION_REPORT.md").write_text(
        "# V2 dev reference verification\n\n"
        "PASS: 8/8 references verified with Rocq 9.2.0, including both additions. "
        "All eight exact tokenizer paths and EOS remain allowed by constrained-v1 within 256 completion tokens. "
        "All eight Prompt v2 serializations plus their targets fit the existing 768-token limit.\n\n"
        + "| Row | Completion + EOS | Total sequence |\n| --- | ---: | ---: |\n"
        + "".join(f"| {r['id']} | {tokens[r['id']]['with_eos']} | {lengths[r['id']]['sequence_length']} |\n" for r in combined)
        + "\nAll six new negative controls fail in Rocq: removal of the premise, removal of the IH, "
        "and replacement of the base premise with reflexivity, for each argument. "
        "These test usefulness in the authored scripts, not global proof minimality.\n\n"
        "All four A/B reference-contract comparisons behave as intended: same variant MATCH, opposite variant MISMATCH. "
        "Both verified variants remain mathematically faithful to the same induction. These are reference QA checks, not model accuracy.\n\n"
        "Schema, canonical alignment joins, features, local scope and public train/dev family checks pass. "
        "The original six dev lines are preserved byte-for-byte as the prefix of dev.jsonl. "
        "No weights were loaded, no model outputs generated, and no holdout or diagnostic proof files opened. "
        "No claim of exact duplicate exclusion against unopened holdout/diagnostic contents is made; see family_review.json.\n")
    require(before == fingerprints(), "Protected source changed during preparation")
    validate_layout(directory)
    manifest = {
        "schema_version": "pilot-dev-manifest-2.0", "release_id": "pilot-v2-dev-extension-1",
        "status": "FROZEN_DEV_EXTENSION_ONLY", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": sha256_file(Path(__file__)), "source_files_sha256": before,
        "files_sha256": {p: sha256_file(directory / p) for p in ARTIFACTS},
        "dev_example_ids": [r["id"] for r in combined], "new_example_ids": NEW_IDS,
        "theorem_ids": sorted({r["theorem_id"] for r in combined}),
        "family_assignment": {FAMILY: "dev"},
        "coverage": {"computational_base_ih_rewrite": 3, "computational_base_ih_congruence": 3,
                     "premise_base_ih_rewrite": 1, "premise_base_ih_congruence": 1},
        "train": {"source": "data/pilot-v1/train.jsonl", "rows": 24, "unchanged": True,
                  "sha256": before["data/pilot-v1/train.jsonl"]},
        "original_dev": {"source": "data/pilot-v1/dev.jsonl", "rows": 6,
                         "sha256": before["data/pilot-v1/dev.jsonl"], "byte_prefix_preserved": True},
        "policy": {"assignment_unit": "generation_family", "training_on_dev_forbidden": True,
                   "checkpoint_selection": "No new selection rule or runner in this release. Frozen v1 scorer still requires its original six rows. A separately frozen v2 training/evaluation protocol is required before using eight rows for selection.",
                   "holdout": "No v2 holdout is created. The consumed v1 holdout remains historical; a fresh independent holdout must be sealed before a later training experiment.",
                   "scope": "Dataset extension only: no CUDA port, inference, training, prompt/decoder/verifier changes."}}
    write_json(directory / "dev_manifest.json", manifest)
    write_json(directory / "release.json", {"status": "FROZEN", "dev_manifest_sha256": sha256_file(directory / "dev_manifest.json")})
    return check(directory)


README = """# ProofBridge v2 development extension

This frozen development-only release has **8 rows / 4 theorems / 1 existing family**.
It preserves the six original dev rows byte-for-byte and adds `pd04_A` and `pd04_B`.
The 24 training rows remain in their original file; no training examples are added.

The new statement is:

```coq
forall n m p : nat,
  m + S p = S (m + p) ->
  (n + m) + S p = S ((n + m) + p)
```

Induct on n. The zero branch reduces to the supplied premise. The successor branch
uses the IH through a forward rewrite (A) or successor congruence and exact IH (B).
The informal paragraphs and each aligned step specify those choices explicitly.
Both variants preserve the same mathematical induction. Opposite successor tactics
can be mathematically faithful while failing the requested proof-step contract.

The theorem generalizes the successor identity in historical seed_028; its premise
is the identity of seed_018. Both already belong to successor_reassociation_variants.
The old family allocation is preserved. No new theorem family, tactic or lemma is
introduced. The public duplicate check finds no exact/definitional/orientation/binder
equivalent statement in the old 30 development seeds or pilot train/dev. The family
review separates this successor law from the consumed conditional-permutation family
without opening holdout files. It is not an exact hidden-statement duplicate scan.

| Base case | IH rewrite | IH congruence |
| --- | ---: | ---: |
| Computation/reflexivity | 3 | 3 |
| Supplied premise | 1 | 1 |

The original pilot-1.0 row schema is retained: formal_statement and informal_statement
describe the theorem; informal_proof is the paragraph; steps retain text/code alignment;
proof_body joins step code; argument_features and argument_contract describe the proof;
generation_family controls grouping; provenance, review, verification and quality_checks
record assistant curation and reference QA. Only theorem and informal proof are model
inputs. IDs, contracts, references and QA metadata must not be included in prompts.

All original v1 files, including their known minor prose issues, remain unchanged.
All new semantic judgments are assistant reviews; human_reviewed is false. The premise
and IH are useful within these scripts; no global minimality or logically indispensable
premise claim is made. There is only one new theorem pair in an exposed family: this
fills a selection-dev coverage gap, not an independent generalization benchmark.

Files: additions.jsonl contains only the two new rows; dev.jsonl is the original six
lines plus those rows. references/ contains the exact verifier-rendered .v sources.
verification_report.json and VERIFICATION_REPORT.md record compiler/token evidence;
family_review.json records public comparisons and manual scope limitations.
dev_manifest.json commits membership, artifacts and protected dependencies; release.json
seals the manifest. This is not a complete pilot-v2 train/dev/holdout protocol.

From the repository root:

```sh
# Verify the release hashes and schema without model work or writing artifacts.
.venv/bin/python -B -m pilot_dev_v2.release check
# Additionally recompile all eight references and check exact tokenizer paths.
.venv/bin/python -B -m pilot_dev_v2.release check --reverify
# Test the extension and its tamper checks without opening holdout/diagnostic data.
.venv/bin/python -B -m unittest discover -s tests -p 'test_pilot_dev_v2.py' -v
```

One-time preparation uses `python -B -m pilot_dev_v2.release prepare`. It refuses an
existing release. To reproduce construction without replacing evidence, use a fresh
`--output /private/tmp/proofbridge-dev-v2-reproduction` directory. Compiler timings and
freeze timestamps vary, so reconstructed manifests are not expected to be identical.

No training or generation entry point is provided. The frozen v1 scorer requires six
dev rows; it is not repurposed silently. Before training, freeze a separate v2 protocol
with explicit selection membership/rule, an original-data control, and a fresh holdout.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--reverify", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        require(not args.reverify, "prepare already verifies references")
        result = prepare(args.output)
    else:
        result = check(args.output, args.reverify)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
