"""Schema, conservative equivalence checks, and train/dev-only access."""

from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from constrained_v1.grammar import accepts
from scripts.seed_schema import argument_features, definitional_key
from verifier import render_source, sha256_file, tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/pilot-v1"
MANIFEST = DATA / "split_manifest.json"
PATHS = {"train": "train.jsonl", "dev": "dev.jsonl", "holdout": "reserved/holdout.jsonl"}
FAMILIES = {
    "equality_transport": {"split": "train", "historical_families": ["premise_reuse", "premise_congruence", "premise_definitional_substitution"],
        "reason": "All direct premise substitution, orientation, constructor exposure, and congruence variants stay together."},
    "right_zero_variants": {"split": "train", "historical_families": ["right_zero_variants"],
        "reason": "Right-zero, repeated-zero, reversed, contextual, and premise-conditioned forms stay together."},
    "successor_reassociation_variants": {"split": "dev", "historical_families": ["successor_reassociation_variants"],
        "reason": "Successor, increment and reassociation forms are connected in the historical family ledger; reserve the whole group for dev."},
    "conditional_permutation": {"split": "holdout", "historical_families": ["conditional_permutation"],
        "reason": "Conditional permutation lifted under a common prefix, including substitutions into the premise, stays entirely in holdout."},
}
PATTERNS = ("forward_rewrite", "reverse_rewrite", "rewrite_before_simpl", "rewrite_after_simpl",
            "premise_in_base_case", "ih_via_rewrite", "ih_via_congruence", "premise_and_induction")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def equivalence_key(statement):
    """Definitional reduction + binder permutation + symmetric equalities.

    This deliberately does NOT prove equivalence by addition laws. Human-curated
    family assignments catch those relationships. All true propositions must not
    be collapsed merely because they are logically equivalent.
    """
    count, *equalities = definitional_key(statement)
    def rename(term, permutation):
        if term[0] == "var":
            return ("var", permutation[term[1]])
        return (term[0], *(rename(t, permutation) for t in term[1:]))
    keys = []
    for permutation in itertools.permutations(range(count)):
        pairs = [tuple(sorted((rename(a, permutation), rename(b, permutation)))) for a, b in equalities]
        keys.append((count, *pairs))
    return min(keys)


def commands(body):
    branch, command, result = 0, [], []
    for token in tokenize(body):
        if token == "-":
            branch += 1
        elif token == ".":
            result.append((branch, command))
            command = []
        else:
            command.append(token)
    return result


def features(statement, body):
    base = argument_features(statement, body)
    cmds = commands(body)
    rewrites = [{"branch": b, "reference": c[-1], "direction": "reverse" if "<-" in c else "forward"}
                for b, c in cmds if c[0] == "rewrite"]
    ih = next((c[-2] for _, c in cmds if c[0] == "induction"), None)
    # Authored pilot uses H for its one premise; schema requires canonical intros.
    premise_base = any(b == 1 and c[0] in {"exact", "rewrite", "apply"} and c[-1] == "H" for b, c in cmds)
    ordering = []
    for index, (branch, command) in enumerate(cmds):
        if command[0] != "rewrite":
            continue
        before = any(b == branch and c[0] == "simpl" for b, c in cmds[:index])
        after = any(b == branch and c[0] == "simpl" for b, c in cmds[index + 1:])
        ordering.append({"command_index": index, "branch": branch, "simpl_before": before, "simpl_after": after})
    return base | {"rewrite_steps": rewrites, "rewrite_simpl_order": ordering,
                   "premise_in_base_case": premise_base,
                   "ih_via_rewrite": bool(ih) and any(c[0] == "rewrite" and c[-1] == ih for _, c in cmds),
                   "ih_via_congruence": bool(ih) and any(b == 2 and c[0] == "f_equal" for b, c in cmds)
                       and any(b == 2 and c[0] in {"exact", "apply"} and c[-1] == ih for b, c in cmds)}


def patterns(row):
    f = row["argument_features"]
    enabled = {"forward_rewrite": any(s["direction"] == "forward" for s in f["rewrite_steps"]),
        "reverse_rewrite": any(s["direction"] == "reverse" for s in f["rewrite_steps"]),
        "rewrite_before_simpl": any(s["simpl_after"] for s in f["rewrite_simpl_order"]),
        "rewrite_after_simpl": any(s["simpl_before"] for s in f["rewrite_simpl_order"]),
        "premise_in_base_case": f["premise_in_base_case"], "ih_via_rewrite": f["ih_via_rewrite"],
        "ih_via_congruence": f["ih_via_congruence"],
        "premise_and_induction": f["uses_premise"] and f["uses_induction"]}
    return [p for p in PATTERNS if enabled[p]]


def validate_row(row, verified=True):
    required = ("id", "theorem_id", "argument_id", "split", "formal_statement", "informal_statement",
                "informal_proof", "proof_body", "steps", "argument_features", "generation_family",
                "provenance", "review", "argument_contract", "contrast_type")
    if any(not row.get(field) for field in required):
        raise ValueError("Missing pilot schema field")
    if row["schema_version"] != "pilot-1.0" or row["split"] != FAMILIES[row["generation_family"]]["split"]:
        raise ValueError("Family/split mismatch")
    if row["proof_body"] != "\n".join(s["code"] for s in row["steps"]):
        raise ValueError("Canonical body differs from step alignment")
    if row["informal_proof"] != " ".join(s["text"].strip() for s in row["steps"]):
        raise ValueError("Informal paragraph differs from step alignment")
    if any(not s["text"].strip() or not s["code"].strip() for s in row["steps"]):
        raise ValueError("Empty aligned step")
    if not accepts(row["formal_statement"], row["proof_body"]):
        raise ValueError("Frozen decoder rejects proof")
    if row["argument_features"] != features(row["formal_statement"], row["proof_body"]):
        raise ValueError("Argument features drifted")
    if row["review"]["primary_target_eligible"] is not True or row["provenance"]["human_reviewed"] is not False:
        raise ValueError("Unreviewed/weak target or inaccurate human provenance")
    if verified:
        result = row["verification"]
        if (result["status"] != "PASS" or result["category"] != "VERIFIED" or not result["kernel_checked"]
                or not result["assumptions_checked"] or result["compiler_version"] != "9.2.0"
                or result["source_sha256"] != digest(render_source(row["formal_statement"], row["proof_body"]))
                or not row["provenance"]["rocq_verified"]):
            raise ValueError("Missing or mismatched Rocq evidence")


def audit_rows(rows, verified=True):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    equivalents, theorems, exact_pairs = defaultdict(list), defaultdict(list), set()
    for row in rows:
        validate_row(row, verified=verified)
        equivalents[equivalence_key(row["formal_statement"])].append(row)
        theorems[row["theorem_id"]].append(row)
        pair = (tuple(tokenize(row["formal_statement"])), row["proof_body"])
        if pair in exact_pairs:
            raise ValueError("Duplicate equivalent statement/proof target")
        exact_pairs.add(pair)
    for group in equivalents.values():
        if len({(r["generation_family"], r["split"]) for r in group}) != 1:
            raise ValueError("Equivalent theorems cross family or split")
    for group in theorems.values():
        if len({r["formal_statement"] for r in group}) != 1 or len({r["argument_id"] for r in group}) != len(group):
            raise ValueError("Malformed same-theorem argument group")
        if len({r["informal_proof"] for r in group}) != len(group):
            raise ValueError("Different targets need different informal arguments")
    return {"status": "PASS", "examples": len(rows), "theorems": len(theorems),
            "paired_theorems": sum(len(g) > 1 for g in theorems.values()),
            "equivalence_classes": len(equivalents), "cross_split_equivalents": 0,
            "cross_split_families": 0, "duplicate_statement_body_pairs": 0}


def coverage(rows):
    return {split: {"examples": len(group), "theorems": len({r["theorem_id"] for r in group}),
            "families": dict(Counter(r["generation_family"] for r in group)),
            "patterns": {p: sum(p in patterns(r) for r in group) for p in PATTERNS},
            "same_theorem_pairs": sum(n > 1 for n in Counter(r["theorem_id"] for r in group).values())}
        for split in PATHS for group in [[r for r in rows if r["split"] == split]]}


def load_split(split):
    if split not in ("train", "dev"):
        raise ValueError("Holdout is reserved until after training; this loader exposes only train/dev")
    manifest = json.loads(MANIFEST.read_text())
    path = DATA / PATHS[split]
    if sha256_file(path) != manifest["files_sha256"][str(path.relative_to(ROOT))]:
        raise ValueError("Frozen data changed")
    rows = read_rows(path)
    audit_rows(rows)
    return rows


def training_records(split="train"):
    """Only formal theorem and informal proof are inputs; no gold metadata leak.

    Deliberately not a chat template and not a training runner. Prompt v2 stays
    unchanged; a later authorized training experiment must fix its own loss mask.
    """
    return [{"id": r["id"], "input": {"formal_statement": r["formal_statement"], "informal_proof": r["informal_proof"]},
             "target": {"proof_body": r["proof_body"]}} for r in load_split(split)]
