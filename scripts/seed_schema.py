"""Deterministic annotations and leakage checks for the fixed 30-seed release.

This module annotates the restricted grammar; Rocq remains the proof checker.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from verifier import REFERENCE_TACTICS, StatementParser, tokenize

SCHEMA_VERSION = "seed-0.2"
FAMILIES = Path(__file__).resolve().parents[1] / "data/seeds/families.json"


def authored_body(seed: dict) -> str:
    return "\n".join(step["code"] for step in seed["steps"])


def informal_paragraph(seed: dict) -> str:
    return " ".join(step["text"].strip() for step in seed["steps"])


def steps_sha256(seed: dict) -> str:
    payload = json.dumps(seed["steps"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_body(seed: dict) -> str:
    """Join unchanged authored steps, then apply explicit, line-preserving edits.

    Only apply-local -> exact-same-local is permitted. Whether exact closes the
    goal is established by compiling both bodies, not inferred by this function.
    """
    lines = authored_body(seed).splitlines()
    used_lines = set()
    for edit in seed["metadata"]["proof_normalizations"]:
        number = edit.get("proof_line")
        if type(number) is not int or not 1 <= number <= len(lines) or number in used_lines:
            raise ValueError(f"{seed['id']}: invalid normalization line")
        before = lines[number - 1]
        match = re.fullmatch(r"(\s*)apply ([A-Za-z][A-Za-z0-9_']*)\.", before)
        if (not match or edit.get("from") != before
                or edit.get("to") != f"{match[1]}exact {match[2]}."
                or not isinstance(edit.get("reason"), str) or not edit["reason"].strip()):
            raise ValueError(f"{seed['id']}: unsupported proof normalization")
        lines[number - 1] = edit["to"]
        used_lines.add(number)
    return "\n".join(lines)


def argument_features(statement: str, body: str) -> dict:
    parser = StatementParser(tokenize(statement))
    _, binder_count = parser.parse()
    tokens = tokenize(body)
    commands = []
    command = []
    for token in tokens:
        if token == ".":
            commands.append(command)
            command = []
        elif token != "-":
            command.append(token)
    introduced = [name for command in commands if command[0] == "intros" for name in command[1:]]
    # A supplied equality premise is distinct from the induction hypothesis.
    premises = set(introduced[len(parser.variables):binder_count])
    induction = next((command for command in commands if command[0] == "induction"), None)
    tactics = {command[0] for command in commands}
    return {
        "uses_induction": induction is not None,
        "induction_variable": induction[1] if induction else None,
        "uses_premise": any(command[0] in REFERENCE_TACTICS and command[-1] in premises for command in commands),
        "uses_rewrite": "rewrite" in tactics,
        "uses_f_equal": "f_equal" in tactics,
        "uses_reflexivity": "reflexivity" in tactics,
    }


class DefinitionalParser(StatementParser):
    """Compare the fragment using only numeral/successor and left-add reduction.

    No associativity, commutativity, right-zero law, or equality proof search.
    StatementParser validates the grammar before this parser is invoked.
    """

    def __init__(self, tokens):
        super().__init__(tokens)
        self.equalities = []

    def atom(self, depth):
        token = self.take()
        if token == "(":
            term = self.term(depth + 1)
            self.take(")")
            return term
        if token == "S":
            return ("S", self.atom(depth + 1))
        if token in {"0", "1", "2", "3"}:
            term = ("0",)
            for _ in range(int(token)):
                term = ("S", term)
            return term
        return ("var", self.variables.index(token))

    @staticmethod
    def add(left, right):
        if left[0] == "0":
            return right
        if left[0] == "S":
            return ("S", DefinitionalParser.add(left[1], right))
        return ("+", left, right)

    def term(self, depth=0):
        term = self.atom(depth)
        while self.peek() == "+":
            self.take("+")
            term = self.add(term, self.atom(depth))
        return term

    def equality(self):
        left = self.term()
        self.take("=")
        self.equalities.append((left, self.term()))


def definitional_key(statement: str) -> tuple:
    tokens = tokenize(statement)
    StatementParser(tokens).parse()
    parser = DefinitionalParser(tokens)
    parser.parse()
    return (len(parser.variables), *parser.equalities)


def audit_families(seeds: list[dict], families_file: Path = FAMILIES) -> dict:
    manifest = json.loads(families_file.read_text())
    declared = {}
    for family, specification in manifest["families"].items():
        for seed_id in specification["members"]:
            if seed_id in declared:
                raise ValueError(f"{seed_id}: repeated family manifest member")
            declared[seed_id] = (family, specification["split_group"])
    if set(declared) != {seed["id"] for seed in seeds}:
        raise ValueError("Family manifest must cover precisely the 30 seed IDs.")
    by_definition = defaultdict(list)
    for seed in seeds:
        if (seed["generation_family"], seed["split_group"]) != declared[seed["id"]]:
            raise ValueError(f"{seed['id']}: generation family/split group disagrees with curated family manifest")
        by_definition[definitional_key(seed["formal_statement"])].append(seed)
    equivalents = []
    for members in by_definition.values():
        if len(members) < 2:
            continue
        if len({(seed["generation_family"], seed["split_group"]) for seed in members}) != 1:
            raise ValueError("Definitionally equivalent variants cross a generation family or split group")
        equivalents.append([seed["id"] for seed in members])
    return {
        "status": "PASS",
        "unique_definitional_statements": len(by_definition),
        "definitional_equivalence_classes": equivalents,
        "generation_families": len(manifest["families"]),
        "split_groups": len({seed["split_group"] for seed in seeds}),
        "family_manifest_checked": True,
        "split_policy": "Keep each split_group intact; generation_family refines it and never licenses splitting it.",
    }
