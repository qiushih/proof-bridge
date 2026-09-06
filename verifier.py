"""Restricted addition-proof verifier. Python standard library only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
LOCK_FILE = ROOT / "environment.lock.json"
PRELUDE = "From Stdlib Require Import Arith.PeanoNat.\n"
THEOREM_NAME = "proof_bridge_target"
MAX_INPUT_CHARS = 16_384
MAX_COMMANDS = 40
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_']*\Z")
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_']*(?:\.[A-Za-z][A-Za-z0-9_']*)*|[0-9]+|->|<-|[():,+=.\[\]|-]")
FORBIDDEN_COMMANDS = {
    "Admitted", "admit", "Abort", "Axiom", "Axioms", "Parameter", "Parameters",
    "Require", "Import", "Export", "From", "Load", "Theorem", "Lemma", "Goal",
    "Proof", "Qed", "Defined", "Definition", "Fixpoint", "Inductive", "CoFixpoint",
    "Conjecture", "Hypothesis", "Variable", "Variables", "Context", "Section",
    "End", "Module", "Ltac", "Ltac2", "Set", "Unset", "Add", "Declare",
    "Notation", "Infix", "Include", "Redirect", "Fail", "Succeed", "Time",
}
FORBIDDEN_TACTICS = {"auto", "eauto", "lia", "nia", "ring", "hammer", "give_up"}
NO_ARGUMENT_TACTICS = {"simpl", "reflexivity", "symmetry", "f_equal"}
REFERENCE_TACTICS = {"rewrite", "exact", "apply"}
RESERVED = FORBIDDEN_COMMANDS | FORBIDDEN_TACTICS | NO_ARGUMENT_TACTICS | REFERENCE_TACTICS | {
    "forall", "nat", "S", "intros", "induction", "as", THEOREM_NAME,
}


class Rejection(Exception):
    def __init__(self, category: str, message: str):
        self.category = category
        super().__init__(message)


@dataclass
class VerificationResult:
    status: str
    category: str
    message: str
    stage: str
    compiler_version: str | None = None
    kernel_checked: bool = False
    assumptions_checked: bool = False
    elapsed_seconds: float = 0.0
    source_sha256: str | None = None
    stdout: str = ""
    stderr: str = ""


def sha256_file(filename: Path) -> str:
    with filename.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokenize(source: str) -> list[str]:
    if not isinstance(source, str):
        raise Rejection("INPUT_ERROR", "Statement and proof must be strings.")
    if len(source) > MAX_INPUT_CHARS:
        raise Rejection("LIMIT_EXCEEDED", "Input exceeds 16,384 characters.")
    # Strip nested Rocq comments, replacing them with whitespace (never join words).
    clean: list[str] = []
    pos = depth = 0
    while pos < len(source):
        if source.startswith("(*", pos):
            depth += 1
            clean.append(" ")
            pos += 2
        elif source.startswith("*)", pos):
            if not depth:
                raise Rejection("SYNTAX_ERROR", "Unmatched comment terminator.")
            depth -= 1
            clean.append(" ")
            pos += 2
        else:
            clean.append(" " if depth else source[pos])
            pos += 1
    if depth:
        raise Rejection("SYNTAX_ERROR", "Unterminated comment.")
    source = "".join(clean)
    tokens: list[str] = []
    pos = 0
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        match = TOKEN.match(source, pos)
        if not match:
            raise Rejection("SYNTAX_ERROR", f"Unsupported character at offset {pos}: {source[pos]!r}")
        token = match.group()
        # Qualified command/tactic names are disallowed as well.
        for component in token.split("."):
            if component in FORBIDDEN_COMMANDS:
                raise Rejection("FORBIDDEN_COMMAND", f"Forbidden command: {component}")
            if component in FORBIDDEN_TACTICS:
                raise Rejection("FORBIDDEN_TACTIC", f"Forbidden tactic: {component}")
        tokens.append(token)
        pos = match.end()
    return tokens


def check_identifier(name: str) -> None:
    if not IDENTIFIER.fullmatch(name) or name in RESERVED or name.startswith("pb_local_"):
        raise Rejection("SYNTAX_ERROR", f"Unsupported local identifier: {name!r}")


class StatementParser:
    """forall 1–3 nat variables, equality, optionally preceded by one equality."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0
        self.variables: list[str] = []

    def take(self, expected: str | None = None) -> str:
        if self.pos == len(self.tokens):
            raise Rejection("SYNTAX_ERROR", "Unexpected end of theorem statement.")
        token = self.tokens[self.pos]
        if expected is not None and token != expected:
            raise Rejection("SYNTAX_ERROR", f"Expected {expected!r}, got {token!r}.")
        self.pos += 1
        return token

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def atom(self, depth: int) -> None:
        if depth > 5:
            raise Rejection("LIMIT_EXCEEDED", "Expression nesting exceeds 5.")
        token = self.take()
        if token == "(":
            self.term(depth + 1)
            self.take(")")
        elif token == "S":
            self.atom(depth + 1)
        elif token not in self.variables and token not in {"0", "1", "2", "3"}:
            raise Rejection("SYNTAX_ERROR", f"Unsupported arithmetic term: {token!r}")

    def term(self, depth: int = 0) -> None:
        self.atom(depth)
        while self.peek() == "+":
            self.take("+")
            self.atom(depth)

    def equality(self) -> None:
        self.term()
        self.take("=")
        self.term()

    def parse(self) -> tuple[str, int]:
        self.take("forall")
        while self.peek() != ":":
            name = self.take()
            check_identifier(name)
            if name in self.variables:
                raise Rejection("SYNTAX_ERROR", "Duplicate quantified variable.")
            self.variables.append(name)
            if len(self.variables) > 3:
                raise Rejection("LIMIT_EXCEEDED", "At most three quantified variables are supported.")
        if not self.variables:
            raise Rejection("SYNTAX_ERROR", "Expected a quantified variable.")
        self.take(":")
        self.take("nat")
        self.take(",")
        self.equality()
        premise_count = 0
        if self.peek() == "->":
            self.take("->")
            self.equality()
            premise_count = 1
        if self.peek() is not None:
            raise Rejection("SYNTAX_ERROR", "Unexpected text after theorem statement.")
        return " ".join(self.tokens), len(self.variables) + premise_count


def render_body(tokens: list[str], binder_count: int) -> str:
    """Parse allowed atomic tactics and render fresh local names with branch scope.

    No global helpers are enabled in milestone 1. References are accepted only
    for explicitly introduced locals; generated names prevent global fallback.
    """
    lines: list[str] = []
    scope: dict[str, str] = {}
    branch_scope: dict[str, str] | None = None
    induction_names: tuple[str, str] | None = None
    branches = command_count = introduced = serial = 0
    pos = 0

    def bind(name: str) -> str:
        nonlocal serial
        check_identifier(name)
        if name in scope:
            raise Rejection("SYNTAX_ERROR", f"Local name already in use: {name}")
        fresh = f"pb_local_{serial}"
        serial += 1
        scope[name] = fresh
        return fresh

    while pos < len(tokens):
        if tokens[pos] == "-":
            if branch_scope is None or branches == 2:
                raise Rejection("SYNTAX_ERROR", "Only the two induction branch bullets are supported.")
            scope = branch_scope.copy()
            branches += 1
            if branches == 2:
                assert induction_names is not None
                # Reuse exactly the names emitted in the induction command.
                scope.update(zip(induction_names, successor_names))
            lines.append("-")
            pos += 1
            continue
        if branch_scope is not None and branches == 0:
            raise Rejection("SYNTAX_ERROR", "An induction must be followed by explicit '-' branches.")
        try:
            end = tokens.index(".", pos)
        except ValueError:
            raise Rejection("SYNTAX_ERROR", "Every tactic must end with a period.") from None
        command = tokens[pos:end]
        pos = end + 1
        command_count += 1
        if command_count > MAX_COMMANDS:
            raise Rejection("LIMIT_EXCEEDED", "At most 40 tactic commands are supported.")
        if not command:
            raise Rejection("SYNTAX_ERROR", "Empty tactic command.")
        tactic, *args = command
        if tactic == "intros":
            if branch_scope is not None or not args:
                raise Rejection("SYNTAX_ERROR", "Use explicit intros names before induction.")
            introduced += len(args)
            if introduced > binder_count:
                raise Rejection("SYNTAX_ERROR", "Too many introduction names.")
            lines.append("intros " + " ".join(bind(name) for name in args) + ".")
        elif tactic == "induction":
            if branch_scope is not None:
                raise Rejection("LIMIT_EXCEEDED", "Only one induction is supported.")
            if (len(args) != 7 or args[1:4] != ["as", "[", "|"] or args[-1] != "]"):
                raise Rejection("SYNTAX_ERROR", "Expected induction n as [| k IH].")
            if introduced != binder_count or args[0] not in scope:
                raise Rejection("SYNTAX_ERROR", "Introduce all theorem binders before induction on a local.")
            variable = scope[args[0]]
            branch_scope = scope.copy()
            induction_names = (args[4], args[5])
            successor_names = (bind(args[4]), bind(args[5]))
            lines.append(f"induction {variable} as [| {successor_names[0]} {successor_names[1]}].")
            scope = branch_scope.copy()
        elif tactic in NO_ARGUMENT_TACTICS:
            if args:
                raise Rejection("SYNTAX_ERROR", f"{tactic} accepts no arguments in this verifier.")
            lines.append(tactic + ".")
        elif tactic in REFERENCE_TACTICS:
            direction = ""
            if tactic == "rewrite" and args and args[0] in {"<-", "->"}:
                direction = args.pop(0) + " "
            if len(args) != 1:
                raise Rejection("SYNTAX_ERROR", f"{tactic} expects one local reference, not a proof term.")
            if args[0] not in scope:
                raise Rejection("FORBIDDEN_HELPER", f"Reference is not an in-scope local: {args[0]}")
            lines.append(f"{tactic} {direction}{scope[args[0]]}.")
        else:
            raise Rejection("FORBIDDEN_TACTIC", f"Tactic is outside the allowlist: {tactic}")
    if branch_scope is not None and branches != 2:
        raise Rejection("SYNTAX_ERROR", "Expected exactly two induction branches.")
    return "\n".join(lines)


def render_source(statement: str, body: str) -> str:
    target, binder_count = StatementParser(tokenize(statement)).parse()
    proof = render_body(tokenize(body), binder_count)
    return (
        PRELUDE + f"Theorem {THEOREM_NAME} : {target}.\nProof.\n{proof}\nQed.\n"
        + f"Check ({THEOREM_NAME} : {target}).\nPrint Assumptions {THEOREM_NAME}.\n"
    )


def run_process(command: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    """Fresh environment; terminate the process group on timeout (POSIX)."""
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TMPDIR": str(cwd)}
    with subprocess.Popen(
        command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, stdout, stderr) from None
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def check_environment(cwd: Path) -> tuple[dict, Path]:
    lock = json.loads(LOCK_FILE.read_text())
    prefix = Path(lock["homebrew_prefix"]) / "Cellar" / "rocq" / lock["rocq_package_version"]
    for relative, expected in lock["runtime_fingerprints"].items():
        filename = prefix / relative
        if not filename.is_file() or sha256_file(filename) != expected:
            raise Rejection("ENVIRONMENT_ERROR", f"Pinned runtime missing or changed: {filename}")
    compiler = prefix / "bin/rocq"
    version = run_process([str(compiler), "compile", "-print-version"], cwd, 5)
    if version.returncode or version.stdout.strip() != lock["compiler_version_output"]:
        raise Rejection("ENVIRONMENT_ERROR", "Compiler/OCaml version differs from environment.lock.json.")
    if hashlib.sha256(PRELUDE.encode()).hexdigest() != lock["prelude_sha256"]:
        raise Rejection("ENVIRONMENT_ERROR", "Prelude differs from the environment lock.")
    return lock, compiler


def verify(statement: str, proof_body: str, timeout: float = 10.0) -> VerificationResult:
    started = time.monotonic()
    stage = "policy"
    version = source_hash = None
    try:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise Rejection("INPUT_ERROR", "Timeout must be a finite positive number of seconds.")
        source = render_source(statement, proof_body)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="proof-bridge-") as temporary:
            work = Path(temporary)
            stage = "environment"
            lock, compiler = check_environment(work)
            version = lock["rocq_package_version"]
            filename = work / "Candidate.v"
            filename.write_text(source, encoding="utf-8")
            stage = "rocq"
            process = run_process(
                [str(compiler), "compile", "-q", "-color", "no", "-noglob",
                 "-coqlib", str(compiler.parent.parent / "lib/ocaml/coq"), filename.name],
                work, timeout,
            )
            output = process.stdout + "\n" + process.stderr
            if process.returncode:
                if "Syntax error:" in output:
                    category = "SYNTAX_ERROR"
                elif "Attempt to save an incomplete proof" in output:
                    category = "INCOMPLETE_PROOF"
                else:
                    category = "PROOF_ERROR"
                result = VerificationResult("FAIL", category, "Rocq rejected the candidate.", stage)
            elif not (work / "Candidate.vo").is_file():
                result = VerificationResult("FAIL", "ENVIRONMENT_ERROR", "Compiler produced no .vo file.", stage)
            elif "Closed under the global context" not in process.stdout.splitlines():
                result = VerificationResult("FAIL", "ASSUMPTIONS_ERROR", "Expected an axiom-free theorem.", stage)
            else:
                result = VerificationResult("PASS", "VERIFIED", "Rocq checked the fixed theorem without global axioms.", stage,
                                            kernel_checked=True, assumptions_checked=True)
            result.stdout = process.stdout
            result.stderr = process.stderr
    except Rejection as error:
        result = VerificationResult("FAIL", error.category, str(error), stage)
    except subprocess.TimeoutExpired as error:
        category = "TIMEOUT" if stage == "rocq" else "ENVIRONMENT_ERROR"
        result = VerificationResult("FAIL", category, f"{stage} process exceeded its timeout.", stage,
                                    stdout=error.stdout or "", stderr=error.stderr or "")
    except (OSError, ValueError, KeyError) as error:
        result = VerificationResult("FAIL", "ENVIRONMENT_ERROR", str(error), stage)
    result.compiler_version = version
    result.source_sha256 = source_hash
    result.elapsed_seconds = round(time.monotonic() - started, 4)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement", help="Fixed theorem statement, without a trailing period")
    parser.add_argument("--proof-file", type=Path, help="UTF-8 file containing only the candidate proof body")
    parser.add_argument("--example", type=Path, help="Existing Proof Bridge JSON example")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.example:
        if args.statement or args.proof_file:
            parser.error("--example cannot be combined with --statement or --proof-file")
        try:
            record = json.loads(args.example.read_text())
            result = verify(record["input"]["formal_statement"], record["target"]["proof_body"], args.timeout)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = VerificationResult("FAIL", "INPUT_ERROR", str(error), "input")
    elif args.statement and args.proof_file:
        try:
            result = verify(args.statement, args.proof_file.read_text(), args.timeout)
        except OSError as error:
            result = VerificationResult("FAIL", "INPUT_ERROR", str(error), "input")
    else:
        parser.error("provide --example OR both --statement and --proof-file")
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
