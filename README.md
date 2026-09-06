# Proof Bridge

Proof Bridge provides a fixed natural-number addition environment, a restricted Python verifier, and the first 30 individually curated informal/formal seed pairs. No model has been trained.

The tested environment is **Rocq 9.2.0, Stdlib 9.1.0, OCaml 5.5.0, and Python 3.13.7**, on Apple Silicon macOS 15.7.3 with Homebrew at `/opt/homebrew`. Rocq's version command abbreviates its package version to `9.2`. Python 3.11 or newer is required; there are no pip dependencies.

The checked-in `environment.lock.json` records exact bottle URLs and SHA-256 digests for Rocq and all four runtime dependencies, the prelude hash, and selected installed runtime fingerprints. The installer prepares a named local Homebrew tap, `proof-bridge/locked`, whose five recipes select those official binary bottles and locked dependencies. It verifies downloads, installs in dependency order, and pins the packages. It refuses conflicting linked versions or modified local recipes instead of replacing them. Source builds are disabled in these recipes. This uses Homebrew's normal named-tap interface without enabling file-path package installation.

Run these commands from the repository root on macOS 15+ ARM64 with Homebrew and Python 3.11+ already installed:

```sh
python3 scripts/setup_rocq.py
python3 scripts/setup_rocq.py --check
/opt/homebrew/Cellar/rocq/9.2.0/bin/rocq --version
/opt/homebrew/Cellar/rocq/9.2.0/bin/rocq compile -q examples/add_zero_right.v
python3 verifier.py --example examples/add_zero_right.json
python3 -m unittest discover -s tests -v
python3 scripts/verify_example.py
```

The standalone compile must exit zero and print `Closed under the global context`. The verifier must return JSON with `"status": "PASS"`, `"category": "VERIFIED"`, and both check flags true. The final command rechecks both representations and updates the existing JSON record with timestamps, hashes, compiler diagnostics, and actual verification status. Generated Rocq artifacts and cached bottles are ignored by Git.

The first seed release is documented in [data/seeds/README.md](data/seeds/README.md). Its editable source is `data/seeds/curated.json`; the exported pairs are in `data/seeds/pairs.jsonl`, with actual results in `data/seeds/verification_report.json`. The 30 pairs cover 6 definition proofs, 9 equality-premise proofs, and 15 single-induction proofs. Each proof is checked independently with the unchanged verifier and no global helpers. They were individually authored and reviewed by the assistant; they are not a human-authored evaluation corpus.

Recheck the stored seed artifacts and recompile every proof:

```sh
python3 scripts/verify_seeds.py --check
python3 scripts/verify_seeds.py --audit
python3 -m unittest discover -s tests -v
```

After intentionally editing a curated pair, rebuild the records and compilation evidence:

```sh
python3 scripts/verify_seeds.py --write
```

The refined [seed-0.2 schema](data/seeds/SCHEMA.md) adds a proof paragraph, canonical proof body, argument features, generation family, and explicit provenance to every record. All original statements and aligned steps are preserved; three canonical bodies normalize direct hypothesis use from `apply` to `exact`, with both versions verified and the edits recorded in metadata. All 30 records remain `split: unassigned`. Seven generation families refine five conservative split groups; four definitionally equivalent pairs are kept together. Keep entire split groups and the related original example together when designing future data splits. Stored PASS metadata is checked against source, environment, verifier, schema/family definitions, generated-code, and dataset hashes before revalidation. Full informal/formal fidelity still requires review; the kernel does not certify English.

To independently exercise the locked downloads without changing installed packages:

```sh
python3 scripts/setup_rocq.py --fetch-only
```

The first implementation was tested against actual Rocq, including a real compiler timeout. The tests cover valid, invalid, incomplete, and malformed proofs; every requested forbidden command/tactic; qualified and unqualified helpers; import/theorem injection; branch-local hypothesis scope; equality rewriting; CLI exit codes; and environment failures. Run the test command above for the current count and results.

The Python interface takes the statement and body separately:

```python
from verifier import verify

result = verify(
    "forall n : nat, n + 0 = n",
    """intros n.
induction n as [| k IH].
- simpl. reflexivity.
- simpl. rewrite IH. reflexivity.""",
    timeout=10.0,
)
print(result.status, result.category)
```

For a body stored in a UTF-8 text file, use:

```sh
python3 verifier.py --statement 'forall n : nat, n + 0 = n' --proof-file /path/to/body.txt --timeout 10
```

The candidate contains only tactics and branch bullets. The verifier parses and re-renders it, supplies the fixed import and theorem wrapper, adds `Qed`, checks the resulting theorem type, and prints its assumptions. It compiles in a fresh temporary directory, with no shell, no startup rcfile, and a minimal subprocess environment. Each compile has a wall-clock timeout; on expiry the POSIX process group is killed. The compiler/environment probe has a separate five-second timeout, so total API time can exceed the requested compilation timeout.

| Result category | Meaning |
| --- | --- |
| `VERIFIED` | PASS: full compile, `.vo` output, and no global axioms reported |
| `PROOF_ERROR` | Rocq rejected the tactic's mathematical/type behavior |
| `INCOMPLETE_PROOF` | Rocq reached `Qed` with unfinished obligations |
| `SYNTAX_ERROR` | Malformed restricted syntax, unsupported term syntax, or a Rocq parser error; `stage` identifies the source |
| `FORBIDDEN_COMMAND` | Admission, declaration, import, theorem modification, or another prohibited command |
| `FORBIDDEN_TACTIC` | Explicitly prohibited tactic or any tactic outside the allowlist |
| `FORBIDDEN_HELPER` | Reference is not an explicitly introduced, in-scope local |
| `TIMEOUT` | Compilation exceeded its time limit |
| `ENVIRONMENT_ERROR` | Missing/changed runtime, lock failure, or compiler startup failure |
| `ASSUMPTIONS_ERROR` | Compilation did not establish the expected axiom-free result |
| `INPUT_ERROR` / `LIMIT_EXCEEDED` | Invalid API input or a configured size/complexity bound |

CLI exit status is 0 for PASS, 1 for a verification/input failure, and 2 for malformed CLI arguments. Result JSON includes the stage, compiler version when reached, elapsed time, generated-source hash, stdout/stderr, and verification flags. A policy failure is rejected before the compiler runs. A syntax test can therefore fail at the restricted parser rather than Rocq itself.

The supported statement form is `forall n m ... : nat, expression = expression`, optionally with one equality premise using `->`. There must be one to three quantified variables. Expressions use those variables, `0`–`3`, `S`, parentheses, and `+`. No other constants, functions, or arithmetic operators are accepted. Parenthesis/successor nesting is limited to five, input strings to 16,384 characters each, and proof bodies to 40 tactic commands.

Allowed proof forms are explicit `intros` names; at most one `induction n as [| k IH]` followed by two `-` branches; argument-free `simpl`, `reflexivity`, `symmetry`, `f_equal`; and `rewrite` (optionally `->` or `<-`), `apply`, or `exact` with one local reference. All binders must be introduced before induction. Nested comments are supported and ignored. Introduced names are re-rendered as fresh reserved names, with separate branch scopes, to prevent a missing local from resolving to a global library helper.

**There are no permitted global helpers in this increment.** Thus even `Nat.add_0_r` is rejected, despite being loaded by the fixed prelude. Arbitrary proof terms, tactic combinators, custom Ltac, additional imports, and candidate-supplied `Proof`/`Qed` are rejected. The allowlist is enforced in code; the example's metadata cannot enable more tactics or lemmas.

Remaining limitations: the grammar is intentionally narrower than Rocq; valid proofs outside it are rejected. The environment lock currently supports only the tested macOS/Homebrew architecture and prefix. Runtime checks fingerprint selected files, not every transitive library or OS component; the installed toolchain and lock file are trusted. This is a subprocess checker, not an OS/container security sandbox, and it has no separate memory limit. Some error categories depend on Rocq 9.2 diagnostic text. Verification establishes the supplied formal theorem, not fidelity to an informal statement or proof. These limitations are separate from the successful kernel checks.

`RESEARCH_PLAN.md` remains the original broader proposal, with its status updated to distinguish this implemented increment from deferred work.
