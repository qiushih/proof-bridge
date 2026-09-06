# Proof Bridge: initial research and pilot plan

Prepared 2026-09-06. Status: research and design; no model has been trained and no Rocq runtime has been installed. Training hardware is undecided.

The proposed first experiment is a small pretrained language model that translates typed English proofs about natural-number equalities into short Rocq scripts. Start with addition and one structural induction; add multiplication once that pipeline works. Evaluate preservation of the user's argument as well as formal correctness. This is a bounded research hypothesis, not an established accuracy claim.

The central research question is: **Can a 1.5–4B parameter model translate previously unseen informal proofs within a fixed arithmetic fragment into kernel-checked Rocq proofs, while preserving the intended induction and rewrite steps?**

Rocq checks a formal proof against a formal statement. It does not establish that either faithfully represents the English input. Its `Qed` command checks the completed proof term; `Admitted` instead declares the goal as an axiom. These distinctions determine the acceptance criteria. [Rocq proof-mode documentation](https://rocq-prover.org/doc/v9.0/refman/proofs/writing-proofs/proof-mode.html)

**Working assumptions.** Inputs are typed text, possibly containing simple LaTeX. Handwritten images and OCR are a later, separately evaluated component. Variables range over `nat`, including zero. The first proof task receives an independently specified formal statement alongside the English proof; a second experiment adds statement translation, yielding the requested text-to-code interface. Keeping these experiments separate makes failures diagnosable without dropping the eventual end-to-end objective.

For the sample input, interpret the duplicated `nn` and equation fragments as copying artifacts, giving: “For every natural number n, prove n + 0 = n. Induct on n; simplify the base case; simplify the successor case and use the induction hypothesis.” Record this normalization. A future interface should flag genuinely ambiguous notation instead of guessing a different theorem.

```coq
From Stdlib Require Import Arith.PeanoNat.

Theorem add_zero_right : forall n : nat, n + 0 = n.
Proof.
  intros n.
  induction n as [| k IH].
  - simpl.
    reflexivity.
  - simpl.
    rewrite IH.
    reflexivity.
Qed.

Print Assumptions add_zero_right.
```

This script is also saved in `examples/add_zero_right.v`. It was reviewed against documented induction patterns but has **not been compiled locally**: neither `rocq` nor `coqc` was found on PATH during this research pass. Software Foundations presents this same mathematical example and induction pattern. [Software Foundations: Induction](https://softwarefoundations.cis.upenn.edu/lf-current/Induction.html)

| Informal step | Formal role |
| --- | --- |
| “For every natural number n” | `forall n : nat` and `intros n` |
| “Induct on n” | `induction n as [\| k IH]` |
| “Base case follows by simplification” | Reduce `0 + 0 = 0`, then close by reflexivity |
| “Assume k + 0 = k” | The induction tactic supplies `IH : k + 0 = k` |
| “Simplify and use the hypothesis” | Reduce the successor goal to `S (k + 0) = S k`, rewrite with `IH`, then close |

“Apply the induction hypothesis” in ordinary mathematics can translate to `rewrite IH` in Rocq. The mapping depends on the goal, not just the English verb. In particular, `apply IH` alone is not the corresponding closing step for `S (k + 0) = S k`.

**Freeze a small language before collecting data.** These are proposed pilot limits and can be revised after the first error analysis.

| Component | First supported fragment |
| --- | --- |
| Objects | Standard unary natural numbers `nat`, with `0` and successor `S` |
| Expressions | Variables, small numerals, successor, addition; multiplication in the next curriculum tier |
| Statements | Universally quantified equalities; up to three variables |
| Hypotheses | Induction hypothesis; optionally one explicit equality premise in a separately labeled rewrite tier |
| Induction | At most one induction on a natural-number variable; explicit base and successor branches |
| Proof size | Initially at most 40 tactic commands and expression depth 5 |
| Tactics | Restricted forms of `intros`, `induction`, `simpl`, `reflexivity`, `rewrite`, `exact`, `apply`, `symmetry`, `f_equal` |
| Lemmas | A small project-owned catalog with exact types and an explicit permitted subset per example |
| Deferred | Nested or strong induction, arbitrary predicates/functions, inequalities, subtraction, division, lists, integers, real numbers, existential reasoning, custom Ltac |

An illustrative grammar is:

```text
term       ::= variable | 0 | 1 | 2 | 3 | S(term)
             | term + term | term * term
equality   ::= term = term
statement  ::= forall variables : nat, equality
             | forall variables : nat, equality -> equality
```

The grammar restricts expressible inputs; acceptance also depends on whether a proof lies within the bounded tactic and induction fragment. Unsupported does not mean mathematically false.

Use the standard definitions rather than maintaining a competing arithmetic implementation. Addition and multiplication recurse on their first natural-number argument; this affects which simplifications and induction choices work. Keep those definitions and simplification behavior fixed. [Rocq natural-number definitions](https://docs.rocq-prover.org/v8.20/stdlib/Coq.Init.Nat.html)

A proposed helper interface follows. The `PB.*` identifiers are **new project names to implement**, not claims about existing standard-library names. All variables below are universally quantified natural numbers.

| Proposed helper | Exact mathematical type |
| --- | --- |
| `PB.add_zero_r` | `n + 0 = n` |
| `PB.add_succ_r` | `n + S m = S (n + m)` |
| `PB.add_assoc` | `(a + b) + c = a + (b + c)` |
| `PB.add_comm` | `a + b = b + a` |
| `PB.mul_zero_r` | `n * 0 = 0` |
| `PB.mul_one_r` | `n * 1 = n` |
| `PB.mul_succ_r` | `n * S m = n + n * m` |
| `PB.mul_add` | `a * (b + c) = a * b + a * c` |
| `PB.add_mul` | `(a + b) * c = a * c + b * c` |
| `PB.mul_comm` | `a * b = b * a` |
| `PB.mul_assoc` | `(a * b) * c = a * (b * c)` |

Prove helpers in a documented dependency order and freeze the catalog. Each task carries its available subset. While proving `n + 0 = n` by induction, do not permit `PB.add_zero_r`, `Nat.add_0_r`, or a renamed copy of that result as a shortcut. Later tasks can legitimately use previously available helpers. Manually review the dependency graph for indirect shortcuts as well as direct copies.

Exclude `auto`, `eauto`, `lia`, `nia`, `ring`, `hammer`, and arbitrary tactic composition from the translation experiment. These can solve goals while revealing little about translation of the supplied argument. A separate automation baseline is useful, but report its results separately. Restrict arguments to `apply`, `exact`, and `rewrite` too: allowing an arbitrary Rocq term would undo the intended small language.

**Use two model outputs as a controlled comparison.** Start with direct generation because it is the quickest baseline. Compare it with a structured proof plan if syntax and branch errors are significant.

```text
Typed statement + informal proof
             |
   normalize and parse statement
             |
   fix the formal target and environment
             |
 small model -> proof body OR structured proof plan
             |
  validate permitted actions; render fixed wrapper
             |
   Rocq check -> success, error, or timeout
             |
 optional bounded repair using actual error / goal
```

The structured representation can contain `intro`, `induct`, `simplify`, `rewrite`, and `close_reflexivity` nodes, with explicit base and successor children. Each significant node can refer to a span of the input proof. A deterministic renderer owns identifiers, bullets, imports, theorem declarations, and `Qed`. This makes malformed output easier to reject and alignment easier to inspect. It remains a proposed design advantage to test against the direct baseline.

For whole-script verification, start with a subprocess invoking `rocq compile` under a fixed environment and resource limit. Rocq documents batch compilation as a standard interface. [Rocq command reference](https://rocq-prover.org/doc/V9.2.0/refman/index.html)

Add Pétanque/Pytanque when current goals, tactic-level traces, and repair justify the integration. Pytanque supports tactic execution, feedback, and state management over the Pétanque protocol. [Pytanque repository](https://github.com/LLM4Rocq/pytanque)

Pin the compiler, core/standard libraries, Python dependencies, and any LSP/Pétanque backend as one tested environment. Rocq 9.2.0 was marked as the latest stable release on the releases page during this research; that is a setup candidate, not a tested compatibility matrix with every Python client. Store exact versions and a prelude hash with every example and result. [Rocq releases](https://github.com/rocq-prover/rocq/releases)

**Build the data from checked proofs.** Begin with 30–50 manually authored and compiled examples covering reflexivity, equality rewriting, addition induction, and the first multiplication identities. This first set validates the environment and data schema; it is too small to support a broad generalization claim.

Then construct roughly 200–500 distinct formal theorem/proof instances through controlled expression substitution and composition of approved proof patterns. Compile each instance, record its environment, and only then generate English explanations. Aim initially for 3–5 paraphrases per accepted instance, giving approximately 600–2,500 aligned pairs. These are planning quantities; adjust them according to diversity and observed learning curves.

The augmentation unit should include a proof trace, not just a true equation. Paraphrases must preserve the induction variable, hypothesis, intermediate equalities, and helper usage. Have a human check alignment for all manually curated seeds and a stratified sample from every synthetic generation batch. Automatically flag references to absent variables, incorrect induction hypotheses, and unsupported lemmas. Quarantine failed checks.

Reverse translation from formal mathematics to natural language is supported as a data-generation strategy by MMA, but its reported autoformalization experiments focus on statements. Extending the idea to aligned proof steps here is our proposed adaptation. [Multilingual Mathematical Autoformalization](https://arxiv.org/abs/2311.03755)

Collect 50–100 independently written human inputs for a separate evaluation set. Include concise arguments, verbose arguments, renamed variables, LaTeX notation, formatting noise, and a separate set of incomplete or invalid arguments. Label a false claim, an unsupported task, an ambiguous statement, a proof gap, and a generation failure differently. A failed proof search is not evidence that the theorem is false. Synthetic English alone will overstate readiness for user writing.

Every record should contain the raw input, normalized input, gold statement, environment identifier, permitted helpers/tactics, formal proof body, proof-method annotation, source/provenance, verification status, and split-group identifier. Store actual proof states later when an interactive backend is in place. The example JSON supplied with this plan is an illustrative schema record and deliberately marks verification as pending.

Keep metadata and targets out of the input prompt unless they will genuinely be available at inference time. The theorem-only baseline gets the same formal target and environment but no informal proof. Gold alignment annotations belong in labels/evaluation, not in its prompt.

Split by theorem and generation family **before** paraphrasing. Keep alpha-renamed statements, their proof variants, and all derived paraphrases together. Reserve additional unseen compositions and proof skeletons for a harder test. A random split over paraphrases mainly measures recognition of already-seen problems. Common textbook facts are also likely present in pretrained models; test novel compositions and avoid claiming that a held-out textbook statement proves absence of pretraining contamination.

**Use existing research selectively.**

| Resource | What it contributes | Limit for this project |
| --- | --- | --- |
| [Software Foundations, Logical Foundations](https://softwarefoundations.cis.upenn.edu/lf-current/Induction.html) | A close match for induction exercises and explicit tactic style | Educational source material; exercises and prose are not automatically a clean paired corpus |
| [CoqGym / Learning to Prove Theorems](https://arxiv.org/abs/1905.09381) | Formal Coq proofs and an interaction-oriented learning environment | Does not directly supply the required aligned informal proof paragraphs; broad historical environments add setup work |
| [ProofNet](https://arxiv.org/abs/2302.12433) | 371 examples with informal statements/proofs and formal Lean 3 statements | Not a ready-made paired Rocq proof-script training set; its mathematics is much broader |
| [miniF2F](https://github.com/openai/miniF2F) and [its Rocq port](https://github.com/LLM4Rocq/miniF2F-rocq) | Useful later formal-theorem benchmarks; a Rocq version exists | Not tailored to this induction fragment or to measuring fidelity to supplied prose |
| [Draft, Sketch, and Prove](https://arxiv.org/abs/2210.12283) | Demonstrates using informal proofs to guide formal proof sketches and automated completion | Its setup/results do not establish performance for a small Rocq translator |
| [NLIR](https://guillaume.baudart.eu/papers/mathai_neurips24.pdf) | Studies natural-language intermediate steps and interactive Coq proving | Relevant architecture evidence, not a guarantee for our supervised translation setting |
| [CoqPilot](https://arxiv.org/abs/2410.19605) and [PALM](https://arxiv.org/abs/2409.14274) | Candidate checking and generation/repair approaches in Coq | Formal proof completion differs from translating a particular user's argument |

**Fine-tune after measuring the baseline.** Use pretrained weights and supervised fine-tuning. An initial comparison can use the published [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) and [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) checkpoints. They offer two practical model sizes for this experiment; the model cards do not establish their Rocq performance, and this pair is not a controlled scaling study because the model families differ.

Start with LoRA adapters; consider QLoRA on compatible hardware when memory is limiting. QLoRA trains adapters through a frozen quantized model. Framework and accelerator support must be checked after hardware is selected. [QLoRA paper](https://arxiv.org/abs/2305.14314), [PEFT quantization documentation](https://huggingface.co/docs/peft/developer_guides/quantization)

A proposed starting configuration is a 2,048-token sequence limit, completion-only training loss, and a small validation-selected sweep over LoRA ranks 8/16 and learning rates 5e-5/1e-4. Run one batch to measure actual peak memory and throughput before scheduling full training. Increase sequence length only when inspected examples need it. Do not assume quantized inference memory equals training memory. Keep model revision, tokenizer, chat template, seeds, decoding settings, and checkpoint selection reproducible.

Train direct proof output first. Then compare structured-plan output with the same examples and verification budget. If logs show recurring local mistakes, add a separately labeled repair dataset: failed attempt + actual goal/error -> corrected proof. Use training/validation problems for generating repairs; keep test feedback out of training. Reinforcement learning can wait until supervised learning and error repair have a reliable baseline.

**Measure three different properties.**

| Property | Measurement |
| --- | --- |
| Statement fidelity | Does the parsed formal statement match the intended variables, quantifiers, assumptions, and equation? Use gold structured statements and human review where needed. |
| Formal validity | Does full compilation check a completed proof of that exact frozen target, within budget, using permitted actions and without added axioms? |
| Argument fidelity | Does the script use the intended induction variable, branches, hypothesis, and important intermediate equalities/lemmas? Use annotations and a human-reviewed sample. |

For compilation acceptance, generate the wrapper outside the model, parse the restricted body or plan, allow only registered references, and reject arbitrary imports, declarations, theorem changes, `admit`/`Admitted`, `Abort`, or checking bypasses. Require the intended theorem to exist after a full compile with `Qed`; interface-only compilation is insufficient. Inspect its assumptions and require no unapproved global axioms. `Print Assumptions` reports assumptions a theorem depends on. [Rocq environment-query documentation](https://rocq-prover.org/doc/v9.0/refman/proof-engine/vernacular-commands.html)

Restriction checks and compilation serve different purposes: a valid proof can still use a forbidden helper. Likewise, a successfully compiled `.v` file can contain admissions unless the harness excludes them. Run each candidate in a fresh task environment so earlier generated theorems cannot leak into later checks.

Argument fidelity is necessarily an operational evaluation, not a kernel theorem about English. Accept equivalent formal implementations, such as rewriting under `S` or using congruence followed by the induction hypothesis. A check that only searches for the word `induction` is insufficient: inspect the actual trace and align key steps. Record inserted gap-filling steps and do not silently replace a flawed argument with an unrelated proof.

Report first-attempt verification rate, success with a fixed maximum of five independent candidates, and success with one initial attempt plus at most two repairs as separate metrics. For every comparison keep the generation/token/checking budget explicit. Include valid-and-faithful success, latency, model memory, error categories, and coverage/abstention. Exact code-string match is only a diagnostic; multiple scripts can express the same argument.

Compare a template-based translator, a prompted pretrained model, the fine-tuned direct model, and the structured-plan variant. Also compare each neural approach with and without the informal proof. A theorem-only model that performs equally well would weaken the claim that the system uses the user's reasoning. Use paired prompts with distinct valid proof methods for the same target, plus deliberately corrupted proofs, to test this more directly.

**Suggested implementation sequence.** These are milestones rather than calendar commitments; hardware and available development time remain unknown.

| Milestone | Deliverable | Exit condition |
| --- | --- | --- |
| 1. Formal contract | Fixed arithmetic grammar, versioned prelude, helper dependency map, checking harness | 30–50 seed proofs compile; malformed, admitted, target-changing, and forbidden-helper candidates are rejected |
| 2. Pilot benchmark | Aligned records, family-based splits, independent human evaluation inputs | Seeds are reviewed; all included positive targets are checked; provenance and leakage checks are recorded |
| 3. Baselines | Template and prompted-model results, theorem-only ablation | Every failure receives a useful category; budgets and metrics are reproducible |
| 4. First fine-tune | One small-model LoRA run and learning curves | Evaluate against the same baselines on locked splits, including argument fidelity |
| 5. Targeted extension | Structured output or bounded repair, chosen from observed failures | Improvement is measured at a declared budget; multiplication expands only after addition performance is understood |

The next concrete task is Milestone 1: implement and test the formal environment and verifier, then author the first checked proof pairs. Training, larger corpora, and hardware spending should follow that small working benchmark.
