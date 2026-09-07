# ProofBridge training protocol v1

This protocol is fixed **before the nine pre-training dev generations and before any weight update**. The current task prepares data serialization, adapter/loss definitions and the baseline; it does not start training. `config.json` is the machine-readable configuration. `frozen.json` binds this document, configuration, implementation, preflight tests and preparation evidence by SHA-256. Any later configuration change requires a new protocol version.

## Base model and actual hardware

- Base: **Qwen/Qwen2.5-Coder-0.5B-Instruct**, revision **`ea3f2471cf1b1f0db85067f1ef93848e38e88c25`**. Load only the local files validated by `baseline/model.lock.json`. Weight file SHA-256: `f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44`.
- Actual host: **MacBook Air, Mac14,2, Apple M2, 8 CPU cores, 8 GiB unified memory**, arm64, macOS **15.7.3 (24G419)**. The read-only hardware evidence is recorded in `hardware.json`.
- Training device: **CPU**, float32, eager attention, 4 Torch compute threads and 1 interop thread. Deterministic algorithms enabled. No quantization, mixed precision, GPU/MPS or remote hardware. MPS was unavailable in this task environment; CPU is an explicit protocol choice.
- Runtime: Python **3.13.7**, Torch **2.9.0**, Transformers **4.57.1**, Safetensors **0.8.0**, with all transitive versions pinned in the unchanged `baseline/requirements.lock`. No PEFT, TRL or Accelerate dependency is introduced. The ordinary LoRA layer is a short repository implementation, frozen and tested on synthetic tensors.

The base model is approximately 0.5B parameters. Rank-limited adapters, activation checkpointing and selecting only supervised output-logit positions reduce future training memory. **A full Qwen training forward/backward and peak-memory/runtime measurement have not been performed.** Resource exhaustion must stop the future run; do not silently change hardware, sequence length, dtype, rank or batch size to continue. This is a reproducible planned configuration, not a claim of benchmarked training throughput.

## Data, prompt and completion-only loss

Use exactly the **24 rows in `data/pilot-v1/train.jsonl`**, in the frozen split manifest. Every row is one equally weighted training example, including A/B variants. No historical examples are appended, no paraphrases are generated, and no theorem-only training examples are added. Only the six rows in `data/pilot-v1/dev.jsonl` may select checkpoints.

Build every training input with the **unchanged selected Prompt v2** (`short_3shot`, demonstrations `seed_002`, `seed_010`, `seed_016`) using the existing `prompt_v2.protocol.build_messages`. The target user turn contains the fixed theorem and that row's informal proof. IDs, features, reference steps, verification diagnostics and family labels are excluded from input. The frozen three demonstration completions stay in context but receive no training loss.

Serialization is implemented in `training_protocol_v1.data.encode_example`:

1. Apply the pinned tokenizer's chat template to the frozen messages with `add_generation_prompt=True`.
2. Encode that rendered prefix with `add_special_tokens=False`, exactly as baseline inference does.
3. Encode `proof_body` separately without special tokens; append **one tokenizer EOS**. Concatenate prefix IDs and completion IDs. No extra closing role text, BOS, explanation or Markdown is added.
4. Assign `-100` to every prefix label, including system text, demonstrations, theorem, informal proof and assistant role header. Supervise **only the proof-body tokens and the final EOS**. Padding, if ever needed, uses attention 0 and label `-100`; batch size 1 needs no padding.

Maximum total sequence length is **768 tokens**, with no packing and no truncation. An oversized row is an error. Tokenization preflight checks all 24 train and 6 dev rows. Observed lengths before freezing: train **497–617**, dev **575–624** tokens, including the target EOS.

Use causal cross-entropy averaged over each example's supervised completion tokens. For prefix length P and total length L, logits at positions **P−1 through L−2** predict target IDs **P through L−1**. The pinned Qwen2 forward supports `logits_to_keep`; `completion_loss` selects exactly those positions before the vocabulary projection. It does not pass labels to the model, avoiding a second causal shift. The labels with `-100` remain the canonical auditable mask. Synthetic tests compare this loss with full-sequence shifted/masked cross-entropy and check that changing prompt labels is rejected.

## LoRA and SFT update configuration

Ordinary, unmerged LoRA updates only **`self_attn.q_proj` and `self_attn.v_proj` in all 24 decoder layers**: 48 wrapped linear modules, **540,672 trainable parameters**. All original weights, biases, embedding and language-model-head weights remain frozen. The update is `base(x) + (alpha/r) * B(A(x))`.

| Setting | Frozen value |
| --- | --- |
| Rank / alpha / scaling | 8 / 16 / 2 |
| Adapter dropout | 0.0 |
| Initialization | A: Kaiming uniform with a = sqrt(5); B: zero |
| Adapter dtype | float32 |
| Activation checkpointing | Enabled; `use_reentrant=False` |
| Training KV cache | Disabled |
| Optimizer | `torch.optim.AdamW`, adapter parameters only |
| Learning rate | 0.0002 |
| Betas / epsilon | (0.9, 0.999) / 1e-8 |
| Weight decay | 0.0 |
| Optimizer implementation | `foreach=False`, `fused=False` |
| Scheduler / warmup | Constant / 0 steps |
| Gradient clipping | Global adapter gradient norm 1.0 |
| Microbatch / accumulation | 1 example / 4 microbatches |
| Effective batch | 4 examples |
| Data-loader workers / drop-last | 0 / false |
| Training budget | 10 complete epochs; 240 microbatches; **60 optimizer steps** |

For each microbatch, divide its mean completion loss by 4 before backpropagation. After four microbatches, clip adapter gradients, perform one AdamW step, and clear gradients with `set_to_none=True`. There are exactly six updates per 24-row epoch, so there is no partial accumulation at the epoch boundary. Use training mode during updates and evaluation mode during checkpoint scoring. Never merge an adapter into the base checkpoint.

Python, NumPy and Torch seeds are all **1729**; adapter initialization uses the Torch seed. Epoch e (zero-based, 0–9) shuffles sorted training IDs with an independent `random.Random(1729 + e)`. Shuffle orders and their hashes are fixed during preparation. No seed sweep, augmentation, automatic retry or early stopping is allowed. Do not use dev outcomes to change the 60-step budget or any hyperparameter.

The current preparation includes a tested adapter/loss definition but **no optimizer loop or training command**. A later authorized training implementation must follow this exact specification; implementing it does not permit tuning the specification from baseline outcomes.

## Checkpoints and selection using six dev rows only

Keep step **0** (the unchanged base model) and save adapter checkpoints after optimizer steps **6, 12, 18, 24, 30, 36, 42, 48, 54, 60**. Save adapter A/B tensors, optimizer and scheduler states, Python/NumPy/Torch RNG states, step/epoch, shuffle order and protocol/data hashes. Do not write or replace base-model weights. Interrupted training must be reported; no unrecorded replay or altered run may replace it.

At each trained checkpoint, generate exactly six dev completions: one for each of `pd01_A`, `pd01_B`, `pd02_A`, `pd02_B`, `pd03_A`, `pd03_B`, with theorem plus that row's informal argument. The six argument-conditioned outputs from this pre-training baseline serve as step zero; do not regenerate them for selection. Use the unchanged Prompt v2, frozen constrained-v1, current pinned generation settings, 256 tokens, one attempt and no repair. Theorem-only controls are excluded from selection.

Choose the lexicographic maximum of these counts, using **only those six rows**:

1. Verified and mathematically faithful complete proofs.
2. Verified proofs matching the requested proof-step contract.
3. Rocq-verified proofs.
4. Negative optimizer step: choose the **earliest** checkpoint on ties, including step zero.

`selection_score` implements this ordering. There is no training-loss, dev-loss, latency, historical-diagnostic or holdout tie-break. Missing/unresolved fidelity reviews cannot count as faithful. Review every checkpoint under the same rubric, saving hash-bound reviewer judgments; do not rewrite criteria after seeing outputs. If step zero wins, report that the pilot did not improve the selection score.

## Three separate outcome measurements

**Rocq verification:** the unchanged `verifier.verify`, with a 10-second timeout and the fixed theorem. Retain PASS/FAIL, category, source hash and kernel/assumption evidence for every output.

**Mathematical argument fidelity:** a separate reviewed judgment of whether a verified proof follows induction on n, proves the base equality by computation and transports the IH through the successor case. Consistent symmetry, definitional conversion, `rewrite IH; reflexivity` and `f_equal; exact/apply IH` may implement the same mathematical argument. Equivalent tactics must not automatically become unfaithful. A genuinely different induction variable/strategy requires a separate unfaithful judgment and reason. Failed/incomplete proofs are `NOT_ESTABLISHED`; record their attempted structure without claiming a faithful complete proof. Theorem-only is `NOT_APPLICABLE`, since no argument was supplied. All current semantic reviews are assistant reviews, not independent human reviews.

**Requested proof-step adherence:** the predeclared ordered branch/tactic trace, normalized for local names, intros grouping, `exact` versus `apply` of the same local, and optional forward `->`. It distinguishes A's forward IH rewrite from B's congruence-plus-IH realization and preserves simplification order. Record both attempted trace matches and **verified** trace matches. Extra/missing/different commands can mismatch this strict step contract while the mathematical argument remains faithful. This metric is not a semantic proof-equivalence test.

The three dev theorems share one family and A/B pairs mainly contrast equivalent IH realizations. Their six rows are correlated. Neither trace matching nor a nine-output baseline establishes broad argument dependence or a high-level strategy switch.

## Pre-training baseline: exactly nine generations

The fixed order is `pd01`, `pd02`, `pd03`; for each theorem: **theorem only, argument A, argument B**. Each condition has one attempt, no repair, greedy decoding, one beam and a **256-token** maximum including EOS. The original `constrained_v1.experiment.generate` and `prompt_v2.experiment.Inference` are called without modification. CPU/float32, eager attention, seed 1729, four compute threads, one interop thread, deterministic algorithms and inference KV cache remain as in `baseline/config.json`.

Freeze this protocol and preflight evidence before the first attempt. Journal starts/finishes and preserve raw text, token IDs, prompts, settings, latency, masks and verifier outcomes. Stop if a run is incomplete; refuse resume, overwrite, repair or additional attempts. Assess the nine saved outputs separately and freeze their raw/assessed artifacts. Reproduction after completion uses token replay, verifier replay and hash checks, **not fresh generations**.

## Sealed holdout and unchanged artifacts

The six-row pilot holdout remains sealed. Do not parse, display, export, generate on or score it in this task or during training/checkpoint selection. Integrity checks may read bytes only to compare existing hashes. The reused diagnostic set and historical development/probe outputs do not choose this protocol, configuration or checkpoint. Frozen Prompt v2, constrained-v1, argument-switch-v1, all pilot data and the model files are preserved by hashes.

Only after training and dev checkpoint selection are complete may a separately authorized holdout evaluation open it once under the frozen pilot split policy. This task does not authorize that opening. Full historical regression tests may replay older diagnostic artifacts; that is distinct from pilot selection and does not inspect the sealed holdout contents.

## Exact preparation and reproduction commands

Run from the repository root. The existing `.venv` and Rocq environment are pinned; their standard setup/check commands are:

```sh
python3 scripts/setup_rocq.py --check
.venv/bin/python -m baseline.setup_model --check
.venv/bin/python -m unittest discover -s tests -p 'test_training_protocol_v1.py' -v
.venv/bin/python -m training_protocol_v1.experiment freeze-protocol
.venv/bin/python -m training_protocol_v1.experiment run
.venv/bin/python -m training_protocol_v1.report
.venv/bin/python -m training_protocol_v1.experiment check --replay-tokens --reverify
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m training_protocol_v1.experiment freeze-results
```

`freeze-protocol`, `run` and `freeze-results` are one-time commands that refuse existing outputs. The report command checks the saved hash-bound review file and never generates samples. After the release exists, reproduce with:

```sh
.venv/bin/python -m training_protocol_v1.experiment check --replay-tokens --reverify
.venv/bin/python -m unittest discover -s tests -p 'test_training_protocol_v1.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

No command above trains a model. Preparation and tests use the tokenizer and synthetic tensors; only the nine baseline attempts load Qwen for inference. No adapters are attached to Qwen and no Qwen forward/backward training pass or optimizer step is performed.
