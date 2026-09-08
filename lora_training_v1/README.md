# LoRA implementation and disposable M2 feasibility

This module implements the frozen training protocol without editing it. `core.py` provides deterministic setup, the actual four-microbatch accumulated update, epoch scheduling, metrics, trainable-parameter audits and adapter-only checkpoint saving. `train_pilot` is a future callable loop with a required checkpoint-evaluation callback; no CLI command starts it. Checkpoint selection continues to use the existing frozen six-dev-row rule.

The only runnable training work here is the separate bounded feasibility worker: **at most three optimizer updates**, no dev generation, no holdout content reads, no saved adapter tensors. Its updates disappear when the worker exits. The full 60-update pilot requires a fresh model/adapter initialization and future authorization; these test updates cannot be resumed or promoted.

The three measurements use the longest frozen training row at its native size, the same row right-padded to a 768-token tensor, and the first four examples from the deterministic epoch-zero ordering. The 24 training rows are loaded through the unchanged loader and encoded with frozen Prompt v2 and its unchanged completion-only mask. No record or informal proof is extended to manufacture a longer example. The padding wrapper only adds attention-masked tensor padding before the model forward; the original target positions remain the sole loss positions.

`plan.json` fixes the update cap, cases and speed-classification threshold before execution. The worker logs forward/backward/optimizer events, finite losses and gradients, adapter changes, and before/after hashes for every original model parameter. It also checks base-model files and all 178 previously tracked files. A two-second monitor records Darwin resident/physical-footprint measurements and system-wide swap. The host already had substantial swap use before this task; report initial use and increases separately.

Run from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_lora_training_v1.py' -v
.venv/bin/python -m lora_training_v1.preflight run
.venv/bin/python -m lora_training_v1.report
.venv/bin/python -m lora_training_v1.preflight check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m lora_training_v1.preflight freeze
```

The one-time `run` needs permission to read macOS system hardware/swap telemetry. It verifies the actual 8-GiB M2 and refuses an existing output directory before loading any model, preventing accidental extra updates. `check`, `report` and the tests reproduce evidence without another real-model update; tensor tests use tiny synthetic layers and temporary checkpoint directories. Tests never load Qwen. Do not delete an existing run to repeat it under the same identity.

The result is saved under `results/m2-feasibility-v1/`, including `M2_FEASIBILITY_REPORT.md`, `assessment.json`, raw metrics and parameter hashes. The classification uses a predeclared threshold of 120 seconds for the median native-length update, corresponding to two hours for 60 updates before evaluation/checkpoint overhead. A non-memory software failure is reported as unresolved rather than mislabeled an out-of-memory result. Kernel RSS/footprint maxima cover the entire worker lifetime, including loading and hash checks; swap is global and cannot be assigned solely to this process.

The frozen protocol document remains historical: its statement that a training loop had not yet been implemented described that earlier checkpoint. This separate module supplies the implementation. The frozen prompt, decoder, dataset, verifier and checkpoint-selection code are unchanged.
