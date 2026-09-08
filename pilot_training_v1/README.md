# First full LoRA pilot

This module calls the unchanged `lora_training_v1.core.train_pilot` with a dev evaluation callback. It starts from the pinned base and fresh seeded rank-8 q/v adapters. Exactly 24 training rows, 10 epochs, 240 microbatches and 60 optimizer updates follow `training_protocol_v1/config.json`. The callback receives checkpoints at steps 6 through 60 in increments of 6 and calls the original constrained decoder six times per checkpoint. Inference uses the exact baseline generation configuration and restores the training RNG stream afterward. No adapters are merged; no feasibility adapter is loaded.

All six argument-conditioned step-zero outputs come from the frozen pre-training baseline. Every new output requires a hash-bound assistant review before selection; mathematical equivalence and strict requested-step adherence remain separate. Selection uses the unchanged `training_protocol_v1.scoring.selection_score` across step zero and all ten trained checkpoints. Missing reviews prevent selection. No training outcome changes the fixed budget. Holdout contents and historical diagnostic examples are excluded from this workflow; frozen artifacts are byte-hashed only for integrity.

From the repository root with the pinned `.venv` and Rocq environment:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_pilot_training_v1.py' -v
.venv/bin/python -m pilot_training_v1.experiment run
# Review each dev-step-*/raw_generations.jsonl against the six frozen dev arguments;
# save manual_reviews.json with the raw-file SHA-256 and each proof-body SHA-256.
.venv/bin/python -m pilot_training_v1.experiment check --replay-tokens --reverify
.venv/bin/python -m pilot_training_v1.report
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pilot_training_v1.experiment freeze
```

The one-time training command refuses existing output, resume or retries. An interruption is recorded and cannot be silently replaced. `check`, `report` and tests never load Qwen for inference or training. Tests use synthetic tensors and stubbed generation; verifier replay compiles saved candidates. Checkpoint files, optimizer/RNG state, token traces, reviews, selection and reports remain under `results/lora-pilot-v1`. The freeze requires a PASS validation record with all 60 token and verifier replays. The holdout remains sealed after this task.
