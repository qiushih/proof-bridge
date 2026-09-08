# M2 LoRA feasibility v1

**Classification: feasible.**

Apple M2 MacBook Air (Mac14,2), 8 GiB RAM; CPU/float32 and the unchanged frozen training configuration. This is a disposable feasibility run, not the 60-update ProofBridge pilot.

Completed **3/3 optimizer updates**, each accumulating four batch-size-one microbatches. The loader used the fixed 24-row training pool. No dev generations or holdout content reads occurred, and no feasibility adapter or optimizer tensors were retained.

| Check | Result |
| --- | --- |
| Pinned model load | PASS |
| Longest actual training row | pt14_B, 617 tokens including target/EOS |
| 768-token tensor, forward, finite loss, backward, optimizer | PASS |
| Trainable weights | 540672 LoRA parameters; query/value only |
| Base parameters unchanged in memory | True |
| Original model files unchanged | True |
| Peak worker RSS | 2.28 GiB |
| Peak charged physical footprint | 2.23 GiB |
| System swap: initial / peak / final | 2.99 GiB / 4.59 GiB / 4.58 GiB |
| Peak system swap increase above initial | 1.60 GiB |

## Per-update timings

| Update | Input | Mean target loss | Forward total | Backward total | Optimizer | Update wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | longest row, native | 0.90960 | 4.59s | 8.47s | 0.021s | 13.17s |
| 2 | longest row, padded to 768 | 0.79957 | 5.58s | 9.73s | 0.005s | 15.34s |
| 3 | first four frozen epoch-zero rows | 0.56863 | 3.85s | 7.14s | 0.008s | 11.01s |

The median of the two native-length update measurements is **12.09s**, projecting approximately **12.1 minutes for 60 updates**, excluding dev evaluation, checkpoint I/O and loading. The predeclared practicality threshold is 120 seconds/update (two hours for 60 updates). This is a small-sample estimate under the observed host load, not a training-time guarantee.

Worker wall time, including loading, integrity hashing and telemetry: **50.12s**. Model-loading time is recorded separately in `worker_result.json`.

## What was exercised

The actual loop implements deterministic seeding, frozen AdamW settings, mean completion-only loss divided by four for accumulation, finite-gradient checks, clipping at norm 1.0, optimizer stepping, constant scheduling and JSONL event metrics. The frozen LoRA helper is reused unchanged. Only 96 adapter tensors (540,672 parameters) are trainable; all base gradients remain absent. Streaming SHA-256 checks compare every deduplicated base parameter before attachment and after updates without cloning the large embedding. Original model-file hashes are also checked before and after.

No training example is naturally 768 tokens. The memory-stress forward right-pads the longest original input tensor to 768 using EOS padding with attention zero. The unchanged completion-only loss still selects only original proof/EOS prediction positions. It does not alter the frozen prompt, proof text, split or dataset. Padding is not supervised, and this does not test a hypothetical 768-token unpadded proof.

The future full epoch loop and adapter-only checkpoint saver are implemented in `lora_training_v1/core.py`. A future authorized caller must supply the frozen six-dev-row checkpoint assessment callback; the existing checkpoint-selection rule remains unchanged. There is intentionally no full-pilot CLI entry here. Checkpoint saving was tested with synthetic layers in temporary directories, but the real feasibility worker never calls it and discards its adapters on exit.

## Measurement limits

RSS comes from Darwin getrusage in bytes. Physical footprint comes from proc_pid_rusage RUSAGE_INFO_V4 and includes charged compressed memory. Both are lifetime maxima for the worker, including loading and checksum passes. Two-second samples also capture current footprint and system-wide swap; lifetime maxima do not depend on catching the exact peak. The host was already swapping before the run, and other applications can change the global swap readings. Do not interpret system swap as per-process swap or extrapolate these losses as model quality.

## Reproduce without extra updates

```sh
.venv/bin/python -m lora_training_v1.preflight check
.venv/bin/python -m lora_training_v1.report
.venv/bin/python -m unittest discover -s tests -p 'test_lora_training_v1.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

The original one-time measurement command was `.venv/bin/python -m lora_training_v1.preflight run`. It requires macOS hardware/swap telemetry permission and refuses an existing output directory, so rechecking cannot accidentally add updates. Plan, source hashes, actual hardware, raw event/memory samples, all losses and parameter hashes are retained. Do not delete the record to repeat this run under the same identity. No full training pilot has started.
