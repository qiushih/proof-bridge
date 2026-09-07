# Constrained-v1 checkpoint

`constrained-v1.json` freezes the existing decoder, preflight, model/runtime
pins, frozen Prompt v2 inputs, development evidence, manual fidelity reviews,
results, reports, and regression tests. No decoder or previous result is edited
to create this checkpoint. The annotated Git tag is
`proofbridge-constrained-v1`.

Check it without inference:

```sh
python3 -m checkpoints.freeze_constrained_v1 check
.venv/bin/python -m baseline.setup_model --check
```

The original freeze command, which refuses an existing checkpoint, is:

```sh
.venv/bin/python -m checkpoints.freeze_constrained_v1 freeze
```

Reproduction of the 60-attempt experiment remains documented in the unchanged
`constrained_v1/README.md`. The checkpoint stores source and artifact hashes;
the model weights remain external in the pinned local model cache. Model
identity and runtime settings are retained, not retrained or modified.

The subsequent argument-switching probe uses separate files under
`argument_switch_v1/` and `results/argument-switch-v1/`. Changes to those probe
files do not alter this checkpoint. The original 12-example diagnostic is not
part of probe selection, generation, or scoring.
