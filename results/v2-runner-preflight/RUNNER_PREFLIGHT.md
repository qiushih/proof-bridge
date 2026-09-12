# V2 runner readiness

PASS: the separate v2 runner is implemented and tested. **Qwen training has not
started.** This is runner-readiness evidence, not a trained-checkpoint result.

- 54 tests passed, zero failures or skips: 26 new runner tests plus 28 existing
  dev-release, baseline and v2 selection/schedule regressions.
- Both exact frozen 240-microbatch streams completed on tiny synthetic tensors:
  120 synthetic optimizer updates total, with 20 temporary adapter-only checkpoint
  roundtrips. These fixtures are not Qwen, contain no learned ProofBridge checkpoint,
  and were removed with their temporary directories.
- Eight predetermined dev reference fixtures reached the unchanged Rocq verifier
  successfully through the callback. Generation was mocked; these were not model
  samples. Tests also cover interruption, environment failure, RNG restoration,
  retry rejection, incomplete selection, mismatched initial adapters and review hashes.
- All 34 unique public serializations were checked for target-only loss masks,
  one EOS, sequence length at most 768 and target length at most 256. The two frozen
  schedules and step-zero prompt/token identities match their released evidence.
- The current pinned Apple M2/Mac14,2 8-GiB CPU/float32 runtime and complete dependency
  lock passed checks. A fixed public reference passed a fresh Rocq 9.2.0 smoke check.
- Qwen loading was explicitly blocked in the recorded test suite. Qwen optimizer
  updates and model generations are zero. The sealed holdout was not parsed;
  old holdout and diagnostic content were not opened.

The runner uses fresh adapters for each future arm, checks their initial hashes
before intervention updates, saves every six updates, and evaluates eight dev rows
per checkpoint under the unchanged prompt/decoder/verifier. It requires all 160
hash-bound reviews before selecting either arm. Verified proofs with unresolved
fidelity count as zero for that metric; a different equivalent tactic is not
automatically mathematically unfaithful. All scoring and success gates remain the
frozen v2 definitions.

The one-time prepare/test/freeze workflow was completed before any Qwen update.
preflight.json records encodings, hardware, model/input/code hashes and the Rocq
smoke check; validation.json and test_output.txt record the tests.
pilot_training_v2/release.json seals these artifacts and the runner source.

From the repository root, reproduce readiness without training:

```sh
.venv/bin/python -B -m pilot_training_v2.experiment check-ready
.venv/bin/python -B -m unittest discover -s tests -p 'test_pilot_training_v2.py' -v
```

A later explicit `python -B -m pilot_training_v2.experiment run` performs both real
arms: 60 updates each and 160 total dev generations. It has not been run here.
Real-model integration, learning improvement and throughput remain unmeasured for
this v2 runner. The holdout stays sealed until both runs and checkpoint selections
are complete and a final evaluation is separately authorized. See the runner README
for the exact future commands and review schema.
