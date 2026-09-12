# ProofBridge v2 controlled training comparison

Predeclared development success: **False**. Holdout remains sealed.

| Subset | Model | Verified | Verified + faithful | Verified + faithful + requested steps | Both-argument successes |
| --- | --- | ---: | ---: | ---: | ---: |
| original_six | base | 6/6 | 6/6 | 3/6 | 0/3 |
| original_six | control | 6/6 | 6/6 | 6/6 | 3/3 |
| original_six | intervention | 6/6 | 6/6 | 6/6 | 3/3 |
| new_two | base | 0/2 | 0/2 | 0/2 | 0/1 |
| new_two | control | 2/2 | 2/2 | 2/2 | 1/1 |
| new_two | intervention | 2/2 | 2/2 | 2/2 | 1/1 |
| all_eight | base | 6/8 | 6/8 | 3/8 | 0/4 |
| all_eight | control | 8/8 | 8/8 | 8/8 | 4/4 |
| all_eight | intervention | 8/8 | 8/8 | 8/8 | 4/4 |

All ten checkpoints and the shared step-zero baseline enter each selection. Ties choose the earliest step.

Selected control: step 36, score [8, 8, 8, -36].
Selected intervention: step 36, score [8, 8, 8, -36].

## Success gates

- strictly_more_verified_faithful_requested_steps: False
- verification_not_lower_than_control: True
- mathematical_fidelity_not_lower_than_control: True
- original_six_preserve_verification_and_fidelity: True
- both_new_arguments_verified_faithful_and_step_adherent: True

Full checkpoint scores, separate step-match counts and percentages are in summary.json. Raw prompts, token usage, latency, verifier diagnostics and decoder traces remain in each dev-step directory. Training logs record loss, update time and memory; timing is descriptive.

## Limits

- One seed; two additional rows in one existing training family.
- Selection and success assessed on the same eight correlated dev rows.
- Equivalent tactics may preserve mathematical fidelity while failing requested-step adherence.
- Update-matched arms have different row exposures and token totals.
- No independent holdout claim or causal informal-proof dependence claim from these dev results.

## Reproduction

```sh
.venv/bin/python -B -m pilot_training_v2.experiment check --replay-tokens --reverify
.venv/bin/python -B -m pilot_training_v2.experiment report
```

These commands replay saved evidence and reviews; they never train or generate new samples.
