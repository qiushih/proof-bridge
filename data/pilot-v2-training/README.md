# V2 pilot training data preparation

Frozen: 24 control rows, 26 intervention rows, the existing eight dev rows and six
fresh reserved holdout rows. Original files remain unchanged. The intervention
copies all original training bytes and appends ptv201_A/B. Dev is referenced at its
existing frozen path, not copied or edited.

The new training pair proves `forall n m p : nat, m = p -> (n + 0) + m = n + p`.
Its useful premise-base induction moves the right-zero residual inside a larger sum
with a variable suffix. A rewrites IH; B uses f_equal and exact IH. This is one
contextual addition in the same right_zero_variants family, not broad data expansion.

Rows retain formal_statement, informal_statement, informal_proof, proof_body, aligned
steps, argument_features, generation_family, argument_contract, review, provenance,
verification and quality_checks. Original rows keep pilot-1.0; new rows use pilot-2.0
with the same fields and an extended family ledger. Only theorem and informal proof
are prompt inputs. Targets contain only proof-body code plus the tokenizer EOS.
All curation/reviews are assistant-produced; human_reviewed is false.

split_manifest.json and release.json freeze membership, protocol, dependencies and
file hashes. schedule.json fixes 240 microbatches for each arm, including exposure
counts and token totals. coverage.json counts proof patterns. references/ contains
new public .v files. The public verification report exposes aggregate reserved QA
only; reserved/ contains the six fresh references and detailed construction checks.
After sealing, the checker hashes those files without parsing them. There is no
holdout loader, training runner, or generation command in this release.

Family separation follows a documented manual ledger. The holdout contraction
family shares prefix-induction mechanisms and associativity connections with dev;
its three theorem pairs are correlated. Old holdout/diagnostic files were never
opened, so exact hidden-statement overlap is not ruled out. This is a fresh-instance
pilot holdout, not evidence of unseen algebra or independent statistical samples.
See family_review.json and training_protocol_v2/V2_TRAINING_PROTOCOL.md for limits.

From the repository root:

```sh
.venv/bin/python -B -m training_protocol_v2.preparation check
.venv/bin/python -B -m training_protocol_v2.preparation check --reverify-public
.venv/bin/python -B -m unittest discover -s tests -p 'test_training_protocol_v2.py' -v
```

The initial prepare command requires a fresh unsealed draft and refuses existing
output. Do not reconstruct it using the sealed holdout. Checks reproduce evidence
without sampling or training; --reverify-public recompiles only 26 train + 8 dev.
