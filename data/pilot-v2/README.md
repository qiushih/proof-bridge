# ProofBridge v2 development extension

This frozen development-only release has **8 rows / 4 theorems / 1 existing family**.
It preserves the six original dev rows byte-for-byte and adds `pd04_A` and `pd04_B`.
The 24 training rows remain in their original file; no training examples are added.

The new statement is:

```coq
forall n m p : nat,
  m + S p = S (m + p) ->
  (n + m) + S p = S ((n + m) + p)
```

Induct on n. The zero branch reduces to the supplied premise. The successor branch
uses the IH through a forward rewrite (A) or successor congruence and exact IH (B).
The informal paragraphs and each aligned step specify those choices explicitly.
Both variants preserve the same mathematical induction. Opposite successor tactics
can be mathematically faithful while failing the requested proof-step contract.

The theorem generalizes the successor identity in historical seed_028; its premise
is the identity of seed_018. Both already belong to successor_reassociation_variants.
The old family allocation is preserved. No new theorem family, tactic or lemma is
introduced. The public duplicate check finds no exact/definitional/orientation/binder
equivalent statement in the old 30 development seeds or pilot train/dev. The family
review separates this successor law from the consumed conditional-permutation family
without opening holdout files. It is not an exact hidden-statement duplicate scan.

| Base case | IH rewrite | IH congruence |
| --- | ---: | ---: |
| Computation/reflexivity | 3 | 3 |
| Supplied premise | 1 | 1 |

The original pilot-1.0 row schema is retained: formal_statement and informal_statement
describe the theorem; informal_proof is the paragraph; steps retain text/code alignment;
proof_body joins step code; argument_features and argument_contract describe the proof;
generation_family controls grouping; provenance, review, verification and quality_checks
record assistant curation and reference QA. Only theorem and informal proof are model
inputs. IDs, contracts, references and QA metadata must not be included in prompts.

All original v1 files, including their known minor prose issues, remain unchanged.
All new semantic judgments are assistant reviews; human_reviewed is false. The premise
and IH are useful within these scripts; no global minimality or logically indispensable
premise claim is made. There is only one new theorem pair in an exposed family: this
fills a selection-dev coverage gap, not an independent generalization benchmark.

Files: additions.jsonl contains only the two new rows; dev.jsonl is the original six
lines plus those rows. references/ contains the exact verifier-rendered .v sources.
verification_report.json and VERIFICATION_REPORT.md record compiler/token evidence;
family_review.json records public comparisons and manual scope limitations.
dev_manifest.json commits membership, artifacts and protected dependencies; release.json
seals the manifest. This is not a complete pilot-v2 train/dev/holdout protocol.

From the repository root:

```sh
# Verify the release hashes and schema without model work or writing artifacts.
.venv/bin/python -B -m pilot_dev_v2.release check
# Additionally recompile all eight references and check exact tokenizer paths.
.venv/bin/python -B -m pilot_dev_v2.release check --reverify
# Test the extension and its tamper checks without opening holdout/diagnostic data.
.venv/bin/python -B -m unittest discover -s tests -p 'test_pilot_dev_v2.py' -v
```

One-time preparation uses `python -B -m pilot_dev_v2.release prepare`. It refuses an
existing release. To reproduce construction without replacing evidence, use a fresh
`--output /private/tmp/proofbridge-dev-v2-reproduction` directory. Compiler timings and
freeze timestamps vary, so reconstructed manifests are not expected to be identical.

No training or generation entry point is provided. The frozen v1 scorer requires six
dev rows; it is not repurposed silently. Before training, freeze a separate v2 protocol
with explicit selection membership/rule, an original-data control, and a fresh holdout.
