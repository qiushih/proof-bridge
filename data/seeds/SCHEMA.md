# Seed schema `seed-0.2`

The release remains exactly `seed_001` through `seed_030`. `curated.json` is the
source array; `pairs.jsonl` contains the same fields plus the verifier input,
rendered target, line alignment, and actual compilation evidence. The dataset ID
remains `mini_nat_addition_seeds_v1`; the schema version changed from `seed-0.1`
to `seed-0.2`. No training split has been assigned.

| Field | Meaning and invariant |
| --- | --- |
| `informal_statement` | Natural-language theorem statement, including any supplied equality premise. It contains the claim to prove. |
| `informal_proof` | One proof paragraph made by joining the ordered `steps[*].text` with spaces. It contains the argument, including the induction hypothesis and cases when present. |
| `formal_statement` | Fixed, universally quantified Rocq proposition in the existing natural-number addition grammar. Unchanged in this refinement. |
| `steps` | Original ordered `{text, code}` objects for step-level alignment. Every original string is preserved. One step can cover several tactic lines. |
| `proof_body` | Canonical candidate body, without `Proof`/`Qed`: join `steps[*].code` with newlines, then apply only the explicitly recorded local-hypothesis normalizations described below. |
| `split_group` | Coarse leakage boundary. All records with this value must stay in the same future split. It is not an assigned split. |
| `generation_family` | Curated argument/variant family within one `split_group`. Related variants and detected definitionally equivalent statements are kept together. The family label never licenses splitting the broader group. |
| `argument_features` | Derived description of tactics and premise use in the canonical proof, as specified below. |
| `metadata` | Provenance, original-step checksum, normalization audit trail, previous grouping labels where changed, and research-only links such as `related_existing_example`. |
| `metadata.provenance` | Explicit authorship, review, and Rocq-verification annotations; a true verification flag is bound to the rendered-source hash and checked compilation evidence. |
| `method` | Existing coarse method: `definition`, `equality_premise`, or `induction`. Retained for compatibility. |
| `induction_variable` | Existing source annotation on induction records only. Must agree with `argument_features.induction_variable`. |

`input.normalized_statement`, `input.normalized_proof`,
`input.formal_statement`, and `target.proof_body` are compatibility copies of
the canonical fields. `input.raw_text` is the informal statement, two newlines,
then the informal proof. Export validation rejects disagreements between the
source and these copies. Consumers can use the canonical fields directly.

`target.rocq_source` is the complete verifier-generated source, including its
fixed prelude and theorem wrapper. Its local names are renamed by the existing
verifier. `alignment[*].proof_line_start` and `proof_line_end` are inclusive,
one-based line numbers in `proof_body`, with the original curated names.
They cover every proof-body line exactly once.

## Argument features

All six fields are required on every record. Five are JSON booleans;
`induction_variable` is a string or JSON `null`.

| Feature | Derivation |
| --- | --- |
| `uses_induction` | A permitted `induction` command occurs in the canonical body. |
| `induction_variable` | Local name used by that command, or `null` for a non-inductive proof. |
| `uses_premise` | A `rewrite`, `exact`, or `apply` command references the explicitly introduced theorem premise. Merely having a premise is insufficient. An induction hypothesis alone does not count. |
| `uses_rewrite` | An explicit `rewrite` tactic occurs, in either direction and with either a premise or induction hypothesis. Definitional reduction by `simpl` does not count. |
| `uses_f_equal` | An explicit `f_equal` tactic occurs. |
| `uses_reflexivity` | An explicit `reflexivity` tactic occurs, including in an induction base case. A proof finished by `exact` does not implicitly count as reflexivity. |

These annotations describe the curated code and its corresponding argument.
They are not a semantic analysis of arbitrary English or of the generated proof
term. For example, seed 017 uses induction but no supplied premise; seed 027
uses both. Seed 009 uses a premise and `f_equal`, but no explicit reflexivity.

## Preserved steps and canonical normalization

The canonical builder first joins the original code. A
`metadata.proof_normalizations` entry then specifies `proof_line`, `from`, `to`,
and `reason`. The only accepted edit is `apply local.` to `exact local.` on the
same line, with the same local reference and indentation. Every edit must match
the original line. The builder cannot insert commands or change the reference.
This text check does not establish that a hypothesis matches a goal: Rocq
compiles both versions to establish that each proves the unchanged theorem.

| Seed | Original line retained in `steps` | Canonical line |
| --- | --- | --- |
| `seed_009` | `apply H.` | `exact H.` |
| `seed_022` | `apply IH.` | `exact IH.` |
| `seed_027` | `apply IH.` | `exact IH.` |

Seed 022 matches modulo the definitional reduction `1 + k = S k`. Each of these
three replacements passes Rocq, and `verification.authored_proof` records a
separate successful compilation of the original. The other 27 canonical bodies
are exactly the newline join of their steps. All line counts are unchanged.

`metadata.authored_steps_sha256` hashes UTF-8 JSON of the original step array
with sorted object keys, `ensure_ascii=False`, and compact separators `,` and
`:`. It detects accidental edits to either original text or code; it is an
integrity check, not a signature or independent review.

## Family policy and duplicate checks

`families.json` declares each family, its members, its parent `split_group`, and
the curation rationale. It covers precisely the 30 existing IDs. There are
seven generation families within five coarse split groups:

| Split group | Generation families | Seeds |
| --- | --- | --- |
| `definitional_addition` | `definitional_reflexivity` | 001–006 |
| `equality_transport` | `premise_reuse`; `premise_congruence`; `premise_definitional_substitution` | 007–008; 009–013; 014–015 |
| `right_identity_induction` | `right_zero_variants` | 016, 020, 021, 027 |
| `addition_structure_induction` | `successor_reassociation_variants` | 017–019, 022–026, 028, 030 |
| `conditional_permutation_induction` | `conditional_permutation` | 029 |

The earlier `equality_reflexivity` group is merged into `definitional_addition`
because 001/002 are definitionally equivalent. The earlier successor and
associativity groups are merged into `addition_structure_induction`: seed 030
combines both arguments and specializes to seed 025. Changed labels are retained
as `metadata.previous_split_group`. These are regroupings of existing records,
not new theorem content. Finer labels are used where the argument supports
them, without splitting these connected variants.

The audit rejects duplicate statements modulo binder names. It also compares
the fragment after removing redundant parentheses, renaming binders by position,
expanding numerals into successors, and reducing addition on `0` or `S` in its
first argument. It detects four equivalent pairs: **001/002, 005/006, 017/022,
and 018/026**, leaving 26 distinct definitional forms. These intentional pairs
are retained and must share both grouping labels. They are not independent
evaluation problems. The audit does not use arithmetic lemmas or decide general
logical equivalence.

All exported records remain `split: "unassigned"`. Seed 016 also records
`metadata.related_existing_example: "examples/add_zero_right.json"`; that
existing example must stay with the right-zero family in any future split.
Family curation is conservative but still needs human review before evaluation.
The five groups should not be interpreted as a claim of statistical independence.

## Provenance and actual verification

Every `metadata.provenance` contains:

```json
{
  "informal_proof_source": "assistant-curated",
  "formal_proof_source": "assistant-curated",
  "human_reviewed": false,
  "rocq_verified": true,
  "rocq_source_sha256": "<SHA-256 of target.rocq_source>",
  "source_file": "data/seeds/curated.json",
  "informal_alignment_status": "assistant-reviewed; not kernel-certified",
  "curation": "Individually selected and authored; no enumeration or paraphrase expansion."
}
```

`rocq_verified` and its source hash are refreshed from actual compilation by
`--write`; they become false/null for a failed canonical proof. The export's
`verification` object records status, category, kernel/assumption check flags,
source hash, timestamp, compiler diagnostics, and elapsed time. For normalized
proofs it also includes the original compilation result. The report fingerprints
the curated source, environment lock, verifier, build/schema scripts, family
manifest, and JSONL output. `--check` rejects stale or contradictory evidence
before compiling again. `--audit` checks annotations and families but does not
certify proof validity or freshness of saved compiler evidence.

The source is deliberately strict: an intentional future proof edit must update
derived annotations and clear its old `rocq_verified`/`rocq_source_sha256` before
rebuilding. Updating a recorded original-step checksum is a deliberate source
revision, not a way to preserve an old verification claim.

`verification.training_eligible` is only a formal-verification flag. It does not
record training, human review, or English fidelity. Kernel success establishes
the fixed formal theorem and absence of global axioms; it does not verify that
the informal statement or proof expresses exactly that theorem and argument.
