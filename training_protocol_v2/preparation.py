"""Prepare/freeze v2 references and protocol; checks never parse the sealed holdout."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from baseline_dev_v2 import experiment as baseline
from constrained_v1.grammar import accepts
from pilot_dev_v2 import release as dev_release
from pilot_v1.dataset import ablations, token_evidence
from pilot_v1.protocol import (ROOT, digest, equivalence_key, features, patterns, read_rows,
                               validate_row as validate_v1)
from training_protocol_v1.data import encode_example
from training_protocol_v1.scoring import step_adherence
from training_protocol_v2.curation import TRAIN_SPEC, make_pair
from training_protocol_v2.scoring import DEV_IDS, selection_score
from verifier import render_source, sha256_file, verify

HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/pilot-v2-training'
FAMILIES = {'equality_transport': 'train', 'right_zero_variants': 'train',
            'successor_reassociation_variants': 'dev',
            'conditional_repeated_sum_contraction': 'holdout'}
NEW_IDS = ['ptv201_A', 'ptv201_B']
HOLDOUT_IDS = [f'phv20{n}_{a}' for n in (1, 2, 3) for a in ('A', 'B')]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def rows_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    result = baseline.sources()
    for name in ('baseline/requirements.lock', 'baseline_dev_v2/experiment.py',
                 'baseline_dev_v2/PROTOCOL.md', 'results/v2-dev-baseline/release.json',
                 'results/v2-dev-baseline/assessed_results.jsonl', 'lora_training_v1/core.py',
                 'training_protocol_v1/adaptation.py', 'pilot_training_v1/evaluation.py'):
        result[name] = sha256_file(ROOT / name)
    return result


def protocol_hashes():
    result = {p.name: sha256_file(p) for p in sorted(HERE.iterdir())
              if p.suffix in ('.py', '.md', '.json')}
    result['../tests/test_training_protocol_v2.py'] = sha256_file(HERE.parent / 'tests/test_training_protocol_v2.py')
    return result


def check_design():
    design = read(HERE / 'design.json')
    require(design['status'] == 'PRECOMMITTED_BEFORE_FRESH_HOLDOUT_CONSTRUCTION', 'Design not precommitted')
    for name, expected in design['files_sha256'].items():
        require(sha256_file(HERE / name) == expected, 'Precommitted design changed')
    return design


def validate_row(row):
    require(FAMILIES.get(row['generation_family']) == row['split'], 'Family/split mismatch')
    if row['schema_version'] == 'pilot-1.0':
        validate_v1(row)
        return
    require(row['schema_version'] == 'pilot-2.0', 'Unknown row schema')
    required = ('id', 'theorem_id', 'argument_id', 'formal_statement', 'informal_statement',
                'informal_proof', 'proof_body', 'steps', 'argument_features', 'provenance',
                'review', 'argument_contract', 'contrast_type', 'verification', 'quality_checks')
    require(all(row.get(k) for k in required), 'Missing row field')
    require(row['proof_body'] == '\n'.join(s['code'] for s in row['steps']), 'Canonical proof body differs')
    require(row['informal_proof'] == ' '.join(s['text'].strip() for s in row['steps']), 'Canonical informal paragraph differs')
    require(all(s['code'].strip() and s['text'].strip() for s in row['steps']), 'Empty aligned step')
    require(row['argument_features'] == features(row['formal_statement'], row['proof_body']), 'Features changed')
    require(accepts(row['formal_statement'], row['proof_body']), 'Frozen decoder rejects body')
    v, p, review = row['verification'], row['provenance'], row['review']
    require(v['status'] == 'PASS' and v['category'] == 'VERIFIED' and v['compiler_version'] == '9.2.0'
            and v['kernel_checked'] and v['assumptions_checked'], 'Reference lacks Rocq verification')
    require(v['source_sha256'] == digest(render_source(row['formal_statement'], row['proof_body']))
            == p['rocq_source_sha256'], 'Reference hash differs')
    require(p['rocq_verified'] and p['human_reviewed'] is False and p['model_generated'] is False
            and review['human_reviewed'] is False and review['primary_target_eligible'] is True,
            'Invalid review/provenance')


def audit(rows):
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate IDs')
    equivalents, theorems = defaultdict(list), defaultdict(list)
    pairs = set()
    for r in rows:
        validate_row(r)
        equivalents[equivalence_key(r['formal_statement'])].append(r)
        theorems[r['theorem_id']].append(r)
        pair = (r['formal_statement'], r['proof_body'])
        require(pair not in pairs, 'Duplicate statement/proof')
        pairs.add(pair)
    for group in equivalents.values():
        require(len({(r['generation_family'], r['split']) for r in group}) == 1, 'Cross-split equivalent')
    for group in theorems.values():
        require(len({r['formal_statement'] for r in group}) == 1, 'Theorem variants differ')
        require(len({r['argument_id'] for r in group}) == len(group), 'Repeated argument')
        require(len({r['informal_proof'] for r in group}) == len(group), 'Distinct targets need distinct arguments')
    return {'status': 'PASS', 'rows': len(rows), 'theorems': len(theorems),
            'equivalence_classes': len(equivalents), 'cross_split_equivalents': 0,
            'cross_split_families': 0, 'duplicate_statement_body_pairs': 0}


def schedule(ids):
    require(len(ids) == len(set(ids)) and len(ids) in (24, 26), 'Unexpected training membership')
    ordered, cycle = [], 0
    while len(ordered) < 240:
        group = sorted(ids)
        random.Random(1729 + cycle).shuffle(group)
        ordered.extend(group)
        cycle += 1
    ordered = ordered[:240]
    return {'microbatches': ordered, 'updates': [ordered[i:i+4] for i in range(0, 240, 4)],
            'counts_by_id': dict(sorted(Counter(ordered).items())), 'optimizer_updates': 60,
            'complete_cycles': 240 // len(ids), 'final_cycle_rows': 240 % len(ids)}


def coverage(rows):
    by_pattern = Counter(p for r in rows for p in patterns(r))
    matrix = Counter(('premise' if r['argument_features'].get('premise_in_base_case') else 'computation') + '/' +
                     ('rewrite' if r['argument_features'].get('ih_via_rewrite') else 'congruence')
                     for r in rows if r['argument_features']['uses_induction'])
    return {'rows': len(rows), 'theorems': len({r['theorem_id'] for r in rows}),
            'families': dict(Counter(r['generation_family'] for r in rows)),
            'equivalence_classes': len({equivalence_key(r['formal_statement']) for r in rows}),
            'patterns': dict(by_pattern), 'induction_matrix': dict(matrix)}


def negative_checks(row):
    evidence = ablations(row)
    for name, bad in (
        ('replace_base_premise_with_reflexivity', row['proof_body'].replace('exact H.', 'reflexivity.')),
        ('direct_substitution_without_induction', 'intros n m p H.\nrewrite H.\nreflexivity.'),
    ):
        v = asdict(verify(row['formal_statement'], bad))
        require(v['status'] == 'FAIL' and v['stage'] == 'rocq' and v['category'] in ('PROOF_ERROR', 'INCOMPLETE_PROOF'), 'Negative reference control passed or failed at parser')
        evidence.append({'ablation': name, 'mutated_body_sha256': digest(bad), 'verification': v})
    return evidence


def qa(rows, new_ids):
    from constrained_v1.experiment import tokenizer_only
    tokens = token_evidence(rows)
    tokenizer = tokenizer_only()
    prompt = read(ROOT / 'prompt_v2/selected.json')
    evidence = []
    for r in rows:
        v = asdict(verify(r['formal_statement'], r['proof_body']))
        require(v['status'] == 'PASS' and v['compiler_version'] == '9.2.0', 'Rocq reference failed: ' + r['id'])
        encoded = encode_example(tokenizer, prompt, r)
        p = encoded['prompt_length']
        require(encoded['labels'][:p] == [-100] * p and encoded['labels'][p:] == encoded['input_ids'][p:]
                and encoded['labels'][-1] == tokenizer.eos_token_id, 'Target-only loss mask incorrect')
        lengths = {k: encoded[k] for k in ('prompt_length', 'target_length', 'sequence_length')}
        record = {'id': r['id'], 'verification': v, 'decoder': tokens[r['id']],
                  'serialization': lengths, 'target_only_mask_valid': True}
        if r['id'] in new_ids:
            r['verification'] = v
            r['provenance'].update(rocq_verified=True, rocq_source_sha256=v['source_sha256'])
            negatives = negative_checks(r)
            r['quality_checks'] = {'decoder': tokens[r['id']], 'serialization': lengths,
                                   'target_only_mask_valid': True, 'negative_controls': negatives}
            record['negative_controls'] = negatives
        else:
            require(v['source_sha256'] == r['verification']['source_sha256'], 'Historical reference changed')
        validate_row(r)
        evidence.append(record)
    return evidence


def contrast_checks(rows):
    results = []
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        require(a['formal_statement'] == b['formal_statement'], 'Malformed A/B pair')
        for requested in (a, b):
            for supplied in (a, b):
                result = step_adherence(requested, supplied['proof_body'], 'argument_' + requested['argument_id'])
                require(result['matched'] == (requested['id'] == supplied['id']), 'Step contracts do not separate A/B')
                results.append({'requested': requested['id'], 'supplied_reference': supplied['id'],
                                'requested_steps': result['status'], 'mathematical_fidelity': 'FAITHFUL',
                                'reviewer': 'assistant', 'human_reviewed': False,
                                'reason': 'Both references use the same premise-base induction; rewrite and congruence are equivalent valid successor realizations.'})
    return results


def baseline_records():
    baseline.check(baseline.OUT)
    records = [r for r in read_rows(baseline.OUT / 'assessed_results.jsonl') if r['model_key'] == 'base']
    require([r['dev_example_id'] for r in records] == DEV_IDS, 'Step-zero baseline membership changed')
    require(selection_score(records, 0) == [6, 3, 6, 0], 'Step-zero assessment changed')
    return records


def load_public(split, directory=DATA):
    require(split in ('control', 'intervention', 'dev'), 'Only control/intervention/dev are public; holdout is sealed')
    manifest = read(directory / 'split_manifest.json')
    require(sha256_file(directory / 'split_manifest.json') == read(directory / 'release.json')['split_manifest_sha256'], 'Manifest seal mismatch')
    if split == 'dev':
        path = ROOT / 'data/pilot-v2/dev.jsonl'
        expected = manifest['source_files_sha256']['data/pilot-v2/dev.jsonl']
    else:
        name = 'train_' + split + '.jsonl'
        path, expected = directory / name, manifest['files_sha256'][name]
    require(sha256_file(path) == expected, 'Frozen public data changed')
    rows = read_rows(path)
    require([r['id'] for r in rows] == manifest['memberships'][split], 'Split membership changed')
    audit(rows)
    return rows


def prepare(directory, draft_path):
    require(not directory.exists(), 'Release already exists; refusing overwrite')
    require(draft_path is not None and 'reserved' not in draft_path.parts, 'Use an unsealed draft, never a sealed holdout')
    design = check_design()
    draft = read(draft_path)
    require(draft['status'] == 'UNSEALED_CONSTRUCTION_DRAFT' and draft['design_sha256'] == sha256_file(HERE / 'design.json'), 'Draft not bound to design')
    require(design['precommitted_at_utc'] < draft['created_at_utc'], 'Holdout predates design')
    source_before, code_before = source_hashes(), protocol_hashes()
    dev_release.check()
    old_bytes = (ROOT / 'data/pilot-v1/train.jsonl').read_bytes()
    control = read_rows(ROOT / 'data/pilot-v1/train.jsonl')
    dev = read_rows(ROOT / 'data/pilot-v2/dev.jsonl')
    new = make_pair(TRAIN_SPEC)
    holdout = [r for spec in draft['specifications'] for r in make_pair(spec)]
    require([r['id'] for r in holdout] == HOLDOUT_IDS and [r['id'] for r in new] == NEW_IDS, 'Wrong new membership')
    all_rows = control + new + dev + holdout
    historical = read_rows(ROOT / 'data/development/pairs.jsonl')
    known_keys = {equivalence_key(r['formal_statement']) for r in control + dev + historical}
    require(all(equivalence_key(r['formal_statement']) not in known_keys for r in new + holdout), 'New reference duplicates public theorem')
    print('Verifying 40 references, decoder paths, loss masks and 32 new-reference negative controls.', flush=True)
    evidence = qa(all_rows, set(NEW_IDS + HOLDOUT_IDS))
    family_audit = audit(all_rows)
    public_evidence = [r for r in evidence if r['id'] not in HOLDOUT_IDS]
    private_evidence = [r for r in evidence if r['id'] in HOLDOUT_IDS]
    public_contrasts, private_contrasts = contrast_checks(new), contrast_checks(holdout)
    step_zero = baseline_records()
    arm_rows = {'control': control, 'intervention': control + new}
    lengths = {r['id']: r['serialization'] for r in public_evidence}
    schedules = {}
    for arm, rows in arm_rows.items():
        s = schedule([r['id'] for r in rows])
        s['supervised_tokens'] = sum(lengths[i]['target_length'] for i in s['microbatches'])
        s['total_sequence_tokens'] = sum(lengths[i]['sequence_length'] for i in s['microbatches'])
        schedules[arm] = s
    directory.mkdir(parents=True)
    (directory / 'train_control.jsonl').write_bytes(old_bytes)
    extra = rows_bytes(new)
    require(old_bytes.endswith(b'\n'), 'Missing original JSONL boundary')
    (directory / 'train_additions.jsonl').write_bytes(extra)
    (directory / 'train_intervention.jsonl').write_bytes(old_bytes + extra)
    (directory / 'reserved').mkdir()
    (directory / 'reserved/holdout.jsonl').write_bytes(rows_bytes(holdout))
    for row in new + holdout:
        prefix = directory / ('reserved/references' if row['split'] == 'holdout' else 'references')
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / (row['id'] + '.v')).write_text(render_source(row['formal_statement'], row['proof_body']))
    write(directory / 'reserved/quality_assessment.json', {'reference_evidence': private_evidence,
          'step_vs_fidelity_reviews': private_contrasts, 'constructed_after_design': True,
          'reviewer': 'assistant', 'human_reviewed': False})
    write(directory / 'schedule.json', schedules)
    write(directory / 'coverage.json', {name: coverage(rows) for name, rows in
          [('control', control), ('intervention', control + new), ('dev', dev), ('holdout', holdout)]})
    write(directory / 'step_zero.json', {'source': 'results/v2-dev-baseline/assessed_results.jsonl',
          'source_sha256': sha256_file(baseline.OUT / 'assessed_results.jsonl'),
          'source_keys': [r['key'] for r in step_zero], 'selection_score': selection_score(step_zero, 0),
          'new_generations': 0, 'v1_step18_is_not_a_new_arm_checkpoint': True})
    write(directory / 'family_review.json', {
          'status': 'PASS_UNDER_DOCUMENTED_MANUAL_LEDGER', 'reviewer': 'assistant', 'human_reviewed': False,
          'allocation': FAMILIES, 'construction_audit': family_audit,
          'new_public_duplicates': 0, 'comparison_rows': len(control + dev + historical),
          'equivalence_method': 'Existing definitional reduction, binder permutation and equality orientation plus manual family review.',
          'training_addition_reason': TRAIN_SPEC['review_notes'],
          'holdout_family_reason': 'A supplied repeated-sum value is contracted after propagation under a prefix. Keep every repeated-context, association, orientation and A/B variant together. No explicit right-zero law, successor identity or permutation premise is used. This is a newly allocated conditional contraction group under the existing manually structural family convention.',
          'limitations': ['Shared prefix-induction templates cross splits intentionally.',
              'Associativity connects the new contraction family to dev mathematics. This is not mathematical independence or unseen algebra.',
              'Fresh holdout pairs are correlated members of one family, including repeated-context and association siblings.',
              'Old holdout and diagnostic files were not opened; exact duplicate exclusion against their hidden statements is untested.',
              'Family assignments and English alignment are assistant judgments, not independent human review.']})
    write(directory / 'verification_report.json', {
          'status': 'PASS', 'rocq_version': '9.2.0', 'references_verified': 40,
          'new_train_verified': 2, 'fresh_holdout_verified_before_seal': 6,
          'public_references': public_evidence, 'public_step_vs_fidelity_reviews': public_contrasts,
          'negative_controls': {'total_failed_as_expected': 32, 'public': 8, 'reserved': 24},
          'all_token_paths_allowed': True, 'all_sequences_fit_768': True,
          'all_targets_fit_256_including_eos': True, 'all_target_only_masks_valid': True,
          'reserved_details': 'reserved/quality_assessment.json (sealed; not for selection)'})
    (directory / 'README.md').write_text(README)
    (directory / 'VERIFICATION_REPORT.md').write_text(VERIFICATION_REPORT)
    write(directory / 'preparation.json', {'prepared_at_utc': now(), 'design_sha256': sha256_file(HERE / 'design.json'),
          'draft_created_at_utc': draft['created_at_utc'], 'draft_sha256': sha256_file(draft_path),
          'construction_holdout_content_review': 'assistant review and automated QA before sealing only',
          'model_generations': 0, 'model_weight_loads': 0, 'optimizer_updates': 0,
          'old_holdout_content_reads': 0, 'diagnostic_content_reads': 0,
          'command': 'python -B -m training_protocol_v2.preparation prepare --output <fresh-output> --holdout-draft <unsealed-draft>',
          'post_seal_reproduction': 'check; optionally --reverify-public; never reconstruct from the sealed holdout'})
    require(source_before == source_hashes() and code_before == protocol_hashes(), 'Inputs changed during preparation')
    files = {str(p.relative_to(directory)): sha256_file(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    manifest = {'schema_version': 'pilot-v2-training-manifest-1', 'status': 'FROZEN_BEFORE_TRAINING',
          'frozen_at_utc': now(), 'memberships': {**{k: [r['id'] for r in v] for k, v in arm_rows.items()},
                                               'dev': DEV_IDS, 'holdout': HOLDOUT_IDS},
          'family_assignment': FAMILIES, 'source_files_sha256': source_before,
          'protocol_files_sha256': code_before, 'files_sha256': files,
          'holdout_policy': 'Post-seal checks hash bytes only; no content reads until both arms and selections complete.',
          'activity': {'training_updates': 0, 'model_generations': 0, 'model_weight_loads': 0}}
    write(directory / 'split_manifest.json', manifest)
    write(directory / 'release.json', {'status': 'FROZEN', 'sealed_at_utc': now(),
                                      'split_manifest_sha256': sha256_file(directory / 'split_manifest.json')})
    # The following check hashes reserved bytes but never parses them.
    return check(directory)


def check(directory=DATA, reverify_public=False):
    design = check_design()
    manifest, seal = read(directory / 'split_manifest.json'), read(directory / 'release.json')
    require(seal['status'] == 'FROZEN' and sha256_file(directory / 'split_manifest.json') == seal['split_manifest_sha256'], 'Manifest seal mismatch')
    require(manifest['status'] == 'FROZEN_BEFORE_TRAINING' and manifest['family_assignment'] == FAMILIES, 'Wrong manifest')
    require(manifest['source_files_sha256'] == source_hashes() and manifest['protocol_files_sha256'] == protocol_hashes(), 'Frozen dependency changed')
    require(set(manifest['files_sha256']) == {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} - {'split_manifest.json', 'release.json'}, 'Artifact membership differs')
    for name, expected in manifest['files_sha256'].items():
        path = directory / name
        require(not path.is_symlink() and sha256_file(path) == expected, 'Frozen artifact changed: ' + name)
    prep = read(directory / 'preparation.json')
    require(design['precommitted_at_utc'] < prep['draft_created_at_utc'] < manifest['frozen_at_utc'] <= seal['sealed_at_utc'], 'Custody timestamps differ')
    control, intervention, dev = (load_public(split, directory) for split in ('control', 'intervention', 'dev'))
    old_bytes = (ROOT / 'data/pilot-v1/train.jsonl').read_bytes()
    extra_bytes = (directory / 'train_additions.jsonl').read_bytes()
    require((directory / 'train_control.jsonl').read_bytes() == old_bytes and
            (directory / 'train_intervention.jsonl').read_bytes() == old_bytes + extra_bytes, 'Original training bytes changed')
    require(len(control) == 24 and len(intervention) == 26 and [r['id'] for r in intervention[24:]] == NEW_IDS
            and [r['id'] for r in dev] == DEV_IDS and manifest['memberships']['holdout'] == HOLDOUT_IDS, 'Split count/membership differs')
    audit(intervention + dev)
    schedule_file = read(directory / 'schedule.json')
    public_evidence = read(directory / 'verification_report.json')['public_references']
    lengths = {r['id']: r['serialization'] for r in public_evidence}
    for arm, rows in [('control', control), ('intervention', intervention)]:
        actual = schedule([r['id'] for r in rows])
        actual['supervised_tokens'] = sum(lengths[i]['target_length'] for i in actual['microbatches'])
        actual['total_sequence_tokens'] = sum(lengths[i]['sequence_length'] for i in actual['microbatches'])
        require(actual == schedule_file[arm], 'Exposure schedule differs')
    for row in intervention[24:]:
        require((directory / 'references' / (row['id'] + '.v')).read_text() == render_source(row['formal_statement'], row['proof_body']), 'Public .v source differs')
    zero = baseline_records()
    require(read(directory / 'step_zero.json')['selection_score'] == selection_score(zero, 0), 'Baseline score differs')
    if reverify_public:
        # No holdout loader exists and no private QA record is parsed here.
        evidence = qa(intervention + dev, set())
        for actual, saved in zip(evidence, public_evidence):
            require(actual['id'] == saved['id'] and actual['serialization'] == saved['serialization'] and
                    actual['decoder'] == saved['decoder'] and
                    actual['verification']['source_sha256'] == saved['verification']['source_sha256'], 'Public QA replay differs')
    return {'status': 'PASS', 'control_rows': 24, 'intervention_rows': 26, 'dev_rows': 8,
            'sealed_holdout_rows': 6, 'holdout_content_parsed': False,
            'public_reverified': 34 if reverify_public else 0,
            'optimizer_updates': 0, 'model_generations': 0}


README = '''# V2 pilot training data preparation

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
'''

VERIFICATION_REPORT = '''# V2 reference verification

PASS: 40/40 unique train/dev/fresh-holdout reference rows compile with Rocq 9.2.0,
including 2/2 new training and 6/6 fresh holdout references before sealing. All target
token paths plus EOS are accepted by frozen constrained-v1 within 256 tokens; all
Prompt v2 plus target serializations fit 768 tokens and pass target-only mask checks.

All 32 negative controls fail within Rocq as intended: remove premise use, remove
IH use, replace the base premise with reflexivity, or attempt direct substitution
without induction, for each of eight new references. These establish useful steps
in the authored script, not global minimality or logical necessity of every premise.

Each new A/B pair has matching own-step contracts and mismatching opposite-step
contracts; both verified variants remain mathematically faithful to the same
induction. Semantic/family judgments are assistant reviews, not independent human
review. Frozen original training/dev text and targets remain byte-identical.

Construction checks found zero cross-split canonical equivalents, duplicate target
pairs, or overlapping family assignments among the 40 rows, and no new statement
equivalent to the public pilot or 30 historical development statements. The manual
family ledger and exact-check limitations are in family_review.json. No comparison
with unopened old holdout/diagnostic statements is claimed. Reserved detailed QA
must remain sealed after construction. No model generation or training occurred.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--output', type=Path, default=DATA)
    parser.add_argument('--holdout-draft', type=Path)
    parser.add_argument('--reverify-public', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        require(not args.reverify_public, 'Preparation already verifies references')
        result = prepare(args.output, args.holdout_draft)
    else:
        require(args.holdout_draft is None, 'Check never accepts holdout input')
        result = check(args.output, args.reverify_public)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
