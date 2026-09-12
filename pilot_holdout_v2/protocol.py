"""Freeze public evidence before the exclusive original-holdout opening."""
from contextlib import contextmanager
import builtins
import hashlib
import inspect
import io
import subprocess

from baseline.setup_model import check_model
from pilot_training_v2.common import (
    ROOT, Path, read, write, now, require, inputs, hardware, verifier_smoke,
    sealed_files, check_files, sha256_file, dev_rows,
)
from pilot_training_v2.audit import check as check_training

HERE = ROOT / 'pilot_holdout_v2'
FROZEN = HERE / 'frozen.json'
PREFLIGHT = ROOT / 'results/holdout-v2-preflight'
OUTPUT = ROOT / 'results/holdout-v2'
TRAINING = ROOT / 'results/lora-pilot-v2'
RESERVED = ROOT / 'data/pilot-v2-training/reserved/holdout.jsonl'
MODELS = ('base', 'control', 'intervention')
IDS = [f'phv20{n}_{a}' for n in (1, 2, 3) for a in ('A', 'B')]
ORIGINAL_OPEN_ALLOWED = False


def code_hashes():
    paths = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.md'))
    paths.append(ROOT / 'tests/test_pilot_holdout_v2.py')
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def selected():
    value = read(TRAINING / 'selected_checkpoints.json')
    expected = {
        'control': '93cee6837152b91b75aeba580a36ab9c85c382a5703c4850fe046bf4e1f343af',
        'intervention': 'ee7255324835cf0b47d9b5cb92dec930f344b8d1988b9354397fc4bbe9c7329c',
    }
    for arm, digest in expected.items():
        row = value[arm]
        require(row['selected_step'] == 36 and row['selected_adapter_sha256'] == digest,
                'Previously selected checkpoint changed')
        require(row['selected_adapter_path_relative_to_run'] == f'{arm}/training/step-0036/adapter.safetensors',
                'Unexpected adapter path')
        require(sha256_file(TRAINING / row['selected_adapter_path_relative_to_run']) == digest,
                'Selected adapter bytes changed')
    return value


def sources():
    result = inputs()
    names = ['baseline_dev_v2/experiment.py', 'pilot_training_v2/report.py',
             'pilot_training_v2/audit.py', 'pilot_training_v2/common.py',
             'pilot_training_v2/release.json', 'results/lora-pilot-v2/release.json',
             'results/lora-pilot-v2/selected_checkpoints.json',
             'results/lora-pilot-v2/summary.json']
    for arm, row in selected().items():
        names += [f'results/lora-pilot-v2/{arm}/training/step-0036/adapter.safetensors',
                  f'results/lora-pilot-v2/{arm}/training/step-0036/checkpoint.json']
    result.update({name: sha256_file(ROOT / name) for name in names})
    return result


def committed(hashes):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for name, expected in hashes.items():
        data = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        require(hashlib.sha256(data).hexdigest() == expected, 'Required evidence is not committed: ' + name)
    return commit


@contextmanager
def holdout_io(reserved=RESERVED, root=ROOT):
    """Workflow guard: historical sets denied; original reads limited to custody.

    Hash-only integrity checks are allowed for new reserved files. This is not
    an OS sandbox or a defense against malicious code.
    """
    root = Path(root).resolve()
    original_builtin, original_io = builtins.open, io.open
    def checked(original, file, mode='r', *args, **kwargs):
        if isinstance(file, (str, bytes, Path)):
            path = Path(file.decode() if isinstance(file, bytes) else file).resolve()
            name = str(path)
            if any(path.is_relative_to(root / s) for s in ('data/pilot-v1/reserved', 'results/holdout-v1', 'data/evaluation')):
                raise PermissionError('Historical holdout and diagnostic content remain forbidden')
            if path.is_relative_to(reserved.resolve().parent):
                frame, hashing = inspect.currentframe().f_back, False
                while frame:
                    if frame.f_code is sha256_file.__code__:
                        hashing = True
                        break
                    frame = frame.f_back
                authorized = ORIGINAL_OPEN_ALLOWED and path == reserved.resolve()
                require(mode == 'rb' and (hashing or authorized), 'Original holdout is sealed')
        return original(file, mode, *args, **kwargs)
    builtins.open = lambda f, mode='r', *a, **kw: checked(original_builtin, f, mode, *a, **kw)
    io.open = lambda f, mode='r', *a, **kw: checked(original_io, f, mode, *a, **kw)
    try:
        yield
    finally:
        builtins.open, io.open = original_builtin, original_io


def freeze():
    require(not FROZEN.exists() and not OUTPUT.exists(), 'Protocol/run exists; no replacement')
    check_training()
    require(read(TRAINING / 'release.json')['status'] == 'FROZEN_TRAINED_DEV_RESULTS', 'Training results must be frozen')
    manifest = read(ROOT / 'data/pilot-v2-training/split_manifest.json')
    require(manifest['memberships']['holdout'] == IDS, 'Unexpected public holdout membership')
    evidence = read(PREFLIGHT / 'preflight.json')
    require(evidence['status'] == 'PASS' and evidence['code_sha256'] == code_hashes()
            and evidence['holdout_content_reads'] == 0 and evidence['model_generations'] == 0,
            'A passing unopened public preflight is required')
    before = sources()
    training_names = ['results/lora-pilot-v2/release.json', 'results/lora-pilot-v2/selected_checkpoints.json']
    training_names += [f'results/lora-pilot-v2/{arm}/training/step-0036/adapter.safetensors' for arm in MODELS[1:]]
    commit = committed({name: before[name] for name in training_names})
    old = read(ROOT / 'results/pretraining-dev-v1/manifest.json')
    write(FROZEN, {'status': 'FROZEN_BEFORE_HOLDOUT_OPENING', 'frozen_at_utc': now(),
          'model_order': list(MODELS), 'example_ids': IDS, 'expected_attempts': 18,
          'model': check_model(), 'selected': selected(), 'training_commit': commit,
          'hardware': hardware(), 'generation_settings': old['generation_settings'],
          'effective_generation_config': old['effective_generation_config'],
          'sources_sha256': before, 'code_sha256': code_hashes(),
          'preflight_sha256': sealed_files(PREFLIGHT),
          'holdout_sha256': manifest['files_sha256']['reserved/holdout.jsonl'],
          'verifier_timeout_seconds': 10.0, 'training_updates': 0,
          'holdout_content_reads': 0, 'attempts_per_row_per_model': 1,
          'repair_attempts': 0, 'reselection': False,
          'authorization': 'User said continue to the predeclared six-row, three-model, 18-generation final holdout comparison.'})
    return {'status': 'FROZEN_BEFORE_HOLDOUT_OPENING', 'expected_attempts': 18, 'holdout_content_reads': 0}


def check_protocol():
    p = read(FROZEN)
    require(p['status'] == 'FROZEN_BEFORE_HOLDOUT_OPENING' and p['model_order'] == list(MODELS)
            and p['example_ids'] == IDS and p['expected_attempts'] == 18, 'Protocol identity changed')
    require(p['code_sha256'] == code_hashes() and p['sources_sha256'] == sources(), 'Frozen code or inputs changed')
    require(p['selected'] == selected() and sha256_file(RESERVED) == p['holdout_sha256'], 'Selected or sealed data changed')
    check_files(PREFLIGHT, p['preflight_sha256'])
    return p


def validate_rows(rows, expected=IDS):
    require(len(rows) == 6 and [r['id'] for r in rows] == expected, 'Exactly six ordered holdout rows required')
    for row in rows:
        require(row['argument_id'] == row['id'][-1] and row['split'] == 'holdout', 'Invalid holdout identity')
        require(row['theorem_id'] == row['id'].rsplit('_', 1)[0], 'Theorem identity differs')
        for key in ('formal_statement', 'informal_statement', 'informal_proof', 'proof_body'):
            require(isinstance(row[key], str) and row[key].strip(), 'Missing proof field: ' + key)
    for a, b in zip(rows[::2], rows[1::2]):
        require(a['argument_id'] == 'A' and b['argument_id'] == 'B'
                and a['theorem_id'] == b['theorem_id'] and a['formal_statement'] == b['formal_statement'],
                'A/B pair membership differs')
    return rows


def open_original_once(output, frozen, reserved=RESERVED):
    global ORIGINAL_OPEN_ALLOWED
    require(sha256_file(reserved) == frozen['holdout_sha256'], 'Original hash changed')
    write(output / 'custody.json', {'status': 'UNSEALED_CONSUMED', 'opened_at_utc': now(),
          'protocol_sha256': sha256_file(FROZEN), 'original_sha256': frozen['holdout_sha256'],
          'authorized_original_content_openings': 1, 'future_tuning_requires_new_holdout': True})
    try:
        ORIGINAL_OPEN_ALLOWED = True
        data = reserved.read_bytes()
    finally:
        ORIGINAL_OPEN_ALLOWED = False
    require(hashlib.sha256(data).hexdigest() == frozen['holdout_sha256'], 'Original changed during opening')
    with (output / 'opened_holdout.jsonl').open('xb') as f:
        f.write(data)
    return opened_rows(output, frozen)


def opened_rows(output=OUTPUT, frozen=None):
    import json
    frozen = frozen or check_protocol()
    custody = read(output / 'custody.json')
    require(custody['status'] == 'UNSEALED_CONSUMED'
            and custody['protocol_sha256'] == sha256_file(FROZEN), 'Missing bound custody record')
    path = output / 'opened_holdout.jsonl'
    require(sha256_file(path) == frozen['holdout_sha256'], 'Opened snapshot changed')
    return validate_rows([json.loads(line) for line in path.read_text().splitlines()])
