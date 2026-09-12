"""Frozen inputs, exclusive evidence files and sealed-data access guard."""
from contextlib import contextmanager
from datetime import datetime, timezone
import builtins
import inspect
import io
import json
from pathlib import Path
import platform
import subprocess

from baseline.run import runtime_versions
from pilot_v1.protocol import ROOT, digest, read_rows
from training_protocol_v2 import preparation as preparation
from training_protocol_v2.scoring import DEV_IDS, STEPS
from verifier import sha256_file

HERE = Path(__file__).resolve().parent
PREFLIGHT = ROOT / 'results/v2-runner-preflight'
OUTPUT = ROOT / 'results/lora-pilot-v2'
ARMS = ('control', 'intervention')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    return read(ROOT / 'training_protocol_v2/config.json')


def inputs():
    preparation.check()
    # These kernels still read v1's config. Prove every value they use is identical;
    # the v1 data loader and fixed-epoch loop are never called by this runner.
    old = read(ROOT / 'training_protocol_v1/config.json')['training']
    new = config()['training']
    require(all(value == old[k] for k, value in new.items() if k != 'shuffle'),
            'Shared numerical kernels would use different training settings')
    extra = [ROOT / 'data/pilot-v2-training/release.json',
             ROOT / 'data/pilot-v2-training/split_manifest.json',
             ROOT / 'data/pilot-v2-training/schedule.json',
             ROOT / 'pilot_training_v1/evaluation.py', ROOT / 'lora_training_v1/telemetry.py']
    result = preparation.source_hashes()
    result.update({str(p.relative_to(ROOT)): sha256_file(p) for p in extra})
    result.update({'training_protocol_v2/' + name: value
                   for name, value in preparation.protocol_hashes().items()})
    return result


def code_hashes():
    result = {p.name: sha256_file(p) for p in sorted(HERE.iterdir())
              if p.suffix in ('.py', '.md')}
    result['../tests/test_pilot_training_v2.py'] = sha256_file(HERE.parent / 'tests/test_pilot_training_v2.py')
    return result


def hardware():
    # Select only non-identifying fields; never persist serial numbers or UUIDs.
    data = json.loads(subprocess.check_output(
        ['/usr/sbin/system_profiler', 'SPHardwareDataType', '-json'], text=True, timeout=30))['SPHardwareDataType'][0]
    current = {'model_identifier': data['machine_model'], 'chip': data['chip_type'],
               'memory': data['physical_memory'], 'architecture': platform.machine(),
               'os': platform.mac_ver()[0], 'os_build': subprocess.check_output(
                   ['/usr/bin/sw_vers', '-buildVersion'], text=True, timeout=5).strip(),
               'python': platform.python_version()}
    expected = read(ROOT / 'training_protocol_v2/hardware.json')
    require(all(expected[k] == v for k, v in current.items()), 'Frozen training hardware/runtime differs')
    packages = runtime_versions()
    require(all(packages[k] == v for k, v in expected['packages'].items()), 'Pinned package differs')
    return current | {'packages': packages}


def schedules():
    result = read(ROOT / 'data/pilot-v2-training/schedule.json')
    for arm in ARMS:
        rows = preparation.load_public(arm)
        expected = preparation.schedule([r['id'] for r in rows])
        require(all(result[arm][k] == v for k, v in expected.items()), 'Frozen schedule differs')
    return result


def public_rows(arm):
    require(arm in ARMS, 'Unknown arm; no holdout access')
    return preparation.load_public(arm)


def dev_rows():
    rows = preparation.load_public('dev')
    require([r['id'] for r in rows] == DEV_IDS, 'Expected eight dev rows')
    return rows


def verifier_smoke():
    from dataclasses import asdict
    from verifier import verify
    row = dev_rows()[0]
    result = verify(row['formal_statement'], row['proof_body'], timeout=10.0)
    require(result.status == 'PASS' and result.compiler_version == '9.2.0'
            and result.kernel_checked and result.assumptions_checked, 'Rocq reference smoke check failed')
    return {'reference_id': row['id'], 'verification': asdict(result)}


def sealed_files(folder, names=None):
    paths = [folder / n for n in names] if names is not None else [p for p in folder.rglob('*') if p.is_file()]
    return {str(p.relative_to(folder)): sha256_file(p) for p in sorted(paths)}


def check_files(folder, hashes):
    for name, expected in hashes.items():
        path = folder / name
        require(path.is_file() and not path.is_symlink() and sha256_file(path) == expected,
                'Saved evidence changed: ' + name)


def check_ready(preflight=PREFLIGHT):
    release = read(HERE / 'release.json')
    require(release['status'] == 'FROZEN_RUNNER_NOT_TRAINED' and release['code_sha256'] == code_hashes(), 'Runner not frozen or code changed')
    require(release['inputs_sha256'] == inputs(), 'Frozen inputs differ')
    check_files(preflight, release['preflight_sha256'])
    evidence = read(preflight / 'preflight.json')
    require(evidence['status'] == 'PASS' and evidence['code_sha256'] == code_hashes()
            and evidence['inputs_sha256'] == release['inputs_sha256'], 'Preflight not bound to current runner')
    return release


@contextmanager
def sealed_io():
    """Allow new reserved bytes only through the existing SHA256 helper.

    This is a workflow guard, not an OS sandbox or a defense against hostile code.
    Old reserved/diagnostic content is denied even for hashing.
    """
    original_builtin, original_io = builtins.open, io.open
    def checked_open(original, file, mode='r', *args, **kwargs):
        if isinstance(file, (str, bytes, Path)):
            path = str(Path(file).resolve()) if not isinstance(file, bytes) else str(Path(file.decode()).resolve())
            if any(part in path for part in ('/data/pilot-v1/reserved/', '/results/holdout-v1/', '/data/evaluation/')):
                raise PermissionError('Historical holdout/diagnostic access forbidden')
            if '/reserved/' in path:
                frame, hashing = inspect.currentframe().f_back, False
                while frame:
                    if frame.f_code is sha256_file.__code__:
                        hashing = True; break
                    frame = frame.f_back
                require(mode == 'rb' and hashing, 'Sealed holdout may only be hashed, never parsed')
        return original(file, mode, *args, **kwargs)
    builtins.open = lambda file, mode='r', *a, **kw: checked_open(original_builtin, file, mode, *a, **kw)
    io.open = lambda file, mode='r', *a, **kw: checked_open(original_io, file, mode, *a, **kw)
    try:
        yield
    finally:
        builtins.open, io.open = original_builtin, original_io
