"""Single-run inference; all failures are preserved and no attempts are retried."""
import argparse
from dataclasses import asdict
import io
import json
import os
import random
import numpy as np
import signal
import subprocess
import sys
import time
import traceback
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from baseline.protocol import extract_body
from baseline_dev_v2.experiment import load_adapter
from constrained_v1.experiment import generate, tokenizer_only
from constrained_v1.tokens import Vocabulary
from prompt_v2.experiment import Inference
from prompt_v2.protocol import build_messages, format_compliance
from training_protocol_v1.scoring import step_adherence
from verifier import verify
from pilot_v1.protocol import digest
from pilot_holdout_v2.protocol import *


def prompt_for(example):
    # Deliberately whitelist the only two per-example inputs to the model.
    public_input = {k: example[k] for k in ('formal_statement', 'informal_proof')}
    return build_messages(read(ROOT / 'prompt_v2/selected.json'), public_input, 'theorem_and_informal')


def generation_record(model_key, example, text, generated, adapter_sha):
    body, extraction = extract_body(text)
    evidence = asdict(verify(example['formal_statement'], body, timeout=10.0))
    condition = 'argument_' + example['argument_id']
    return {'key': f'holdout_v2:{model_key}:{example["id"]}', 'model_key': model_key,
            'example_id': example['id'], 'theorem_id': example['theorem_id'],
            'condition': condition, 'formal_statement': example['formal_statement'],
            'checkpoint_adapter_sha256': adapter_sha, 'attempt': 1, 'repair_attempts': 0,
            'generated_text': text, 'proof_body': body, 'proof_body_sha256': digest(body),
            'output_extraction': extraction, 'verification': evidence,
            'failure_category': None if evidence['status'] == 'PASS' else evidence['category'],
            'format_compliance': format_compliance(example['formal_statement'], text),
            'requested_proof_steps': step_adherence(example, body, condition)} | generated


def generate_rows(folder, model_key, rows, engine, vocabulary, adapter_sha):
    with (folder / 'raw_generations.jsonl').open('x') as raw, (folder / 'attempts.jsonl').open('x') as journal:
        for e in rows:
            key = f'holdout_v2:{model_key}:{e["id"]}'
            journal.write(json.dumps({'key': key, 'event': 'started', 'attempt': 1, 'at_utc': now()}) + '\n')
            journal.flush()
            print('Generating ' + key, flush=True)
            text, generated = generate(engine, vocabulary, e['formal_statement'], prompt_for(e))
            row = generation_record(model_key, e, text, generated, adapter_sha)
            raw.write(json.dumps(row, ensure_ascii=False) + '\n'); raw.flush()
            journal.write(json.dumps({'key': key, 'event': 'finished', 'attempt': 1, 'at_utc': now()}) + '\n')
            journal.flush()
            print(f"{key}: {row['verification']['status']} / {row['verification']['category']}", flush=True)
            require(generated['generation_error'] is None, 'Runtime generation error recorded; no retry')
            require(row['verification']['category'] != 'ENVIRONMENT_ERROR', 'Verifier environment failure; no retry')


def worker(model_key, output=OUTPUT):
    from lora_training_v1.core import base_hashes, tensor_digest
    p = check_protocol()
    require(model_key in MODELS and (output / 'run_started.json').exists(), 'Invalid worker/run')
    require(not (output / 'interruption.json').exists(), 'Run is interrupted')
    index = MODELS.index(model_key)
    for previous in MODELS[:index]:
        require(read(output / previous / 'completion.json')['status'] == 'COMPLETE', 'Model order differs')
    require(all(not (output / later).exists() for later in MODELS[index+1:]), 'Model order differs')
    folder = output / model_key
    folder.mkdir(exist_ok=False)
    try:
        require(hardware() == p['hardware'], 'Hardware/runtime changed')
        engine = Inference(p['generation_settings'])
        require(engine.lock == p['model'] and engine.generation.to_dict() == p['effective_generation_config'], 'Inference settings differ')
        before = base_hashes(engine.model)
        named, adapter_before, adapter_sha = {}, {}, None
        if model_key != 'base':
            s = p['selected'][model_key]
            path = TRAINING / s['selected_adapter_path_relative_to_run']
            named, adapter_before = load_adapter(engine.model, path)
            require(adapter_before == read(path.parent / 'checkpoint.json')['adapter_tensor_hashes'], 'Loaded adapter differs from selected checkpoint')
            adapter_sha = s['selected_adapter_sha256']
        engine.model.requires_grad_(False)
        engine.model.eval()
        require(base_hashes(engine.model) == before, 'Adapter attachment changed base')
        require(sum(t.numel() for t in engine.model.parameters() if t.requires_grad) == 0, 'Inference parameters must be frozen')
        random.seed(p['generation_settings']['seed'])
        np.random.seed(p['generation_settings']['seed'])
        engine.torch.manual_seed(p['generation_settings']['seed'])
        vocabulary = Vocabulary(engine.tokenizer)
        write(folder / 'manifest.json', {'model_key': model_key, 'runtime': engine.runtime,
              'model': engine.lock, 'generation_settings': p['generation_settings'],
              'effective_generation_config': engine.generation.to_dict(),
              'protocol_sha256': sha256_file(FROZEN), 'expected_attempts': 6,
              'checkpoint_adapter_sha256': adapter_sha, 'base_hashes_before': before,
              'adapter_hashes_before': adapter_before, 'trainable_parameters': 0,
              'vocabulary_policy': 'Fresh per model process; no reference token-path prewarming'})
        generate_rows(folder, model_key, opened_rows(output, p), engine, vocabulary, adapter_sha)
        after = base_hashes(engine.model)
        adapter_after = {n: tensor_digest(t) for n, t in named.items()}
        require(before == after and adapter_before == adapter_after, 'Inference changed weights')
        require(check_model() == p['model'], 'On-disk model changed')
        check_protocol()
        write(folder / 'completion.json', {'status': 'COMPLETE', 'model_key': model_key,
              'attempts': 6, 'optimizer_updates': 0, 'weights_unchanged': True,
              'base_hashes_after': after, 'adapter_hashes_after': adapter_after,
              'files_sha256': sealed_files(folder)})
    except BaseException as error:
        write(folder / 'interruption.json', {'status': 'INTERRUPTED', 'error': str(error),
              'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise


def launch(model_key):
    child = subprocess.Popen([sys.executable, '-B', '-m', 'pilot_holdout_v2.experiment',
                              'worker', '--model', model_key], cwd=ROOT, start_new_session=True)
    try:
        require(child.wait() == 0, 'Worker failed; no retries or later models')
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        raise


def run():
    require(not OUTPUT.exists(), 'Run already exists; refusing resume, overwrite or retry')
    p = check_protocol()
    check_training()
    require(hardware() == p['hardware'] and check_model() == p['model'], 'Pre-run environment differs')
    verifier_smoke()
    bound = p['code_sha256'] | {str(FROZEN.relative_to(ROOT)): sha256_file(FROZEN)}
    bound.update({str((PREFLIGHT / name).relative_to(ROOT)): value for name, value in p['preflight_sha256'].items()})
    commit = committed(bound)
    OUTPUT.mkdir()
    started = time.perf_counter()
    write(OUTPUT / 'run_started.json', {'started_at_utc': now(), 'runner_commit': commit,
          'protocol_sha256': sha256_file(FROZEN), 'expected_attempts': 18, 'optimizer_updates': 0})
    try:
        open_original_once(OUTPUT, p)
        for model_key in MODELS:
            launch(model_key)
        check_protocol()
        write(OUTPUT / 'generation_complete.json', {'status': 'COMPLETE_AWAITING_REVIEWS',
              'attempts': 18, 'model_order': list(MODELS), 'optimizer_updates': 0,
              'finished_at_utc': now(), 'monotonic_seconds': time.perf_counter() - started,
              'protocol_sha256': sha256_file(FROZEN), 'files_sha256': sealed_files(OUTPUT)})
    except BaseException as error:
        write(OUTPUT / 'interruption.json', {'status': 'INTERRUPTED', 'error': str(error),
              'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise
    return {'status': 'COMPLETE_AWAITING_REVIEWS', 'new_generations': 18, 'optimizer_updates': 0}


def preflight():
    require(not PREFLIGHT.exists() and not FROZEN.exists() and not OUTPUT.exists(), 'Preflight/freeze/run exists')
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_pilot_holdout_v2.py')
    with patch.object(Inference, '__init__', side_effect=AssertionError('Qwen weight loading forbidden in public preflight')), redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    require(result.wasSuccessful(), log.getvalue())
    check_training()
    hardware()
    verifier_smoke()
    PREFLIGHT.mkdir()
    with (PREFLIGHT / 'test_output.txt').open('x') as f:
        f.write(log.getvalue())
    write(PREFLIGHT / 'preflight.json', {'status': 'PASS', 'completed_at_utc': now(),
          'tests_run': result.testsRun, 'failures': 0, 'errors': 0, 'skipped': len(result.skipped),
          'code_sha256': code_hashes(), 'holdout_content_reads': 0,
          'model_generations': 0, 'model_weight_loads': 0,
          'public_reference_fixtures': 8, 'scope': 'New runner tests, frozen training audit, public Rocq/token fixtures only'})
    return {'status': 'PASS', 'tests': result.testsRun, 'holdout_content_reads': 0, 'model_generations': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'freeze', 'check-ready', 'run', 'worker', 'check', 'report', 'freeze-results'))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--replay-tokens', action='store_true')
    parser.add_argument('--reverify', action='store_true')
    args = parser.parse_args()
    with holdout_io():
        if args.action == 'preflight': result = preflight()
        elif args.action == 'freeze': result = freeze()
        elif args.action == 'check-ready':
            check_protocol(); result = {'status': 'READY', 'holdout_content_reads': 0}
        elif args.action == 'run': result = run()
        elif args.action == 'worker': worker(args.model); return
        else:
            from pilot_holdout_v2.assessment import check, report, freeze_results
            if args.action == 'check': result = check(replay_tokens=args.replay_tokens, reverify=args.reverify)
            elif args.action == 'report': result = report()
            else: result = freeze_results()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
