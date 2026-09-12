"""Eight dev attempts per checkpoint; frozen decoder, no repairs or holdout loader."""
from dataclasses import asdict
import json
import random
import numpy as np
import torch

from baseline.protocol import extract_body
from constrained_v1.experiment import generate
from constrained_v1.tokens import Vocabulary
from lora_training_v1.core import audit_trainable, tensor_digest
from pilot_training_v1.evaluation import evaluation_scope, make_engine
from prompt_v2.protocol import build_messages, format_compliance
from training_protocol_v1.scoring import mathematical_structure, step_adherence
from pilot_training_v2.common import (ROOT, DEV_IDS, read, require, write, now, digest,
                                     dev_rows, sealed_files, sha256_file)


class CheckpointEvaluator:
    def __init__(self, output, arm, tokenizer):
        self.output, self.arm, self.tokenizer = output, arm, tokenizer
        self.examples = dev_rows()
        self.prompt = read(ROOT / 'prompt_v2/selected.json')
        self.vocabulary = None
        self.completed_steps = []

    def __call__(self, model, checkpoint, step):
        require(step == 6 * (len(self.completed_steps) + 1) and step <= 60, 'Checkpoint repeated or out of order')
        folder = self.output / f'dev-step-{step:04d}'
        folder.mkdir(exist_ok=False)
        named, audit = audit_trainable(model)
        before = {n: tensor_digest(p) for n, p in named.items()}
        require(before == read(checkpoint / 'checkpoint.json')['adapter_tensor_hashes'], 'In-memory checkpoint differs from saved adapter')
        engine = make_engine(model, self.tokenizer)
        if self.vocabulary is None:
            self.vocabulary = Vocabulary(self.tokenizer)
        rows = []
        with evaluation_scope(model), (folder / 'raw_generations.jsonl').open('x') as raw, (folder / 'attempts.jsonl').open('x') as journal:
            # Restore the training RNG stream on exit, even after failure.
            random.seed(1729); np.random.seed(1729); torch.manual_seed(1729)
            for example in self.examples:
                condition = 'argument_' + example['argument_id']
                key = f"pilot_training_v2:{self.arm}:step-{step:04d}:{example['id']}"
                journal.write(json.dumps({'key': key, 'event': 'started', 'attempt': 1, 'at_utc': now()}) + '\n'); journal.flush()
                print('Generating ' + key, flush=True)
                text, generated = generate(engine, self.vocabulary, example['formal_statement'],
                                            build_messages(self.prompt, example, 'theorem_and_informal'))
                body, extraction = extract_body(text)
                evidence = asdict(verify_candidate(example['formal_statement'], body))
                row = {'key': key, 'arm': self.arm, 'optimizer_step': step,
                       'theorem_id': example['theorem_id'], 'dev_example_id': example['id'],
                       'condition': condition, 'formal_statement': example['formal_statement'],
                       'attempt': 1, 'repair_attempts': 0,
                       'checkpoint_adapter_sha256': sha256_file(checkpoint / 'adapter.safetensors'),
                       'generated_text': text, 'proof_body': body, 'proof_body_sha256': digest(body),
                       'output_extraction': extraction, 'verification': evidence,
                       'failure_category': None if evidence['status'] == 'PASS' else evidence['category'],
                       'format_compliance': format_compliance(example['formal_statement'], text),
                       'requested_proof_steps': step_adherence(example, body, condition),
                       'mathematical_structure_review_aid': mathematical_structure(example['formal_statement'], body),
                       'requested_base_action_review_aid': example['argument_contract']['base']} | generated
                raw.write(json.dumps(row, ensure_ascii=False) + '\n'); raw.flush()
                journal.write(json.dumps({'key': key, 'event': 'finished', 'attempt': 1, 'at_utc': now()}) + '\n'); journal.flush()
                rows.append(row)
                print(f"{key}: {evidence['status']} / {evidence['category']}", flush=True)
                require(generated['generation_error'] is None, 'Runtime generation error recorded; no retry')
                require(evidence['category'] != 'ENVIRONMENT_ERROR', 'Verifier environment failure recorded; stop without retry')
        require(before == {n: tensor_digest(p) for n, p in named.items()}, 'Evaluation changed an adapter')
        audit_trainable(model)
        require([r['dev_example_id'] for r in rows] == DEV_IDS, 'Wrong dev attempts')
        write(folder / 'completion.json', {'status': 'COMPLETE', 'arm': self.arm, 'optimizer_step': step,
              'attempts': 8, 'adapter_weights_unchanged': True, 'adapter_hashes_before': before,
              'adapter_hashes_after': {n: tensor_digest(p) for n, p in named.items()},
              'trainable_audit': audit, 'files_sha256': sealed_files(folder)})
        self.completed_steps.append(step)


def verify_candidate(statement, body):
    from verifier import verify
    return verify(statement, body, timeout=10.0)
