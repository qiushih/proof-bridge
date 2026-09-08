"""Six dev attempts on an in-memory checkpoint; no model reload or repairs."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import random
from types import SimpleNamespace

import numpy as np
import torch
from transformers import GenerationConfig

from baseline.dataset import digest
from baseline.protocol import extract_body
from constrained_v1.experiment import generate
from constrained_v1.tokens import Vocabulary
from lora_training_v1.core import audit_trainable, tensor_digest
from pilot_v1.protocol import ROOT
from prompt_v2.protocol import format_compliance
from training_protocol_v1.experiment import dev_groups, messages
from training_protocol_v1.scoring import mathematical_structure, step_adherence
from verifier import sha256_file, verify


@contextmanager
def evaluation_scope(model):
    """Evaluation cannot consume the subsequent training RNG stream."""
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    training = model.training
    model.eval()
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        model.train(training)


def make_engine(model, tokenizer):
    saved = json.loads((ROOT / 'results/pretraining-dev-v1/manifest.json').read_text())
    settings = json.loads((ROOT / 'baseline/config.json').read_text())['generation']
    generation = GenerationConfig(do_sample=False, num_beams=1,
        max_new_tokens=settings['max_new_tokens'], use_cache=settings['use_cache'],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id)
    if generation.to_dict() != saved['effective_generation_config'] or settings != saved['generation_settings']:
        raise ValueError('Inference configuration differs from frozen baseline')
    return SimpleNamespace(model=model, tokenizer=tokenizer, torch=torch, generation=generation)


class CheckpointEvaluator:
    def __init__(self, output):
        self.output = output
        self.vocabulary = None
        self.completed_steps = []

    def __call__(self, model, tokenizer, checkpoint, step):
        expected = 6 * (len(self.completed_steps) + 1)
        if step != expected or step > 60:
            raise ValueError('Checkpoint order changed or repeated')
        destination = self.output / f'dev-step-{step:04d}'
        destination.mkdir(exist_ok=False)
        named, audit = audit_trainable(model)
        before = {name:tensor_digest(p) for name,p in named.items()}
        engine = make_engine(model, tokenizer)
        if self.vocabulary is None:
            self.vocabulary = Vocabulary(tokenizer)
        rows = []
        with evaluation_scope(model), (destination/'raw_generations.jsonl').open('x') as raw, (destination/'attempts.jsonl').open('x') as journal:
            for group in dev_groups():
                for argument in ('A','B'):
                    example = group['arguments'][argument]
                    condition = 'argument_' + argument
                    key = f"pilot_training_v1:step-{step:04d}:{example['id']}"
                    journal.write(json.dumps({'key':key, 'event':'started', 'attempt':1, 'at_utc':datetime.now(timezone.utc).isoformat()})+'\n')
                    journal.flush()
                    print(f'Generating {key}', flush=True)
                    text, generated = generate(engine, self.vocabulary, example['formal_statement'], messages(group,condition))
                    body, extraction = extract_body(text)
                    evidence = asdict(verify(example['formal_statement'],body,timeout=10.0))
                    row = {'key':key, 'optimizer_step':step, 'theorem_id':group['theorem_id'],
                        'dev_example_id':example['id'], 'condition':condition,
                        'formal_statement':example['formal_statement'], 'attempt':1, 'repair_attempts':0,
                        'checkpoint_adapter_sha256':sha256_file(checkpoint/'adapter.safetensors'),
                        'generated_text':text, 'proof_body':body, 'proof_body_sha256':digest(body),
                        'output_extraction':extraction, 'verification':evidence,
                        'failure_category':None if evidence['status']=='PASS' else evidence['category'],
                        'format_compliance':format_compliance(example['formal_statement'],text),
                        'requested_proof_steps':step_adherence(example,body,condition),
                        'mathematical_structure_review_aid':mathematical_structure(example['formal_statement'],body)} | generated
                    raw.write(json.dumps(row,ensure_ascii=False)+'\n'); raw.flush()
                    journal.write(json.dumps({'key':key,'event':'finished','attempt':1})+'\n'); journal.flush()
                    rows.append(row)
                    print(f"{key}: {evidence['status']} {evidence['category']}; steps {row['requested_proof_steps']['status']}",flush=True)
                    if generated['generation_error']:
                        raise RuntimeError('Generation runtime error recorded; no retry or altered run')
        if before != {name:tensor_digest(p) for name,p in named.items()}:
            raise ValueError('Evaluation changed an adapter')
        audit_trainable(model)
        (destination/'completion.json').write_text(json.dumps({'status':'COMPLETE','optimizer_step':step,
            'attempts':len(rows), 'adapter_weights_unchanged_by_evaluation':True, 'trainable_audit':audit,
            'files_sha256':{p.name:sha256_file(p) for p in destination.iterdir() if p.is_file()}},indent=2)+'\n')
        self.completed_steps.append(step)
