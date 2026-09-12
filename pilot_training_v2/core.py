"""Fresh model setup and exact update-based stream; no import-time computation."""
import random
import time
import numpy as np
import torch
from safetensors.torch import save_file

from baseline.run import runtime_versions
from baseline.setup_model import MODEL_DIR, check_model
from lora_training_v1.core import (audit_trainable, base_hashes, make_optimizer,
                                   optimizer_update, seed_runtime, tensor_digest)
from lora_training_v1.telemetry import process_memory, swap_snapshot
from training_protocol_v1.adaptation import attach_for_future_training
from training_protocol_v1.data import completion_loss, encode_example
from pilot_training_v2.common import (ROOT, config, public_rows, read, require, write, now, sealed_files)


def setup(arm):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lock = check_model()
    seed_runtime()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.float32, attn_implementation='eager').to('cpu')
    before = base_hashes(model)
    torch.manual_seed(config()['training']['torch_seed'])
    attach_for_future_training(model)
    model.train()
    named, audit = audit_trainable(model)
    require(all(torch.count_nonzero(p).item() == 0 for n, p in named.items() if n.endswith('.B')),
            'Adapters must start with zero B, not a trained checkpoint')
    require(base_hashes(model) == before and model.is_gradient_checkpointing and model.config.use_cache is False,
            'Base or checkpointing configuration differs')
    prompt = read(ROOT / 'prompt_v2/selected.json')
    rows = public_rows(arm)
    encoded = {r['id']: encode_example(tokenizer, prompt, r) for r in rows}
    initial = {n: tensor_digest(p) for n, p in named.items()}
    runtime = {'arm': arm, 'model': lock, 'packages': runtime_versions(),
               'load_and_prepare_seconds': time.perf_counter() - start,
               'training_rows': len(encoded), 'trainable_audit': audit,
               'initial_adapter_hashes': initial, 'base_hashes_before': before,
               'fresh_adapter_initialization': True, 'feasibility_or_v1_adapter_loaded': False,
               'device': str(model.device), 'dtype': str(model.dtype),
               'torch_build': torch.__config__.show(), 'gradient_checkpointing': True,
               'gradient_checkpointing_use_reentrant': False, 'training_use_cache': False}
    return model, tokenizer, encoded, before, runtime


def matching_initial(control_runtime, intervention_runtime):
    require(control_runtime['initial_adapter_hashes'] == intervention_runtime['initial_adapter_hashes']
            and control_runtime['base_hashes_before'] == intervention_runtime['base_hashes_before'],
            'Arms did not start from identical base and adapter tensors')


def save_checkpoint(model, optimizer, scheduler, output, step, stream, arm, bindings, strict=True):
    require(step in range(6, 61, 6), 'Unexpected checkpoint step')
    require(len(stream) == 240 and len(set(stream)) in (24, 26), 'Checkpoint needs entire frozen stream')
    named, audit = audit_trainable(model, strict=strict)
    require(all(torch.isfinite(p).all() for p in named.values()), 'Non-finite adapter checkpoint')
    path = output / f'step-{step:04d}'
    path.mkdir(parents=True, exist_ok=False)
    save_file({n: p.detach().contiguous() for n, p in named.items()}, str(path / 'adapter.safetensors'))
    consumed = 4 * step
    row_count = len(set(stream))
    state = {'arm': arm, 'optimizer_step': step, 'completed_microbatches': consumed,
             'next_microbatch_index': consumed, 'completed_cycles': consumed // row_count,
             'next_cycle_offset': consumed % row_count, 'microbatch_stream': stream,
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(), 'bindings': bindings}
    torch.save(state, path / 'trainer_state.pt')
    write(path / 'checkpoint.json', {'status': 'COMPLETE', 'arm': arm, 'optimizer_step': step,
          'completed_microbatches': consumed, 'next_microbatch_index': consumed,
          'completed_cycles': state['completed_cycles'], 'next_cycle_offset': state['next_cycle_offset'],
          'adapter_parameters_only': True, 'base_weights_saved': False, 'trainable_audit': audit,
          'adapter_tensor_hashes': {n: tensor_digest(p) for n, p in named.items()},
          'bindings': bindings, 'files_sha256': sealed_files(path)})
    return path


def run_updates(model, optimizer, scheduler, encoded, schedule, output, arm, before,
                bindings, on_checkpoint, strict=True, loss_fn=completion_loss):
    """One stream, no epoch truncation; test fixtures can substitute a tiny loss/model."""
    stream, groups = schedule['microbatches'], schedule['updates']
    require(arm in ('control', 'intervention') and len(groups) == 60 and len(stream) == 240
            and all(len(g) == 4 for g in groups) and sum(groups, []) == stream
            and set(stream) == set(encoded), 'Invalid stream or accumulation groups')
    require(not optimizer.state and scheduler.last_epoch == 0, 'Optimizer/scheduler must be fresh')
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    updates = []
    with (output / 'training_metrics.jsonl').open('x') as log:
        def emit(event, **values):
            record = {'event': event, 'arm': arm, 'at_utc': now(), **values}
            if event == 'optimizer_finished':
                record['memory'] = process_memory()
            log.write(__import__('json').dumps(record) + '\n'); log.flush()
        emit('training_started', expected_optimizer_updates=60, expected_microbatches=240,
             memory=process_memory(), system_swap=swap_snapshot())
        for step, ids in enumerate(groups, 1):
            result = optimizer_update(model, optimizer, scheduler, [encoded[i] for i in ids], emit,
                                      step, strict=strict, loss_fn=loss_fn)
            require(all(torch.isfinite(p).all() for p in audit_trainable(model, strict)[0].values()),
                    'Optimizer created a non-finite adapter')
            updates.append(result)
            if step % 6 == 0:
                require(base_hashes(model) == before, 'Base weights changed during training')
                checkpoint = save_checkpoint(model, optimizer, scheduler, output, step, stream, arm, bindings, strict)
                on_checkpoint(model, checkpoint, step)
                require(base_hashes(model) == before, 'Evaluation changed base weights')
                require(model.training, 'Evaluation did not restore training mode')
                emit('checkpoint_evaluated', optimizer_step=step, completed_microbatches=4*step,
                     memory=process_memory(), system_swap=swap_snapshot())
        require(base_hashes(model) == before, 'Final base weights changed')
        emit('training_completed', optimizer_updates=60, microbatches=240, base_weights_unchanged=True)
    return {'optimizer_updates': len(updates), 'microbatches': 240,
            'wall_seconds': time.perf_counter() - start, 'base_hashes_after': base_hashes(model),
            'final_memory': process_memory(), 'final_swap': swap_snapshot()}
