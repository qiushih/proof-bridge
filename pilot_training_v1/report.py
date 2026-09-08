"""Compare all reviewed dev checkpoints using the unchanged selection rule."""

import json
import statistics

from pilot_training_v1.experiment import OUTPUT, ROOT, check, read_rows
from training_protocol_v1.scoring import attach_reviews, selection_score
from verifier import sha256_file


def metrics(rows):
    passed=lambda r:r['verification']['status']=='PASS'
    return {'rows':len(rows),'verified':sum(passed(r) for r in rows),
        'verified_mathematically_faithful':sum(passed(r) and r['mathematical_argument_fidelity']['status']=='FAITHFUL' for r in rows),
        'attempted_step_matches':sum(r['requested_proof_steps']['matched'] is True for r in rows),
        'verified_requested_steps':sum(passed(r) and r['requested_proof_steps']['matched'] is True for r in rows),
        'mean_generation_latency_seconds':statistics.mean(r['generation_latency_seconds'] for r in rows)}


def select(checkpoints):
    scores={step:selection_score(rows,step) for step,rows in checkpoints.items()}
    if set(scores)!=set(range(0,61,6)):
        raise ValueError('All ten checkpoints and step zero are required; no early selection')
    selected=max(scores,key=lambda step:scores[step])
    return selected,scores


def build_report():
    manifest,completion,updates,raw=check()
    baseline=[json.loads(line) for line in (ROOT/manifest['step_zero_source']).read_text().splitlines()]
    baseline=[r for r in baseline if r['condition']!='theorem_only']
    checkpoints={0:baseline}
    assessed=[]
    for step in range(6,61,6):
        folder=OUTPUT/f'dev-step-{step:04d}'
        reviews=json.loads((folder/'manual_reviews.json').read_text())
        if reviews['raw_generations_sha256']!=sha256_file(folder/'raw_generations.jsonl'):
            raise ValueError('Unbound checkpoint reviews')
        if reviews['human_reviewed'] or reviews['reviewer']!='assistant':
            raise ValueError('These are assistant reviews, not independent human reviews')
        rows=attach_reviews(read_rows(folder),reviews['reviews'])
        checkpoints[step]=rows
        assessed.extend(rows)
    selected,scores=select(checkpoints)
    baseline_metrics=metrics(baseline)
    selected_metrics=metrics(checkpoints[selected])
    pairs=[]
    for theorem in ('pd01','pd02','pd03'):
        rows=[r for r in checkpoints[selected] if r['theorem_id']==theorem]
        pairs.append({'theorem_id':theorem,'both_verified_and_step_adherent':all(r['verification']['status']=='PASS' and r['requested_proof_steps']['matched'] is True for r in rows),
            'body_changed':rows[0]['proof_body']!=rows[1]['proof_body']})
    comparison={key:selected_metrics[key]-baseline_metrics[key] for key in ('verified','verified_mathematically_faithful','verified_requested_steps')}
    improves=(comparison['verified_requested_steps']>0 and comparison['verified']>=0 and comparison['verified_mathematically_faithful']>=0)
    checkpoint_path=OUTPUT/f'training/step-{selected:04d}/adapter.safetensors' if selected else None
    summary={'experiment_id':manifest['experiment_id'],'selected_step':selected,
        'selected_adapter_path':str(checkpoint_path.relative_to(ROOT)) if checkpoint_path else None,
        'selected_adapter_sha256':sha256_file(checkpoint_path) if checkpoint_path else None,
        'selection_score':scores[selected],'baseline_selection_score':scores[0],
        'selection_score_improved':scores[selected]>scores[0],
        'proof_step_adherence_improved_without_verification_or_fidelity_loss':improves,
        'baseline':baseline_metrics,'selected':selected_metrics,'count_changes':comparison,
        'checkpoints':[{'optimizer_step':s,'selection_score':scores[s],**metrics(checkpoints[s])} for s in sorted(checkpoints)],
        'selected_step_pairs':pairs, 'optimizer_updates':60,'training_microbatches':240,'new_dev_generations':60,
        'training_wall_seconds':sum(u['update_wall_seconds'] for u in updates),
        'total_run_wall_seconds':completion['wall_seconds'],
        'epoch_mean_training_loss':[statistics.mean(u['mean_completion_loss'] for u in updates[i:i+6]) for i in range(0,60,6)],
        'base_weights_unchanged':completion['base_weights_unchanged'],'holdout_content_reads':0,
        'holdout_evaluated':False,'human_reviewed':False,
        'limitations':['Six correlated dev rows, three theorems in one family; selected on these same rows, not an independent estimate.',
            'A/B distinguish equivalent IH tactic realizations, not different high-level mathematical strategies.',
            'Assistant fidelity review, not independent human review.',
            'No new theorem-only or diagnostic generations; no claims about held-out generalization.']}
    answer='Yes, on the six selection-dev rows.' if improves else 'No improvement meeting all three criteria on the six selection-dev rows.'
    lines=['# First LoRA pilot v1','',f'**Did proof-step adherence improve while preserving verification and mathematical fidelity? {answer}**','',
        f"Selected **optimizer step {selected}**, using the unchanged lexicographic dev rule. Selected score: `{scores[selected]}`; step-zero score: `{scores[0]}`. All ten checkpoints were evaluated before selection; ties choose the earliest step, including zero.",'',
        '| Metric | Frozen pre-training baseline | Selected checkpoint |','| --- | ---: | ---: |']
    for label,key in [('Rocq verification','verified'),('Verified + mathematically faithful','verified_mathematically_faithful'),('Verified + requested proof steps','verified_requested_steps')]:
        lines.append(f"| {label} | {baseline_metrics[key]}/6 ({100*baseline_metrics[key]/6:.1f}%) | {selected_metrics[key]}/6 ({100*selected_metrics[key]/6:.1f}%) |")
    lines+=['','## Checkpoint comparison','','| Step | Verified | Verified + faithful | Verified + requested steps | Selection score |','| ---: | ---: | ---: | ---: | --- |']
    for c in summary['checkpoints']:
        lines.append(f"| {c['optimizer_step']} | {c['verified']}/6 | {c['verified_mathematically_faithful']}/6 | {c['verified_requested_steps']}/6 | `{c['selection_score']}` |")
    lines+=['','## Selected proofs and reviews','','| Dev row | Rocq | Mathematical fidelity | Requested steps |','| --- | --- | --- | --- |']
    for r in checkpoints[selected]:
        lines.append(f"| {r['dev_example_id']} | {r['verification']['status']} | {r['mathematical_argument_fidelity']['status']} | {r['requested_proof_steps']['status']} |")
    lines+=['',f"Both A/B step contracts were verified on **{sum(p['both_verified_and_step_adherent'] for p in pairs)}/3** selected theorem pairs. Equivalent IH rewriting and `f_equal` followed by `exact`/`apply` can remain mathematically faithful even when the strict step contract differs.",'',
        '## Training and custody','',
        'Fresh seeded rank-8 adapters were attached to the pinned Qwen2.5-Coder-0.5B-Instruct base; feasibility updates were not loaded. The unchanged loop trained on the frozen 24 rows for 10 complete epochs: 240 batch-size-one microbatches, accumulation four, exactly 60 AdamW updates. All 540,672 trainable parameters belong to query/value LoRA modules. Base weights remained unchanged. Ten adapter-only checkpoints retain optimizer/scheduler/RNG state; no base weights were overwritten.',
        '',f"Training-update time: **{summary['training_wall_seconds']:.2f}s**. Total run time including loading, checkpoint I/O, integrity checks and dev evaluation: **{summary['total_run_wall_seconds']:.2f}s**. Peak process RSS: **{completion['final_process_memory']['peak_rss_bytes']/1024**3:.2f} GiB**.",'',
        'Each checkpoint received exactly six one-attempt argument-conditioned generations, with frozen Prompt v2, constrained-v1, greedy decoding, 256-token budget, and no repair. Evaluation saved/restored training RNG state. Step zero reuses the original six argument-conditioned outputs; no baseline regeneration or theorem-only control enters selection.',
        '', 'All 60 generated candidates have raw prompts, token IDs, constraint traces, latency, Rocq evidence and separate hash-bound assistant fidelity reviews. The sealed holdout was only byte-hashed for integrity; its content was not inspected, generated on or scored. Historical artifacts remain unchanged.','',
        '## Reproduction','', 'Run from the repository root in the pinned environment. The completed run is checked without training or new generations:', '', '```sh',
        '.venv/bin/python -m pilot_training_v1.experiment check --replay-tokens --reverify',
        '.venv/bin/python -m pilot_training_v1.report',
        ".venv/bin/python -m unittest discover -s tests -p 'test_pilot_training_v1.py' -v",
        '.venv/bin/python -m unittest discover -s tests -v','```','',
        'The original one-time command was `.venv/bin/python -m pilot_training_v1.experiment run`. It refuses an existing run; do not delete evidence to rerun under the same identity. `selected_checkpoint.json` identifies the selected adapter without copying or merging it. Protocol, dataset, prompt, decoder and checkpoint-selection code are unchanged.','', '## Limits','']
    lines += ['- '+s for s in summary['limitations']]
    lines.append('')
    return '\n'.join(lines),summary,assessed


if __name__=='__main__':
    text,summary,rows=build_report()
    outputs={OUTPUT/'LORA_PILOT_REPORT.md':text,OUTPUT/'summary.json':json.dumps(summary,indent=2)+'\n',
        OUTPUT/'assessed_results.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),
        OUTPUT/'selected_checkpoint.json':json.dumps({key:summary[key] for key in ('selected_step','selected_adapter_path','selected_adapter_sha256','selection_score','baseline_selection_score')},indent=2)+'\n'}
    for path,body in outputs.items():
        if path.exists() and path.read_text()!=body: raise ValueError('Refusing to overwrite a different selection/report')
        path.write_text(body)
    print(f"Selected step {summary['selected_step']}: {summary['selection_score']}; baseline {summary['baseline_selection_score']}")
