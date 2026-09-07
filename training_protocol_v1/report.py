"""Assess saved baseline outputs; no model calls or holdout content access."""

import json

from baseline.dataset import write_json
from training_protocol_v1.experiment import OUTPUT, check
from training_protocol_v1.scoring import attach_reviews, summarize


def build_report():
    manifest, raw = check()
    reviews = json.loads((OUTPUT / "manual_reviews.json").read_text())
    rows = attach_reviews(raw, reviews["reviews"])
    summary = summarize(rows)
    lines = ["# Pre-training dev baseline v1", "",
        "Exactly **9 generations**: three frozen pilot dev theorems × theorem only / argument A / argument B. One attempt each, no repair, 256 tokens, frozen Prompt v2 and unchanged constrained-v1, model and verifier.", "",
        "The training protocol was frozen before inference. No adapters were attached to Qwen, no training forward/backward or optimizer step ran, and the base model files remain unchanged. The sealed holdout was not inspected or evaluated; integrity checks used existing byte hashes only.", "",
        "| Condition | Rocq verified | Verified + mathematically faithful | Requested steps matched | Verified + requested steps | Mean latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for condition, m in summary["by_condition"].items():
        fmt = lambda value: "N/A" if value is None else f"{value}/3"
        lines.append(f"| {condition} | {m['verified']}/3 | {fmt(m['verified_mathematically_faithful'])} | {fmt(m['requested_steps_matched'])} | {fmt(m['verified_requested_steps'])} | {m['mean_latency_seconds']:.2f}s |")
    total = summary["all_outputs"]
    lines += ["", f"Overall verification: **{total['verified']}/9 ({100*total['verification_accuracy']:.1f}%)**. On the six argument-conditioned rows: **{total['verified_mathematically_faithful']}/6** verified and mathematically faithful; **{total['verified_requested_steps']}/6** verified and step-adherent. The theorem-only controls have no supplied argument and are excluded from both fidelity denominators and checkpoint selection.", "",
        "## Per-output assessment", "",
        "| Theorem | Condition | Verification | Mathematical fidelity | Requested steps |",
        "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['theorem_id']} | {r['condition']} | {r['verification']['status']} / {r['verification']['category']} | {r['mathematical_argument_fidelity']['status']} | {r['requested_proof_steps']['status']} |")
    lines += ["", "## Fidelity review", "",
        "These are hash-bound assistant reviews, not independent human reviews. Rocq verification, mathematical argument fidelity and strict requested-step adherence are separate fields. Equivalent IH rewriting and successor congruence are accepted as the same mathematical induction. A proof can therefore be mathematically faithful while missing the requested tactic/step realization. Failed proofs cannot establish faithful complete arguments.", ""]
    for r in rows:
        if r["condition"] != "theorem_only":
            lines.append(f"- **{r['theorem_id']} {r['condition']}**: {r['mathematical_argument_fidelity']['reason']}")
    pairs = sum(p["both_verified_and_step_adherent"] for p in summary["step_switch_pairs"])
    changed = sum(p["body_changed"] for p in summary["step_switch_pairs"])
    lines += ["", f"A/B bodies changed on **{changed}/3** theorems. Both variants verified and followed their respective step contracts on **{pairs}/3**. This is a step-realization test; A and B share the same high-level induction argument.", "",
        f"The frozen step-zero checkpoint-selection tuple is **{summary['step_zero_selection_score']}**: verified faithful, verified step-adherent, verified, negative optimizer step. This is the comparator for the future fixed 60-update LoRA pilot, not evidence of a training improvement.", "",
        "## Reproduction and custody", "",
        "Raw generations, prompts, token IDs, constraint traces, latency and verifier evidence are in `raw_generations.jsonl`; one-attempt events are in `attempts.jsonl`. `assessed_results.jsonl` adds the separate reviewed mathematical judgments. `summary.json` is machine-readable. Protocol and results have separate freezes so the configuration demonstrably precedes inference.", "",
        "From the repository root:", "", "```sh",
        ".venv/bin/python -m training_protocol_v1.experiment check --replay-tokens --reverify",
        ".venv/bin/python -m unittest discover -s tests -p 'test_training_protocol_v1.py' -v",
        ".venv/bin/python -m unittest discover -s tests -p 'test_pretraining_v1_results.py' -v",
        ".venv/bin/python -m unittest discover -s tests -v", "```", "",
        "These commands check saved artifacts and replay token/verifier evidence; they do not generate additional samples. The original `run` command refuses an existing output directory, and there is no training command. The complete one-time sequence and frozen hyperparameters are in `training_protocol_v1/TRAINING_PROTOCOL.md`.", "",
        "## Limits", "",
        "Three correlated dev theorems in one family are a checkpoint-selection set, not an independent test. The protocol deliberately preserves the frozen prose and its minor grammatical defects. The planned Apple M2 / 8 GiB CPU training configuration has serialization and synthetic adapter/loss checks but has not been benchmarked with a Qwen backward pass. No statement about future training gains, memory peak or wall-clock training time is supported yet.", ""]
    return "\n".join(lines), rows, summary


if __name__ == "__main__":
    report, rows, summary = build_report()
    outputs = {OUTPUT / "PRETRAINING_DEV_BASELINE.md": report,
        OUTPUT / "assessed_results.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        OUTPUT / "summary.json": json.dumps(summary, indent=2, ensure_ascii=False) + "\n"}
    for path, text in outputs.items():
        if path.exists() and path.read_text() != text:
            raise ValueError("Refusing to replace different assessed results")
        path.write_text(text)
    print("Assessed nine saved outputs; zero new generations")
