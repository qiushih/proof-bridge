"""Classify measured feasibility; do not train, generate, or open holdout."""

import json
import statistics

from lora_training_v1.preflight import OUTPUT, check


def classify(result, completion, plan):
    if result.get("failure_kind")=="MEMORY":
        return "infeasible due to memory",None
    if result.get("status")=="PASS" and result.get("base_parameters_unchanged"):
        native=[u["update_wall_seconds"] for u in result["updates"] if u["pad_to"] is None]
        seconds=statistics.median(native)
        return ("feasible but impractically slow" if seconds>plan["slow_threshold_native_update_seconds"] else "feasible"),seconds
    if completion["timed_out"] and len(result.get("updates",[]))>=2:
        return "feasible but impractically slow",None
    # Never mislabel an implementation error or unexplained external kill as OOM.
    raise ValueError("No defensible feasibility class; diagnose the incomplete run")


def build_report():
    manifest,completion=check()
    result=json.loads((OUTPUT/"worker_result.json").read_text()) if (OUTPUT/"worker_result.json").exists() else {}
    samples=[json.loads(line) for line in (OUTPUT/"memory_samples.jsonl").read_text().splitlines()]
    classification,native_seconds=classify(result,completion,manifest["plan"])
    peak_rss=max(s["process"]["peak_rss_bytes"] for s in samples)
    footprint=[s["process"].get("peak_physical_footprint_bytes") for s in samples]
    peak_footprint=max(v for v in footprint if v is not None) if any(v is not None for v in footprint) else None
    swap_start=manifest["initial_system_swap"].get("used_bytes")
    swap_end=completion["final_system_swap"].get("used_bytes")
    swap_samples=[s["swap"].get("used_bytes") for s in samples if s["swap"].get("available")]
    swap_peak=max(swap_samples+[v for v in (swap_start,swap_end) if v is not None]) if swap_samples else None
    summary={"classification":classification,"completed_optimizer_updates":result.get("optimizer_updates_completed",0),
        "model_loaded":result.get("model_loaded",False),"longest_training_example":result.get("longest_training_example"),
        "all_forward_backward_optimizer_steps_passed":result.get("status")=="PASS",
        "peak_rss_bytes":peak_rss,"peak_physical_footprint_bytes":peak_footprint,
        "system_swap_start_bytes":swap_start,"system_swap_peak_bytes":swap_peak,"system_swap_end_bytes":swap_end,
        "system_swap_peak_increase_bytes":swap_peak-swap_start if swap_peak is not None and swap_start is not None else None,
        "native_update_median_seconds":native_seconds,"projected_60_update_seconds":native_seconds*60 if native_seconds else None,
        "total_worker_wall_seconds":result.get("wall_seconds",completion["wall_seconds"]),
        "base_parameters_unchanged":result.get("base_parameters_unchanged"),"base_model_files_unchanged":completion["base_model_files_unchanged"],
        "updates":result.get("updates",[]),"trained_checkpoints_saved":0,"holdout_content_reads":0,"full_pilot_started":False,
        "limitation":"Small CPU feasibility run on a loaded shared-memory host; global swap cannot be attributed solely to this worker. Right padding tests tensor allocation, not a new proof or 768 active content tokens."}
    gib=lambda n:"unavailable" if n is None else f"{n/1024**3:.2f} GiB"
    longest=summary["longest_training_example"] or {}
    lines=["# M2 LoRA feasibility v1","",f"**Classification: {classification}.**","",
        "Apple M2 MacBook Air (Mac14,2), 8 GiB RAM; CPU/float32 and the unchanged frozen training configuration. This is a disposable feasibility run, not the 60-update ProofBridge pilot.","",
        f"Completed **{summary['completed_optimizer_updates']}/3 optimizer updates**, each accumulating four batch-size-one microbatches. The loader used the fixed 24-row training pool. No dev generations or holdout content reads occurred, and no feasibility adapter or optimizer tensors were retained.","",
        "| Check | Result |","| --- | --- |",
        f"| Pinned model load | {'PASS' if summary['model_loaded'] else 'FAIL'} |",
        f"| Longest actual training row | {longest.get('id','unavailable')}, {longest.get('native_length','unavailable')} tokens including target/EOS |",
        f"| 768-token tensor, forward, finite loss, backward, optimizer | {'PASS' if result.get('status')=='PASS' else result.get('error','incomplete')} |",
        f"| Trainable weights | {result.get('runtime',{}).get('trainable_audit',{}).get('trainable_parameters','unavailable')} LoRA parameters; query/value only |",
        f"| Base parameters unchanged in memory | {summary['base_parameters_unchanged']} |",
        f"| Original model files unchanged | {summary['base_model_files_unchanged']} |",
        f"| Peak worker RSS | {gib(peak_rss)} |",
        f"| Peak charged physical footprint | {gib(peak_footprint)} |",
        f"| System swap: initial / peak / final | {gib(swap_start)} / {gib(swap_peak)} / {gib(swap_end)} |",
        f"| Peak system swap increase above initial | {gib(summary['system_swap_peak_increase_bytes'])} |","",
        "## Per-update timings","","| Update | Input | Mean target loss | Forward total | Backward total | Optimizer | Update wall time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for u in result.get("updates",[]):
        fwd=sum(t["forward_seconds"] for t in u["microbatch_timings"])
        bwd=sum(t["backward_seconds"] for t in u["microbatch_timings"])
        kind="longest row, native" if u["update"]==1 else "longest row, padded to 768" if u["update"]==2 else "first four frozen epoch-zero rows"
        lines.append(f"| {u['update']} | {kind} | {u['mean_completion_loss']:.5f} | {fwd:.2f}s | {bwd:.2f}s | {u['optimizer_seconds']:.3f}s | {u['update_wall_seconds']:.2f}s |")
    if native_seconds is not None:
        lines += ["",f"The median of the two native-length update measurements is **{native_seconds:.2f}s**, projecting approximately **{native_seconds:.1f} minutes for 60 updates**, excluding dev evaluation, checkpoint I/O and loading. The predeclared practicality threshold is 120 seconds/update (two hours for 60 updates). This is a small-sample estimate under the observed host load, not a training-time guarantee."]
    lines += ["",f"Worker wall time, including loading, integrity hashing and telemetry: **{summary['total_worker_wall_seconds']:.2f}s**. Model-loading time is recorded separately in `worker_result.json`.","",
        "## What was exercised","",
        "The actual loop implements deterministic seeding, frozen AdamW settings, mean completion-only loss divided by four for accumulation, finite-gradient checks, clipping at norm 1.0, optimizer stepping, constant scheduling and JSONL event metrics. The frozen LoRA helper is reused unchanged. Only 96 adapter tensors (540,672 parameters) are trainable; all base gradients remain absent. Streaming SHA-256 checks compare every deduplicated base parameter before attachment and after updates without cloning the large embedding. Original model-file hashes are also checked before and after.","",
        "No training example is naturally 768 tokens. The memory-stress forward right-pads the longest original input tensor to 768 using EOS padding with attention zero. The unchanged completion-only loss still selects only original proof/EOS prediction positions. It does not alter the frozen prompt, proof text, split or dataset. Padding is not supervised, and this does not test a hypothetical 768-token unpadded proof.","",
        "The future full epoch loop and adapter-only checkpoint saver are implemented in `lora_training_v1/core.py`. A future authorized caller must supply the frozen six-dev-row checkpoint assessment callback; the existing checkpoint-selection rule remains unchanged. There is intentionally no full-pilot CLI entry here. Checkpoint saving was tested with synthetic layers in temporary directories, but the real feasibility worker never calls it and discards its adapters on exit.","",
        "## Measurement limits","",
        "RSS comes from Darwin getrusage in bytes. Physical footprint comes from proc_pid_rusage RUSAGE_INFO_V4 and includes charged compressed memory. Both are lifetime maxima for the worker, including loading and checksum passes. Two-second samples also capture current footprint and system-wide swap; lifetime maxima do not depend on catching the exact peak. The host was already swapping before the run, and other applications can change the global swap readings. Do not interpret system swap as per-process swap or extrapolate these losses as model quality.","",
        "## Reproduce without extra updates","","```sh",
        ".venv/bin/python -m lora_training_v1.preflight check",
        ".venv/bin/python -m lora_training_v1.report",
        ".venv/bin/python -m unittest discover -s tests -p 'test_lora_training_v1.py' -v",
        ".venv/bin/python -m unittest discover -s tests -v","```","",
        "The original one-time measurement command was `.venv/bin/python -m lora_training_v1.preflight run`. It requires macOS hardware/swap telemetry permission and refuses an existing output directory, so rechecking cannot accidentally add updates. Plan, source hashes, actual hardware, raw event/memory samples, all losses and parameter hashes are retained. Do not delete the record to repeat this run under the same identity. No full training pilot has started.",""]
    return "\n".join(lines),summary


if __name__=="__main__":
    text,summary=build_report()
    for path,body in ((OUTPUT/"M2_FEASIBILITY_REPORT.md",text),(OUTPUT/"assessment.json",json.dumps(summary,indent=2)+"\n")):
        if path.exists() and path.read_text()!=body:
            raise ValueError("Refusing to overwrite different assessment")
        path.write_text(body)
    print(summary["classification"])
