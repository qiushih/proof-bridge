"""One bounded feasibility worker; no route from this CLI to the full pilot."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from baseline.setup_model import check_model
from pilot_v1.protocol import ROOT
from verifier import sha256_file

MODULE=ROOT/"lora_training_v1"
PLAN=MODULE/"plan.json"
OUTPUT=ROOT/"results/m2-feasibility-v1"


def check_inputs():
    protected=json.loads((MODULE/"protected_baselines.json").read_text())
    for name,expected in protected["files_sha256"].items():
        if sha256_file(ROOT/name)!=expected:
            raise ValueError(f"Frozen input changed: {name}")
    return len(protected["files_sha256"])


def fingerprints():
    paths=[p for p in MODULE.iterdir() if p.is_file() and p.name!="release.json"]
    paths.append(ROOT/"tests/test_lora_training_v1.py")
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(paths)}


def run():
    from lora_training_v1.telemetry import sysctl, swap_snapshot
    if OUTPUT.exists():
        raise ValueError("Feasibility run exists; no retry, resume, overwrite or extra updates")
    check_inputs()
    model=check_model()
    # This check runs before output creation and before loading any model. The
    # desktop sandbox requires a tool approval for accurate OS/swap telemetry.
    hardware={k:sysctl(k) for k in ("hw.model","hw.memsize","machdep.cpu.brand_string")}
    if hardware!={"hw.model":"Mac14,2","hw.memsize":"8589934592","machdep.cpu.brand_string":"Apple M2"}:
        raise ValueError("This measurement is pinned to the actual 8-GiB M2 host")
    plan=json.loads(PLAN.read_text())
    OUTPUT.mkdir(parents=True)
    manifest={"experiment_id":plan["experiment_id"],"started_at_utc":datetime.now(timezone.utc).isoformat(),
              "plan":plan,"hardware":hardware,"model":model,"fingerprints":fingerprints(),
              "initial_system_swap":swap_snapshot(),"full_pilot_enabled":False,
              "checkpoint_saving_enabled":False,"holdout_content_reads":0}
    (OUTPUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    started=time.perf_counter()
    with (OUTPUT/"worker.log").open("x") as log:
        child=subprocess.Popen([sys.executable,"-m","lora_training_v1.worker","--output",str(OUTPUT)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try:
            returncode=child.wait(timeout=plan["wall_timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(child.pid,signal.SIGTERM)
            try:
                returncode=child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                returncode=child.wait(timeout=10)
    completion={"worker_exit_code":returncode,"timed_out":timed_out,"wall_seconds":time.perf_counter()-started,
        "finished_at_utc":datetime.now(timezone.utc).isoformat(),"final_system_swap":swap_snapshot(),
        "base_model_files_unchanged":check_model()==model,"historical_files_unchanged":check_inputs(),
        "files_sha256":{p.name:sha256_file(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}}
    (OUTPUT/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
    print(f"Feasibility worker stopped: exit={returncode}, timeout={timed_out}; no pilot checkpoint saved")


def check():
    check_inputs()
    manifest=json.loads((OUTPUT/"manifest.json").read_text())
    if manifest["fingerprints"]!=fingerprints():
        raise ValueError("Implementation/plan changed since measurement")
    completion=json.loads((OUTPUT/"completion.json").read_text())
    for name,expected in completion["files_sha256"].items():
        if sha256_file(OUTPUT/name)!=expected:
            raise ValueError(f"Measurement changed: {name}")
    events=[json.loads(line) for line in (OUTPUT/"events.jsonl").read_text().splitlines()]
    starts=[e for e in events if e["event"]=="optimizer_started"]
    finishes=[e for e in events if e["event"]=="optimizer_finished"]
    if len(starts)>3 or len(finishes)>3 or len({e["update"] for e in starts})!=len(starts):
        raise ValueError("Exceeded bounded optimizer-update budget or retried a step")
    if any(p.suffix in (".pt",".bin",".safetensors") for p in OUTPUT.rglob("*")):
        raise ValueError("Feasibility must not retain model/adapter/checkpoint tensors")
    if (OUTPUT/"worker_result.json").exists():
        result=json.loads((OUTPUT/"worker_result.json").read_text())
        if result["status"]=="PASS":
            if len(finishes)!=3 or result["optimizer_updates_completed"]!=3:
                raise ValueError("Incomplete successful preflight")
            if result["base_parameter_sha256_before"]!=result["base_parameter_sha256_after"]:
                raise ValueError("Base tensor hashes changed")
            for u in result["updates"]:
                if u["microbatches"]!=4 or len(u["losses"])!=4 or u["trainable_audit"]["trainable_parameters"]!=540672:
                    raise ValueError("Accumulation or trainable-parameter audit differs")
        if result["adapter_weights_saved"] or result["trained_checkpoints_saved"] or result["full_pilot_started"]:
            raise ValueError("Preflight escaped its scope")
    release=MODULE/"release.json"
    if release.exists():
        for name,expected in json.loads(release.read_text())["files_sha256"].items():
            if sha256_file(ROOT/name)!=expected:
                raise ValueError(f"Frozen feasibility release changed: {name}")
    return manifest,completion


def freeze():
    if (MODULE/"release.json").exists():
        raise ValueError("Already frozen")
    check()
    from lora_training_v1.report import build_report
    text,data=build_report()
    if (OUTPUT/"M2_FEASIBILITY_REPORT.md").read_text()!=text or json.loads((OUTPUT/"assessment.json").read_text())!=data:
        raise ValueError("Assessment/report is stale")
    validation=json.loads((OUTPUT/"validation.json").read_text())
    if validation["status"]!="PASS":
        raise ValueError("Regression validation required")
    paths=[p for folder in (MODULE,OUTPUT) for p in folder.iterdir() if p.is_file()]
    paths.append(ROOT/"tests/test_lora_training_v1.py")
    (MODULE/"release.json").write_text(json.dumps({"status":"FROZEN","release_id":"proofbridge-m2-feasibility-v1",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),"classification":data["classification"],
        "files_sha256":{str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(paths)}},indent=2)+"\n")
    print("Frozen feasibility evidence; no trained checkpoint retained")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["run","check","freeze"])
    args=parser.parse_args()
    if args.action=="run": run()
    elif args.action=="freeze": freeze()
    else:
        check()
        print("PASS: bounded updates, unchanged inputs and saved evidence; no new computation")
