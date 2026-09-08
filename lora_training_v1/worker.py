"""Disposable adapter updates, capped at three; no adapter/checkpoint files."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from baseline.setup_model import check_model
from lora_training_v1.core import audit_trainable, base_hashes, make_optimizer, optimizer_update, setup_model
from lora_training_v1.telemetry import Monitor, process_memory, swap_snapshot
from training_protocol_v1.data import epoch_order


def execute(output):
    from lora_training_v1.preflight import check_inputs, PLAN
    plan=json.loads(PLAN.read_text())
    if plan["optimizer_update_limit"]!=3 or plan["checkpoints_enabled"] or plan["adapter_serialization_enabled"]:
        raise ValueError("Invalid disposable feasibility plan")
    check_inputs()
    updates=[]
    result={"purpose":"disposable_local_feasibility_only","status":"RUNNING","model_loaded":False,
            "base_parameters_unchanged":None,"optimizer_updates_completed":0,"updates":updates,
            "full_pilot_started":False,"trained_checkpoints_saved":0,"adapter_weights_saved":False,
            "holdout_content_reads":0,"dev_generations":0}
    model=None
    before=None
    start=time.perf_counter()
    with (output/"events.jsonl").open("x") as events, Monitor(output/"memory_samples.jsonl",plan["sampling_seconds"]):
        def emit(event,**values):
            record={"event":event,"elapsed_seconds":time.perf_counter()-start,
                    "at_utc":datetime.now(timezone.utc).isoformat(),"memory":process_memory(),**values}
            events.write(json.dumps(record)+"\n"); events.flush()
            if event in ("model_loaded","forward_started","backward_finished","optimizer_finished"):
                print(f"{event}: update={values.get('update','-')} micro={values.get('microbatch','-')} elapsed={record['elapsed_seconds']:.1f}s",flush=True)
        try:
            emit("model_load_started")
            model,tokenizer,encoded,before,runtime=setup_model()
            result.update(model_loaded=True,runtime=runtime)
            emit("model_loaded",**runtime)
            optimizer,scheduler=make_optimizer(model)
            longest=max(encoded.values(),key=lambda e:(e["sequence_length"],e["id"]))
            order=epoch_order(0)
            batches=[([longest]*4,None),([longest]*4,768),([encoded[i] for i in order[:4]],None)]
            result["longest_training_example"]={"id":longest["id"],"native_length":longest["sequence_length"],
                                                 "target_tokens":longest["target_length"],"stress_tensor_length":768}
            result["encoded_training_rows"]=len(encoded)
            result["base_parameter_sha256_before"]=before
            for number,(examples,pad_to) in enumerate(batches,1):
                if number>3:
                    raise ValueError("Feasibility update cap exceeded")
                measured=optimizer_update(model,optimizer,scheduler,examples,emit,number,pad_to,tokenizer.eos_token_id)
                updates.append(measured)
                result["optimizer_updates_completed"]=number
            result["status"]="PASS"
        except Exception as error:
            text=str(error)
            memory_error=isinstance(error,MemoryError) or any(s in text.lower() for s in ("out of memory","can't allocate memory","cannot allocate memory","defaultcpuallocator","std::bad_alloc"))
            result.update(status="FAIL",failure_kind="MEMORY" if memory_error else "IMPLEMENTATION_OR_RUNTIME",
                          error_type=type(error).__name__,error=text,traceback=traceback.format_exc())
            emit("failure",failure_kind=result["failure_kind"],error=text)
        finally:
            if model is not None and before is not None:
                after=base_hashes(model)
                result["base_parameter_sha256_after"]=after
                result["base_parameters_unchanged"]=after==before
                _,result["final_trainable_audit"]=audit_trainable(model)
                if after!=before:
                    result.update(status="FAIL",failure_kind="BASE_WEIGHT_CHANGED")
            check_model()
            check_inputs()
            result.update(base_model_files_unchanged=True,wall_seconds=time.perf_counter()-start,
                          final_memory=process_memory(),final_swap=swap_snapshot())
            emit("worker_finished",status=result["status"],optimizer_updates_completed=len(updates),
                 base_parameters_unchanged=result["base_parameters_unchanged"])
            (output/"worker_result.json").write_text(json.dumps(result,indent=2)+"\n")
    # Exiting the worker discards all ephemeral adapter updates and optimizer state.
    return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(execute(args.output))
