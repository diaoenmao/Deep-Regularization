from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "experiments", "backbone_tier2_synthetic_full.py")
RESULTS_DIR = os.path.join(ROOT, "results", "mentor_axes")
LOG_DIR = os.path.join(RESULTS_DIR, "backbone_tier2_task_logs")
os.makedirs(LOG_DIR, exist_ok=True)

DATASETS_CONFIG = [
    ("xor", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ("ring+xor+sum", [6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
]

DEFAULT_METHODS = [
    "gated_token_transformer",
    "gated_token_transformer_pretrained",
    "fsnet",
    "e2efs",
    "cae",
    "tabnet",
]


def _all_task_keys() -> list[str]:
    tasks = []
    for ds_name, dims in DATASETS_CONFIG:
        for m in dims:
            tasks.append(f"{ds_name}_m{m}")
    return tasks


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_results(path: str) -> dict:
    if not os.path.exists(path):
        return {"results": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--warmup-epochs", type=int, default=60)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = os.path.join(RESULTS_DIR, args.output)
    status_path = os.path.join(LOG_DIR, "task_queue_status.json")
    master_log_path = os.path.join(LOG_DIR, "task_queue_master.log")

    status = {
        "state": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "output": output_path,
        "methods": args.methods,
        "current_method": None,
        "current_task": None,
        "completed_pairs": 0,
    }
    _write_json(status_path, status)

    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] start output={output_path}\n"
        )
        all_tasks = _all_task_keys()
        for method in args.methods:
            results = _load_results(output_path).get("results", {})
            pending = [t for t in all_tasks if method not in results.get(t, {})]
            master_log.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] method={method} pending={len(pending)}\n"
            )
            master_log.flush()
            for task_key in pending:
                status["current_method"] = method
                status["current_task"] = task_key
                _write_json(status_path, status)
                safe_task = (
                    task_key.replace("+", "_plus_").replace(":", "_").replace("/", "_")
                )
                log_path = os.path.join(LOG_DIR, f"{method}__{safe_task}.log")
                err_path = os.path.join(LOG_DIR, f"{method}__{safe_task}.err")
                cmd = [
                    sys.executable,
                    RUNNER,
                    "--device",
                    args.device,
                    "--epochs",
                    str(args.epochs),
                    "--warmup-epochs",
                    str(args.warmup_epochs),
                    "--methods",
                    method,
                    "--task-keys",
                    task_key,
                    "--output",
                    args.output,
                    "--resume",
                ]
                master_log.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] run {method}/{task_key}\n"
                )
                master_log.flush()
                with (
                    open(log_path, "w", encoding="utf-8") as log_f,
                    open(err_path, "w", encoding="utf-8") as err_f,
                ):
                    proc = subprocess.run(
                        cmd,
                        cwd=os.path.dirname(ROOT),
                        stdout=log_f,
                        stderr=err_f,
                        text=True,
                    )
                if proc.returncode != 0:
                    status["state"] = "failed"
                    status["failed_method"] = method
                    status["failed_task"] = task_key
                    status["failed_at"] = datetime.now().isoformat(timespec="seconds")
                    _write_json(status_path, status)
                    master_log.write(
                        f"[{datetime.now().isoformat(timespec='seconds')}] failed {method}/{task_key} code={proc.returncode}\n"
                    )
                    raise SystemExit(proc.returncode)
                results = _load_results(output_path).get("results", {})
                if method not in results.get(task_key, {}):
                    status["state"] = "failed"
                    status["failed_method"] = method
                    status["failed_task"] = task_key
                    status["failed_at"] = datetime.now().isoformat(timespec="seconds")
                    status["reason"] = "subprocess returned 0 but result not written"
                    _write_json(status_path, status)
                    master_log.write(
                        f"[{datetime.now().isoformat(timespec='seconds')}] missing output {method}/{task_key}\n"
                    )
                    raise SystemExit(2)
                status["completed_pairs"] += 1
                status["current_method"] = None
                status["current_task"] = None
                _write_json(status_path, status)
                master_log.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] completed {method}/{task_key}\n"
                )
                master_log.flush()

    status["state"] = "completed"
    status["completed_at"] = datetime.now().isoformat(timespec="seconds")
    status["current_method"] = None
    status["current_task"] = None
    _write_json(status_path, status)


if __name__ == "__main__":
    main()
