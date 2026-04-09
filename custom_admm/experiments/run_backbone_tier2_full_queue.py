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
LOG_DIR = os.path.join(RESULTS_DIR, "backbone_tier2_full_logs")
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_METHODS = [
    "gated_mlp",
    "gated_token_transformer",
    "gated_token_transformer_pretrained",
    "fsnet",
    "e2efs",
    "cae",
    "tabnet",
]


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--warmup-epochs", type=int, default=60)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output or f"backbone_tier2_synthetic_full_{stamp}.json"
    output_path = os.path.join(RESULTS_DIR, output_name)
    status_path = os.path.join(LOG_DIR, "queue_status.json")
    master_log_path = os.path.join(LOG_DIR, "queue_master.log")

    status = {
        "state": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "output": output_path,
        "methods": args.methods,
        "completed": [],
        "current": None,
    }
    _write_json(status_path, status)

    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] start output={output_path}\n"
        )
        for method in args.methods:
            status["current"] = method
            _write_json(status_path, status)
            log_path = os.path.join(LOG_DIR, f"{method}.log")
            err_path = os.path.join(LOG_DIR, f"{method}.err")
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
                "--output",
                output_name,
                "--resume",
            ]
            master_log.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] run {method}: {' '.join(cmd)}\n"
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
                status["failed_at"] = datetime.now().isoformat(timespec="seconds")
                _write_json(status_path, status)
                master_log.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] failed {method} code={proc.returncode}\n"
                )
                raise SystemExit(proc.returncode)
            status["completed"].append(method)
            status["current"] = None
            _write_json(status_path, status)
            master_log.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] completed {method}\n"
            )
            master_log.flush()

    status["state"] = "completed"
    status["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(status_path, status)


if __name__ == "__main__":
    main()
