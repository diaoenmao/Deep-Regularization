"""Serial appendix-regeneration queue with per-step logs.

Designed for long-running appendix jobs. It can optionally wait for an
already-running process (e.g. an in-flight dropout ablation) before starting
the remaining regeneration steps.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results", "appendix_regen")
os.makedirs(RESULTS_DIR, exist_ok=True)


def pid_exists(pid: int) -> bool:
    """Check process existence on Windows via tasklist."""
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in result.stdout


def write_status(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    args = parser.parse_args()

    queue = [
        (
            "dropout_xor512",
            [sys.executable, "-u", os.path.join(ROOT, "run_dropout_xor512.py")],
        ),
        (
            "l1_ablation",
            [
                sys.executable,
                "-u",
                os.path.join(ROOT, "run_ablations.py"),
                "--l1-ablation",
            ],
        ),
        (
            "multiseed_dag",
            [
                sys.executable,
                "-u",
                os.path.join(ROOT, "run_multiseed_dag.py"),
                "--methods",
                "admm_input_group",
                "stg",
                "rf",
                "treeshap",
            ],
        ),
        (
            "penalty_ablation",
            [sys.executable, "-u", os.path.join(ROOT, "run_penalty_ablation.py")],
        ),
        (
            "bounded_gate_ablation",
            [sys.executable, "-u", os.path.join(ROOT, "run_bounded_gate_ablation.py")],
        ),
    ]

    status_path = os.path.join(RESULTS_DIR, "queue_status.json")
    master_log_path = os.path.join(RESULTS_DIR, "queue_master.log")

    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] queue start\n"
        )
        if args.wait_pid:
            master_log.write(f"waiting for pid {args.wait_pid}\n")
            master_log.flush()
            while pid_exists(args.wait_pid):
                write_status(
                    status_path,
                    {
                        "state": "waiting",
                        "wait_pid": args.wait_pid,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                time.sleep(30)

        for step_name, cmd in queue:
            out_log = os.path.join(RESULTS_DIR, f"{step_name}.log")
            err_log = os.path.join(RESULTS_DIR, f"{step_name}.err")
            write_status(
                status_path,
                {
                    "state": "running",
                    "current_step": step_name,
                    "command": cmd,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            master_log.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] START {step_name}: {' '.join(cmd)}\n"
            )
            master_log.flush()
            with (
                open(out_log, "w", encoding="utf-8") as out_f,
                open(err_log, "w", encoding="utf-8") as err_f,
            ):
                proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, check=False)
            master_log.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] END {step_name}: rc={proc.returncode}\n"
            )
            master_log.flush()
            if proc.returncode != 0:
                write_status(
                    status_path,
                    {
                        "state": "failed",
                        "current_step": step_name,
                        "returncode": proc.returncode,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                return proc.returncode

        write_status(
            status_path,
            {
                "state": "completed",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        master_log.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] queue completed\n"
        )
        master_log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
