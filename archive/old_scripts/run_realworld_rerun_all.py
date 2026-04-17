#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequential fresh rerun for the paper's real-world SADMM-FS/STG experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "custom_admm" / "results" / "realworld_rerun_logs"
STATUS_PATH = ROOT / "custom_admm" / "results" / "realworld_rerun_status.json"

MODERN_DATASETS = ["fashion", "coil20", "isolet", "mice", "har"]


def _write_status(status: dict) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "name": "nips_admm",
            "cmd": [
                sys.executable,
                "-u",
                str(ROOT / "custom_admm" / "run_realworld_admm.py"),
                "--method",
                "admm_input_group",
                "--seeds",
                "5",
                "--force",
            ],
        },
        {
            "name": "nips_stg",
            "cmd": [
                sys.executable,
                "-u",
                str(ROOT / "custom_admm" / "run_realworld_admm.py"),
                "--method",
                "stg",
                "--seeds",
                "5",
                "--force",
            ],
        },
        {
            "name": "modern_admm",
            "cmd": [
                sys.executable,
                "-u",
                str(ROOT / "custom_admm" / "run_modern_admm_stg.py"),
                "--methods",
                "admm_input_group",
                "--datasets",
                *MODERN_DATASETS,
                "--seeds",
                "5",
                "--force",
            ],
        },
        {
            "name": "modern_stg",
            "cmd": [
                sys.executable,
                "-u",
                str(ROOT / "custom_admm" / "run_modern_admm_stg.py"),
                "--methods",
                "stg",
                "--datasets",
                *MODERN_DATASETS,
                "--seeds",
                "5",
                "--force",
            ],
        },
    ]

    status = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "jobs": [],
        "state": "running",
    }
    _write_status(status)

    for job in jobs:
        log_path = LOG_DIR / f"{job['name']}.log"
        err_path = LOG_DIR / f"{job['name']}.err"
        for path in [log_path, err_path]:
            if path.exists():
                path.unlink()

        job_state = {
            "name": job["name"],
            "cmd": job["cmd"],
            "log": str(log_path),
            "err": str(err_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
        }
        status["jobs"].append(job_state)
        _write_status(status)

        t0 = time.time()
        with (
            open(log_path, "ab", buffering=0) as log_file,
            open(err_path, "ab", buffering=0) as err_file,
        ):
            proc = subprocess.run(
                job["cmd"],
                cwd=str(ROOT),
                stdout=log_file,
                stderr=err_file,
            )

        job_state["elapsed_sec"] = time.time() - t0
        job_state["returncode"] = proc.returncode
        job_state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        job_state["status"] = "completed" if proc.returncode == 0 else "failed"
        _write_status(status)

        if proc.returncode != 0:
            status["state"] = "failed"
            _write_status(status)
            raise SystemExit(proc.returncode)

    status["state"] = "completed"
    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_status(status)


if __name__ == "__main__":
    main()
