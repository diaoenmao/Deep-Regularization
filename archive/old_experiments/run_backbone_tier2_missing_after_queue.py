from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "results", "mentor_axes", "backbone_tier2_full_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--warmup-epochs", type=int, default=60)
    args = parser.parse_args()

    log_path = os.path.join(LOG_DIR, "missing_tier2_chain.log")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] waiting for pid={args.wait_pid}\n"
        )
        log.flush()
        while True:
            try:
                os.kill(args.wait_pid, 0)
                time.sleep(60)
            except OSError:
                break

        cmd = [
            sys.executable,
            os.path.join(ROOT, "experiments", "run_backbone_tier2_full_queue.py"),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--warmup-epochs",
            str(args.warmup_epochs),
            "--methods",
            "cae",
            "tabnet",
            "--output",
            os.path.basename(args.output),
        ]
        log.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] starting missing tier2: {' '.join(cmd)}\n"
        )
        log.flush()
        subprocess.run(cmd, cwd=os.path.dirname(ROOT), check=False)


if __name__ == "__main__":
    main()
