#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download the real-world datasets used by the paper into the repo-level data dir.

This is a standalone replacement for the removed benchmark downloader so that
current ``custom_admm`` experiment scripts can be rerun without modifying the
benchmark tree.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data"

NIPS2003_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
ISOLET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/"
ISOLET_ZIP_URL = "https://archive.ics.uci.edu/static/public/54/isolet.zip"
MICE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
COIL20_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"

MNIST_FAMILY = {
    "fashion": {
        "dir": "fashion",
        "files": {
            "train-images-idx3-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        },
    },
    "mnist": {
        "dir": "mnist",
        "files": {
            "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        },
    },
}

NIPS2003 = {
    "arcene": {
        "base_url": f"{NIPS2003_BASE}/arcene/ARCENE",
        "valid_labels_url": f"{NIPS2003_BASE}/arcene/arcene_valid.labels",
        "files": ["arcene_train.data", "arcene_valid.data", "arcene_train.labels"],
    },
    "madelon": {
        "base_url": f"{NIPS2003_BASE}/madelon/MADELON",
        "valid_labels_url": f"{NIPS2003_BASE}/madelon/madelon_valid.labels",
        "files": ["madelon_train.data", "madelon_valid.data", "madelon_train.labels"],
    },
    "dexter": {
        "base_url": f"{NIPS2003_BASE}/dexter/DEXTER",
        "valid_labels_url": f"{NIPS2003_BASE}/dexter/dexter_valid.labels",
        "files": ["dexter_train.data", "dexter_valid.data", "dexter_train.labels"],
    },
    "gisette": {
        "base_url": f"{NIPS2003_BASE}/gisette/GISETTE",
        "valid_labels_url": f"{NIPS2003_BASE}/gisette/gisette_valid.labels",
        "files": ["gisette_train.data", "gisette_valid.data", "gisette_train.labels"],
    },
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, str(dest))


def _download_and_gunzip(url: str, dest_dir: Path, gz_name: str) -> None:
    gz_path = dest_dir / gz_name
    final_path = dest_dir / gz_name.replace(".gz", "")
    if final_path.exists():
        print(f"[skip] {final_path}")
        return
    _download(url, gz_path)
    with gzip.open(gz_path, "rb") as src, open(final_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink(missing_ok=True)


def _download_and_unzip(url: str, dest_dir: Path, zip_name: str) -> None:
    zip_path = dest_dir / zip_name
    _download(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)


def download_nips2003(name: str) -> None:
    info = NIPS2003[name]
    sub_dir = DATA_PATH / name / name.upper()
    sub_dir.mkdir(parents=True, exist_ok=True)
    for fname in info["files"]:
        _download(f"{info['base_url']}/{fname}", sub_dir / fname)
    _download(info["valid_labels_url"], DATA_PATH / name / f"{name}_valid.labels")


def download_mnist_family(name: str) -> None:
    info = MNIST_FAMILY[name]
    dest_dir = DATA_PATH / info["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    for gz_name, url in info["files"].items():
        _download_and_gunzip(url, dest_dir, gz_name)


def download_isolet() -> None:
    dest_dir = DATA_PATH / "isolet"
    dest_dir.mkdir(parents=True, exist_ok=True)
    train_path = dest_dir / "isolet1+2+3+4.data"
    test_path = dest_dir / "isolet5.data"
    if train_path.exists() and test_path.exists():
        print("[skip] ISOLET")
        return

    try:
        _download_and_unzip(ISOLET_ZIP_URL, dest_dir, "isolet.zip")
    except Exception:
        for fname in ["isolet1+2+3+4.data", "isolet5.data"]:
            _download(f"{ISOLET_URL}{fname}", dest_dir / fname)
        return

    nested_dir = dest_dir / "isolet"
    if nested_dir.exists() and nested_dir.is_dir():
        for child in nested_dir.iterdir():
            shutil.move(str(child), str(dest_dir / child.name))
        nested_dir.rmdir()

    compressed_train = dest_dir / "isolet1+2+3+4.data.Z"
    compressed_test = dest_dir / "isolet5.data.Z"
    if compressed_train.exists() and not train_path.exists():
        shutil.move(str(compressed_train), str(train_path))
    if compressed_test.exists() and not test_path.exists():
        shutil.move(str(compressed_test), str(test_path))


def download_mice() -> None:
    dest_dir = DATA_PATH / "mice+protein+expression"
    dest_dir.mkdir(parents=True, exist_ok=True)
    _download(MICE_URL, dest_dir / "Data_Cortex_Nuclear.xls")


def download_har() -> None:
    target = DATA_PATH / "UCI HAR Dataset"
    if (target / "train" / "X_train.txt").exists():
        print("[skip] HAR")
        return
    _download_and_unzip(HAR_URL, DATA_PATH, "UCI_HAR_Dataset.zip")


def download_coil20() -> None:
    target = DATA_PATH / "coil-20-proc"
    if target.exists() and any(target.glob("*.png")):
        print("[skip] COIL-20")
        return
    target.mkdir(parents=True, exist_ok=True)
    _download_and_unzip(COIL20_URL, DATA_PATH, "coil-20-proc.zip")
    nested = target / "coil-20-proc"
    if nested.exists() and nested.is_dir():
        for child in nested.iterdir():
            shutil.move(str(child), str(target / child.name))
        nested.rmdir()


DOWNLOADERS = {
    "arcene": lambda: download_nips2003("arcene"),
    "madelon": lambda: download_nips2003("madelon"),
    "dexter": lambda: download_nips2003("dexter"),
    "gisette": lambda: download_nips2003("gisette"),
    "fashion": lambda: download_mnist_family("fashion"),
    "mnist": lambda: download_mnist_family("mnist"),
    "isolet": download_isolet,
    "mice": download_mice,
    "har": download_har,
    "coil20": download_coil20,
}

PAPER_DATASETS = [
    "madelon",
    "arcene",
    "gisette",
    "dexter",
    "fashion",
    "isolet",
    "har",
    "coil20",
    "mice",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="Specific datasets to download")
    parser.add_argument("--list", action="store_true", help="List supported datasets")
    args = parser.parse_args()

    if args.list:
        for name in DOWNLOADERS:
            print(name)
        return

    targets = args.datasets or PAPER_DATASETS
    for name in targets:
        if name not in DOWNLOADERS:
            raise SystemExit(f"Unknown dataset: {name}")
        print(f"\n=== {name} ===")
        DOWNLOADERS[name]()


if __name__ == "__main__":
    main()
