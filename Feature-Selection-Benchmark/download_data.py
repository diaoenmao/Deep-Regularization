#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_data.py — Auto-download all 11 real datasets for real-data-benchmark.py

Usage:
    python download_data.py          # download all
    python download_data.py mnist    # download specific dataset
    python download_data.py --list   # show status of all datasets

Datasets:
  NIPS 2003 (5): arcene, madelon, dexter, dorothea, gisette
  Image (3):     mnist, fashion, coil20
  Other (3):     isolet, mice, har
"""

import os
import sys
import gzip
import shutil
import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data"

# ──────────────────────────────────────────────────────────
# NIPS 2003 Feature Selection Challenge datasets
# ──────────────────────────────────────────────────────────

NIPS2003_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"

NIPS2003_DATASETS = {
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
    "dorothea": {
        "base_url": f"{NIPS2003_BASE}/dorothea/DOROTHEA",
        "valid_labels_url": f"{NIPS2003_BASE}/dorothea/dorothea_valid.labels",
        "files": ["dorothea_train.data", "dorothea_valid.data", "dorothea_train.labels"],
    },
    "gisette": {
        "base_url": f"{NIPS2003_BASE}/gisette/GISETTE",
        "valid_labels_url": f"{NIPS2003_BASE}/gisette/gisette_valid.labels",
        "files": ["gisette_train.data", "gisette_valid.data", "gisette_train.labels"],
    },
}

# ──────────────────────────────────────────────────────────
# MNIST / Fashion-MNIST  (IDX format for `mnist` Python package)
# ──────────────────────────────────────────────────────────

MNIST_URLS = {
    "mnist": {
        "dir": "mnist",
        "files": {
            "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz":  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz":  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        },
    },
    "fashion": {
        "dir": "fashion",
        "files": {
            "train-images-idx3-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        },
    },
}

# ──────────────────────────────────────────────────────────
# Other datasets
# ──────────────────────────────────────────────────────────

ISOLET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/"
MICE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
COIL20_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str = ""):
    """Download a file with progress indication."""
    if dest.exists():
        print(f"    [skip] {desc or dest.name} (already exists)")
        return
    print(f"    Downloading {desc or dest.name} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"    Done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    FAILED: {e}")
        if dest.exists():
            dest.unlink()
        raise


def _download_and_gunzip(url: str, dest_dir: Path, gz_name: str):
    """Download a .gz file and decompress it."""
    gz_path = dest_dir / gz_name
    final_path = dest_dir / gz_name.replace(".gz", "")
    if final_path.exists():
        print(f"    [skip] {final_path.name} (already exists)")
        return
    _download(url, gz_path, gz_name)
    print(f"    Decompressing {gz_name} ...")
    with gzip.open(str(gz_path), "rb") as f_in:
        with open(str(final_path), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()


def _download_and_unzip(url: str, dest_dir: Path, desc: str = ""):
    """Download a .zip file and extract it."""
    zip_path = dest_dir / "temp_download.zip"
    _download(url, zip_path, desc)
    print(f"    Extracting ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))
    zip_path.unlink()


# ──────────────────────────────────────────────────────────
# Per-dataset download functions
# ──────────────────────────────────────────────────────────

def download_nips2003(name: str):
    """Download a NIPS 2003 challenge dataset.

    Structure:
        data/{name}/{NAME}/{name}_train.data
        data/{name}/{NAME}/{name}_train.labels
        data/{name}/{NAME}/{name}_valid.data
        data/{name}/{name}_valid.labels        <- parent folder
    """
    info = NIPS2003_DATASETS[name]
    sub_dir = DATA_PATH / name / name.upper()
    sub_dir.mkdir(parents=True, exist_ok=True)

    for fname in info["files"]:
        url = f"{info['base_url']}/{fname}"
        _download(url, sub_dir / fname, fname)

    # valid labels go in the parent directory
    vl_name = f"{name}_valid.labels"
    vl_dest = DATA_PATH / name / vl_name
    _download(info["valid_labels_url"], vl_dest, vl_name)


def download_mnist_family(name: str):
    """Download MNIST or Fashion-MNIST IDX files.

    Structure:
        data/{name}/train-images-idx3-ubyte
        data/{name}/train-labels-idx1-ubyte
        data/{name}/t10k-images-idx3-ubyte
        data/{name}/t10k-labels-idx1-ubyte
    """
    info = MNIST_URLS[name]
    dest_dir = DATA_PATH / info["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    for gz_name, url in info["files"].items():
        _download_and_gunzip(url, dest_dir, gz_name)


def download_isolet():
    """Download ISOLET dataset (CSV format).

    Structure:
        data/isolet/isolet1+2+3+4.data
        data/isolet/isolet5.data
    """
    dest_dir = DATA_PATH / "isolet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["isolet1+2+3+4.data", "isolet5.data"]:
        url = f"{ISOLET_URL}{fname}"
        _download(url, dest_dir / fname, fname)


def download_mice():
    """Download Mice Protein Expression dataset.

    Structure:
        data/mice+protein+expression/Data_Cortex_Nuclear.xls
    """
    dest_dir = DATA_PATH / "mice+protein+expression"
    dest_dir.mkdir(parents=True, exist_ok=True)
    _download(MICE_URL, dest_dir / "Data_Cortex_Nuclear.xls",
              "Data_Cortex_Nuclear.xls")


def download_har():
    """Download HAR (Human Activity Recognition) dataset.

    Structure:
        data/UCI HAR Dataset/train/X_train.txt
        data/UCI HAR Dataset/train/y_train.txt
        data/UCI HAR Dataset/test/X_test.txt
        data/UCI HAR Dataset/test/y_test.txt
    """
    target = DATA_PATH / "UCI HAR Dataset"
    if (target / "train" / "X_train.txt").exists():
        print("    [skip] HAR (already exists)")
        return
    _download_and_unzip(HAR_URL, DATA_PATH, "UCI_HAR_Dataset.zip")
    # Handle potential folder naming differences
    if not target.exists():
        for candidate in DATA_PATH.iterdir():
            if "har" in candidate.name.lower() and candidate.is_dir():
                candidate.rename(target)
                break


def download_coil20():
    """Download COIL-20 processed image dataset.

    Structure:
        data/coil-20-proc/obj1__0.png  ...  obj20__71.png
    """
    target = DATA_PATH / "coil-20-proc"
    if target.exists() and any(target.glob("*.png")):
        print("    [skip] COIL-20 (already exists)")
        return
    target.mkdir(parents=True, exist_ok=True)
    _download_and_unzip(COIL20_URL, DATA_PATH, "coil-20-proc.zip")
    # If zip extracts into a nested folder, move files up
    nested = target / "coil-20-proc"
    if nested.exists() and nested.is_dir():
        for f in nested.iterdir():
            shutil.move(str(f), str(target / f.name))
        nested.rmdir()


# ──────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────

DOWNLOAD_MAP = {
    "arcene":   lambda: download_nips2003("arcene"),
    "madelon":  lambda: download_nips2003("madelon"),
    "dexter":   lambda: download_nips2003("dexter"),
    "dorothea": lambda: download_nips2003("dorothea"),
    "gisette":  lambda: download_nips2003("gisette"),
    "mnist":    lambda: download_mnist_family("mnist"),
    "fashion":  lambda: download_mnist_family("fashion"),
    "isolet":   download_isolet,
    "mice":     download_mice,
    "har":      download_har,
    "coil20":   download_coil20,
}

ALL_DATASETS = list(DOWNLOAD_MAP.keys())


def check_status():
    """Print availability status of all datasets."""
    checks = {
        "arcene":   (DATA_PATH / "arcene" / "ARCENE" / "arcene_train.data").exists(),
        "madelon":  (DATA_PATH / "madelon" / "MADELON" / "madelon_train.data").exists(),
        "dexter":   (DATA_PATH / "dexter" / "DEXTER" / "dexter_train.data").exists(),
        "dorothea": (DATA_PATH / "dorothea" / "DOROTHEA" / "dorothea_train.data").exists(),
        "gisette":  (DATA_PATH / "gisette" / "GISETTE" / "gisette_train.data").exists(),
        "mnist":    (DATA_PATH / "mnist" / "train-images-idx3-ubyte").exists(),
        "fashion":  (DATA_PATH / "fashion" / "train-images-idx3-ubyte").exists(),
        "isolet":   (DATA_PATH / "isolet" / "isolet1+2+3+4.data").exists(),
        "mice":     (DATA_PATH / "mice+protein+expression" / "Data_Cortex_Nuclear.xls").exists(),
        "har":      (DATA_PATH / "UCI HAR Dataset" / "train" / "X_train.txt").exists(),
        "coil20":   (DATA_PATH / "coil-20-proc").exists() and
                    any((DATA_PATH / "coil-20-proc").glob("*.png"))
                    if (DATA_PATH / "coil-20-proc").exists() else False,
    }
    print(f"\n{'Dataset':<12} {'Status':<10}")
    print("-" * 30)
    for name in ALL_DATASETS:
        ok = checks[name]
        marker = "OK" if ok else "MISSING"
        sym = "+" if ok else "-"
        print(f"  [{sym}] {name:<12} {marker}")
    n_ok = sum(checks.values())
    print(f"\n{n_ok}/{len(checks)} datasets available.\n")
    return checks


def main():
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            check_status()
            return
        targets = sys.argv[1:]
        for t in targets:
            if t not in DOWNLOAD_MAP:
                print(f"Unknown dataset: {t}")
                print(f"Available: {', '.join(ALL_DATASETS)}")
                sys.exit(1)
    else:
        targets = ALL_DATASETS

    print(f"Downloading {len(targets)} dataset(s) to {DATA_PATH}/\n")

    succeeded, failed = [], []
    for name in targets:
        print(f"[{name}]")
        try:
            DOWNLOAD_MAP[name]()
            print(f"  + {name} ready\n")
            succeeded.append(name)
        except Exception as e:
            print(f"  - {name} FAILED: {e}\n")
            failed.append(name)

    print("=" * 50)
    check_status()

    if failed:
        print(f"Failed datasets: {', '.join(failed)}")
        print("You may need to download these manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
