#!/usr/bin/env python3
"""
Check and compare the number of output files with the list of jobs to run.

For each dataset, the voxels listed in voxels_<dataset>.csv are the jobs.
A job is done if a result file with the same coordinate tag exists in the
results directory of the dataset's fidelity (results/lf or results/hf).

Files are only listed, never opened, so this runs in seconds even with
O(10k) files. "Done" means the file exists, not that it is readable or
contains hits.

Usage:
    python check_jobs.py <voxel-set-folder>
"""

import os
import re
import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = [
    "LF_train",
    "LF2_train",
    "HF_train",
    "CNP_val",
    "GP_val",
]

# Fidelity each dataset is meant to use. Only result files from that
# fidelity directory count for the dataset (None = accept any).
DATASET_FIDELITY = {
    "LF_train": "lf",
    "LF2_train": "lf",
    "HF_train": "hf",
    "CNP_val": "lf",
    "GP_val": "hf",
}

# Expected number of voxels per dataset (used for sanity checks only).
EXPECTED_N = {
    "LF_train": 6400,
    "LF2_train": 6400,
    "HF_train": 640,
    "CNP_val": 500,
    "GP_val": 500,
}

RESULT_DIRS = {
    "lf": os.path.join("results", "lf"),
    "hf": os.path.join("results", "hf"),
}

RESULT_EXTENSIONS = (".lh5", ".h5", ".hdf5")

COORD_RE = re.compile(
    r"(x[+-]\d+\.\d+_y[+-]\d+\.\d+_z[+-]\d+\.\d+)"
)


# ============================================================
# COORDINATE TAGS
# ============================================================

def extract_coordinate_tag(value):
    """Extract x/y/z coordinate tag from a filename or manifest value."""
    match = COORD_RE.search(os.path.basename(str(value)))
    return match.group(1) if match else None


def coordinate_tag_from_mm(x, y, z):
    """Build a coordinate tag from voxel-center coordinates in mm."""
    return (
        f"x{x / 1000:+.3f}_"
        f"y{y / 1000:+.3f}_"
        f"z{z / 1000:+.3f}"
    )


def _pct(num, den):
    return 100.0 * num / den if den else float("nan")


# ============================================================
# LOAD MANIFESTS
# ============================================================

def load_manifests(folder):
    """Return {"tags": {dataset: set of tags}, "rows": {dataset: n rows}}."""
    manifest_tags = {}
    manifest_rows = {}

    for dataset in DATASETS:
        manifest = os.path.join(folder, f"voxels_{dataset}.csv")

        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        df = pd.read_csv(manifest)
        manifest_rows[dataset] = len(df)
        manifest_tags[dataset] = set()

        for _, row in df.iterrows():
            tag = None

            if "tag" in df.columns and pd.notna(row.get("tag")):
                tag = extract_coordinate_tag(row["tag"])

            if (
                tag is None
                and "stp_file" in df.columns
                and pd.notna(row.get("stp_file"))
            ):
                tag = extract_coordinate_tag(row["stp_file"])

            if tag is None:
                tag = coordinate_tag_from_mm(
                    row["x_mm"], row["y_mm"], row["z_mm"]
                )

            manifest_tags[dataset].add(tag)

    print(f"Loaded {len(DATASETS)} manifests.")
    return {"tags": manifest_tags, "rows": manifest_rows}


# ============================================================
# JOB COMPLETION CHECK (manifest jobs vs. result files on disk)
# ============================================================

def print_job_completion(folder, manifest_info, save_missing=True):
    """
    Compare the jobs requested in each manifest with the result files found.

    A job = one unique voxel tag in the manifest. A job counts as done if a
    result file with that tag exists in the dataset's fidelity directory.
    """
    # Tags found on disk, per fidelity directory
    found = {}
    n_files = {}
    for fidelity, relative_dir in RESULT_DIRS.items():
        result_dir = os.path.join(folder, relative_dir)
        found[fidelity] = set()
        n_files[fidelity] = 0

        if not os.path.isdir(result_dir):
            print(f"Results directory not found: {result_dir}")
            continue

        for name in os.listdir(result_dir):
            if not name.endswith(RESULT_EXTENSIONS):
                continue
            n_files[fidelity] += 1
            tag = extract_coordinate_tag(name)
            if tag:
                found[fidelity].add(tag)

    print("\nJOB COMPLETION (manifest jobs vs. result files on disk)")
    print("Result files per fidelity directory: "
          + ", ".join(f"{k.upper()}={v}" for k, v in n_files.items()))

    header = (f"{'dataset':<12}{'fid':<5}{'jobs':>7}{'done':>7}"
              f"{'missing':>9}{'done %':>8}   status")
    print(header)
    print("-" * len(header))

    outdir = Path(folder).resolve() / "hits_distributions"

    for dataset in DATASETS:
        wanted = manifest_info["tags"].get(dataset, set())
        fidelity = DATASET_FIDELITY.get(dataset)

        if fidelity is None:
            on_disk = set().union(*found.values())
        else:
            on_disk = found.get(fidelity, set())

        done = wanted & on_disk
        missing = wanted - on_disk
        n_jobs = len(wanted)

        # Voxels that ran, but only at the wrong fidelity
        wrong_fid = set()
        if fidelity is not None:
            for other, tags in found.items():
                if other != fidelity:
                    wrong_fid |= (missing & tags)

        status = "COMPLETE" if not missing else f"{len(missing)} jobs to run"
        print(f"{dataset:<12}{(fidelity or 'any').upper():<5}{n_jobs:>7}"
              f"{len(done):>7}{len(missing):>9}"
              f"{_pct(len(done), n_jobs):>7.1f}%   {status}")

        expected = EXPECTED_N.get(dataset)
        if expected is not None and n_jobs != expected:
            print(f"  WARNING: manifest has {n_jobs} unique voxels, "
                  f"expected {expected}")
        if manifest_info["rows"].get(dataset, n_jobs) != n_jobs:
            print(f"  WARNING: {manifest_info['rows'][dataset] - n_jobs} "
                  f"duplicate rows in manifest")

        if wrong_fid:
            print(f"  note: {len(wrong_fid)} of the missing voxels exist "
                  f"only in the other fidelity directory")

        if save_missing and missing:
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f"missing_jobs_{dataset}.txt"
            path.write_text("\n".join(sorted(missing)) + "\n")
            print(f"  missing tags saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare manifest jobs with result files on disk."
    )
    parser.add_argument("folder",
                        help="Voxel-set folder containing manifests and results/")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the missing_jobs_<dataset>.txt files")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        parser.error(f"Folder does not exist: {folder}")

    manifest_info = load_manifests(folder)
    print_job_completion(folder, manifest_info, save_missing=not args.no_save)


if __name__ == "__main__":
    main()