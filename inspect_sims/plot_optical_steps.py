#!/usr/bin/env python3
"""
Optical-hit and SiPM channel-coverage analysis for LF/HF datasets.

Designed for O(10k) simulation files:
  * per-file stacked-bar plot uses a fixed figure size (step-filled area for
    large N, individual bars for small N) and can be zoomed with
    --coverage-start / --coverage-stop
  * all summaries are vectorised on per-dataset DataFrames
"""

import os
import re
import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401: registers HDF5 compression filters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# CONFIGURATION
# ============================================================

MAX_BINS = 25
LABEL_BARS_UNDER = 20
BARS_BELOW = 150          # <= this many files: individual bars; above: step area
PROGRESS_EVERY = 1000
MAX_ERRORS_PRINTED = 10

DATASETS = [
    "LF_train",
    "LF2_train",
    "HF_train",
    "CNP_val",
    "GP_val",
]

# Fidelity each dataset is meant to use. Only result files from that
# fidelity directory are counted for the dataset (None = accept any).
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

COORD_RE = re.compile(
    r"(x[+-]\d+\.\d+_y[+-]\d+\.\d+_z[+-]\d+\.\d+)"
)

COORD_VALUES_RE = re.compile(
    r"x([+-]\d+\.\d+)_y([+-]\d+\.\d+)_z([+-]\d+\.\d+)"
)

SINGLE_HIT_COLOR = "lightblue"
REPEATED_HIT_COLOR = "darkblue"


# ============================================================
# COORDINATE TAGS
# ============================================================

def extract_coordinate_tag(value):
    """Extract x/y/z coordinate tag from a filename or manifest value."""
    match = COORD_RE.search(os.path.basename(str(value)))
    return match.group(1) if match else None


def coords_mm_from_tag(tag):
    """Parse (x, y, z) in mm from a coordinate tag (tag values are in m)."""
    match = COORD_VALUES_RE.search(str(tag))
    if not match:
        return (np.nan, np.nan, np.nan)
    return tuple(round(float(v) * 1000.0, 1) for v in match.groups())


def coordinate_tag_from_mm(x, y, z):
    """Build a coordinate tag from voxel-center coordinates in mm."""
    return (
        f"x{x / 1000:+.3f}_"
        f"y{y / 1000:+.3f}_"
        f"z{z / 1000:+.3f}"
    )


# ============================================================
# LOAD MANIFESTS
# ============================================================

def load_manifests(folder):
    """Map each coordinate tag to every dataset containing that voxel."""
    tag_to_datasets = {}
    manifest_tags = {}
    manifest_rows = {}

    for dataset in DATASETS:
        manifest = os.path.join(folder, "hits_distributions", f"voxels_{dataset}.csv")

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

            tag_to_datasets.setdefault(tag, set()).add(dataset)
            manifest_tags[dataset].add(tag)

    overlaps = {
        tag: datasets
        for tag, datasets in tag_to_datasets.items()
        if len(datasets) > 1
    }

    print(f"Loaded {len(DATASETS)} manifests.")
    print(f"Unique coordinate tags: {len(tag_to_datasets)}")
    print(f"Coordinate tags shared between datasets: {len(overlaps)}")

    return tag_to_datasets, {"tags": manifest_tags, "rows": manifest_rows}


# ============================================================
# SCAN SIMULATION RESULTS
# ============================================================

def scan_results(folder, tag_to_datasets):
    """
    Read optical results and compute per-file SiPM statistics.

    Each entry in stp/optical/det_uid is one optical hit. A detector ID that
    appears more than once in a file is a channel with repeated hits.

    Returns
    -------
    optical_steps : dict[dataset] -> list of hit counts (one per file)
    frames        : dict[dataset] -> DataFrame, one row per file
    """
    optical_steps = {dataset: [] for dataset in DATASETS}
    records = {dataset: [] for dataset in DATASETS}
    unreadable = {dataset: 0 for dataset in DATASETS}
    unmatched = []
    excluded = {dataset: 0 for dataset in DATASETS}
    n_errors_printed = 0

    for fidelity, relative_dir in RESULT_DIRS.items():
        result_dir = os.path.join(folder, relative_dir)

        if not os.path.isdir(result_dir):
            print(f"Results directory not found: {result_dir}")
            continue

        files = sorted(
            os.path.join(result_dir, name)
            for name in os.listdir(result_dir)
            if name.endswith((".lh5", ".h5", ".hdf5"))
        )

        print(f"\nScanning {result_dir}: {len(files)} files")

        for i, filename in enumerate(files, start=1):
            if i % PROGRESS_EVERY == 0:
                print(f"  ... {i}/{len(files)}")

            tag = extract_coordinate_tag(filename)
            all_datasets = tag_to_datasets.get(tag) if tag else None

            if not all_datasets:
                unmatched.append(filename)
                continue

            # Keep only datasets that use this fidelity; skip the rest
            # without opening the file.
            datasets = {
                d for d in all_datasets
                if DATASET_FIDELITY.get(d) in (None, fidelity)
            }
            for d in all_datasets - datasets:
                excluded[d] += 1

            if not datasets:
                continue

            try:
                with h5py.File(filename, "r") as f:
                    n_evtid = len(f["stp/optical/evtid"])
                    det_uid = f["stp/optical/det_uid"][:]

                if n_evtid != len(det_uid):
                    raise ValueError(
                        f"evtid length ({n_evtid}) differs from "
                        f"det_uid length ({len(det_uid)})"
                    )

                n_hits = len(det_uid)
                _, per_channel = np.unique(det_uid, return_counts=True)
                n_channels = len(per_channel)
                n_single = int(np.count_nonzero(per_channel == 1))
                n_repeated = int(np.count_nonzero(per_channel > 1))

                x_mm, y_mm, z_mm = coords_mm_from_tag(tag)

                record = {
                    "filename": filename,
                    "tag": tag,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "z_mm": z_mm,
                    "fidelity": fidelity,
                    "n_hits": n_hits,
                    "n_channels": n_channels,
                    "n_single": n_single,
                    "n_repeated": n_repeated,
                }

                # Shared voxels contribute to every matching dataset.
                for dataset in datasets:
                    optical_steps[dataset].append(n_hits)
                    records[dataset].append(record)

            except Exception as exc:
                for dataset in datasets:
                    unreadable[dataset] += 1

                if n_errors_printed < MAX_ERRORS_PRINTED:
                    print(f"Could not read {filename}: {exc}")
                    n_errors_printed += 1
                elif n_errors_printed == MAX_ERRORS_PRINTED:
                    print("(further read errors suppressed)")
                    n_errors_printed += 1

    print(f"\nUnmatched files (no tag / not in any manifest): {len(unmatched)}")
    for filename in unmatched[:5]:
        print(f"  {filename}")

    for dataset in DATASETS:
        if unreadable[dataset]:
            print(f"Unreadable files in {dataset}: {unreadable[dataset]}")

    print("\nFiles ignored because they have the wrong fidelity for the dataset:")
    for dataset in DATASETS:
        print(f"  {dataset} (uses {DATASET_FIDELITY.get(dataset, 'any')}): "
              f"{excluded[dataset]} ignored")

    frames = {}
    columns = ["filename", "tag", "x_mm", "y_mm", "z_mm", "fidelity",
               "n_hits", "n_channels", "n_single", "n_repeated"]
    for dataset in DATASETS:
        df = pd.DataFrame(records[dataset], columns=columns)
        df["has_repeated"] = df["n_repeated"] > 0
        df["all_single"] = (df["n_hits"] > 0) & (df["n_repeated"] == 0)
        frames[dataset] = df

    return optical_steps, frames


# ============================================================
# SUMMARIES
# ============================================================

def _pct(num, den):
    return 100.0 * num / den if den else float("nan")


def print_no_hit_files(frames):
    """Number of files with zero optical hits, per dataset."""
    print("\nFILES WITH NO HITS")

    for dataset in DATASETS:
        df = frames[dataset]
        n_files = len(df)

        if n_files == 0:
            print(f"{dataset}: no readable files")
            continue

        n_zero = int((df["n_hits"] == 0).sum())
        print(
            f"{dataset}: {n_zero}/{n_files} files with no hits "
            f"({_pct(n_zero, n_files):.1f}%); "
            f"{n_files - n_zero} with hits"
        )


def print_repeated_hit_stats(frames):
    """How often repeated hits (>1 hit in one SiPM channel) occur."""
    print("\nREPEATED HITS (>1 hit in the same SiPM channel within a voxel)")

    for dataset in DATASETS:
        df = frames[dataset]
        n_files = len(df)

        if n_files == 0:
            print(f"{dataset}: no readable files")
            continue

        n_with_hits = int((df["n_hits"] > 0).sum())
        n_rep = int(df["has_repeated"].sum())

        print(
            f"{dataset}: repeated hits in {n_rep}/{n_files} files "
            f"({_pct(n_rep, n_files):.1f}% of all files; "
            f"{_pct(n_rep, n_with_hits):.1f}% of the {n_with_hits} "
            f"files with hits)"
        )


def print_single_hit_stats(frames):
    """
    Single-hit statistics.

    (a) Channel-voxel observations, pooled over files: how many distinct
        (channel, voxel) pairs received exactly one hit.
    (b) Files in which every hit channel received exactly one hit.
    """
    print("\nSINGLE-HIT CHANNEL-VOXEL OBSERVATIONS")

    for dataset in DATASETS:
        df = frames[dataset]
        n_files = len(df)

        if n_files == 0:
            print(f"{dataset}: no readable files")
            continue

        n_with_hits = int((df["n_hits"] > 0).sum())
        n_obs = int(df["n_channels"].sum())
        n_single_obs = int(df["n_single"].sum())
        n_all_single = int(df["all_single"].sum())

        print(f"{dataset}:")
        print(
            f"  Observations with exactly 1 hit: {n_single_obs}/{n_obs} "
            f"({_pct(n_single_obs, n_obs):.1f}%); "
            f"with >1 hit: {n_obs - n_single_obs}"
        )
        print(
            f"  Files where all hit channels are single-hit: "
            f"{n_all_single}/{n_files} "
            f"({_pct(n_all_single, n_files):.1f}% of all files; "
            f"{_pct(n_all_single, n_with_hits):.1f}% of files with hits)"
        )


def save_summary_csv(folder, frames):
    """Save one row per dataset with all headline numbers."""
    rows = []

    for dataset in DATASETS:
        df = frames[dataset]
        n_files = len(df)
        n_zero = int((df["n_hits"] == 0).sum())
        n_with_hits = n_files - n_zero
        n_obs = int(df["n_channels"].sum())
        n_single_obs = int(df["n_single"].sum())
        n_rep = int(df["has_repeated"].sum())
        n_all_single = int(df["all_single"].sum())

        rows.append({
            "dataset": dataset,
            "n_files": n_files,
            "n_no_hits": n_zero,
            "pct_no_hits": _pct(n_zero, n_files),
            "n_with_hits": n_with_hits,
            "n_files_repeated_hits": n_rep,
            "pct_files_repeated_of_all": _pct(n_rep, n_files),
            "pct_files_repeated_of_with_hits": _pct(n_rep, n_with_hits),
            "n_observations": n_obs,
            "n_single_hit_observations": n_single_obs,
            "pct_single_hit_observations": _pct(n_single_obs, n_obs),
            "n_files_all_single": n_all_single,
            "median_channels": float(df["n_channels"].median())
            if n_files else np.nan,
            "max_channels": int(df["n_channels"].max()) if n_files else 0,
        })

    path = os.path.join(folder, "hits_distributions", "coverage_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.3f")
    print(f"\nSaved: {path}")


# ============================================================
# BINNING FOR OPTICAL-STEP DISTRIBUTION
# ============================================================

def choose_bin_width(vmax, max_bins):
    """Smallest nice integer width so [0, vmax] fits in <= max_bins bins."""
    needed = int(np.ceil((vmax + 1) / max_bins))

    for exp in range(12):
        for m in (1, 2, 5):
            width = m * 10**exp
            if width >= needed:
                return width

    return needed


def make_bins(all_values, max_bins):
    """Shared, integer-aligned bins for all datasets."""
    vmax = int(max(all_values))
    width = choose_bin_width(vmax, max_bins)
    n_bins = int(np.ceil((vmax + 1) / width))
    edges = -0.5 + width * np.arange(n_bins + 1)

    if width == 1:
        labels = [str(i) for i in range(n_bins)]
    else:
        labels = [f"{i * width}-{(i + 1) * width - 1}" for i in range(n_bins)]

    return edges, width, labels


def percentage_table(optical_steps, edges, labels):
    """Percentage of simulations per optical-step bin."""
    table = {}

    for dataset in DATASETS:
        values = np.asarray(optical_steps[dataset], dtype=int)

        if len(values) == 0:
            table[dataset] = np.full(len(labels), np.nan)
            continue

        counts, _ = np.histogram(values, bins=edges)
        table[dataset] = 100.0 * counts / len(values)

    return pd.DataFrame(table, index=pd.Index(labels, name="optical_steps"))


# ============================================================
# SAVE HELPER
# ============================================================

def save_figure(fig, folder, stem):
    outdir = Path(folder).resolve() / "hits_distributions"
    outdir.mkdir(parents=True, exist_ok=True)
    path = os.path.join(outdir, f"{stem}.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")

# ============================================================
# PLOT 1: OPTICAL-HIT DISTRIBUTIONS
# ============================================================

def make_plot(folder, optical_steps, max_bins=MAX_BINS):
    """One panel per dataset: % of files vs. number of optical hits."""
    all_values = [v for d in DATASETS for v in optical_steps[d]]

    if not all_values:
        raise RuntimeError("No readable simulation results matched the manifests.")

    edges, width, labels = make_bins(all_values, max_bins)
    n_bins = len(labels)
    centers = 0.5 * (edges[:-1] + edges[1:])
    table = percentage_table(optical_steps, edges, labels)

    print("\nPERCENTAGE OF SIMULATIONS PER BIN")
    print(table.to_string(float_format=lambda x: f"{x:6.2f}"))
    table.to_csv(os.path.join(folder, "hits_distributions", "optical_steps_percentages.csv"),
                 float_format="%.4f")

    colors = plt.get_cmap("tab10").colors
    ymax = np.nanmax(table.values) * 1.18

    fig, axes = plt.subplots(
        len(DATASETS), 1,
        figsize=(7, 2.1 * len(DATASETS) + 1),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[:, 0]

    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        values = np.asarray(optical_steps[dataset], dtype=int)

        if len(values) == 0:
            ax.text(0.5, 0.5, f"{dataset}: no data",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        pct = table[dataset].values
        ax.bar(centers, pct, width=width * 0.9, color=colors[i])

        if n_bins <= LABEL_BARS_UNDER:
            for x, p in zip(centers, pct):
                if p > 0:
                    ax.text(x, p + ymax * 0.01, f"{p:.1f}",
                            ha="center", va="bottom", fontsize=7)

        ax.set_ylim(0, ymax)
        ax.set_ylabel("% of sims")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.98, 0.92,
            f"{dataset}: N={len(values)}, "
            f"mean={values.mean():.1f}, median={np.median(values):.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                      edgecolor="0.7"),
        )

    step = max(1, int(np.ceil(n_bins / 20)))
    axes[-1].set_xticks(centers[::step])
    axes[-1].set_xticklabels(
        labels[::step],
        rotation=0 if width == 1 else 45,
        ha="center" if width == 1 else "right",
    )
    axes[-1].set_xlim(edges[0], edges[-1])
    axes[-1].set_xlabel(
        "Number of optical hits per voxel"
        + ("" if width == 1 else f" (bin width = {width})")
    )

    fig.tight_layout()
    save_figure(fig, folder, "optical_steps_all_datasets")
    return fig


# ============================================================
# GROUPS (DATASET x FIDELITY) AND FIDELITY CHECK
# ============================================================

def get_groups(frames):
    """(label, DataFrame) for every dataset x fidelity combination present."""
    groups = []

    for dataset in DATASETS:
        df = frames[dataset]

        for fidelity in sorted(df["fidelity"].unique()):
            sub = df[df["fidelity"] == fidelity]
            if len(sub):
                groups.append((f"{dataset} [{fidelity.upper()}]", sub))

    return groups


def print_fidelity_breakdown(frames):
    """Files per dataset and fidelity, plus voxels that appear more than once."""
    print("\nFILES PER DATASET AND FIDELITY")

    for dataset in DATASETS:
        df = frames[dataset]

        if len(df) == 0:
            print(f"{dataset}: no readable files")
            continue

        counts = df["fidelity"].value_counts().sort_index()
        parts = ", ".join(f"{k.upper()}={v}" for k, v in counts.items())
        n_unique = df["tag"].nunique()
        n_multi_fid = int((df.groupby("tag")["fidelity"].nunique() > 1).sum())

        print(f"{dataset}: {len(df)} files ({parts}); {n_unique} unique voxels")

        if n_unique < len(df):
            print(
                f"  WARNING: {len(df) - n_unique} extra files; "
                f"{n_multi_fid} voxels have results at more than one fidelity"
            )


def print_dataset_check(frames, manifest_info):
    """Compare manifests, expected sizes and the simulation files found."""
    print("\nDATASET CHECK (manifest vs simulation files)")

    for dataset in DATASETS:
        df = frames[dataset]
        rows = manifest_info["rows"].get(dataset, 0)
        tags = manifest_info["tags"].get(dataset, set())
        found = set(df["tag"]) if len(df) else set()
        expected = EXPECTED_N.get(dataset)

        print(f"{dataset} (fidelity={DATASET_FIDELITY.get(dataset, 'any')}):")
        print(f"  manifest rows={rows}, unique voxels={len(tags)}, "
              f"expected={expected}")
        print(f"  files read={len(df)}, voxels with a file={len(found)}, "
              f"voxels without a file={len(tags - found)}, "
              f"extra files on duplicate voxels={len(df) - len(found)}")

        if rows != len(tags):
            print(f"  WARNING: {rows - len(tags)} duplicate rows in manifest")
        if expected is not None and len(tags) != expected:
            print(f"  WARNING: manifest has {len(tags)} voxels, "
                  f"expected {expected}")
        if len(df) > len(found):
            print("  WARNING: several files share one voxel tag "
                  "(repeated runs?)")

    print("\nVOXELS SHARED BETWEEN MANIFESTS")
    any_overlap = False
    for i, a in enumerate(DATASETS):
        for b in DATASETS[i + 1:]:
            n = len(manifest_info["tags"].get(a, set())
                    & manifest_info["tags"].get(b, set()))
            if n:
                any_overlap = True
                na = len(manifest_info["tags"][a])
                nb = len(manifest_info["tags"][b])
                print(f"  {a} & {b}: {n} voxels "
                      f"({100 * n / na:.1f}% of {a}, {100 * n / nb:.1f}% of {b})")
    if not any_overlap:
        print("  none")


def print_coordinate_ranges(frames):
    """Coordinate range per dataset, to compare with the intended box."""
    print("\nCOORDINATE RANGES (mm)")

    for dataset in DATASETS:
        df = frames[dataset]
        if len(df) == 0 or df["x_mm"].isna().all():
            print(f"{dataset}: no data")
            continue

        parts = [
            f"{c[0]} in [{df[c].min():.0f}, {df[c].max():.0f}]"
            for c in ("x_mm", "y_mm", "z_mm")
        ]
        print(f"{dataset}: " + ", ".join(parts))


def _grid_step(values):
    """Spacing of a coordinate grid (falls back to 5 mm)."""
    u = np.unique(values[np.isfinite(values)])
    d = np.diff(u)
    return float(d.min()) if len(d) else 5.0


# ============================================================
# PLOT 2: COVERAGE VS POSITION (MARGINALS)
# ============================================================

def plot_coverage_vs_position(folder, frames):
    """
    Rows: dataset x fidelity.  Columns: x, y, z.

    Each bar is one coordinate slice. Height = mean number of distinct SiPM
    channels hit per voxel in that slice (averaged over the other two
    coordinates): light blue = channels with exactly 1 hit, dark blue =
    channels with more than 1 hit. Y axis is shared along each row.
    """
    groups = [
        (label, df) for label, df in get_groups(frames)
        if df["x_mm"].notna().any()
    ]

    if not groups:
        print("No coordinate information available for position plots.")
        return None

    coords = ["x_mm", "y_mm", "z_mm"]
    fig, axes = plt.subplots(
        len(groups), 3,
        figsize=(13, 2.1 * len(groups) + 0.8),
        sharey="row", squeeze=False, layout="constrained",
    )

    for r, (label, df) in enumerate(groups):
        for c, coord in enumerate(coords):
            ax = axes[r, c]
            g = df.groupby(coord)[["n_single", "n_repeated"]].mean()
            step = _grid_step(g.index.to_numpy(dtype=float))

            ax.bar(g.index, g["n_single"], width=0.9 * step,
                   color=SINGLE_HIT_COLOR,
                   label="Exactly 1 hit in the same SiPM channel")
            ax.bar(g.index, g["n_repeated"], bottom=g["n_single"],
                   width=0.9 * step, color=REPEATED_HIT_COLOR,
                   label="More than 1 hit in the same SiPM channel")

            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

            if r == 0:
                ax.set_title(f"vs {coord[0]} position", fontsize=10)
            if r == len(groups) - 1:
                ax.set_xlabel(f"{coord[0]} (mm)")
            if c == 0:
                ax.set_ylabel("mean distinct\nSiPM channels")
                ax.text(0.98, 0.95, f"{label}  (N={len(df)})",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white",
                                  alpha=0.85, edgecolor="0.7"))

    axes[0, -1].legend(loc="upper right", fontsize=7)
    save_figure(fig, folder, "coverage_vs_position")
    return fig


# ============================================================
# PLOT 3: COVERAGE MAPS (2D PROJECTIONS)
# ============================================================

def plot_coverage_maps(folder, frames, metric="channels"):
    """
    Rows: dataset x fidelity.  Columns: xy, xz, yz projections.

    metric="channels": mean distinct SiPM channels per voxel
    metric="no_hits":  % of files with no hits
    Values are averaged over the coordinate not shown; blank cells are
    positions with no simulated voxel.
    """
    groups = [
        (label, df) for label, df in get_groups(frames)
        if df["x_mm"].notna().any()
    ]

    if not groups:
        print("No coordinate information available for map plots.")
        return None

    if metric == "channels":
        cmap, cbar_label, stem = "viridis", "mean distinct channels", "coverage_maps_channels"
    else:
        cmap, cbar_label, stem = "magma_r", "% of files with no hits", "coverage_maps_no_hits"

    projections = [("x_mm", "y_mm"), ("x_mm", "z_mm"), ("y_mm", "z_mm")]

    fig, axes = plt.subplots(
        len(groups), 3,
        figsize=(13, 2.6 * len(groups) + 0.8),
        squeeze=False, layout="constrained",
    )

    for r, (label, df) in enumerate(groups):
        df = df.copy()
        df["value"] = (
            df["n_channels"] if metric == "channels"
            else 100.0 * (df["n_hits"] == 0)
        )
        vmax = None if metric == "channels" else 100.0
        mesh = None

        for c, (h, v) in enumerate(projections):
            ax = axes[r, c]
            s = df.groupby([v, h])["value"].mean().unstack(h)

            if s.shape[0] < 2 or s.shape[1] < 2:
                ax.text(0.5, 0.5, "not enough range", transform=ax.transAxes,
                        ha="center", va="center")
            else:
                mesh = ax.pcolormesh(
                    s.columns.to_numpy(float), s.index.to_numpy(float),
                    s.to_numpy(), shading="nearest", cmap=cmap,
                    vmin=0, vmax=vmax,
                )

            ax.set_aspect("equal", adjustable="box")
            if r == len(groups) - 1:
                ax.set_xlabel(f"{h[0]} (mm)")
            ax.set_ylabel(f"{v[0]} (mm)")
            if c == 0:
                ax.set_title(f"{label} (N={len(df)})", loc="left", fontsize=9)

        if mesh is not None:
            fig.colorbar(mesh, ax=axes[r, :].tolist(), pad=0.01,
                         fraction=0.03, label=cbar_label)

    save_figure(fig, folder, stem)
    return fig


# ============================================================
# OPTIONAL: PER-FILE STACKED BARS (FILE INDEX)
# ============================================================

def plot_channel_coverage(folder, frames, start=None, stop=None):
    """
    One stacked bar per file, one panel per dataset x fidelity.
    Light blue: channels with exactly 1 hit; dark blue: more than 1 hit.
    x axis = 1-based file index within the group (sorted filename order).
    """
    groups = get_groups(frames)

    if not groups:
        return None

    fig, axes = plt.subplots(
        len(groups), 1,
        figsize=(14, 2.3 * len(groups) + 0.5),
        sharey=False, squeeze=False, layout="constrained",
    )
    axes = axes[:, 0]

    for ax, (label, df) in zip(axes, groups):
        n_total = len(df)
        lo = 1 if start is None else max(1, start)
        hi = n_total if stop is None else min(n_total, stop)
        sub = df.iloc[lo - 1:hi]

        if len(sub) == 0:
            ax.text(0.5, 0.5, f"{label}: empty range",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        x = np.arange(lo, hi + 1)
        single = sub["n_single"].to_numpy()
        repeated = sub["n_repeated"].to_numpy()
        l1 = "Exactly 1 hit in the same SiPM channel"
        l2 = "More than 1 hit in the same SiPM channel"

        if len(sub) <= BARS_BELOW:
            ax.bar(x, single, width=0.85, color=SINGLE_HIT_COLOR, label=l1)
            ax.bar(x, repeated, bottom=single, width=0.85,
                   color=REPEATED_HIT_COLOR, label=l2)
        else:
            ax.fill_between(x, 0, single, step="mid", linewidth=0,
                            color=SINGLE_HIT_COLOR, label=l1)
            ax.fill_between(x, single, single + repeated, step="mid",
                            linewidth=0, color=REPEATED_HIT_COLOR, label=l2)

        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))

        n_zero = int((df["n_hits"] == 0).sum())
        n_rep = int(df["has_repeated"].sum())
        ax.set_ylabel("Distinct SiPM\nchannels")
        ax.set_title(
            f"{label}: {n_total} files; {n_zero} no hits "
            f"({_pct(n_zero, n_total):.1f}%); "
            f"{n_rep} with repeated hits ({_pct(n_rep, n_total):.1f}%)",
            loc="left", fontsize=10,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("File index within group (sorted filename order)")
    save_figure(fig, folder, "sipm_channel_coverage")
    return fig


# ============================================================
# PLOT 4: DISTINCT SIPM CHANNELS PER VOXEL
# ============================================================

def plot_distinct_channels(folder, frames, log_y=False):
    """Histogram: number of files vs. distinct SiPM channels hit in a voxel."""
    all_counts = np.concatenate(
        [frames[d]["n_channels"].to_numpy() for d in DATASETS]
    )

    if len(all_counts) == 0:
        print("No per-file channel data available for histogram.")
        return None

    max_channels = int(all_counts.max())
    bins = np.arange(-0.5, max_channels + 1.5, 1)
    label_bars = max_channels + 1 <= LABEL_BARS_UNDER

    fig, axes = plt.subplots(
        len(DATASETS), 1,
        figsize=(9, 2.4 * len(DATASETS)),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, dataset in zip(axes, DATASETS):
        counts = frames[dataset]["n_channels"].to_numpy()

        if len(counts) == 0:
            ax.text(0.5, 0.5, f"{dataset}: no data",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        n, _, _ = ax.hist(counts, bins=bins, color="steelblue",
                          edgecolor="black", linewidth=0.5)

        if label_bars:
            for xc, h in zip(np.arange(max_channels + 1), n):
                if h > 0:
                    ax.text(xc, h, f"{int(h)}", ha="center", va="bottom",
                            fontsize=7)

        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, n.max() * 1.15)

        n_zero = int((counts == 0).sum())
        ax.set_ylabel("Number of files")
        ax.set_title(
            f"{dataset}: N={len(counts)}; {n_zero} with 0 channels "
            f"({_pct(n_zero, len(counts)):.1f}%); "
            f"median={np.median(counts):.0f}; max={counts.max()}",
            loc="left", fontsize=10,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[-1].set_xlim(-0.5, max_channels + 0.5)
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    axes[-1].set_xlabel("Distinct SiPM channels hit in one voxel")

    fig.tight_layout()
    save_figure(fig, folder, "distinct_sipm_channels")
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optical hits and SiPM channel coverage for LF/HF datasets."
    )
    parser.add_argument("folder",
                        help="Voxel-set folder containing manifests and results/")
    parser.add_argument("--coverage-start", type=int, default=None,
                        help="First file index (1-based) for the stacked-bar plot")
    parser.add_argument("--coverage-stop", type=int, default=None,
                        help="Last file index (1-based, inclusive) for the stacked-bar plot")
    parser.add_argument("--per-file", action="store_true",
                        help="Also make the per-file-index stacked-bar plot")
    parser.add_argument("--log-y", action="store_true",
                        help="Log y axis for the distinct-channel histogram")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without opening windows")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        parser.error(f"Folder does not exist: {folder}")

    tag_to_datasets, manifest_info = load_manifests(folder)
    optical_steps, frames = scan_results(folder, tag_to_datasets)

    print_dataset_check(frames, manifest_info)
    print_coordinate_ranges(frames)
    print_no_hit_files(frames)
    print_fidelity_breakdown(frames)
    print_repeated_hit_stats(frames)
    print_single_hit_stats(frames)
    save_summary_csv(folder, frames)

    make_plot(folder, optical_steps)
    plot_coverage_vs_position(folder, frames)
    plot_coverage_maps(folder, frames, metric="channels")
    plot_coverage_maps(folder, frames, metric="no_hits")
    plot_distinct_channels(folder, frames, log_y=args.log_y)
    if args.per_file:
        plot_channel_coverage(folder, frames,
                              start=args.coverage_start,
                              stop=args.coverage_stop)

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()