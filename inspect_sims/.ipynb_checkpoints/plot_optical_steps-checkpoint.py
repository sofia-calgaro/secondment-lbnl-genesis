
#!/usr/bin/env python3

import argparse
import os
import re

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASETS = [
    "LF_train",
    "LF2_train",
    "HF_train",
    "CNP_val",
    "GP_val",
]

RESULT_DIRS = {
    "lf": os.path.join("results", "lf"),
    "hf": os.path.join("results", "hf"),
}

COORD_RE = re.compile(
    r"(x[+-]\d+\.\d+_y[+-]\d+\.\d+_z[+-]\d+\.\d+)"
)


def extract_coordinate_tag(value):
    """Extract x/y/z coordinate tag from a filename, path, or manifest tag."""
    match = COORD_RE.search(os.path.basename(str(value)))
    return match.group(1) if match else None


def coordinate_tag_from_mm(x, y, z):
    """Fallback: build a coordinate tag from voxel-center coordinates in mm."""
    return (
        f"x{x / 1000:+.3f}_"
        f"y{y / 1000:+.3f}_"
        f"z{z / 1000:+.3f}"
    )


def load_manifests(folder):
    """
    Map each coordinate tag to every dataset containing that voxel.
    Shared voxels are intentionally allowed across some datasets.
    """
    tag_to_datasets = {}

    for dataset in DATASETS:
        manifest = os.path.join(folder, f"voxels_{dataset}.csv")

        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        df = pd.read_csv(manifest)

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

    overlaps = {
        tag: datasets
        for tag, datasets in tag_to_datasets.items()
        if len(datasets) > 1
    }

    print(f"Loaded {len(DATASETS)} manifests.")
    print(f"Unique coordinate tags: {len(tag_to_datasets)}")
    print(f"Coordinate tags shared between datasets: {len(overlaps)}")

    return tag_to_datasets


def scan_results(folder, tag_to_datasets):
    """Read available result files and count optical-step entries."""
    optical_steps = {dataset: [] for dataset in DATASETS}
    unreadable = {dataset: [] for dataset in DATASETS}
    unmatched = []

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

        for filename in files:
            tag = extract_coordinate_tag(filename)

            if tag is None:
                unmatched.append(filename)
                print(f"Could not extract coordinate tag: {filename}")
                continue

            datasets = tag_to_datasets.get(tag)

            if not datasets:
                unmatched.append(filename)
                print(f"No manifest entry for {os.path.basename(filename)}")
                continue

            try:
                with h5py.File(filename, "r") as f:
                    n_steps = len(f["stp/optical/evtid"])

                # A shared voxel contributes to every matching dataset.
                for dataset in datasets:
                    optical_steps[dataset].append(n_steps)

            except Exception as exc:
                for dataset in datasets:
                    unreadable[dataset].append(filename)

                print(f"Could not read {filename}: {exc}")

    print("\nSUMMARY")
    for dataset in DATASETS:
        values = np.asarray(optical_steps[dataset], dtype=int)

        if len(values):
            print(
                f"{dataset}: N={len(values)}, "
                f"mean={values.mean():.1f}, "
                f"median={np.median(values):.1f}, "
                f"min={values.min()}, max={values.max()}"
            )
        else:
            print(f"{dataset}: no readable simulation files")

        print(f"  Unreadable files: {len(unreadable[dataset])}")

    print(f"\nUnmatched files: {len(unmatched)}")
    for filename in unmatched[:10]:
        print(f"  {filename}")

    return optical_steps


def make_plot(folder, optical_steps, bins=25):
    """Plot the optical-step distributions and save PDF and PNG."""
    all_values = [
        value
        for dataset in DATASETS
        for value in optical_steps[dataset]
    ]

    if not all_values:
        raise RuntimeError(
            "No readable simulation results matched the manifests. "
            "Check the result filenames and manifest tags."
        )

    bin_edges = np.histogram_bin_edges(all_values, bins=bins)

    fig, ax = plt.subplots(figsize=(11, 7))

    for dataset in DATASETS:
        values = np.asarray(optical_steps[dataset], dtype=int)

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.45,
            weights=np.ones(len(values)) / len(values),
            label=(
                f"{dataset}: N={len(values)}, "
                f"mean={values.mean():.0f}, "
                f"median={np.median(values):.0f}"
            ),
        )

    ax.set_xlabel("Number of optical steps per simulation")
    ax.set_ylabel("Fraction of simulations per bin")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("optical_steps_all_datasets.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot optical-step distributions for LF/HF training "
            "and validation datasets."
        )
    )
    parser.add_argument(
        "folder",
        help="Path to the voxel-set folder containing manifests and results/",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="Number of histogram bins (default: 25)",
    )

    args = parser.parse_args()
    folder = os.path.abspath(args.folder)

    if not os.path.isdir(folder):
        parser.error(f"Folder does not exist: {folder}")

    if args.bins < 1:
        parser.error("--bins must be at least 1")

    tag_to_datasets = load_manifests(folder)
    optical_steps = scan_results(folder, tag_to_datasets)
    make_plot(folder, optical_steps, bins=args.bins)


if __name__ == "__main__":
    main()
