#!/usr/bin/env python3
"""Plot and audit the five voxel datasets created by sample_voxels_fixed.py.

Usage:
    python plot_voxel_samples.py voxel_sets_<grid-name>
    python plot_voxel_samples.py voxel_sets_<grid-name> --outdir qc_plots
    python plot_voxel_samples.py voxel_sets_<grid-name> --show

Writes combined and individual 2D projections, combined and individual 3D views,
2D occupancy heatmaps, 1D coordinate-coverage histograms, and a text overlap audit.
Coordinates are voxel centres in mm.
"""
import argparse
import csv
import json
import os
from itertools import combinations

import numpy as np

NAMES = ["LF_train", "LF2_train", "HF_train", "CNP_val", "GP_val"]
STYLE = {
    "LF_train": {"color": "#7F77DD", "marker": "o", "size": 18},
    "LF2_train": {"color": "#3C8DBC", "marker": "o", "size": 18},
    "HF_train": {"color": "#D85A30", "marker": "s", "size": 42},
    "CNP_val": {"color": "#1D9E75", "marker": "^", "size": 24},
    "GP_val": {"color": "#E8B72B", "marker": "D", "size": 36},
}
PLANES = [
    (0, 1, "x [mm]", "y [mm]", "XY"),
    (0, 2, "x [mm]", "z [mm]", "XZ"),
    (1, 2, "y [mm]", "z [mm]", "YZ"),
]


def read_points(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    p = np.array([[float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])] for r in rows])
    idx = np.array([[int(r["ix"]), int(r["iy"]), int(r["iz"])] for r in rows])
    return p, idx


def set_keyset(indices):
    return {tuple(map(int, row)) for row in indices}


def audit(data, outdir):
    keys = {n: set_keyset(data[n][1]) for n in NAMES}
    allowed_pairs = {("LF2_train", "HF_train"), ("CNP_val", "GP_val")}
    pairs = []
    for a, b in combinations(NAMES, 2):
        count = len(keys[a] & keys[b])
        allowed = (a, b) in allowed_pairs
        pairs.append((a, b, count, allowed))
    lines = ["Voxel overlap audit", "===================", "", "Dataset counts:"]
    for n in NAMES:
        lines.append(f"  {n:12s}: {len(keys[n]):6d}")
    lines += ["", "Pairwise shared voxel indices:"]
    for a, b, count, allowed in pairs:
        status = "ALLOWED" if allowed else "MUST BE ZERO"
        lines.append(f"  {a:12s} vs {b:12s}: {count:6d}  [{status}]")
        if count and not allowed:
            raise AssertionError(f"Forbidden overlap: {a} and {b} share {count} voxels")
    if not (keys["LF2_train"] & keys["HF_train"]):
        lines.append("\nNOTE: LF2_train and HF_train have no shared voxels.")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "overlap_audit.txt"), "w") as f:
        f.write(report)
    print(report)


def load_bounds(sample_dir, all_points):
    """Use configured sampling-box bounds when available, otherwise data bounds."""
    cfg_path = os.path.join(sample_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            lo, hi = np.asarray(cfg["box_min"], float), np.asarray(cfg["box_max"], float)
            if lo.shape == (3,) and hi.shape == (3,) and np.all(hi > lo):
                return lo, hi
        except (OSError, ValueError, KeyError, TypeError):
            pass
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    pad = np.maximum((hi - lo) * 0.03, 1.0)
    return lo - pad, hi + pad


def draw_box_3d(ax, lo, hi, color="black", linestyle="--", linewidth=0.8):
    # Draw all 12 edges of the configured sampling box.
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            if np.count_nonzero(np.abs(a - b) > 1e-9) == 1:
                ax.plot(*zip(a, b), color=color, linestyle=linestyle,
                        linewidth=linewidth, alpha=0.65)


def finish_3d(ax, lo, hi):
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    #ax.set_box_aspect(np.maximum(hi - lo, 1.0))
    ax.view_init(elev=23, azim=-50)
    draw_box_3d(ax, lo, hi)


def _nearest_slice(data, axis, requested=None):
    """Choose an actual sampled voxel-centre layer nearest the requested coordinate."""
    coords = np.concatenate([data[n][0][:, axis] for n in NAMES if len(data[n][0])])
    if coords.size == 0:
        raise ValueError("No sampled points available for slice selection")
    target = float(np.median(coords)) if requested is None else float(requested)
    layers = np.unique(coords)
    return float(layers[np.argmin(np.abs(layers - target))])


def plot_combined_2d(data, outdir, lo, hi, plt):
    """Plot every sampled voxel projected across the full volume (original behavior)."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (i, j, xl, yl, plane) in zip(axes, PLANES):
        for name in NAMES:
            p = data[name][0]
            st = STYLE[name]
            ax.scatter(p[:, i], p[:, j], c=st["color"], marker=st["marker"],
                       s=st["size"], alpha=0.75, linewidths=0,
                       label=f"{name} (n={len(p)})", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{plane} full-volume projection")
        ax.set_xlim(lo[i], hi[i])
        ax.set_ylim(lo[j], hi[j])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Voxel sample locations: all datasets, full-volume projections")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "voxel_samples_2d_combined.pdf"), dpi=180)
    plt.close(fig)


def plot_sliced_2d(data, outdir, lo, hi, plt, slice_x=None, slice_y=None, slice_z=None):
    """Plot one voxel-centre layer per plane in a separate PDF.

    XY selects a single z layer, XZ a single y layer, and YZ a single x layer.
    """
    requested = {0: slice_x, 1: slice_y, 2: slice_z}
    slice_axes = {"XY": 2, "XZ": 1, "YZ": 0}
    slice_values = {
        plane: _nearest_slice(data, axis, requested[axis])
        for plane, axis in slice_axes.items()
    }
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (i, j, xl, yl, plane) in zip(axes, PLANES):
        omitted_axis = slice_axes[plane]
        slice_value = slice_values[plane]
        all_coords = np.concatenate([data[n][0][:, omitted_axis] for n in NAMES if len(data[n][0])])
        unique = np.unique(all_coords)
        diffs = np.diff(unique)
        diffs = diffs[diffs > 1e-8]
        tol = max(1e-7, float(np.min(diffs)) * 1e-5 if diffs.size else 1e-7)
        for name in NAMES:
            p = data[name][0]
            st = STYLE[name]
            selected = p[np.isclose(p[:, omitted_axis], slice_value, rtol=0.0, atol=tol)]
            if len(selected):
                ax.scatter(selected[:, i], selected[:, j], c=st["color"], marker=st["marker"],
                           s=st["size"], alpha=0.88, linewidths=0.45,
                           edgecolors="black", label=f"{name} (n={len(selected)})",
                           rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{plane} slice at {'xyz'[omitted_axis]} = {slice_value:g} mm")
        ax.set_xlim(lo[i], hi[i])
        ax.set_ylim(lo[j], hi[j])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Voxel sample locations: combined single-layer 2D views")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "voxel_samples_2d_slices.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Single-layer slices saved to voxel_samples_2d_slices.pdf:")
    for plane, value in slice_values.items():
        print(f"  {plane}: {'xyz'[slice_axes[plane]]} = {value:g} mm")


def plot_individual_2d(data, outdir, lo, hi, plt):
    # One figure per dataset, with XY/XZ/YZ panels. Easier to see sparse sets than overlaying.
    for name in NAMES:
        p = data[name][0]
        st = STYLE[name]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
        for ax, (i, j, xl, yl, plane) in zip(axes, PLANES):
            ax.scatter(p[:, i], p[:, j], c=st["color"], marker=st["marker"],
                       s=st["size"], alpha=0.82, linewidths=0, rasterized=True)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"{plane} projection")
            ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[j], hi[j])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
        fig.suptitle(f"{name}: {len(p)} voxel centres")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name.lower()}_2d.pdf"), dpi=180)
        plt.close(fig)


def plot_3d(data, outdir, lo, hi, plt):
    # Individual panels and an overlay.
    fig = plt.figure(figsize=(20, 12))
    for k, name in enumerate(NAMES, start=1):
        ax = fig.add_subplot(2, 3, k, projection="3d")
        p, st = data[name][0], STYLE[name]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=st["color"], marker=st["marker"],
                   s=st["size"], alpha=0.82, depthshade=False)
        ax.set_title(f"{name}: {len(p)} voxels")
        finish_3d(ax, lo, hi)
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    for name in NAMES:
        p, st = data[name][0], STYLE[name]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=st["color"], marker=st["marker"],
                   s=st["size"], alpha=0.72, label=name, depthshade=False)
    ax.set_title("All datasets overlaid")
    ax.legend(fontsize=8, loc="best")
    finish_3d(ax, lo, hi)
    fig.suptitle("Voxel sample locations: 3D views (dashed outline = sampling box)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "voxel_samples_3d.pdf"), dpi=180)
    plt.close(fig)


def plot_occupancy(data, outdir, lo, hi, plt):
    """Binned projections make spatial clumping easier to diagnose than point clouds alone."""
    nbins = 12
    fig, axes = plt.subplots(len(NAMES), 3, figsize=(16, 3.0 * len(NAMES)))
    for row_i, name in enumerate(NAMES):
        p, st = data[name][0], STYLE[name]
        for col_i, (i, j, xl, yl, plane) in enumerate(PLANES):
            ax = axes[row_i, col_i]
            h = ax.hist2d(p[:, i], p[:, j], bins=nbins,
                          range=[[lo[i], hi[i]], [lo[j], hi[j]]], cmin=0)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"{name}: {plane} occupancy")
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.03, label="voxel count")
    fig.suptitle("Spatial occupancy by dataset (same box and binning for every dataset)", y=1.002)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spatial_occupancy_heatmaps.pdf"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # A uniformly sampled rectangular box should have approximately flat 1D counts.
    fig, axes = plt.subplots(len(NAMES), 3, figsize=(15, 2.6 * len(NAMES)))
    for row_i, name in enumerate(NAMES):
        p, st = data[name][0], STYLE[name]
        for col_i, (axis_i, axis_name) in enumerate(((0, "x"), (1, "y"), (2, "z"))):
            ax = axes[row_i, col_i]
            ax.hist(p[:, axis_i], bins=nbins, range=(lo[axis_i], hi[axis_i]),
                    color=st["color"], edgecolor="black", linewidth=0.35)
            ax.set_xlim(lo[axis_i], hi[axis_i])
            ax.set_xlabel(f"{axis_name} [mm]"); ax.set_ylabel("count")
            ax.set_title(f"{name}: {axis_name} coverage")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Coordinate coverage (flat is expected for a fully eligible rectangular box)", y=1.002)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "coordinate_coverage.pdf"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)



def plot_depth_coloured_projections(data, outdir, lo, hi, plt):
    """Show the third coordinate as colour in each 2D projection.

    XY is coloured by Z, XZ by Y, and YZ by X. This makes the coordinate
    omitted from each projection visible and helps diagnose apparent overlaps.
    """
    specs = [
        # horizontal axis, vertical axis, colour axis, labels, plane
        (0, 1, 2, "x [mm]", "y [mm]", "z [mm]", "XY (colour = Z)"),
        (0, 2, 1, "x [mm]", "z [mm]", "y [mm]", "XZ (colour = Y)"),
        (1, 2, 0, "y [mm]", "z [mm]", "x [mm]", "YZ (colour = X)"),
    ]
    fig, axes = plt.subplots(len(NAMES), 3, figsize=(18, 3.35 * len(NAMES)), squeeze=False)
    for row_i, name in enumerate(NAMES):
        p = data[name][0]
        for col_i, (i, j, k, xl, yl, cl, title) in enumerate(specs):
            ax = axes[row_i, col_i]
            sc = ax.scatter(p[:, i], p[:, j], c=p[:, k], cmap="viridis",
                            vmin=lo[k], vmax=hi[k], s=20, alpha=0.9,
                            linewidths=0, rasterized=True)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{name}: {title}  (n={len(p)})")
            ax.set_xlim(lo[i], hi[i])
            ax.set_ylim(lo[j], hi[j])
            # Auto aspect uses the available panel space and avoids hiding
            # structure when the selected x/y/z ranges have very different sizes.
            ax.set_aspect("auto")
            ax.grid(alpha=0.2)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.025)
            cb.set_label(cl)
    fig.suptitle("Voxel projections with the third coordinate encoded by colour", y=1.002)
    fig.tight_layout()
    path = os.path.join(outdir, "depth_coloured_projections_XY_XZ_YZ.pdf")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Also save a focused XY/Z diagnostic: XY position coloured by Z, with
    # horizontal Z slices summarized separately to expose layering/gaps.
    fig, axes = plt.subplots(1, len(NAMES), figsize=(5 * len(NAMES), 5), squeeze=False)
    for col_i, name in enumerate(NAMES):
        p = data[name][0]
        ax = axes[0, col_i]
        sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="viridis",
                        vmin=lo[2], vmax=hi[2], s=22, alpha=0.9,
                        linewidths=0, rasterized=True)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{name}: XY coloured by Z")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("auto")
        ax.grid(alpha=0.2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.025)
        cb.set_label("z [mm]")
    fig.suptitle("XY distribution: check how sampled voxels span Z", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "XY_projection_coloured_by_Z.pdf"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample_dir", help="directory containing voxels_<DATASET>.csv")
    ap.add_argument("--outdir", default=None, help="plot output directory; default: <sample_dir>/plots")
    ap.add_argument("--show", action="store_true", help="open interactive windows after saving")
    ap.add_argument("--slice-x", type=float, default=None,
                    help="slice coordinate in mm for the YZ panel; defaults to median sampled x layer")
    ap.add_argument("--slice-y", type=float, default=None,
                    help="slice coordinate in mm for the XZ panel; defaults to median sampled y layer")
    ap.add_argument("--slice-z", type=float, default=None,
                    help="slice coordinate in mm for the XY panel; defaults to median sampled z layer")
    a = ap.parse_args()
    try:
        import matplotlib
        if not a.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required: pip install matplotlib") from e

    outdir = os.path.abspath(a.outdir or os.path.join(a.sample_dir, "plots"))
    os.makedirs(outdir, exist_ok=True)
    data = {}
    for name in NAMES:
        path = os.path.join(a.sample_dir, f"voxels_{name}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}; run sample_voxels_fixed.py first")
        data[name] = read_points(path)

    audit(data, outdir)
    all_points = np.vstack([data[n][0] for n in NAMES])
    lo, hi = load_bounds(a.sample_dir, all_points)
    plot_combined_2d(data, outdir, lo, hi, plt)
    plot_sliced_2d(data, outdir, lo, hi, plt,
                   slice_x=a.slice_x, slice_y=a.slice_y, slice_z=a.slice_z)
    plot_individual_2d(data, outdir, lo, hi, plt)
    plot_3d(data, outdir, lo, hi, plt)
    plot_occupancy(data, outdir, lo, hi, plt)
    plot_depth_coloured_projections(data, outdir, lo, hi, plt)

    print(f"Plots and audit saved to: {outdir}")
    print("  voxel_samples_2d_combined.pdf  (all sets, full-volume XY/XZ/YZ projections)")
    print("  voxel_samples_2d_slices.pdf    (all sets, fixed-layer XY/XZ/YZ projections)")
    print("  voxel_samples_3d.pdf           (individual 3D panels + overlay)")
    print("  spatial_occupancy_heatmaps.pdf (XY/XZ/YZ occupancy heatmaps per dataset)")
    print("  coordinate_coverage.pdf        (x/y/z coverage histograms per dataset)")
    print("  depth_coloured_projections_XY_XZ_YZ.pdf (all planes, omitted coordinate shown by colour)")
    print("  XY_projection_coloured_by_Z.pdf (focused XY/Z diagnostic)")
    for name in NAMES:
        print(f"  {name.lower()}_2d.pdf" + " " * max(1, 24 - len(name)) + "(individual XY/XZ/YZ)")
    print("  overlap_audit.txt")

    if a.show:
        # Optional convenience: show a final combined figure interactively from the saved image.
        img = plt.imread(os.path.join(outdir, "voxel_samples_2d_combined.pdf"))
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(img); ax.axis("off"); plt.show(); plt.close(fig)


if __name__ == "__main__":
    main()
