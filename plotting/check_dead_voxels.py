"""
Plot the voxel centres stored in a voxel-space file (output of make_voxel_space.py)
together with the ULAr boundary from the GDML, cut at --zcut.

  python plot_voxel_space.py try_dead_mass --gdml l1000.gdml --vox 10
  python plot_voxel_space.py try_dead_mass --gdml l1000.gdml --vox 10 \
      --dead try_dead_mass_dead_voxels.csv.gz --zcut 1500

Writes <prefix>_rz.png, <prefix>_xz.png, <prefix>_3d.png (and <prefix>_dead_xy.png with --dead)
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_file(path):
    for cand in (path, path + ".csv", path + ".csv.gz"):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"voxel file not found: {path}")


def load_idx(path):
    """Voxel-space file with a header (ix,iy,iz)."""
    path = find_file(path)
    df = pd.read_csv(path, comment="#")
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if all(c in cols for c in ("ix", "iy", "iz")):
        idx = df[["ix", "iy", "iz"]].to_numpy()
    else:
        idx = df.select_dtypes("number").iloc[:, :3].to_numpy()
    print(f"read {len(idx):,} voxels from {path} (columns: {cols[:6]})")
    return idx.astype(np.int64)


def load_dead_idx(path):
    """*_dead_voxels.csv.gz is written by np.savetxt with a '# ix,iy,iz' comment header,
    so read it with loadtxt (pandas comment='#' would swallow the first voxel)."""
    idx = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    print(f"read {len(idx):,} dead voxels from {path}")
    return idx


def load_profile(gdml, name):
    import pyg4ometry as pg
    from pyg4ometry.gdml import Units
    reg = pg.gdml.Reader(gdml).getRegistry()
    s = reg.logicalVolumeDict[name].solid
    f = Units.unit(s.lunit)
    return (np.array([e.eval() for e in s.pR]) * f,
            np.array([e.eval() for e in s.pZ]) * f)


def clip_profile(r, z, zcut):
    """Cut the (r, z) profile at z = zcut; r is interpolated at the crossing."""
    if z.max() <= zcut:
        return r, z
    i = int(np.argmax(z > zcut))            # first point above the cut
    t = (zcut - z[i - 1]) / (z[i] - z[i - 1])
    r_cut = r[i - 1] + t * (r[i] - r[i - 1])
    return np.r_[r[:i], r_cut], np.r_[z[:i], zcut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space", help="voxel-space file (e.g. try_dead_mass)")
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--ular-name", default="liquid_argon_underground")
    ap.add_argument("--vox", type=float, default=10.0, help="voxel size [mm]")
    ap.add_argument("--gdml-offset", type=float, nargs=3, default=[0, 0, 0])
    ap.add_argument("--dead", default=None, help="optional *_dead_voxels.csv.gz to overlay")
    ap.add_argument("--zcut", type=float, default=1500.0,
                    help="cut the ULAr boundary at this z [mm] (detector frame)")
    ap.add_argument("--slab", type=float, default=None,
                    help="half-thickness of the |y| slab in the xz view [mm] (default: voxel size)")
    ap.add_argument("--max-points", type=int, default=300_000, help="max points in the 2D plots")
    ap.add_argument("--max-points-3d", type=int, default=20_000,
                    help="max voxel centres in the 3D plot (only with --show-centres)")
    ap.add_argument("--max-dead-3d", type=int, default=5_000,
                    help="max dead voxels in the 3D plot")
    ap.add_argument("--show-centres", action="store_true",
                    help="also draw the (subsampled) voxel centres in the 3D plot")
    ap.add_argument("--out", default=None, help="output prefix (default: space file name)")
    a = ap.parse_args()

    prefix = a.out or a.space.replace(".csv.gz", "").replace(".csv", "")
    slab = a.slab if a.slab is not None else a.vox
    off = np.array(a.gdml_offset, float)
    rng = np.random.default_rng(1)

    cen = (load_idx(a.space) + 0.5) * a.vox           # grid origin (0,0,0): centre = (i+0.5)*size
    dead = None
    if a.dead and os.path.isfile(a.dead):
        dead = (load_dead_idx(a.dead) + 0.5) * a.vox
        print(f"{len(dead):,} dead voxels overlaid")

    pr, pz = load_profile(a.gdml, a.ular_name)
    pz = pz - off[2]                                   # GDML frame -> detector frame
    pr, pz = clip_profile(pr, pz, a.zcut)              # cut the boundary at zcut
    zlo, zhi = pz.min(), pz.max()

    def sub(c, n):
        return c if len(c) <= n else c[rng.choice(len(c), n, replace=False)]

    print(f"voxel centres: x[{cen[:,0].min():g},{cen[:,0].max():g}] "
          f"y[{cen[:,1].min():g},{cen[:,1].max():g}] z[{cen[:,2].min():g},{cen[:,2].max():g}] mm")

    # ---- string / level counting from the dead voxels ---------------------------------------
    if dead is not None:
        from scipy import ndimage
        di = np.floor(dead / a.vox).astype(int)
        xy = np.unique(di[:, :2], axis=0)
        mn = xy.min(0)
        grid = np.zeros(tuple(xy.max(0) - mn + 3), bool)
        grid[xy[:, 0] - mn[0] + 1, xy[:, 1] - mn[1] + 1] = True
        lab, n_str = ndimage.label(grid, structure=np.ones((3, 3)))
        print("strings found:", n_str)                                  # expect 42
        counts = []
        for k in range(1, n_str + 1):
            m = lab[di[:, 0] - mn[0] + 1, di[:, 1] - mn[1] + 1] == k
            zs = np.unique(di[m, 2])
            counts.append(1 + int((np.diff(zs) > 1).sum()))
        print("levels per string:", sorted(Counter(counts).items()))    # expect {8: 42}

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(dead[:, 0], dead[:, 1], s=1, c="C3", rasterized=True)
        ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"Dead voxels projected on xy ({n_str} strings)")
        fig.savefig(prefix + "_dead_xy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- 1. (r, z): every voxel against the ULAr profile -------------------------------
    c = sub(cen, a.max_points)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(np.hypot(c[:, 0], c[:, 1]), c[:, 2], s=1, c="C0", alpha=0.3,
               rasterized=True, label=f"voxel centres ({len(cen):,})")
    if dead is not None:
        d = sub(dead, a.max_points)
        ax.scatter(np.hypot(d[:, 0], d[:, 1]), d[:, 2], s=2, c="C3", alpha=0.6,
                   rasterized=True, label=f"dead voxels ({len(dead):,})")
    ax.plot(pr, pz, "k-", lw=1.2, label=f"ULAr boundary (z ≤ {a.zcut:g} mm)")
    ax.plot([0, pr[-1]], [pz[-1], pz[-1]], "k-", lw=1.2)            # cut line
    ax.set_ylim(zlo - 100, zhi + 100)
    ax.set_xlabel("r = sqrt(x²+y²) [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title("Voxel centres in (r, z)")
    ax.legend(loc="lower left", markerscale=6)
    fig.savefig(prefix + "_rz.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. x-z slab through y = 0, profile mirrored ----------------------------------------
    m = np.abs(cen[:, 1]) <= slab
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(cen[m, 0], cen[m, 2], s=2, c="C0", alpha=0.6, rasterized=True,
               label=f"voxel centres, |y| ≤ {slab:g} mm")
    if dead is not None:
        md = np.abs(dead[:, 1]) <= slab
        ax.scatter(dead[md, 0], dead[md, 2], s=3, c="C3", rasterized=True, label="dead voxels")
    ax.plot(pr, pz, "k-", lw=1.2, label=f"ULAr boundary (z ≤ {a.zcut:g} mm)")
    ax.plot(-pr, pz, "k-", lw=1.2)
    ax.plot([-pr[-1], pr[-1]], [pz[-1], pz[-1]], "k-", lw=1.2)      # cut line
    ax.set_ylim(zlo - 100, zhi + 100)
    ax.set_aspect("equal"); ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title("x–z slab")
    ax.legend(loc="lower right", markerscale=5)
    fig.savefig(prefix + "_xz.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. 3D: dead voxels (+ optional centres) + sparse wireframe --------------------------
    fig = plt.figure(figsize=(8, 9))
    ax3 = fig.add_subplot(111, projection="3d")
    n_shown = []
    if a.show_centres:
        c3 = sub(cen, a.max_points_3d)
        ax3.scatter(c3[:, 0], c3[:, 1], c3[:, 2], s=1, c="C0", alpha=0.25, rasterized=True)
        n_shown.append(f"{len(c3):,} of {len(cen):,} centres")
    if dead is not None:
        d3 = sub(dead, a.max_dead_3d)
        ax3.scatter(d3[:, 0], d3[:, 1], d3[:, 2], s=2, c="C3", alpha=0.8, rasterized=True)
        n_shown.append(f"{len(d3):,} of {len(dead):,} dead voxels")
    for phi in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ax3.plot(pr * np.cos(phi), pr * np.sin(phi), pz, "k-", lw=0.6)
    ph = np.linspace(0, 2 * np.pi, 120)
    for k in sorted(set(range(0, len(pr), 30)) | {len(pr) - 1}):   # last ring = the cut
        if pr[k] > 0:
            ax3.plot(pr[k] * np.cos(ph), pr[k] * np.sin(ph), np.full_like(ph, pz[k]),
                     "k-", lw=0.6)
    R = pr.max()
    ax3.set_xlim(-R, R); ax3.set_ylim(-R, R); ax3.set_zlim(zlo, zhi)
    ax3.set_box_aspect((2 * R, 2 * R, zhi - zlo))
    ax3.set_xlabel("x [mm]"); ax3.set_ylabel("y [mm]"); ax3.set_zlabel("z [mm]")
    ax3.set_title("ULAr (z ≤ %g mm); showing %s" % (a.zcut, "; ".join(n_shown) or "boundary only"))
    fig.savefig(prefix + "_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("wrote", prefix + "_rz.png,", prefix + "_xz.png,", prefix + "_3d.png")


if __name__ == "__main__":
    main()
