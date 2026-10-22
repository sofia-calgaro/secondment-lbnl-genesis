"""
Shared voxel-grid definition for the optical-map project.  DO NOT COPY OR EDIT PER USER.

Convention (GRID_VERSION 1)
---------------------------
* Frame:   detector frame, millimetres.
* Origin:  (0, 0, 0) -- fixed, not a parameter.
* Lattice: cubic voxels of edge `size` mm.  Voxel (i, j, k) spans
           [i*size, (i+1)*size] x [j*size, (j+1)*size] x [k*size, (k+1)*size]
           and its centre is ((i+.5)*size, (j+.5)*size, (k+.5)*size).
* One grid per voxel size.  Grids whose sizes divide each other are nested
  (every 10 mm voxel is exactly 2x2x2 voxels of the 5 mm grid).

The *region* (box limits, exclusions, GDML/LAr check) is not part of the grid: it is fixed once
in step 1 (make_voxel_space.py) and stored as a VoxelSpace file; sampling (sample_voxels.py)
only reads that file.

Usage
-----
    from voxel_grid import VoxelGrid
    g = VoxelGrid(5.0)
    idx = g.index_of([712.3, -336.0, -494.9])      # -> array([142, -68, -99])
    g.centers(idx)                                  # -> voxel centres [mm]
    g.corners(idx)                                  # -> (N, 8, 3) corner coordinates [mm]
    g.plot(center_mm=(0, 0, 0), extent_mm=100, path="grid.pdf")   # cube, 3D + xy/xz/yz planes, PDF
    g.plot(limits_mm=((-800, 400), (100, 500), (500, 700)), path="box.pdf")   # box with limits
    g.plot(extent_mm=(100, 60, 40), path="slab.pdf")                  # non-cubic extent

CLI:
    python voxel_grid.py 5 --point 712.3 -336 -494.9
    python voxel_grid.py 10 --check voxel_sets_box/config.json
    python voxel_grid.py 5 --plot grid5.pdf [--center X Y Z] [--extent E | EX EY EZ] [--xlim X1 X2 --ylim Y1 Y2 --zlim Z1 Z2]
                                [--centers] [--no-planes]
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

GRID_VERSION = 1
ORIGIN_MM = (0.0, 0.0, 0.0)          # fixed for the whole project
_EPS = 1e-9                          # tolerance in voxel units against float round-off

_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)


@dataclass(frozen=True)
class VoxelGrid:
    size: float                      # voxel edge length [mm]

    def __post_init__(self):
        if not (math.isfinite(self.size) and self.size > 0):
            raise ValueError(f"voxel size must be a positive number, got {self.size}")
        object.__setattr__(self, "size", float(self.size))

    # -- identity ----------------------------------------------------------
    @property
    def name(self):
        """e.g. 'vox5mm', 'vox2.5mm' -- handy for directory / file names."""
        return f"vox{self.size:g}mm"

    def describe(self):
        """Serialisable definition; store it in every config.json."""
        return {"grid_version": GRID_VERSION, "origin_mm": list(ORIGIN_MM),
                "voxel_mm": self.size, "convention": "voxel i spans [i*size,(i+1)*size]"}

    @classmethod
    def from_description(cls, d):
        """Rebuild the grid from a `describe()` dict; refuses a different convention."""
        if d.get("grid_version") != GRID_VERSION or list(d.get("origin_mm", ORIGIN_MM)) != list(ORIGIN_MM):
            raise ValueError(f"grid definition differs from this module "
                             f"(version {GRID_VERSION}, origin {ORIGIN_MM})")
        return cls(d["voxel_mm"])

    @classmethod
    def from_config(cls, path):
        """Rebuild the grid from a config.json written by the sampling script."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_description(d.get("grid", d))

    # -- position <-> index ------------------------------------------------
    def index_of(self, pos_mm):
        """Index of the voxel containing each position [mm]; shape (..., 3) -> (..., 3) int."""
        p = (np.asarray(pos_mm, float) - ORIGIN_MM) / self.size
        return np.floor(p + _EPS).astype(int)

    def lower(self, idx):
        """Lower corner [mm] of each voxel."""
        return np.asarray(idx) * self.size + ORIGIN_MM

    def centers(self, idx):
        """Voxel centres [mm]."""
        return (np.asarray(idx) + 0.5) * self.size + ORIGIN_MM

    def corners(self, idx):
        """All 8 corners [mm]; shape (N, 8, 3)."""
        idx = np.asarray(idx)
        return (idx[:, None, :] + _CORNERS[None]) * self.size + ORIGIN_MM

    def index_range_in_box(self, lo_mm, hi_mm):
        """Inclusive (imin, imax) index bounds of voxels lying completely inside the box.
        Empty along an axis if imax < imin."""
        lo = (np.asarray(lo_mm, float) - ORIGIN_MM) / self.size
        hi = (np.asarray(hi_mm, float) - ORIGIN_MM) / self.size
        return (np.ceil(lo - _EPS).astype(int), np.floor(hi + _EPS).astype(int) - 1)

    # -- region helpers ----------------------------------------------------
    def inside_box(self, idx, lo_mm, hi_mm):
        """True for voxels lying completely inside the box [lo_mm, hi_mm] (per-axis limits)."""
        c = self.corners(np.asarray(idx).reshape(-1, 3))
        return ((c >= np.asarray(lo_mm) - _EPS * self.size)
                & (c <= np.asarray(hi_mm) + _EPS * self.size)).all((1, 2))

    # -- visualisation -----------------------------------------------------
    def plot(self, center_mm=None, extent_mm=100.0, limits_mm=None, highlight=None,
             show_centers=False, planes=True, full_3d=None, path=None, ax=None,
             max_lines=60000):
        """Draw the voxel lattice in a box region: 3D view + the xy, xz, yz planes.

        Region (snapped to whole voxels, so it starts and ends on grid planes):
          * limits_mm=((x1,x2),(y1,y2),(z1,z2)): the voxels lying completely inside these
            limits (same rule as the generator). Nothing is drawn beyond them.
          * otherwise a box around center_mm (default: origin; region centre when limits
            are given) with side extent_mm: a single number (cube) or (ex, ey, ez).
        highlight: None -> shade the voxel containing center_mm; otherwise an (N,3) array of
                   voxel indices to shade ([] for none). Only voxels inside the region are drawn.
        show_centers: also mark voxel centres (only sensible for few voxels).
        planes: add the 2D panels (ignored when `ax` is given: then only the 3D view is drawn).
        full_3d: draw every grid line in 3D. Default: only for <= 4000 voxels; larger regions
                 show the box edges + the grid on the three central planes instead.
        path: save there; PDF by default (".pdf" is appended if there is no extension).
        ax: existing 3D axes to draw into. Returns the 3D axes."""
        import matplotlib
        if path and ax is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

        O = np.array(ORIGIN_MM)
        if limits_mm is not None:
            lim = np.asarray(limits_mm, float).reshape(3, 2)
            if (lim[:, 1] <= lim[:, 0]).any():
                raise ValueError("each limit must be [min, max] with min < max")
            lo_idx, hi_idx = self.index_range_in_box(lim[:, 0], lim[:, 1])
            n = hi_idx - lo_idx + 1
            if (n < 1).any():
                raise ValueError(f"limits {lim.tolist()} do not contain a whole "
                                 f"{self.size:g} mm voxel along every axis")
            c = lim.mean(1) if center_mm is None else np.asarray(center_mm, float)
        else:
            c = O.copy() if center_mm is None else np.asarray(center_mm, float)
            ext = np.broadcast_to(np.asarray(extent_mm, float), (3,))
            n = np.maximum(1, np.rint(ext / self.size).astype(int))
            lo_idx = np.rint((c - O) / self.size - n / 2).astype(int)
        n_lines = sum((n[b] + 1) * (n[d] + 1) for b, d in ((1, 2), (0, 2), (0, 1)))
        if n_lines > max_lines:
            raise ValueError(f"{n.tolist()} voxels needs {n_lines} grid lines (> {max_lines}); "
                             f"reduce the region or increase the voxel size / max_lines")
        edges = [(lo_idx[a] + np.arange(n[a] + 1)) * self.size + O[a] for a in range(3)]
        lo = np.array([e[0] for e in edges])
        hi = np.array([e[-1] for e in edges])

        if highlight is None:
            highlight = self.index_of(c)[None, :]
        highlight = np.asarray(highlight, int).reshape(-1, 3)
        highlight = highlight[((highlight >= lo_idx) & (highlight < lo_idx + n)).all(1)]

        own_fig = ax is None
        if own_fig:
            fig = plt.figure(figsize=(14, 12) if planes else (9, 8))
            ax = fig.add_subplot(2, 2, 1, projection="3d") if planes \
                else fig.add_subplot(projection="3d")
        else:
            fig = ax.figure

        # ---- 3D panel ----
        # small regions: the full lattice; large ones (opaque blob otherwise): the box edges
        # plus the grid on the three planes through the centre layer (as in the 2D panels).
        segs = []

        def seg(a, fixed):
            """line along axis a, other two coordinates given in `fixed` {axis: value}"""
            p0, p1 = np.zeros(3), np.zeros(3)
            p0[a], p1[a] = lo[a], hi[a]
            for ax_i, val in fixed.items():
                p0[ax_i] = p1[ax_i] = val
            segs.append((p0, p1))

        if full_3d is None:
            full_3d = int(n.prod()) <= 4000
        if full_3d:
            for a in range(3):
                b, d = [i for i in range(3) if i != a]
                for u in edges[b]:
                    for v in edges[d]:
                        seg(a, {b: u, d: v})
        else:
            for a in range(3):                                  # outer box edges
                b, d = [i for i in range(3) if i != a]
                for u in (lo[b], hi[b]):
                    for v in (lo[d], hi[d]):
                        seg(a, {b: u, d: v})
            for k in range(3):                                  # centre-layer planes
                a, b = [i for i in range(3) if i != k]
                kc = edges[k][int(np.clip(np.floor((c[k] - O[k]) / self.size + _EPS)
                                          - lo_idx[k], 0, n[k] - 1))]
                for u in edges[b]:
                    seg(a, {b: u, k: kc})
                for u in edges[a]:
                    seg(b, {a: u, k: kc})
        ax.add_collection3d(Line3DCollection(segs, colors="0.55", linewidths=0.4))
        if len(highlight):
            faces_idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                         [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
            polys = [cr[f] for cr in self.corners(highlight) for f in faces_idx]
            ax.add_collection3d(Poly3DCollection(polys, facecolors="#D85A30", alpha=0.45,
                                                 edgecolors="k", linewidths=0.8))
        if show_centers:
            g = np.stack(np.meshgrid(*[lo_idx[a] + np.arange(n[a]) for a in range(3)],
                                     indexing="ij"), -1).reshape(-1, 3)
            cc = self.centers(g)
            ax.scatter(cc[:, 0], cc[:, 1], cc[:, 2], s=3, c="#185FA5", depthshade=False)
        if ((O >= lo) & (O <= hi)).all():
            ax.scatter(*O, s=60, c="red", marker="*", depthshade=False,
                       label="detector origin (0,0,0)")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))                     # true proportions, no distortion
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
        ax.set_title(f"{self.name}: {n[0]}x{n[1]}x{n[2]} = {int(n.prod())} voxels\n"
                     f"x[{lo[0]:g},{hi[0]:g}] y[{lo[1]:g},{hi[1]:g}] z[{lo[2]:g},{hi[2]:g}] mm",
                     fontsize=10)
        ax.view_init(elev=22, azim=-40)

        # ---- 2D planes through the layer containing center_mm ----
        if own_fig and planes:
            nm = "xyz"
            for pos, (a, b, k) in zip((2, 3, 4), ((0, 1, 2), (0, 2, 1), (1, 2, 0))):
                pa = fig.add_subplot(2, 2, pos)
                layer = int(np.clip(np.floor((c[k] - O[k]) / self.size + _EPS),
                                    lo_idx[k], lo_idx[k] + n[k] - 1))
                k0 = layer * self.size + O[k]
                pa.vlines(edges[a], lo[b], hi[b], colors="0.55", linewidths=0.5)
                pa.hlines(edges[b], lo[a], hi[a], colors="0.55", linewidths=0.5)
                for v in highlight[highlight[:, k] == layer]:
                    pa.add_patch(Rectangle((v[a] * self.size + O[a], v[b] * self.size + O[b]),
                                           self.size, self.size, fc="#D85A30", alpha=0.45,
                                           ec="k", lw=0.8))
                if show_centers:
                    ga, gb = np.meshgrid((lo_idx[a] + np.arange(n[a]) + 0.5) * self.size + O[a],
                                         (lo_idx[b] + np.arange(n[b]) + 0.5) * self.size + O[b])
                    pa.scatter(ga, gb, s=4, c="#185FA5", zorder=3)
                if lo[a] <= O[a] <= hi[a] and lo[b] <= O[b] <= hi[b] and k0 <= O[k] <= k0 + self.size:
                    pa.plot(O[a], O[b], "r*", ms=10, zorder=4)
                pa.set_xlim(lo[a], hi[a]); pa.set_ylim(lo[b], hi[b])
                pa.set_aspect("equal")
                pa.set_xlabel(f"{nm[a]} [mm]"); pa.set_ylabel(f"{nm[b]} [mm]")
                pa.set_title(f"{nm[a]}{nm[b]} plane, {nm[k]} layer {layer} "
                             f"({k0:g} to {k0 + self.size:g} mm)", fontsize=10)

        if path:
            if not os.path.splitext(path)[1]:
                path += ".pdf"
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
        return ax


# ----------------------------------------------------------------------------
# Voxel space: the set of voxels of one grid that a simulation campaign may use.
# Written by make_voxel_space.py (step 1), read by sample_voxels.py (step 2).
# ----------------------------------------------------------------------------
SPACE_FORMAT = 1
_SPACE_TAG = "# voxel_space "


def _open_text(path, mode):
    return gzip.open(path, mode + "t") if str(path).endswith(".gz") else open(path, mode)


class VoxelSpace:
    """All allowed voxels (integer indices) of one VoxelGrid, plus the box they were cut from.

    File format (CSV, or CSV.GZ if the name ends in .gz):
        line 1:  '# voxel_space {json metadata}'   (grid definition, box limits, sha256, ...)
        line 2:  ix,iy,iz
        rest:    one voxel index per line, sorted
    The sha256 of the index list is stored in the metadata and re-checked on load, so a
    hand-edited / truncated file is detected, and two people can compare spaces by hash."""

    def __init__(self, grid, idx, box_min, box_max, meta=None):
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        if len(idx) == 0:
            raise ValueError("empty voxel space")
        self.idx = idx[np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))]
        self.grid = grid
        self.box_min = np.asarray(box_min, float)
        self.box_max = np.asarray(box_max, float)
        self.meta = dict(meta or {})
        self._i0 = self.idx.min(0)
        self._dims = self.idx.max(0) - self._i0 + 1
        self._keys = self.keys(self.idx)                 # sorted, because idx is

    def __len__(self):
        return len(self.idx)

    @property
    def sha256(self):
        return hashlib.sha256(np.ascontiguousarray(self.idx, dtype="<i8").tobytes()).hexdigest()

    def keys(self, idx):
        """Unique integer key per voxel index (valid for indices inside the space's index
        range); use with np.isin to compare voxel sets quickly."""
        r = np.asarray(idx, np.int64).reshape(-1, 3) - self._i0
        return (r[:, 0] * self._dims[1] + r[:, 1]) * self._dims[2] + r[:, 2]

    def contains(self, idx):
        """Boolean mask: which of the given voxel indices belong to the space."""
        q = np.asarray(idx, np.int64).reshape(-1, 3)
        inr = ((q >= self._i0) & (q < self._i0 + self._dims)).all(1)
        out = np.zeros(len(q), bool)
        kq = self.keys(q[inr])
        pos = np.minimum(np.searchsorted(self._keys, kq), len(self._keys) - 1)
        out[inr] = self._keys[pos] == kq
        return out

    def save(self, path):
        meta = {**self.meta, "format": SPACE_FORMAT, "grid": self.grid.describe(),
                "box_min": self.box_min.tolist(), "box_max": self.box_max.tolist(),
                "n_voxels": len(self), "sha256": self.sha256}
        with _open_text(path, "w") as f:
            f.write(_SPACE_TAG + json.dumps(meta) + "\n")
            f.write("ix,iy,iz\n")
            np.savetxt(f, self.idx, fmt="%d", delimiter=",")
        self.meta = meta
        return meta

    @classmethod
    def load(cls, path):
        with _open_text(path, "r") as f:
            first = f.readline()
            if not first.startswith(_SPACE_TAG):
                raise ValueError(f"{path} is not a voxel-space file (missing '{_SPACE_TAG.strip()}' header)")
            meta = json.loads(first[len(_SPACE_TAG):])
            idx = np.loadtxt(f, dtype=np.int64, delimiter=",", skiprows=1, ndmin=2)
        if meta.get("format") != SPACE_FORMAT:
            raise ValueError(f"{path}: unsupported space format {meta.get('format')}")
        sp = cls(VoxelGrid.from_description(meta["grid"]), idx,
                 meta["box_min"], meta["box_max"], meta)
        if len(sp) != meta["n_voxels"] or sp.sha256 != meta["sha256"]:
            raise ValueError(f"{path}: contents do not match the stored checksum "
                             f"(file edited or truncated?)")
        return sp


def _main():
    ap = argparse.ArgumentParser(description="Inspect the shared voxel grid.")
    ap.add_argument("vox", type=float, help="voxel size [mm]")
    ap.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="print the voxel containing this detector-frame point [mm]")
    ap.add_argument("--check", metavar="CONFIG_JSON",
                    help="verify a generator config.json uses this grid definition")
    ap.add_argument("--plot", metavar="PDF", help="save a PDF with the 3D lattice and the xy/xz/yz planes")
    ap.add_argument("--no-planes", action="store_true", help="plot: 3D panel only")
    ap.add_argument("--center", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="plot: region centre [mm] (default origin, "
                    "or the middle of the limits)")
    ap.add_argument("--extent", type=float, nargs="+", default=[100.0],
                    help="plot: box side [mm]; one value (cube) or three (EX EY EZ)")
    for ax_ in "xyz":
        ap.add_argument(f"--{ax_}lim", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help=f"plot: do not go beyond [MIN, MAX] mm in {ax_} (give all three limits)")
    ap.add_argument("--centers", action="store_true", help="plot: mark voxel centres")
    a = ap.parse_args()
    g = VoxelGrid(a.vox)
    print(json.dumps(g.describe(), indent=2))
    if a.point:
        i = g.index_of(a.point)
        print("index:", i.tolist(), " centre [mm]:", g.centers(i).tolist(),
              " lower corner [mm]:", g.lower(i).tolist())
    if a.check:
        other = VoxelGrid.from_config(a.check)
        if other != g:
            sys.exit(f"MISMATCH: {a.check} uses {other.name}, expected {g.name}")
        print(f"OK: {a.check} matches {g.name}")
    if a.plot:
        lims = [a.xlim, a.ylim, a.zlim]
        if any(lims) and not all(lims):
            sys.exit("give --xlim, --ylim and --zlim together")
        if len(a.extent) not in (1, 3):
            sys.exit("--extent takes 1 or 3 values")
        g.plot(a.center, a.extent[0] if len(a.extent) == 1 else a.extent,
               limits_mm=lims if all(lims) else None, show_centers=a.centers,
               planes=not a.no_planes, path=a.plot)
        print("wrote", a.plot)


if __name__ == "__main__":
    _main()
