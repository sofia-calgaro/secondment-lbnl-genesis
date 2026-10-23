"""
STEP 1 -- define the space that is divided into voxels.

Enumerates every voxel of the shared grid (voxel_grid.py: origin (0,0,0), detector frame, mm)
that lies COMPLETELY inside the box [xrange] x [yrange] x [zrange], optionally removes voxels
that touch excluded objects, lie outside the ULAr volume, sit entirely inside dead volumes
(HPGe diodes / Cu rods) and/or anything that is not LAr in the GDML, and writes ONE file
listing the surviving voxels.  Nothing is sampled here.

    python make_voxel_space.py --vox 5 --xrange -800 400 --yrange 100 500 --zrange 500 700
    python make_voxel_space.py --vox 10 --geo-check --gdml l1000.gdml --out space_10mm.csv.gz
    python make_voxel_space.py --vox 5 --gdml l1000.gdml --dead-check

With --gdml, voxels not completely inside the ULAr volume (--ular-names) are removed
automatically; use --no-ular-check to disable.  If the detector frame differs from the GDML
world frame, pass --gdml-offset X Y Z (added to voxel coordinates before querying the GDML).

Output:
  <out>                  the voxel-space file (CSV, or CSV.GZ if the name ends in .gz)
                         default name: voxel_space_vox<size>mm.csv
  <out stem>_qc.pdf      fraction of kept voxels projected on xy / xz / yz (needs matplotlib)
  <out stem>_dead_voxels.csv.gz   (with --dead-check) ids of voxels dropped as dead volume

Give the file to STEP 2:   python sample_voxels.py <out> --n-lf ... --n-hf-train ...
Share the file (and its sha256, printed below) so everyone samples from the same space.

--exclusions takes a JSON list, detector-frame mm, e.g.
  [{"type": "cylinder", "x": 0, "y": 0, "r": 100, "z0": 0, "z1": 1000},
   {"type": "box", "min": [100, 100, 0], "max": [150, 150, 300]}]
A voxel is dropped if any corner or its centre is inside (or within --clearance of) one.
"""
import argparse
import io
import json
import os
import re
import sys
import time
import warnings
from collections import Counter

import numpy as np

from voxel_grid import VoxelGrid, VoxelSpace

warnings.filterwarnings("ignore")

_C8 = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)


# ----------------------------------------------------------------------------
# Optional GDML-based checks (Geant4 navigator via geant4_pybind)
# ----------------------------------------------------------------------------
def import_g4():
    """Import geant4_pybind; it asks about downloading datasets (not needed), so say no."""
    old = sys.stdin
    sys.stdin = io.StringIO("n\n")
    try:
        import geant4_pybind as g4
    except ImportError:
        raise SystemExit("geant4_pybind not found. Install it in a separate env, e.g. "
                         "'pip install geant4_pybind', and run this script with that python.")
    finally:
        sys.stdin = old
    return g4


class GdmlLAr:
    """One parsed GDML geometry, with three voxel tests (all use the Geant4 navigator):

      voxel_in_ular : all 8 inset corners are inside the ULAr volume (or a daughter of it)
      voxel_dead    : all 8 inset corners are inside volumes matching the dead regex
      voxel_ok      : all n^3 test points are LAr (names / material in `names`)

    Voxel coordinates are shifted by `offset` (detector frame -> GDML world frame)."""

    def __init__(self, path, names, grid, n=3, inset=0.05, offset=(0.0, 0.0, 0.0)):
        if not os.path.isfile(path):
            raise SystemExit(f"GDML file not found: {path}")
        self.g4 = g4 = import_g4()
        parser = g4.G4GDMLParser()
        parser.Read(path, False)
        self.world = parser.GetWorldVolume()
        g4.G4GeometryManager.GetInstance().CloseGeometry(True)
        self.nav = g4.G4Navigator()
        self.nav.SetWorldVolume(self.world)
        self.names = set(names)
        self.grid = grid
        self.inset = inset
        self.offset = np.asarray(offset, float)
        self.t = self._points(n)        # n^3 points per voxel (--geo-check)
        self.t8 = self._points(2)       # the 8 inset corners (ULAr / dead checks)
        self.rejected = Counter()       # --geo-check: volumes that rejected a voxel
        self.ulnames = set()            # ULAr names (set from main)
        self.outside = Counter()        # ULAr check: volumes that rejected a voxel
        self.rx = None                  # dead-volume regex (set_dead_regex)
        self.dead_names = Counter()

    def _points(self, n):
        t = np.linspace(self.inset / self.grid.size, 1.0 - self.inset / self.grid.size, n)
        return np.array([[a, b, c] for a in t for b in t for c in t]) * self.grid.size

    def locate(self, p):
        q = p + self.offset
        return self.nav.LocateGlobalPointAndSetup(
            self.g4.G4ThreeVector(float(q[0]), float(q[1]), float(q[2])), None, False)

    # ---- LAr (any volume / material in `names`) ----
    def is_lar(self, pv):
        if pv is None:
            return False
        lv = pv.GetLogicalVolume()
        return (pv.GetName() in self.names or lv.GetName() in self.names
                or lv.GetMaterial().GetName() in self.names)

    def voxel_ok(self, idx):
        """All test points of the voxel must be LAr."""
        out = np.zeros(len(idx), bool)
        lower = self.grid.lower(idx)
        for k in range(len(idx)):
            ok = True
            for p in lower[k] + self.t:
                pv = self.locate(p)
                if not self.is_lar(pv):
                    ok = False
                    self.rejected[pv.GetName() if pv else "outside world"] += 1
                    break
            out[k] = ok
        return out

    # ---- inside the ULAr boundary ----
    def in_ular(self, pv):
        """True if the located volume is ULAr itself or any descendant of it."""
        if pv is None:
            return False

        def match(v):
            return v.GetName() in self.ulnames or v.GetLogicalVolume().GetName() in self.ulnames

        if match(pv):                    # fast path: point is in the argon itself
            return True
        th = self.nav.CreateTouchableHistory()
        for d in range(1, th.GetHistoryDepth() + 1):   # walk up the mother chain
            if match(th.GetVolume(d)):
                return True
        return False

    def voxel_in_ular(self, idx):
        """All 8 corners must be inside ULAr (daughters such as diodes count as inside)."""
        out = np.zeros(len(idx), bool)
        lower = self.grid.lower(idx)
        for k in range(len(idx)):
            ok = True
            for p in lower[k] + self.t8:
                pv = self.locate(p)
                if not self.in_ular(pv):
                    ok = False
                    self.outside[pv.GetName() if pv else "outside world"] += 1
                    break
            out[k] = ok
        return out

    # ---- dead volumes (HPGe diodes, Cu rods) ----
    def set_dead_regex(self, pattern):
        self.rx = re.compile(pattern)

    def is_dead(self, pv):
        if pv is None:
            return None
        lv = pv.GetLogicalVolume()
        for nm in (pv.GetName(), lv.GetName()):
            if self.rx.search(nm):
                return nm
        return None

    def voxel_dead(self, idx):
        """True if ALL 8 corners lie inside dead volumes."""
        out = np.zeros(len(idx), bool)
        lower = self.grid.lower(idx)
        for k in range(len(idx)):
            hit = []
            for p in lower[k] + self.t8:
                nm = self.is_dead(self.locate(p))
                if nm is None:
                    break                      # early exit: at least one corner is not dead
                hit.append(nm)
            else:
                out[k] = True
                self.dead_names.update(set(hit))
        return out

    def dead_corner_counts(self, idx, step=300_000):
        """Number (0-8) of corners of every voxel inside dead volumes.
        One navigator call per unique grid vertex, not per voxel corner."""
        out = np.zeros(len(idx), np.uint8)
        t0 = time.time()
        for s in range(0, len(idx), step):
            sub = np.asarray(idx[s:s + step], dtype=np.int64)
            v = (sub[:, None, :] + _C8[None]).reshape(-1, 3)      # (8n, 3) vertex indices
            uv, inv = np.unique(v, axis=0, return_inverse=True)
            pts = uv * self.grid.size + self.inset
            flag = np.fromiter((self.is_dead(self.locate(p)) is not None for p in pts),
                               bool, len(pts))
            out[s:s + step] = flag[inv.ravel()].reshape(-1, 8).sum(1)
            print(f"  dead-corner count {min(s + step, len(idx)):,}/{len(idx):,}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
        return out


def inspect_gdml(path, names):
    """Print what sits inside the LAr volume(s), grouped by logical volume."""
    g = GdmlLAr(path, names, VoxelGrid(1.0))
    mats, lar, seen, stack = set(), [], set(), [g.world]
    while stack:
        pv = stack.pop()
        lv = pv.GetLogicalVolume()
        if lv.GetName() in seen:
            continue
        seen.add(lv.GetName())
        mats.add(lv.GetMaterial().GetName())
        if g.is_lar(pv):
            lar.append(pv)
        stack += [lv.GetDaughter(i) for i in range(lv.GetNoDaughters())]
    print("materials containing 'argon':", sorted(m for m in mats if "argon" in m.lower()
                                                   or "lar" in m.lower()) or "none")
    print("matched LAr volumes:", [pv.GetName() for pv in lar] or
          "none -> pass the right name(s) with --lar-names")
    for pv in lar:
        lv = pv.GetLogicalVolume()
        groups = Counter()
        for i in range(lv.GetNoDaughters()):
            d = lv.GetDaughter(i)
            dl = d.GetLogicalVolume()
            groups[(dl.GetName(), dl.GetMaterial().GetName(), dl.GetSolid().GetEntityType())] += 1
        print(f"\n{pv.GetName()} (logical {lv.GetName()}, material "
              f"{lv.GetMaterial().GetName()}): {lv.GetNoDaughters()} daughters")
        for (ln, mn, st), c in sorted(groups.items(), key=lambda kv: -kv[1]):
            print(f"  {c:5d} x {ln:40s} {mn:28s} {st}")


def load_ular_profile(path, name):
    """(r, z) polycone profile of the ULAr solid, in mm, from the GDML (via pyg4ometry)."""
    import pyg4ometry as pg
    from pyg4ometry.gdml import Units
    reg = pg.gdml.Reader(path).getRegistry()
    s = reg.logicalVolumeDict[name].solid
    f = Units.unit(s.lunit)
    return (np.array([e.eval() for e in s.pR]) * f,
            np.array([e.eval() for e in s.pZ]) * f)


def ular_inside(grid, idx, r, z, offset, chunk=500_000):
    """True if all 8 corners are inside the polycone profile (vectorised, no Geant4)."""
    from matplotlib.path import Path
    off = np.asarray(offset, float)
    rmax = np.empty(len(idx))
    zlo = np.empty(len(idx))
    zhi = np.empty(len(idx))
    for s in range(0, len(idx), chunk):
        c = grid.corners(idx[s:s + chunk]) + off          # (N, 8, 3)
        rmax[s:s + chunk] = np.hypot(np.abs(c[..., 0]).max(1), np.abs(c[..., 1]).max(1))
        zlo[s:s + chunk] = c[..., 2].min(1)
        zhi[s:s + chunk] = c[..., 2].max(1)
    ur, ir = np.unique(np.round(rmax, 6), return_inverse=True)
    uz, iz = np.unique(np.round(np.r_[zlo, zhi], 6), return_inverse=True)
    poly = Path(np.c_[np.r_[r, r[0]], np.r_[z, z[0]]])    # closes along the axis
    R, Z = np.meshgrid(ur, uz, indexing="ij")
    tab = poly.contains_points(np.c_[R.ravel(), Z.ravel()]).reshape(R.shape)
    n = len(idx)
    return tab[ir, iz[:n]] & tab[ir, iz[n:]]


def chunked(fn, idx, label, step=20_000):
    """Apply a per-chunk boolean test fn(idx_chunk) -> bool array, with progress output."""
    out = np.zeros(len(idx), bool)
    t0 = time.time()
    for s in range(0, len(idx), step):
        out[s:s + step] = fn(idx[s:s + step])
        done = min(s + step, len(idx))
        print(f"  {label} {done:,}/{len(idx):,}  ({time.time() - t0:.0f} s)", flush=True)
    return out


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def apply_radial_constraint(grid, idx, radius_mm, chunk=200_000):
    """Keep only voxels whose eight corners lie within the XY cylinder."""
    keep = np.zeros(len(idx), dtype=bool)
    radius2 = radius_mm**2

    for s in range(0, len(idx), chunk):
        sub = idx[s:s + chunk]
        corners = grid.corners(sub)

        r2 = corners[:, :, 0]**2 + corners[:, :, 1]**2
        keep[s:s + chunk] = np.all(
            r2 <= radius2 + 1e-9, axis=1
        )

    return keep


def inside_excluded(p, e, clearance):
    """Points p (..., 3) inside excluded object e (grown by clearance)?"""
    if e["type"] == "cylinder":
        rr = np.hypot(p[..., 0] - e["x"], p[..., 1] - e["y"])
        return ((rr <= e["r"] + clearance)
                & (p[..., 2] >= e["z0"] - clearance)
                & (p[..., 2] <= e["z1"] + clearance))
    if e["type"] == "box":
        lo = np.array(e["min"], float) - clearance
        hi = np.array(e["max"], float) + clearance
        return ((p >= lo) & (p <= hi)).all(-1)
    raise SystemExit(f"unknown exclusion type: {e['type']!r} (use 'cylinder' or 'box')")


def apply_exclusions(grid, idx, exclusions, clearance, chunk=200_000):
    keep = np.ones(len(idx), bool)
    for s in range(0, len(idx), chunk):
        sub = idx[s:s + chunk]
        pts = np.concatenate([grid.corners(sub), grid.centers(sub)[:, None, :]], axis=1)
        for e in exclusions:
            keep[s:s + chunk] &= ~inside_excluded(pts, e, clearance).any(1)
    return keep


def qc_projection(path, space, imin, nfull):
    """Plot the selected voxel space: 3D view and central XY/XZ/YZ slices."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        return False

    g = space.grid
    idx = space.idx
    size = g.size
    names = "xyz"

    # Use the grid layer containing the origin for the central slices.
    origin_idx = g.index_of([0.0, 0.0, 0.0])
    slice_specs = [
        (0, 1, 2, origin_idx[2], "XY"),
        (0, 2, 1, origin_idx[1], "XZ"),
        (1, 2, 0, origin_idx[0], "YZ"),
    ]

    fig = plt.figure(figsize=(15, 11))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")

    # Draw voxel centres in 3D. Subsample very large spaces to keep
    # the PDF manageable; the 2D slices below use all selected voxels.
    max_points = 100_000
    if len(idx) > max_points:
        rng = np.random.default_rng(12345)
        sample = rng.choice(len(idx), max_points, replace=False)
        idx3d = idx[sample]
    else:
        idx3d = idx

    centres = g.centers(idx3d)
    ax3d.scatter(
        centres[:, 0], centres[:, 1], centres[:, 2],
        s=15, alpha=0.55, depthshade=False
    )

    lo = g.lower(imin)
    hi = lo + nfull * size

    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    #ax3d.set_box_aspect(hi - lo)
    ax3d.set_box_aspect((3, 3, 3))
    ax3d.set_xlabel("x [mm]")
    ax3d.set_ylabel("y [mm]")
    ax3d.set_zlabel("z [mm]")
    ax3d.view_init(elev=22, azim=-40)
    ax3d.set_title(
        f"3D voxel space ({len(idx):,} voxels)"
        + (f"; showing {len(idx3d):,} centres"
           if len(idx3d) < len(idx) else "")
    )

    for panel, (a, b, fixed, layer, label) in enumerate(slice_specs, start=2):
        ax = fig.add_subplot(2, 2, panel)

        # Select the voxels intersecting the central grid layer.
        selected = idx[idx[:, fixed] == layer]
        mask = np.zeros((int(nfull[a]), int(nfull[b])), dtype=bool)

        if len(selected):
            ia = selected[:, a] - imin[a]
            ib = selected[:, b] - imin[b]
            valid = (
                (ia >= 0) & (ia < nfull[a]) &
                (ib >= 0) & (ib < nfull[b])
            )
            mask[ia[valid], ib[valid]] = True

        edges_a = (imin[a] + np.arange(int(nfull[a]) + 1)) * size
        edges_b = (imin[b] + np.arange(int(nfull[b]) + 1)) * size

        # Draw selected cells and the underlying voxel grid.
        ax.pcolormesh(
            edges_a, edges_b, mask.T.astype(float),
            shading="flat",
            edgecolors="0.55",
            linewidth=0.25,
            vmin=0, vmax=1,
        )

        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        if label in ("XZ", "YZ"):
            ax.set_aspect("auto")
        else:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{names[a]} [mm]")
        ax.set_ylabel(f"{names[b]} [mm]")

        layer_lo = layer * size
        ax.set_title(
            f"{label} slice: {names[fixed]} in "
            f"[{layer_lo:g}, {layer_lo + size:g}] mm"
        )

        # Optional reference circle in the XY view.
        radius = space.meta.get("radius_mm")
        if label == "XY" and radius is not None:
            ax.add_patch(Circle(
                (0, 0), radius,
                fill=False, edgecolor="red", linewidth=2
            ))

    fig.suptitle(
        f"{g.name}: {len(idx):,} selected voxels "
        f"from x[{lo[0]:g},{hi[0]:g}], "
        f"y[{lo[1]:g},{hi[1]:g}], "
        f"z[{lo[2]:g},{hi[2]:g}] mm"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vox", type=float, default=5.0, help="voxel edge length [mm]")
    ap.add_argument("--xrange", type=float, nargs=2, default=[-800.0, 400.0], metavar=("MIN", "MAX"))
    ap.add_argument("--yrange", type=float, nargs=2, default=[100.0, 500.0], metavar=("MIN", "MAX"))
    ap.add_argument("--zrange", type=float, nargs=2, default=[500.0, 700.0], metavar=("MIN", "MAX"))
    ap.add_argument("--out", default=None,
                    help="output file (default voxel_space_vox<size>mm.csv; '.gz' to compress)")
    ap.add_argument("--exclusions", default=None, help="JSON file: list of cylinders / boxes to remove")
    ap.add_argument("--clearance", type=float, default=0.0,
                    help="extra margin [mm] around excluded objects")
    ap.add_argument("--geo-check", action="store_true",
                    help="drop voxels touching anything that is not LAr in --gdml (slow: "
                         "geant4 navigator on n^3 points per voxel)")
    ap.add_argument("--dead-check", action="store_true",
                    help="drop voxels whose 8 corners are all inside HPGe diodes / Cu rods "
                         "(needs --gdml); dropped voxel ids go to <out stem>_dead_voxels.csv.gz")
    ap.add_argument("--dead-bbox", type=float, nargs=6, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="only run the (slow) dead-volume check on voxels whose centre is in "
                         "this box [mm, detector frame]; voxels outside are assumed live")
    ap.add_argument("--dead-regex", default=r"V\d{5}[A-Z]|hpge_string_support_hanger_copper",
                    help="regex matched against physical and logical volume names")
    ap.add_argument("--no-ular-check", action="store_true",
                    help="do not remove voxels outside the ULAr volume (the check runs "
                         "automatically whenever --gdml is given)")
    ap.add_argument("--ular-names", default="liquid_argon_underground",
                    help="comma-separated physical/logical name(s) of the ULAr volume")
    ap.add_argument("--gdml-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"),
                    help="mm added to voxel coordinates before querying the GDML "
                         "(detector frame -> GDML world frame)")
    ap.add_argument("--gdml", default=None, help="GDML file for the GDML checks / --inspect")
    ap.add_argument("--lar-names", default="liquid_argon,liquid_argon_underground")
    ap.add_argument("--geo-samples", type=int, default=3,
                    help="n: test n^3 points inside every voxel (3 -> 27)")
    ap.add_argument("--inspect", action="store_true",
                    help="print the contents of the LAr volume in --gdml and exit")
    ap.add_argument("--max-voxels", type=int, default=30_000_000,
                    help="refuse to enumerate more voxels than this")
    ap.add_argument("--no-qc", action="store_true", help="do not write the QC pdf")
    ap.add_argument(
        "--radius",
        type=float,
        default=None,
        help="maximum XY radius [mm]; keep only fully contained voxels"
    )
    a = ap.parse_args()

    names = [n.strip() for n in a.lar_names.split(",") if n.strip()]
    ulnames = [n.strip() for n in a.ular_names.split(",") if n.strip()]
    if (a.inspect or a.geo_check or a.dead_check) and not a.gdml:
        raise SystemExit("--gdml is required with --geo-check / --dead-check / --inspect")
    if a.inspect:
        inspect_gdml(a.gdml, names)
        return

    grid = VoxelGrid(a.vox)
    box_min = np.array([a.xrange[0], a.yrange[0], a.zrange[0]], float)
    box_max = np.array([a.xrange[1], a.yrange[1], a.zrange[1]], float)
    if (box_max <= box_min).any():
        raise SystemExit("each range must be MIN < MAX")
    imin, imax = grid.index_range_in_box(box_min, box_max)
    nfull = imax - imin + 1
    if (nfull < 1).any():
        raise SystemExit(f"the box does not contain a whole {grid.size:g} mm voxel along "
                         f"every axis")

    # Determine the full rectangular grid dimensions.
    total_box = int(nfull.prod())
    out = a.out or f"voxel_space_{grid.name}.csv"

    if a.radius is not None and a.radius <= 0:
        raise SystemExit("--radius must be positive")

    if a.radius is None:
        total = total_box
        if total > a.max_voxels:
            raise SystemExit(
                f"{total:,} voxels in the box "
                f"(> --max-voxels {a.max_voxels:,})"
            )

        ranges = [
            np.arange(imin[i], imax[i] + 1)
            for i in range(3)
        ]
        idx = np.stack(
            np.meshgrid(*ranges, indexing="ij"), -1
        ).reshape(-1, 3)

    else:
        # Find XY voxels fully contained inside the circle.
        ix = np.arange(imin[0], imax[0] + 1)
        iy = np.arange(imin[1], imax[1] + 1)

        s = grid.size
        x0, x1 = ix * s, (ix + 1) * s
        y0, y1 = iy * s, (iy + 1) * s

        # The furthest corner from the origin determines
        # whether a rectangular XY voxel fits inside the circle.
        xmax2 = np.maximum(x0**2, x1**2)
        ymax2 = np.maximum(y0**2, y1**2)

        valid_xy = (
            xmax2[:, None] + ymax2[None, :]
            <= a.radius**2 + 1e-9
        )

        ix2, iy2 = np.nonzero(valid_xy)
        xy = np.column_stack((ix[ix2], iy[iy2]))

        nz = int(nfull[2])
        total = len(xy) * nz

        if total > a.max_voxels:
            raise SystemExit(
                f"{total:,} voxels inside the cylinder "
                f"(> --max-voxels {a.max_voxels:,}); "
                "increase the limit only if sufficient RAM is available"
            )

        print(
            f"XY voxels inside circle: {len(xy):,}; "
            f"Z layers: {nz:,}; total: {total:,}",
            flush=True,
        )

        # Preallocate the final list rather than creating the
        # full rectangular 3D grid and subsequently filtering it.
        idx = np.empty((total, 3), dtype=np.int64)
        z_indices = np.arange(imin[2], imax[2] + 1)

        for k, zidx in enumerate(z_indices):
            start = k * len(xy)
            stop = start + len(xy)
            idx[start:stop, :2] = xy
            idx[start:stop, 2] = zidx

    print(
        f"{grid.name}: box {box_min.tolist()} .. "
        f"{box_max.tolist()} mm; {len(idx):,} voxels selected",
        flush=True,
    )

    # ---- 1. hand-written exclusions (JSON) ----
    exclusions = []
    if a.exclusions:
        with open(a.exclusions) as f:
            exclusions = json.load(f)
        keep = apply_exclusions(grid, idx, exclusions, a.clearance)
        print(f"exclusions ({len(exclusions)} objects, clearance {a.clearance:g} mm): "
              f"removed {int((~keep).sum()):,}")
        idx = idx[keep]

    # ---- one GDML load shared by all GDML checks ----
    run_ular = bool(a.gdml) and not a.no_ular_check
    geo = None
    if a.gdml and (a.dead_check or a.geo_check):
        geo = GdmlLAr(a.gdml, names, grid, a.geo_samples, offset=a.gdml_offset)
        #geo.ulnames = set(ulnames)
        geo.set_dead_regex(a.dead_regex)
    elif not a.gdml:
        print("WARNING: no --gdml given -> voxels are NOT checked against the ULAr boundary")

    # ---- 2. ULAr boundary ----
    ular_meta = None
    if run_ular:
        t0 = time.time()
        pr, pz = load_ular_profile(a.gdml, ulnames[0])
        inside = ular_inside(grid, idx, pr, pz, a.gdml_offset)
        print(f"ULAr boundary check ({ulnames[0]}): removed {int((~inside).sum()):,} voxels "
              f"not fully inside  ({time.time() - t0:.0f} s)")
        ular_meta = {"gdml": os.path.abspath(a.gdml), "ular_names": ulnames,
                     "method": "analytic polycone profile",
                     "n_removed": int((~inside).sum())}
        idx = idx[inside]
        if len(idx) == 0:
            raise SystemExit("no voxel inside ULAr (wrong --ular-names or --gdml-offset?)")
    """
    ular_meta = None
    if run_ular:
        inside = chunked(geo.voxel_in_ular, idx, "ULAr boundary check")
        print(f"ULAr boundary check ({', '.join(ulnames)}): removed "
              f"{int((~inside).sum()):,} voxels not fully inside; volumes hit:")
        for n, c in geo.outside.most_common(15):
            print(f"  {c:8d}  {n}")
        ular_meta = {"gdml": os.path.abspath(a.gdml), "ular_names": ulnames,
                     "n_removed": int((~inside).sum()),
                     "rejected_by_volume": dict(geo.outside.most_common(15))}
        idx = idx[inside]
        if len(idx) == 0:
            raise SystemExit("no voxel inside ULAr (wrong --ular-names or --gdml-offset?)")
    """

    # ---- 3. dead volumes (HPGe diodes, Cu rods) ----
    dead_meta = None
    n_after_ular = len(idx)                    # voxels fully inside the ULAr boundary
    if a.dead_check:
        sel = np.ones(len(idx), bool)
        if a.dead_bbox:
            b = np.array(a.dead_bbox).reshape(3, 2)
            cen = grid.centers(idx)
            sel = ((cen >= b[:, 0]) & (cen <= b[:, 1])).all(1)
            print(f"dead-volume check restricted to bbox: {int(sel.sum()):,} of {len(idx):,} voxels")
        ncorner = np.zeros(len(idx), np.uint8)
        ncorner[sel] = geo.dead_corner_counts(idx[sel])
        isdead = ncorner == 8
        ispartial = (ncorner > 0) & (ncorner < 8)
        n_dead, n_part = int(isdead.sum()), int(ispartial.sum())
        n_live = int((ncorner == 0).sum())

        pct = lambda n: f"{100 * n / max(n_after_ular, 1):6.2f}%"
        print("\nVoxel statistics")
        print(f"  initial selection                      {total:10,d}")
        print(f"  outside the ULAr boundary (removed)    {total - n_after_ular:10,d}")
        print(f"  fully inside the ULAr boundary         {n_after_ular:10,d}  (100%)")
        print(f"    fully in dead mass (8/8, removed)    {n_dead:10,d}  {pct(n_dead)}")
        print(f"    partially in dead mass (1-7/8, kept) {n_part:10,d}  {pct(n_part)}")
        print(f"    no dead corner (0/8, kept)           {n_live:10,d}  {pct(n_live)}")

        base = out[:-3] if out.endswith(".gz") else out
        stem = os.path.splitext(base)[0]
        if isdead.any():
            np.savetxt(stem + "_dead_voxels.csv.gz", idx[isdead], fmt="%d",
                       delimiter=",", header="ix,iy,iz")
            print("wrote", stem + "_dead_voxels.csv.gz")
        if ispartial.any():
            np.savetxt(stem + "_partial_voxels.csv.gz", idx[ispartial], fmt="%d",
                       delimiter=",", header="ix,iy,iz")
            print("wrote", stem + "_partial_voxels.csv.gz")
        dead_meta = {"gdml": os.path.abspath(a.gdml), "regex": a.dead_regex,
                     "n_dead": n_dead, "n_partial": n_part, "n_no_dead_corner": n_live}
        idx = idx[~isdead]

    # ---- 4. strict "everything is LAr" check ----
    geo_meta = None
    if a.geo_check:
        ok = chunked(geo.voxel_ok, idx, "GDML check")
        print(f"GDML check: removed {int((~ok).sum()):,}; volumes hit:")
        for n, c in geo.rejected.most_common(15):
            print(f"  {c:8d}  {n}")
        geo_meta = {"gdml": os.path.abspath(a.gdml), "lar_names": names,
                    "samples_per_axis": a.geo_samples,
                    "rejected_by_volume": dict(geo.rejected.most_common(15))}
        idx = idx[ok]

    if len(idx) == 0:
        raise SystemExit("no voxel survived")
    space = VoxelSpace(grid, idx, box_min, box_max, meta={
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_voxels_in_box": total_box,
        "n_voxels_selected_before_exclusions_and_geo_check": total,
        "radius_mm": a.radius,
        "exclusions": exclusions,
        "clearance_mm": a.clearance,
        "gdml_offset_mm": list(a.gdml_offset),
        "ular_check": ular_meta,
        "dead_check": dead_meta,
        "geo_check": geo_meta,
        "command": " ".join(sys.argv),
    })
    meta = space.save(out)
    print(f"\nwrote {out}: {len(space):,} voxels kept")
    print(f"Initial selection: {total:,} voxels")
    print(f"Full rectangular box: {total_box:,} voxels")
    print(f"sha256 {meta['sha256']}")
    if not a.no_qc:
        base = out[:-3] if out.endswith(".gz") else out
        qc = os.path.splitext(base)[0] + "_qc.pdf"
        if qc_projection(qc, space, imin, nfull):
            print("wrote", qc)


if __name__ == "__main__":
    main()
