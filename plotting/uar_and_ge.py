import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pyg4ometry.gdml import Units
from pyg4ometry import transformation as tr
import pyg4ometry as pg

# ---- command line ---------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--box", type=float, nargs=6, default=None,
                metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                help="optional box to draw in red [mm], same frame as the GDML ULAr volume")
args = ap.parse_args()
BOX = None
if args.box is not None:
    b = np.array(args.box, float).reshape(3, 2)
    if (b[:, 1] <= b[:, 0]).any():
        raise SystemExit("--box needs MIN < MAX on every axis")
    BOX = b                                   # rows: x, y, z; columns: min, max

ZCUT = 1500.0
GE_RE = re.compile(r"^V\d{5}[A-Z]$")
SIPM_RE = re.compile(r"^S\d{4}[A-Z]$")        # physvol names like S0102B
ROD_PREFIX = "hpge_string_support_hanger_copper"

# ---- style knobs -------------------------------------------------------------
N_MERID, RING_STEP = 6, 30          # LAr wireframe density (3D)
GE_COLOR, GE_FILL_A, GE_LW = "0.25", 0.05, 0.6
ROD_COLOR = (0.72, 0.45, 0.2)
ROD_FILL_A, ROD_LW = 0.10, 0.6
ROD_ANGLE = 10                      # deg: edge sharpness threshold for rods
BOX_COLOR, BOX_LW = "red", 1.5
SIPM_COLOR, SIPM_S2D, SIPM_S3D = "green", 6, 5   # SiPM dot colour / sizes

reader = pg.gdml.Reader("l1000.gdml")
reg = reader.getRegistry()
lv = reg.logicalVolumeDict["liquid_argon_underground"]
solid = lv.solid

f = Units.unit(solid.lunit)
r = np.array([e.eval() for e in solid.pR]) * f
z = np.array([e.eval() for e in solid.pZ]) * f

if z.max() > ZCUT:
    i = int(np.argmax(z > ZCUT))
    t = (ZCUT - z[i - 1]) / (z[i] - z[i - 1])
    rc = np.r_[r[:i], r[i - 1] + t * (r[i] - r[i - 1])]
    zc = np.r_[z[:i], ZCUT]
else:
    rc, zc = r, z

# ---- helpers -----------------------------------------------------------------
def to_vec(v):
    return np.array([c.eval() if hasattr(c, "eval") else c for c in v], dtype=float)

def mesh_tris(lv_):
    verts, polys, _ = lv_.mesh.localmesh.toVerticesAndPolygons()
    verts = np.array([[v[0], v[1], v[2]] for v in verts])
    tris = []
    for p in polys:
        for k in range(1, len(p) - 1):
            tris.append((p[0], p[k], p[k + 1]))
    return verts, np.array(tris)

def feature_edges(tri_xyz, angle_deg):
    flat = tri_xyz.reshape(-1, 3)
    _, inv = np.unique(np.round(flat, 4), axis=0, return_inverse=True)
    ids = inv.reshape(-1, 3)
    n = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    n[ok] /= ln[ok][:, None]
    edges = {}
    for ti in np.flatnonzero(ok):
        a, b, c = ids[ti]
        for u, v, pu, pw in ((a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)):
            if u == v:
                continue
            k = (u, v) if u < v else (v, u)
            edges.setdefault(k, []).append((ti, pu, pw))
    cos_thr = np.cos(np.radians(angle_deg))
    segs = []
    for lst in edges.values():
        if len(lst) == 1 or np.dot(n[lst[0][0]], n[lst[1][0]]) < cos_thr:
            ti, pu, pw = lst[0]
            segs.append((tri_xyz[ti, pu], tri_xyz[ti, pw]))
    return np.array(segs) if segs else np.empty((0, 2, 3))

def get_profile(sol):
    """(r, z) profile of a polycone; for booleans use the first operand
    (so small cut planes on some detectors are ignored)."""
    if hasattr(sol, "pR") and hasattr(sol, "pZ"):
        fu = Units.unit(sol.lunit)
        return (np.array([e.eval() for e in sol.pR]) * fu,
                np.array([e.eval() for e in sol.pZ]) * fu)
    if hasattr(sol, "obj1"):
        return get_profile(sol.obj1)
    return None

def profile_geometry(rp, zp, R, T, phis, rings=True):
    """Outline segments [N,2,3] and fill polygons for a solid of revolution."""
    tf = lambda p: p @ R.T + T
    n = len(rp)
    segs, polys = [], []
    for phi in phis:
        m = tf(np.c_[rp * np.cos(phi), rp * np.sin(phi), zp])
        polys.append(m)
        for i in range(n):
            j = (i + 1) % n
            if rp[i] < 1e-9 and rp[j] < 1e-9:      # skip the axis edge
                continue
            segs.append((m[i], m[j]))
    if rings:
        rmax = rp.max()
        ph = np.linspace(0, 2 * np.pi, 48)
        for zz in np.unique(np.round(zp[np.isclose(rp, rmax)], 6)):
            ring = tf(np.c_[rmax * np.cos(ph), rmax * np.sin(ph), np.full_like(ph, zz)])
            segs += list(zip(ring[:-1], ring[1:]))
    return np.array(segs), polys

def box_edges(b):
    """The 12 edges of the box b (3x2 array) as (start, end) point pairs."""
    corners = np.array([[x, y, z_] for x in b[0] for y in b[1] for z_ in b[2]])
    return [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
            if np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-9) == 1]

def children_of(lv_):
    """Daughter placements of a logical volume; also handles assembly volumes."""
    d = getattr(lv_, "daughterVolumes", None)
    return d if d else getattr(lv_, "pvList", [])

# ---- collect objects (in the LAr frame) ----------------------------------------
ge2d_s = []
ge2d_p = []
ge3d_s = []
ge3d_p = []
rod_tris = []
sipm_pos = []                       # one (x, y, z) per SiPM placement
sipm_names = []
n_ge = n_ge_fallback = 0

def walk(parent_lv, R, T):
    global n_ge, n_ge_fallback
    for pv in children_of(parent_lv):
        child = getattr(pv, "logicalVolume", None)
        if child is None:
            continue
        rot = tr.tbxyz2matrix(to_vec(pv.rotation)) if hasattr(pv, "rotation") else np.eye(3)
        pos = to_vec(pv.position) if hasattr(pv, "position") else np.zeros(3)
        R2, T2 = R @ rot.T, R @ pos + T            # swap rot.T -> rot if mirrored
        sname = getattr(getattr(child, "solid", None), "name", "")
        pvname = getattr(pv, "name", "") or ""

        if SIPM_RE.match(pvname): # or "sipm" in child.name.lower():
            sipm_pos.append(T2)                    # dot at the placement origin
            sipm_names.append(pvname or child.name)
        elif GE_RE.match(child.name) or GE_RE.match(sname):
            prof = get_profile(child.solid)
            if prof is None:
                n_ge_fallback += 1
                continue
            rp, zp = prof
            s2, p2 = profile_geometry(rp, zp, R2, T2, [0, np.pi], rings=False)
            s3, p3 = profile_geometry(rp, zp, R2, T2, [0, np.pi / 2], rings=True)
            if s3[..., 2].min() < ZCUT:
                ge2d_s.append(s2)
                ge2d_p.extend(p2)
                ge3d_s.append(s3)
                ge3d_p.extend(p3)
                n_ge += 1
        elif child.name.startswith(ROD_PREFIX):
            verts, tris = mesh_tris(child)
            vw = verts @ R2.T + T2
            if vw[..., 2].min() < ZCUT:
                rod_tris.append(vw[tris])
        else:
            walk(child, R2, T2)

walk(lv, np.eye(3), np.zeros(3))
sipm_pos = np.array(sipm_pos).reshape(-1, 3)
sipm_pos = sipm_pos[sipm_pos[:, 2] < ZCUT]
print(f"Ge detectors: {n_ge}  (skipped, no polycone profile: {n_ge_fallback})")
print(f"copper hanger rods: {len(rod_tris)}")
print(f"SiPM channels: {len(sipm_pos)}" +
      (f"  (z from {sipm_pos[:, 2].min():.0f} to {sipm_pos[:, 2].max():.0f} mm; "
       f"first names: {sipm_names[:3]})" if len(sipm_pos) else "  -> none found"))

ge2d_s = np.concatenate(ge2d_s) if ge2d_s else np.empty((0, 2, 3))
ge3d_s = np.concatenate(ge3d_s) if ge3d_s else np.empty((0, 2, 3))
rod_edges = [feature_edges(t, ROD_ANGLE) for t in rod_tris]
rod_edges = np.concatenate(rod_edges) if rod_edges else np.empty((0, 2, 3))
rod_all = np.concatenate(rod_tris) if rod_tris else np.empty((0, 3, 3))

# ---- 2D axial view ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 7))
ax.plot( rc, zc, "C0-", lw=1)
ax.plot(-rc, zc, "C0-", lw=1)
ax.plot([-rc[-1], rc[-1]], [zc[-1], zc[-1]], "C0-", lw=1)

ax.add_collection(PolyCollection([p[:, [0, 2]] for p in ge2d_p], facecolor="0.5",
                                 edgecolor="none", alpha=GE_FILL_A))
ax.add_collection(LineCollection(ge2d_s[:, :, [0, 2]], colors=GE_COLOR, linewidths=GE_LW))
if len(rod_all):
    ax.add_collection(PolyCollection(rod_all[:, :, [0, 2]], facecolor=ROD_COLOR,
                                     edgecolor="none", alpha=ROD_FILL_A))
    ax.add_collection(LineCollection(rod_edges[:, :, [0, 2]], colors=[ROD_COLOR],
                                     linewidths=ROD_LW))
if len(sipm_pos):                                     # SiPM channels: dots (x-z projection)
    ax.scatter(sipm_pos[:, 0], sipm_pos[:, 2], s=SIPM_S2D, c=SIPM_COLOR, zorder=5) #label=f"SiPM ({len(sipm_pos)})")
    #ax.legend(loc="lower right", fontsize=8)
if BOX is not None:                                   # optional box: x-z projection
    ax.add_patch(Rectangle((BOX[0, 0], BOX[2, 0]), BOX[0, 1] - BOX[0, 0],
                           BOX[2, 1] - BOX[2, 0], fill=False,
                           edgecolor=BOX_COLOR, linewidth=BOX_LW))
ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
ax.set_aspect("equal"); ax.autoscale_view()
ax.set_title(f"liquid_argon_underground (z ≤ {ZCUT:.0f} mm)")
plt.savefig("ular_2d.pdf", bbox_inches="tight")

# ---- 3D view ------------------------------------------------------------------------
fig = plt.figure(figsize=(7, 8))
ax3 = fig.add_subplot(111, projection="3d")
for phi in np.linspace(0, 2 * np.pi, N_MERID, endpoint=False):
    ax3.plot(rc * np.cos(phi), rc * np.sin(phi), zc, "C0-", lw=0.6)
phis = np.linspace(0, 2 * np.pi, 120)
for k in sorted(set(range(0, len(rc), RING_STEP)) | {len(rc) - 1}):
    if rc[k] > 0:
        ax3.plot(rc[k] * np.cos(phis), rc[k] * np.sin(phis),
                 np.full_like(phis, zc[k]), "C0-", lw=0.6)

ax3.add_collection3d(Poly3DCollection(ge3d_p, facecolor="0.5", edgecolor="none",
                                      alpha=GE_FILL_A))
ax3.add_collection3d(Line3DCollection(ge3d_s, colors=GE_COLOR, linewidths=GE_LW))
if len(rod_all):
    ax3.add_collection3d(Poly3DCollection(rod_all, facecolor=ROD_COLOR, edgecolor="none",
                                          alpha=ROD_FILL_A))
    ax3.add_collection3d(Line3DCollection(rod_edges, colors=[ROD_COLOR], linewidths=ROD_LW))
if len(sipm_pos):                                     # SiPM channels: dots
    ax3.scatter(sipm_pos[:, 0], sipm_pos[:, 1], sipm_pos[:, 2], s=SIPM_S3D,
                c=SIPM_COLOR, depthshade=False)
if BOX is not None:                                   # optional box: 12 red edges
    ax3.add_collection3d(Line3DCollection(box_edges(BOX), colors=BOX_COLOR,
                                          linewidths=BOX_LW))

zlo, zhi = zc.min(), zc.max()
if BOX is not None:                                   # keep a box outside the z range visible
    zlo, zhi = min(zlo, BOX[2, 0]), max(zhi, BOX[2, 1])
ax3.set_box_aspect((2 * rc.max(), 2 * rc.max(), zhi - zlo))
ax3.set_xlim(-rc.max(), rc.max()); ax3.set_ylim(-rc.max(), rc.max())
ax3.set_zlim(zlo, zhi)
ax3.set_xlabel("x [mm]"); ax3.set_ylabel("y [mm]"); ax3.set_zlabel("z [mm]")
ax3.set_title(f"liquid_argon_underground (z ≤ {ZCUT:.0f} mm)")
plt.savefig("ular_3d.pdf", bbox_inches="tight")

# ---- x-y view (top-down projection): Ge strings, ULAr wall, optional box -------------
from matplotlib.patches import Circle

ge_xy = []                                    # (x, y, rmax) of every Ge detector

def walk_xy(parent_lv, R, T):
    for pv in children_of(parent_lv):
        child = getattr(pv, "logicalVolume", None)
        if child is None:
            continue
        rot = tr.tbxyz2matrix(to_vec(pv.rotation)) if hasattr(pv, "rotation") else np.eye(3)
        pos = to_vec(pv.position) if hasattr(pv, "position") else np.zeros(3)
        R2, T2 = R @ rot.T, R @ pos + T
        sname = getattr(getattr(child, "solid", None), "name", "")
        if GE_RE.match(child.name) or GE_RE.match(sname):
            prof = get_profile(child.solid)
            if prof is not None:
                ge_xy.append((T2[0], T2[1], prof[0].max()))
        elif not ("sipm" in child.name.lower() or child.name.startswith(ROD_PREFIX)):
            walk_xy(child, R2, T2)

walk_xy(lv, np.eye(3), np.zeros(3))
ge_xy = np.unique(np.round(np.array(ge_xy), 1), axis=0)   # one circle per string position
print(f"x-y view: {len(ge_xy)} distinct Ge (x, y) positions")

fig, ax = plt.subplots(figsize=(7, 7))
ax.add_patch(Circle((0, 0), rc.max(), fill=False, edgecolor="C0", linewidth=1))
for x_, y_, rm in ge_xy:
    ax.add_patch(Circle((x_, y_), rm, facecolor="0.5", alpha=GE_FILL_A * 4,
                        edgecolor=GE_COLOR, linewidth=GE_LW))
if BOX is not None:
    ax.add_patch(Rectangle((BOX[0, 0], BOX[1, 0]), BOX[0, 1] - BOX[0, 0],
                           BOX[1, 1] - BOX[1, 0], fill=False,
                           edgecolor=BOX_COLOR, linewidth=BOX_LW))
ax.set_xlim(-rc.max() * 1.05, rc.max() * 1.05)
ax.set_ylim(-rc.max() * 1.05, rc.max() * 1.05)
ax.set_aspect("equal")
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title("liquid_argon_underground: x-y projection")
plt.savefig("ular_xy.pdf", bbox_inches="tight")
plt.show()
