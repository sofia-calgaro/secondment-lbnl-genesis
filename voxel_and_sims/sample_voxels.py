"""
STEP 2 -- sample the voxels of every training / validation set from a voxel-space file and write
one remage macro per voxel that needs a simulation.

The space (grid, box limits, allowed voxels) is fixed in STEP 1 (make_voxel_space.py).  Sets
(LF = --prim-lf primaries / photons, HF = --prim-hf):

  Training step 1   LF_train   n_lf voxels, LF statistics                         SIMULATE
  Training step 2   LF2_train  n_lf2 other voxels, LF statistics                  no simulation:
                                                                                  locations only
                    HF_train   n_hf voxels, HF statistics                         SIMULATE
                               default: 50% share locations with LF2_train; remaining HF
                               locations are new and disjoint from all other sets
  Validation        CNP_val    n_cnp_val voxels, LF statistics                    SIMULATE
                    GP_val     n_gp_val voxels, HF statistics                     SIMULATE

LF_train and LF2_train are disjoint. HF_train overlaps LF2_train at exactly the requested
fraction; its remaining locations come from a separate group and overlap no other set.
Validation voxels are disjoint from every training location, while CNP_val and GP_val may overlap.
Group membership is assigned by a repeating 3D parity pattern, so every location group
is interleaved throughout the box rather than forming spatial clusters. A seeded uniform permutation of the actual eligible CSV-grid voxels selects locations within each group.

    python make_voxel_space.py --vox 50 --xrange ... --out space.csv
    python sample_voxels.py space.csv --n-lf 800 --n-lf2 800 --n-hf 80 --n-cnp-val 500 --n-gp-val 500
    python sample_voxels.py space.csv --lf-frac 0.10 --hf-frac 0.01     # fractions of the space

EXTENDABLE: each set is a deterministic prefix of a seeded uniform permutation of eligible
voxels in its interleaved location group. Increasing counts with the same output directory, seed, box,
voxel-space file, and sampler preserves earlier voxel locations and adds new ones. Do not change
those settings for an extension. Simulation seeds are deterministic per location/tier; job lists
skip finished outputs. The script refuses to overwrite a previous campaign if its fixed settings
change or if a requested count decreases.

Optional "core": a box shrunk by --core-margin mm from every face of the space's box.  The LF
sets are sampled core:outer with density ratio --ratio (default margin 0 -> uniform sampling);
validation sets are uniform.

Output (in --outdir, default voxel_sets_<grid name>), same layout for results/, maps/, macros/,
jobs/, logs/:
  voxels_LF_train.csv, voxels_LF2_train.csv, voxels_HF_train.csv, voxels_CNP_val.csv,
  voxels_GP_val.csv, voxels_all.csv (= the sets that are simulated)
  manifest.json   which set feeds which training / validation step, counts, simulate yes/no
  macros/{lf,hf}/run00001_x+0.712_y-0.336_z-0.494.mac   (x,y,z = voxel centre, detector frame, m)
  results/{lf,hf}/<same stem>.stp.lh5   maps/{lf,hf}/<same stem>.map.lh5
  jobs/{lf,hf}/{remage,maps}.txt and {remage,maps}_chunk_XX.txt, jobs/all_{remage,maps}.txt
  config.json (grid definition + space file path and sha256), qc.png, qc_3d.png
With --scratch-dir, results/, maps/ and logs/ are stored there and symlinked into --outdir.
Submit on NERSC with submit_nersc.sh.

Seeds (the /RMG/Manager/Randomization/Seed line of every macro; also given for LF2_train) are a
hash of (--sim-seed, tier, voxel size, ix, iy, iz), a stable hash of simulation seed, tier, voxel size and voxel indices: a voxel keeps its
seed whatever the rest of the campaign looks like.
"""
import argparse
import csv
import hashlib
import json
import os
import warnings

import numpy as np

from voxel_grid import VoxelSpace

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Globals (set from the space file / command line in main)
# ----------------------------------------------------------------------------
VOX = 5.0
BOX_MIN = np.array([-800.0, 100.0, 500.0])
BOX_MAX = np.array([400.0, 500.0, 700.0])
CORE_MARGIN = 0.0
SAMPLER = "uniform"
DEFAULT_GDML = ("/global/cfs/cdirs/m2676/users/calgaro/uzh/secondment/"
                "optmap_demo/legend-pygeom-l1000/l1000.gdml")


# ----------------------------------------------------------------------------
# Space wrapper / sampling
# ----------------------------------------------------------------------------
class Grid:
    """The loaded voxel space plus the core/outer zones."""

    def __init__(self, space):
        self.space = space
        self.g = space.grid

    def centers(self, idx):
        return self.g.centers(idx)

    def zone_core(self, idx):
        c = self.centers(idx)
        lo = BOX_MIN + CORE_MARGIN
        hi = BOX_MAX - CORE_MARGIN
        return ((c >= lo) & (c <= hi)).all(1)

    def valid(self, idx):
        """Voxel belongs to the space and its centre lies inside the requested sub-box."""
        c = self.centers(idx)
        in_subbox = ((c >= BOX_MIN) & (c < BOX_MAX)).all(1)
        return self.space.contains(idx) & in_subbox


class VoxelSource:
    """Seeded uniform permutation of the actual eligible voxels in the CSV grid.

    Unlike point-then-snap sampling, this samples each eligible grid voxel at most once
    and cannot silently undersample regions because random points landed in the same voxel.
    The candidate list is built from the selected XYZ box and the loaded space.csv. Its
    permutation is deterministic, so increasing counts preserves the earlier selections.
    The fourth value is uniform in [0, 1) for optional core/outer thinning.
    """
    def __init__(self, grid, seed, m=12):
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        # Find the grid-index extent of the requested physical box, then enumerate every
        # grid cell in that extent. The voxel space CSV remains the authority on valid cells.
        eps = max(float(grid.g.size) * 1e-9, 1e-12)
        corner_idx = np.asarray(grid.g.index_of(np.vstack([BOX_MIN, BOX_MAX - eps])), dtype=int)
        lo = np.min(corner_idx, axis=0)
        hi = np.max(corner_idx, axis=0)
        axes = [np.arange(lo[d], hi[d] + 1, dtype=int) for d in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        candidates = np.column_stack([m.ravel() for m in mesh])
        # Filter against both the saved voxel space and the exact requested centre bounds.
        keep = grid.valid(candidates)
        candidates = candidates[keep]
        if len(candidates) == 0:
            raise SystemExit("No eligible voxels from space.csv fall inside the selected XYZ box.")
        # Spatially stratify the ordering into fixed 10-voxel bins in each axis. Shuffle
        # within each bin, then round-robin across bins. This ensures early samples cover
        # the full selected volume instead of relying on random points to happen to spread.
        # The bins are tied to the saved grid indices, not requested sample counts, so extending
        # a campaign preserves the exact earlier prefix.
        bin_width = 10
        bins = {}
        for row in candidates:
            key = tuple((row // bin_width).tolist())
            bins.setdefault(key, []).append(row)
        keys = sorted(bins)
        for key in keys:
            rows = np.asarray(bins[key], dtype=int)
            bins[key] = rows[self.rng.permutation(len(rows))]
        # Randomize the order in which spatial bins are visited, then take one voxel per bin
        # per pass; this is a randomized round-robin stratified permutation.
        if keys:
            keys = [keys[i] for i in self.rng.permutation(len(keys))]
        ordered = []
        max_depth = max(len(bins[k]) for k in keys)
        for depth in range(max_depth):
            for key in keys:
                if depth < len(bins[key]):
                    ordered.append(bins[key][depth])
        self.order = np.asarray(ordered, dtype=int)
        self.cursor = 0

    def batch(self):
        if self.cursor >= len(self.order):
            raise SystemExit("Ran out of eligible grid voxels. Reduce requested counts or enlarge the box.")
        stop = min(self.cursor + 4096, len(self.order))
        idx = self.order[self.cursor:stop]
        self.cursor = stop
        return idx, self.rng.random(len(idx))


GROUPS = ("lf1", "lf2", "hfextra", "val")  # mutually exclusive location groups
LF_GROUPS = ("lf1", "lf2")                 # these get the core/outer thinning


def voxel_group(grid, seed, v):
    """Spatially interleave four mutually exclusive groups at voxel scale.

    The parity pattern repeats in all three coordinates: adjacent voxels are distributed
    among groups instead of assigning groups through a random hash (which can leave visible
    spatial clumps). The seed is intentionally not used for ownership: changing the selection
    seed changes the uniform random ordering, not which spatial regions are eligible for a group.
    """
    ix, iy, iz = (int(x) for x in v)
    g = (ix % 2 + 2 * (iy % 2) + iz % 2) % len(GROUPS)
    return GROUPS[g]


def make_sets(grid, counts, ratio, seed):
    """Sample mutually exclusive location groups plus independently sampled validation sets.

    LF_train, LF2_train, and extra HF locations come from different interleaved spatial groups.
    CNP_val and GP_val are independently sampled from the validation group and may overlap.
    """
    chosen = {"lf1": [], "lf2": [], "hfextra": [], "cnp": [], "gp": []}
    seen = {g: set() for g in chosen}

    # Training sets use one fixed uniform-permutation stream; their spatially interleaved groups are mutually exclusive.
    src = VoxelSource(grid, seed + 7)
    tries = 0
    while any(len(chosen[g]) < counts[g] for g in ("lf1", "lf2", "hfextra")):
        idx, ua = src.batch()
        core = grid.zone_core(idx)
        ok = grid.valid(idx)
        for v, u, c, o in zip(idx, ua, core, ok):
            if not o:
                continue
            g = voxel_group(grid, seed, v)
            if g not in ("lf1", "lf2", "hfextra") or len(chosen[g]) >= counts[g]:
                continue
            t = tuple(v)
            if t in seen[g]:
                continue
            if g in LF_GROUPS and not (c or u < 1.0 / ratio):
                continue
            seen[g].add(t)
            chosen[g].append(t)
        tries += 1
        if tries > 5000:
            short = {g: (len(chosen[g]), counts[g]) for g in ("lf1", "lf2", "hfextra")
                     if len(chosen[g]) < counts[g]}
            raise SystemExit(f"Could not fill {short} (have, wanted): space too small / mostly "
                             "excluded? Each hash group uses about a quarter of the space.")

    # Independent deterministic uniform-permutation streams over the shared validation group. This allows
    # CNP/GP overlap, while hash-group separation guarantees no overlap with training.
    for g, stream_seed in (("cnp", seed + 1007), ("gp", seed + 2007)):
        src = VoxelSource(grid, stream_seed)
        tries = 0
        while len(chosen[g]) < counts[g]:
            idx, _ = src.batch()
            ok = grid.valid(idx)
            for v, o in zip(idx, ok):
                if not o or voxel_group(grid, seed, v) != "val":
                    continue
                t = tuple(v)
                if t in seen[g]:
                    continue
                seen[g].add(t)
                chosen[g].append(t)
                if len(chosen[g]) >= counts[g]:
                    break
            tries += 1
            if tries > 5000:
                raise SystemExit(f"Could not fill {g}: validation group too small / mostly excluded.")
    return {g: np.array(chosen[g], int).reshape(-1, 3) for g in chosen}


# ----------------------------------------------------------------------------
# Macros / tables
# ----------------------------------------------------------------------------
MACRO = """# {vid}  ({kind})  voxel index ({ix},{iy},{iz})  centre (detector frame) {cx:.1f} {cy:.1f} {cz:.1f} mm
/RMG/Processes/OpticalPhysics true
/RMG/Processes/OpticalPhysicsMaxOneWLSPhoton true
/RMG/Geometry/RegisterDetectorsFromGDML Optical
/RMG/Geometry/GDMLDisableOverlapCheck true
/run/initialize

/RMG/Output/NtuplePerDetector false
/RMG/Output/Vertex/StorePrimaryParticleInformation true
/RMG/Output/Vertex/StoreSinglePrecisionEnergy true
/RMG/Output/Vertex/StoreSinglePrecisionPosition true
/RMG/Output/Vertex/StorePrimaryDirection true
{seedline}

/RMG/Generator/Confine Volume # activates vertex confinement

/RMG/Generator/Confinement/SamplingMode IntersectPhysicalWithGeometrical
/RMG/Generator/Confinement/ForceContainmentCheck true

/RMG/Generator/Confinement/Physical/AddVolume liquid_argon_underground
/RMG/Generator/Confinement/Geometrical/AddSolid Box
/RMG/Generator/Confinement/Geometrical/CenterPositionX {x_mm:.3f} mm
/RMG/Generator/Confinement/Geometrical/CenterPositionY {y_mm:.3f} mm
/RMG/Generator/Confinement/Geometrical/CenterPositionZ {z_mm:.3f} mm
/RMG/Generator/Confinement/Geometrical/Box/XLength {vox:g} mm
/RMG/Generator/Confinement/Geometrical/Box/YLength {vox:g} mm
/RMG/Generator/Confinement/Geometrical/Box/ZLength {vox:g} mm

/RMG/Generator/Select GPS
/gps/particle     opticalphoton
/gps/ene/type     Gauss
/gps/ene/mono     9.68 eV # 128nm (VUV) LAr scintillation
/gps/ene/sigma    0.22 eV # gaussian width
/gps/ang/type     iso

/run/beamOn {n}
"""


def macro_text(row, seed_template):
    return MACRO.format(
        vid=row["voxel_id"], kind=row["set"],
        ix=row["ix"], iy=row["iy"], iz=row["iz"],
        cx=row["x_mm"], cy=row["y_mm"], cz=row["z_mm"],
        x_mm=row["x_mm"], y_mm=row["y_mm"], z_mm=row["z_mm"],
        vox=VOX,
        seedline=seed_template.format(seed=row["seed"]) if seed_template else "",
        n=row["n_primaries"],
    )


def stable_voxel_seeds(idx, tier, sim_seed, voxel_size_mm):
    """Stable per-voxel seeds without requiring a custom VoxelGrid.voxel_seed method."""
    seeds = np.empty(len(idx), dtype=np.int64)
    for i, (ix, iy, iz) in enumerate(np.asarray(idx, dtype=np.int64)):
        payload = f"{int(sim_seed)}|{tier}|{float(voxel_size_mm):.9g}|{int(ix)}|{int(iy)}|{int(iz)}"
        # Keep seeds positive and within the conventional signed 31-bit range.
        value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") & 0x7fffffff
        seeds[i] = value or 1
    return seeds


def make_rows(idx, kind, prefix, grid, nprim, ratio, sim_seed, partner=None):
    rows = []
    cen = grid.centers(idx)
    core = grid.zone_core(idx)
    # Seed is a stable hash of simulation seed, tier, voxel size and voxel indices.
    # It is independent of sample order/count and needs no VoxelGrid-specific seed method.
    tier = {"LF_train": "lf", "CNP_val": "lf", "LF2_train": "lf2"}.get(kind, "hf")
    seeds = stable_voxel_seeds(idx, tier, sim_seed, grid.g.size)
    for k in range(len(idx)):
        x, y, z = cen[k]
        rows.append({
            "voxel_id": f"{prefix}_{k:04d}", "set": kind,
            "ix": int(idx[k, 0]), "iy": int(idx[k, 1]), "iz": int(idx[k, 2]),
            "x_mm": x, "y_mm": y, "z_mm": z,
            "x_rel": x - BOX_MIN[0], "y_rel": y - BOX_MIN[1], "z_rel": z - BOX_MIN[2],
            "zone": "core" if core[k] else "outer",
            "weight": 1.0 if core[k] else ratio,
            "n_primaries": nprim, "seed": int(seeds[k]),
            "partner_lf_id": partner[k] if partner else "",
        })
    return rows


RUN_WIDTH = 5          # run00001: fixed width, so names never change when the counts grow
VAL_RUN_OFFSET = 10000  # validation runs are numbered 10001, 10002, ... (training: 1, 2, ...)
TIER = {"LF_train": "lf", "CNP_val": "lf", "HF_train": "hf", "GP_val": "hf",
        "LF2_train": None}   # None: not simulated, so no files



def generate_lf2_event_inputs(outdir, rows, n_events, voxel_size_mm, base_seed):
    """Generate CNP input events at LF2 locations without running Geant4.

    Positions are uniform in the voxel cube; photon directions are isotropic and energies
    follow the same Gaussian parameters as the remage GPS macro. This is a source sampler,
    not a transport simulation. It assumes the voxel cube is an acceptable source volume;
    for voxels intersecting non-LAr geometry, use geometry-aware sampling instead.
    """
    event_dir = os.path.join(outdir, "events", "lf2")
    os.makedirs(event_dir, exist_ok=True)
    half = voxel_size_mm / 2.0
    for row in rows:
        # Use the stable per-voxel LF2 seed, independent of row order.
        rng = np.random.default_rng((int(row["seed"]) ^ int(base_seed)) % (2**32))
        center = np.array([row["x_mm"], row["y_mm"], row["z_mm"]], dtype=np.float64)
        position = rng.uniform(center - half, center + half, size=(n_events, 3)).astype(np.float32)
        cos_theta = rng.uniform(-1.0, 1.0, size=n_events)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_events)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
        direction = np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))
        energy = np.maximum(rng.normal(9.68, 0.22, size=n_events), 1e-6).astype(np.float32)
        momentum = (direction * energy[:, None]).astype(np.float32)  # photon momentum in eV/c
        path = os.path.join(event_dir, row["tag"] + ".npz")
        np.savez_compressed(path, position_mm=position, momentum_eV_c=momentum,
                            energy_eV=energy)
        row["event_file"] = os.path.relpath(path, outdir)
        row["n_events"] = n_events


def assign_names(sets):
    """Add tier (lf/hf), run_id, tag (file stem) and relative file paths to every row.
    Tag: run00001_x+0.712_y-0.336_z-0.494 (detector-frame voxel centre in metres).
    Run numbers depend only on the position within the set, never on the other sets or on the
    counts: training sets 1.., validation sets 10001.. (LF_train and CNP_val share the 'lf'
    tier, HF_train and GP_val the 'hf' tier). HF_train is sampled from both LF training sets;
    its partner_lf_id identifies the LF row at the same location. LF2_train is not simulated:
    it gets run ids / tags but no file paths. With a fixed seed and overlap fraction, growing
    counts keeps the selected prefix of each deterministic selection stream."""
    for name, rows in sets.items():
        start = VAL_RUN_OFFSET + 1 if name.endswith("_val") else 1
        room = 10 ** RUN_WIDTH - start if name.endswith("_val") else VAL_RUN_OFFSET - 1
        if len(rows) > room:
            raise SystemExit(f"{name}: {len(rows)} voxels exceed the run-number range ({room}); "
                             f"raise VAL_RUN_OFFSET / RUN_WIDTH")
        for n, r in enumerate(rows, start=start):
            x, y, z = (round(r[k] / 1000.0, 3) + 0.0 for k in ("x_mm", "y_mm", "z_mm"))
            tag = f"run{n:0{RUN_WIDTH}d}_x{x:+.3f}_y{y:+.3f}_z{z:+.3f}"
            tier = TIER[name]
            r["tier"] = tier or "none"
            r["run_id"] = f"run{n:0{RUN_WIDTH}d}"
            r["tag"] = tag
            r["stp_file"] = f"results/{tier}/{tag}.stp.lh5" if tier else ""
            r["map_file"] = f"maps/{tier}/{tag}.map.lh5" if tier else ""


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def write_jobs(outdir, name, cmds_by_tier, chunk):
    """jobs/<tier>/<name>.txt, jobs/<tier>/<name>_chunk_XX.txt and jobs/all_<name>.txt"""
    allc = []
    for tier, cmds in cmds_by_tier.items():
        d = os.path.join(outdir, "jobs", tier)
        write_lines(os.path.join(d, f"{name}.txt"), cmds)
        for i in range(0, len(cmds), chunk):
            write_lines(os.path.join(d, f"{name}_chunk_{i // chunk:02d}.txt"), cmds[i:i + chunk])
        allc += cmds
    write_lines(os.path.join(outdir, "jobs", f"all_{name}.txt"), allc)


def setup_dirs(outdir, scratch):
    """macros/ and jobs/ live in outdir. results/, maps/, logs/ are real directories in
    outdir, or, with --scratch-dir, live on scratch and are symlinked into outdir (all
    paths in the CSVs / job lists stay <outdir>/results/<tier>/... and keep working)."""
    for t in ("lf", "hf"):
        for d in ("macros", "jobs"):
            os.makedirs(os.path.join(outdir, d, t), exist_ok=True)
    for d in ("results", "maps", "logs"):
        link = os.path.join(outdir, d)
        if scratch:
            real = os.path.join(scratch, os.path.basename(outdir), d)
            for t in ("lf", "hf"):
                os.makedirs(os.path.join(real, t), exist_ok=True)
            if os.path.islink(link):
                if os.path.realpath(link) != os.path.realpath(real):
                    os.remove(link)
                    os.symlink(real, link)
            elif os.path.lexists(link):
                raise SystemExit(f"{link} already exists as a real directory; "
                                 f"move it to {real} or remove it")
            else:
                os.symlink(real, link)
        else:
            for t in ("lf", "hf"):
                os.makedirs(os.path.join(link, t), exist_ok=True)


def write_csv(path, rows):
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]) for c in cols})


# ----------------------------------------------------------------------------
# QC plots
# ----------------------------------------------------------------------------
STYLE = {"LF_train": ("#7F77DD", 6, "o"), "LF2_train": ("#3C8DBC", 6, "o"),
         "HF_train": ("#D85A30", 36, "s"), "CNP_val": ("#1D9E75", 14, "^"),
         "GP_val": ("#E8B72B", 40, "D")}


def qc_plot(path, sets):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
    for name, rows in sets.items():
        c, s, m = STYLE[name]
        P = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in rows])
        for a, (i, j, ni, nj) in zip(ax, pairs):
            a.scatter(P[:, i], P[:, j], s=s, c=c, marker=m, alpha=0.8,
                      label=f"{name} ({len(rows)})")
    for a, (i, j, ni, nj) in zip(ax, pairs):
        lo, hi = BOX_MIN, BOX_MAX
        a.plot([lo[i], hi[i], hi[i], lo[i], lo[i]], [lo[j], lo[j], hi[j], hi[j], lo[j]],
               "k-", lw=0.8)
        if CORE_MARGIN > 0:
            l2, h2 = lo + CORE_MARGIN, hi - CORE_MARGIN
            a.plot([l2[i], h2[i], h2[i], l2[i], l2[i]], [l2[j], l2[j], h2[j], h2[j], l2[j]],
                   "k--", lw=0.8)
        a.set_xlabel(f"{ni} [mm]")
        a.set_ylabel(f"{nj} [mm]")
        a.set_aspect("equal")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _box_outline(ax, lo, hi, style):
    for a in (0, 1):
        for b in (0, 1):
            xs = [lo[0] if a == 0 else hi[0]] * 2
            ys = [lo[1] if b == 0 else hi[1]] * 2
            ax.plot(xs, ys, [lo[2], hi[2]], style, lw=0.6)
            ax.plot([lo[0], hi[0]], [ys[0]] * 2, [lo[2] if a == 0 else hi[2]] * 2, style, lw=0.6)
            ax.plot([xs[0]] * 2, [lo[1], hi[1]], [lo[2] if b == 0 else hi[2]] * 2, style, lw=0.6)


def _local_density(P):
    """Gaussian-KDE density of the sampled positions, in points per litre."""
    from scipy.stats import gaussian_kde
    if len(P) < 5:
        return np.ones(len(P))
    try:
        return gaussian_kde(P.T)(P.T) * len(P) * 1e6   # 1 L = 1e6 mm^3
    except np.linalg.LinAlgError:
        return np.ones(len(P))


def qc_plot_3d(path, sets):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    size = BOX_MAX - BOX_MIN
    fig = plt.figure(figsize=(20, 11))

    def finish(ax):
        _box_outline(ax, BOX_MIN, BOX_MAX, "k-")
        if CORE_MARGIN > 0:
            _box_outline(ax, BOX_MIN + CORE_MARGIN, BOX_MAX - CORE_MARGIN, "k--")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
        ax.set_xlim(BOX_MIN[0], BOX_MAX[0]); ax.set_ylim(BOX_MIN[1], BOX_MAX[1])
        ax.set_zlim(BOX_MIN[2], BOX_MAX[2])
        ax.set_box_aspect(tuple(size)); ax.view_init(elev=22, azim=-40)

    panels = [("LF_train", "viridis", 10), ("LF2_train", "viridis", 10),
              ("HF_train", "plasma", 55), ("CNP_val", "viridis", 14), ("GP_val", "plasma", 55)]
    for i, (name, cmap, sz) in enumerate(panels, start=1):
        rows = sets[name]
        P = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in rows])
        d = _local_density(P)
        o = np.argsort(d)
        ax = fig.add_subplot(2, 3, i, projection="3d")
        sc = ax.scatter(P[o, 0], P[o, 1], P[o, 2], c=d[o], cmap=cmap, s=sz,
                        depthshade=False, edgecolors="none" if name in ("LF_train", "LF2_train", "CNP_val") else "k",
                        linewidths=0.3)
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, label="KDE density [points / L]")
        ax.set_title(f"{name} ({len(rows)} voxels)")
        finish(ax)
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    for name, rows in sets.items():
        c, s, m = STYLE[name]
        ax.scatter([r["x_mm"] for r in rows], [r["y_mm"] for r in rows],
                   [r["z_mm"] for r in rows], c=c, s=s, marker=m, alpha=0.75,
                   label=f"{name} ({len(rows)})", depthshade=False)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("All sets")
    finish(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
def check_existing_campaign(outdir, current):
    """Prevent accidental resampling/relabeling when extending an existing campaign."""
    path = os.path.join(outdir, "config.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path) as f:
            old = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Cannot read existing campaign config {path}: {e}")
    if old.get("sampling_algorithm") != "uniform-grid-permutation-v1":
        raise SystemExit(
            f"{outdir} was not created with the current spatially balanced sampler. "
            "Use a NEW --outdir for this revised sampling design; do not mix old and new samples.")
    old_space = old.get("space_info", {})
    new_space = current.get("space_info", {})
    fixed = ["seed", "sim_seed", "sampler", "ratio", "hf_overlap_lf2", "prim_lf", "prim_hf",
             "core_margin", "xrange", "yrange", "zrange"]
    mismatches = [k for k in fixed if old.get(k) != current.get(k)]
    if old_space.get("sha256") != new_space.get("sha256"):
        mismatches.append("space.csv contents")
    if old.get("box_min") != current.get("box_min") or old.get("box_max") != current.get("box_max"):
        mismatches.append("sampling box")
    if old.get("vox") != current.get("vox"):
        mismatches.append("voxel size")
    if mismatches:
        raise SystemExit("Cannot extend this campaign because fixed sampling settings changed: "
                         + ", ".join(mismatches) + ". Use the original settings or a new --outdir.")
    for arg, key in [("n_lf", "n-lf"), ("n_lf2", "n-lf2"), ("n_hf", "n-hf"),
                     ("n_cnp_val", "n-cnp-val"), ("n_gp_val", "n-gp-val")]:
        previous = int(old.get(arg, 0))
        requested = int(current.get(arg, 0))
        if requested < previous:
            raise SystemExit(f"Cannot reduce --{key} from {previous} to {requested} in an existing "
                             "campaign: this would drop previously selected voxels. Use a new --outdir.")


# ----------------------------------------------------------------------------
def main():
    global VOX, BOX_MIN, BOX_MAX, CORE_MARGIN, SAMPLER
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space", help="voxel-space file written by make_voxel_space.py")
    ap.add_argument("--outdir", default=None, help="default: voxel_sets_<grid name>")
    ap.add_argument("--xrange", nargs=2, type=float, metavar=("XMIN", "XMAX"),
                    help="select voxel centres in this x range [mm]; default: full space")
    ap.add_argument("--yrange", nargs=2, type=float, metavar=("YMIN", "YMAX"),
                    help="select voxel centres in this y range [mm]; default: full space")
    ap.add_argument("--zrange", nargs=2, type=float, metavar=("ZMIN", "ZMAX"),
                    help="select voxel centres in this z range [mm]; default: full space")
    ap.add_argument("--seed", type=int, default=12345,
                    help="seed of the voxel SELECTION (which voxels are picked)")
    ap.add_argument("--sim-seed", type=int, default=12345,
                    help="base of the per-voxel simulation seeds written to the macros. Seed = "
                         "hash(sim-seed, tier, voxel size, ix, iy, iz): the same voxel always "
                         "gets the same seed, independent of --seed, counts or the space file. "
                         "Keep it fixed for the whole project.")
    ap.add_argument("--core-margin", type=float, default=0.0,
                    help="shell thickness [mm] at the box faces treated as 'outer' "
                         "(0 -> whole box is core / uniform)")
    ap.add_argument("--n-lf", type=int, default=800,
                    help="LF_train voxels (training step 1), LF statistics, simulated")
    ap.add_argument("--n-lf2", type=int, default=None,
                    help="LF2_train voxels (training step 2, LF statistics, NOT simulated, "
                         "disjoint from LF_train); default: same as --n-lf")
    ap.add_argument("--n-hf", type=int, default=80,
                    help="HF_train voxels (training step 2), HF statistics, simulated")
    ap.add_argument("--hf-overlap-lf2", type=float, default=0.5,
                    help="fraction of HF_train locations shared with LF2_train (default 0.5); "
                         "remaining HF locations are new and disjoint from every other set. "
                         "Must be between 0 and 1.")
    ap.add_argument("--n-cnp-val", type=int, default=500,
                    help="CNP_val voxels (validation), LF statistics")
    ap.add_argument("--n-gp-val", type=int, default=500,
                    help="GP_val voxels (validation), HF statistics")
    ap.add_argument("--lf-frac", type=float, default=None,
                    help="LF_train as a fraction of all voxels in the space (overrides --n-lf)")
    ap.add_argument("--lf2-frac", type=float, default=None,
                    help="LF2_train as a fraction of all voxels (overrides --n-lf2; default: "
                         "--lf-frac if given)")
    ap.add_argument("--hf-frac", type=float, default=None,
                    help="HF_train as a fraction of all voxels in the space (overrides --n-hf)")
    ap.add_argument("--ratio", type=float, default=2.0, help="core:outer LF density ratio")
    ap.add_argument("--prim-lf", type=int, default=5000)
    ap.add_argument("--prim-hf", type=int, default=20000)
    ap.add_argument("--seed-line", default="/RMG/Manager/Randomization/Seed {seed}",
                    help="macro line template for the RNG seed ('' to omit)")
    ap.add_argument("--gdml", default=None,
                    help="GDML for the remage jobs (default: the one used for the GDML check "
                         "in step 1, else the NERSC l1000.gdml)")
    ap.add_argument("--cmd", default="pixi run remage -w --procs ${{PROCS:-1}} --output-file {out} --flat-output "
                    "--gdml-files {gdml} -- {macro}",
                    help="job command template; fields: gdml, out, macro (literal braces doubled)")
    ap.add_argument("--map-settings", default="map-settings.yaml")
    ap.add_argument("--map-geom", default=None, help="GDML for reboost-optmap (default --gdml)")
    ap.add_argument("--map-cmd",
                    default="pixi run reboost-optmap create --settings {settings} "
                            "--geom {geom} {input} {output}",
                    help="fields: settings, geom, input, output")
    ap.add_argument("--chunk", type=int, default=50, help="voxels per job chunk file")
    ap.add_argument("--scratch-dir", default="",
                    help="put results/, maps/, logs/ under <scratch-dir>/<outdir name>/ "
                         "(e.g. '$PSCRATCH/optmap'); symlinked into --outdir")
    ap.add_argument("--sampler", choices=("uniform",), default="uniform",
                    help="uniform random permutation of eligible voxels in the predefined CSV grid; "
                         "deterministic and extendable (same seed/settings preserves earlier voxels)")
    a = ap.parse_args()

    a.space = os.path.abspath(a.space)
    space = VoxelSpace.load(a.space)
    gridsp = Grid(space)
    VOX = space.grid.size
    # The grid spacing is fixed by space.csv; these options only restrict which of its
    # voxels are eligible for sampling. Bounds are in detector coordinates, in mm.
    full_min, full_max = np.asarray(space.box_min, float), np.asarray(space.box_max, float)
    ranges = (a.xrange, a.yrange, a.zrange)
    BOX_MIN = np.array([r[0] if r is not None else full_min[i]
                        for i, r in enumerate(ranges)], dtype=float)
    BOX_MAX = np.array([r[1] if r is not None else full_max[i]
                        for i, r in enumerate(ranges)], dtype=float)
    if np.any(BOX_MAX <= BOX_MIN):
        raise SystemExit("Each selected range must have MAX > MIN")
    if np.any(BOX_MIN < full_min) or np.any(BOX_MAX > full_max):
        raise SystemExit(f"Selected sub-box {BOX_MIN.tolist()} .. {BOX_MAX.tolist()} mm must lie "
                         f"inside the space bounds {full_min.tolist()} .. {full_max.tolist()} mm")
    CORE_MARGIN = a.core_margin
    SAMPLER = a.sampler
    a.gdml = a.gdml or (space.meta.get("geo_check") or {}).get("gdml") or DEFAULT_GDML
    a.outdir = os.path.abspath(a.outdir or f"voxel_sets_{space.grid.name}")
    a.scratch_dir = os.path.abspath(os.path.expandvars(a.scratch_dir)) if a.scratch_dir else ""
    if "$" in a.scratch_dir:
        raise SystemExit(f"unexpanded variable in --scratch-dir: {a.scratch_dir}")
    print(f"space {os.path.basename(a.space)}: {space.grid.name}, {len(space):,} voxels, "
          f"sha256 {space.sha256[:12]}")
    print(f"sampling sub-box {BOX_MIN.tolist()} .. {BOX_MAX.tolist()} mm; "
          f"voxel side {VOX:g} mm (inherited from space.csv)")

    if a.lf_frac is not None:
        a.n_lf = int(round(a.lf_frac * len(space)))
    if a.lf2_frac is not None:
        a.n_lf2 = int(round(a.lf2_frac * len(space)))
    elif a.n_lf2 is None:
        a.n_lf2 = a.n_lf
    if a.hf_frac is not None:
        a.n_hf = int(round(a.hf_frac * len(space)))
    if not 0.0 <= a.hf_overlap_lf2 <= 1.0:
        raise SystemExit("--hf-overlap-lf2 must be between 0 and 1")
    n_hf_lf2 = int(round(a.n_hf * a.hf_overlap_lf2))
    n_hf_extra = a.n_hf - n_hf_lf2
    if n_hf_lf2 > a.n_lf2:
        raise SystemExit(f"HF/LF2 overlap requires {n_hf_lf2} shared voxels, but LF2 has only "
                         f"{a.n_lf2}; increase --n-lf2 or reduce --hf-overlap-lf2/--n-hf")
    if min(a.n_lf, a.n_lf2, a.n_hf, a.n_cnp_val, a.n_gp_val) < 0:
        raise SystemExit("All requested voxel counts must be non-negative")
    if a.n_lf == 0 or a.n_lf2 == 0 or a.n_hf == 0 or a.n_cnp_val == 0 or a.n_gp_val == 0:
        raise SystemExit("All five dataset counts must be positive")
    if a.ratio <= 0:
        raise SystemExit("--ratio must be positive")
    if len(space) < a.n_lf + a.n_lf2 + n_hf_extra + max(a.n_cnp_val, a.n_gp_val):
        raise SystemExit(f"Space has {len(space)} voxels; requested disjoint locations need at "
                         f"least n_lf + n_lf2 + HF-extra + max(validation counts) = "
                         f"{a.n_lf + a.n_lf2 + n_hf_extra + max(a.n_cnp_val, a.n_gp_val)}. "
                         "The shared HF/LF2 locations are counted only once.")
    # Existing output directory means EXTEND this exact campaign, not draw a new sample.
    # Count values have now been resolved (including any --*-frac options).
    current_config = {**vars(a), "grid": space.grid.describe(),
                      "space_info": {"file": a.space, "sha256": space.sha256,
                                     "n_voxels": len(space)},
                      "box_min": BOX_MIN.tolist(), "box_max": BOX_MAX.tolist(),
                      "vox": VOX, "core_margin": CORE_MARGIN,
                      "sampling_algorithm": "uniform-grid-permutation-v1"}
    check_existing_campaign(a.outdir, current_config)
    setup_dirs(a.outdir, a.scratch_dir)
    grid = gridsp

    # Training sets use disjoint interleaved spatial groups. Validation sets use independent deterministic
    # streams over a shared validation group, so CNP/GP may overlap each other but never training.
    S = make_sets(grid, {"lf1": a.n_lf, "lf2": a.n_lf2, "hfextra": n_hf_extra,
                       "cnp": a.n_cnp_val, "gp": a.n_gp_val}, a.ratio, a.seed)
    # Training step 1 (LF, simulated)
    lf_idx = S["lf1"]
    lf_rows = make_rows(lf_idx, "LF_train", "LFT", grid, a.prim_lf, a.ratio, a.sim_seed)
    lf_core = np.array([r["zone"] == "core" for r in lf_rows])
    # Training step 2, LF: other voxels, locations only (no simulation)
    lf2_idx = S["lf2"]
    lf2_rows = make_rows(lf2_idx, "LF2_train", "LF2", grid, a.prim_lf, a.ratio, a.sim_seed)
    # Training step 2, HF: exact requested overlap with LF2_train only. Remaining HF
    # locations are drawn from the disjoint hfextra spatial group.
    rng_hf = np.random.default_rng(a.seed + 3007)
    n_hf_lf2 = int(round(a.n_hf * a.hf_overlap_lf2))
    n_hf_extra = a.n_hf - n_hf_lf2
    lf2_pos = rng_hf.permutation(len(lf2_idx))[:n_hf_lf2] if n_hf_lf2 else np.array([], int)
    hf_overlap_idx = lf2_idx[lf2_pos]
    hf_extra_idx = S["hfextra"]
    if len(hf_extra_idx) != n_hf_extra:
        raise SystemExit(f"Internal sampling error: got {len(hf_extra_idx)} extra HF locations, "
                         f"expected {n_hf_extra}")
    hf_idx = np.vstack([hf_overlap_idx, hf_extra_idx])
    hf_partner_ids = ([lf2_rows[k]["voxel_id"] for k in lf2_pos]
                       + [""] * len(hf_extra_idx))
    hf_rows = make_rows(hf_idx, "HF_train", "HFT", grid, a.prim_hf, a.ratio, a.sim_seed,
                        partner=hf_partner_ids)
    # Validation: independent of each other and of every training voxel (never overlapping)
    cnp_idx = S["cnp"]
    cnp_rows = make_rows(cnp_idx, "CNP_val", "CNV", grid, a.prim_lf, a.ratio, a.sim_seed)
    gp_idx = S["gp"]
    gp_rows = make_rows(gp_idx, "GP_val", "GPV", grid, a.prim_hf, a.ratio, a.sim_seed)

    tup = lambda x: {tuple(v) for v in x}
    for idx in (lf_idx, lf2_idx, hf_idx, cnp_idx, gp_idx):
        assert grid.valid(idx).all()
    train = tup(lf_idx) | tup(lf2_idx) | tup(hf_idx)
    assert len(tup(lf_idx)) == len(lf_idx) and len(tup(lf2_idx)) == len(lf2_idx)
    assert not (tup(lf_idx) & tup(lf2_idx))
    assert len(tup(hf_idx)) == len(hf_idx)
    assert not (tup(cnp_idx) & train) and not (tup(gp_idx) & train)
    assert not (tup(hf_idx) & tup(lf_idx)), "HF_train must never overlap step-1 LF_train"
    assert len(tup(hf_idx) & tup(lf2_idx)) == n_hf_lf2
    assert len(tup(hf_idx) - tup(lf2_idx)) == n_hf_extra
    assert not (tup(hf_idx) & tup(cnp_idx)) and not (tup(hf_idx) & tup(gp_idx))
    # CNP_val and GP_val may overlap each other; this is intentional.

    sets = {"LF_train": lf_rows, "LF2_train": lf2_rows, "HF_train": hf_rows,
            "CNP_val": cnp_rows, "GP_val": gp_rows}
    allrows = [r for name, rows in sets.items() if TIER[name] for r in rows]   # simulated sets
    # seeds only need to be unique within a tier (the tier is part of the hash)
    for tier in ("lf", "hf"):
        sd = [r["seed"] for r in allrows if TIER[r["set"]] == tier]
        if len(set(sd)) != len(sd):
            raise SystemExit("two voxels of the same tier got the same simulation seed (31-bit "
                             "hash collision); use a different --sim-seed (and then keep it)")
    assign_names(sets)
    # LF2 needs event-level source inputs for CNP inference, but no Geant4 simulation.
    generate_lf2_event_inputs(a.outdir, lf2_rows, a.prim_lf, VOX, a.sim_seed)
    for name, rows in sets.items():
        write_csv(os.path.join(a.outdir, f"voxels_{name}.csv"), rows)
    write_csv(os.path.join(a.outdir, "voxels_all.csv"), allrows)
    manifest = {
        "grid": space.grid.describe(), "sampler": SAMPLER, "seed": a.seed, "sim_seed": a.sim_seed,
        "training_step_1": {"LF": {"set": "LF_train", "file": "voxels_LF_train.csv",
                                   "n_voxels": len(lf_rows), "primaries": a.prim_lf,
                                   "simulate": True}},
        "training_step_2": {
            "LF": {"set": "LF2_train", "file": "voxels_LF2_train.csv", "n_voxels": len(lf2_rows),
                   "photons": a.prim_lf, "simulate": False,
                   "note": "no Geant4 transport simulation; event-level positions and photon momenta are "
                           "sampled and saved as compressed NPZ files in events/lf2/"},
            "HF": {"set": "HF_train", "file": "voxels_HF_train.csv", "n_voxels": len(hf_rows),
                   "primaries": a.prim_hf, "simulate": True,
                   "overlap_with_LF_train": 0,
                   "overlap_with_LF2_train": n_hf_lf2,
                   "new_locations_not_in_LF2_train": n_hf_extra,
                   "note": "HF overlaps LF2_train only; remaining HF locations are new"}},
        "validation": {
            "CNP": {"set": "CNP_val", "file": "voxels_CNP_val.csv", "n_voxels": len(cnp_rows),
                    "primaries": a.prim_lf, "simulate": True},
            "GP": {"set": "GP_val", "file": "voxels_GP_val.csv", "n_voxels": len(gp_rows),
                   "primaries": a.prim_hf, "simulate": True,
                   "note": "independent of CNP_val"}},
        "rules": "LF_train, LF2_train, and extra HF locations are disjoint groups; HF_train shares "
                 "exactly round(n_hf * hf_overlap_lf2) locations with LF2_train and never overlaps "
                 "LF_train; validation is disjoint from all training sets; CNP_val and GP_val may overlap"}
    with open(os.path.join(a.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    map_geom = a.map_geom or a.gdml
    cmds = {"lf": [], "hf": []}
    map_cmds = {"lf": [], "hf": []}
    for r in allrows:
        t, tag = r["tier"], r["tag"]
        mpath = os.path.join(a.outdir, "macros", t, tag + ".mac")
        with open(mpath, "w") as f:
            f.write(macro_text(r, a.seed_line))
        out = os.path.join(a.outdir, r["stp_file"])
        mout = os.path.join(a.outdir, r["map_file"])
        log = os.path.join(a.outdir, "logs", t, tag)
        # skip finished outputs so a resubmission only runs what is missing
        cmds[t].append(f"test -s {out} || " + a.cmd.format(gdml=a.gdml, out=out, macro=mpath)
                       + f" > {log}.remage.log 2>&1")
        map_cmds[t].append(f"test -s {mout} || " + a.map_cmd.format(
            settings=a.map_settings, geom=map_geom, input=out, output=mout)
            + f" > {log}.map.log 2>&1")
    write_jobs(a.outdir, "remage", cmds, a.chunk)
    write_jobs(a.outdir, "maps", map_cmds, a.chunk)

    with open(os.path.join(a.outdir, "config.json"), "w") as f:
        json.dump({**vars(a), "grid": space.grid.describe(),
                   "space_info": {"file": a.space, "sha256": space.sha256,
                                  "n_voxels": len(space)},
                   "box_min": BOX_MIN.tolist(), "box_max": BOX_MAX.tolist(),
                   "vox": VOX, "core_margin": CORE_MARGIN,
                   "sampling_algorithm": "uniform-grid-permutation-v1"},
                  f, indent=2)

    qc_plot(os.path.join(a.outdir, "qc.png"), sets)
    qc_plot_3d(os.path.join(a.outdir, "qc_3d.png"), sets)

    tot = sum(r["n_primaries"] for r in allrows)
    print(f"box {BOX_MIN.tolist()} .. {BOX_MAX.tolist()} mm, voxel {VOX:g} mm")
    print(f"voxels in space: {len(space)}  sampler {SAMPLER}")
    print(f"training 1  LF_train  {len(lf_rows)}  LF statistics, simulated  (core "
          f"{int(lf_core.sum())}, outer {int((~lf_core).sum())})")
    print(f"training 2  LF2_train {len(lf2_rows)}  LF event inputs generated without Geant4; "
          f"{a.prim_lf:,} source events per voxel saved under events/lf2/")
    print(f"training 2  HF_train  {len(hf_rows)}  HF statistics, simulated")
    print(f"  HF locations shared with step-1 LF_train: {len(tup(hf_idx) & tup(lf_idx))} (must be 0)")
    print(f"  HF locations shared with step-2 LF2_train: {len(tup(hf_idx) & tup(lf2_idx))} "
          f"({len(tup(hf_idx) & tup(lf2_idx))/max(len(hf_rows), 1):.1%}; target "
          f"{a.hf_overlap_lf2:.1%}, rounded to {n_hf_lf2}/{a.n_hf})")
    print(f"  HF new locations outside LF2_train: {n_hf_extra} (must be disjoint from every other set)")
    val_overlap = len(tup(cnp_idx) & tup(gp_idx))
    print(f"validation  CNP_val   {len(cnp_rows)}  LF statistics, simulated")
    print(f"validation  GP_val    {len(gp_rows)}  HF statistics, simulated")
    print(f"  CNP/GP validation shared locations: {val_overlap} "
          f"({val_overlap/max(min(len(cnp_rows), len(gp_rows)), 1):.1%} of the smaller set)")
    print(f"  validation/training overlap checks: CNP=0, GP=0 (asserted)")
    print("  overlap policy: only HF_train <-> LF2_train may overlap among training sets; "
          "CNP_val <-> GP_val overlap is allowed")
    print(f"total primaries: {tot:,}   macros (simulations): {len(allrows)}   output: {a.outdir}/")


if __name__ == "__main__":
    main()
