# Voxels sampling and optical photon simulations

This repository contains tools to:
- build a global voxel grid,
- remove voxels outside the ULAr volume or fully inside dead mass (HPGe diodes / Cu support rods),
- sample LF/HF training and validation voxels uniformly in `(x, y, z)`,
- generate remage macros and job lists,
- inspect simulation outputs and optical-step statistics.

Each simulated voxel produces a `.lh5` output file. After the remage modification used here, the files include photon momentum information and the optical step records needed for downstream analysis.

Example structure of a generated `.lh5` file:

```bash
sh-5.1$ ~/.local/bin/lh5ls --depth 2 /tmp/test_direction.stp.lh5
├── number_of_simulated_events · real
├── particles · table{ekin,evtid,particle,px,py,pz,vertexid}
│ ├── ekin · array<1>{real}
│ ├── evtid · array<1>{real}
│ ├── particle · array<1>{real}
│ ├── px · array<1>{real} ### momentum components (px^2+py^2+pz^2 = E^2)
│ ├── py · array<1>{real}
│ ├── pz · array<1>{real}
│ └── vertexid · array<1>{real}
├── stp · struct{optical}
│ ├── by_uid · struct{det2001010}
│ └── optical · table{det_uid,evtid,time,wavelength}
└── vtx · table{evtid,n_part,time,xdir,xloc,ydir,yloc,zdir,zloc}
├── evtid · array<1>{real}
├── n_part · array<1>{real}
├── time · array<1>{real}
├── xdir · array<1>{real} ### unit direction vectors (norm ~ 1)
├── ydir · array<1>{real}
├── zdir · array<1>{real}
├── xloc · array<1>{real} ### vertex position
├── yloc · array<1>{real}
└── zloc · array<1>{real}
```

## Repository layout

- `voxel_and_sims/`
  - `make_voxel_space.py` — defines the admissible voxel space
  - `sample_voxels.py` — samples LF/HF train and validation voxels
  - `voxel_grid.py` — shared grid definition + `VoxelSpace` format
- `plotting/`
  - `check_dead_voxels.py` — dead-voxel / ULAr validation plots
  - `plot_voxel_samples.py` — sample-set diagnostics and overlap audit
  - `uar_and_ge.py` — GDML-based ULAr / detector visualization
  - `voxel_borehole.py` — sketch of a detector-borehole overlap case
  - `photon_absorption.py` — simple attenuation model helper
- `inspect_sims/`
  - `plot_optical_steps.py` — optical-step analysis across datasets
- root files
  - `README.md`

## Requirements

Python packages used by the scripts:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `pyg4ometry` (GDML profile parsing)
- `geant4_pybind` (dead-volume checking)

A GDML file is required for ULAr and dead-volume checks, for example `l1000.gdml` generated via `legend-pygeom-l1000`.

## Workflow

1. **Step 1:** build the voxel space with `voxel_and_sims/make_voxel_space.py`
2. **Step 2:** sample the datasets with `voxel_and_sims/sample_voxels.py`
3. **Validation / inspection:** run the plotting and inspection utilities in `plotting/` and `inspect_sims/`

### Step 1: `voxel_and_sims/make_voxel_space.py`

This script builds the full voxel grid inside a selected box, removes voxels:
- outside the ULAr volume,
- partially or fully overlapping the dead mass,
- outside the allowed geometry as determined from the GDML profile and/or Geant4 checks.

Example:

```bash
python voxel_and_sims/make_voxel_space.py --vox 10 \
  --xrange -957 957 --yrange -957 957 --zrange -1350 1500 \
  --gdml l1000.gdml --dead-check --radius 957 \
  --dead-bbox -850 850 -850 850 -1000 800 \
  --out voxel_20cmBox_10mm_21102026.csv.gz
```

Important notes:
- `--dead-bbox` limits the expensive dead-volume check to a smaller box.
- Output files are written with the same prefix as `--out`:
  - `<out>` — voxel-space file
  - `<out stem>_qc.pdf`
  - `<out stem>_dead_voxels.csv.gz`
  - `<out stem>_partial_voxels.csv.gz`
  - additional removal metadata / summary files

### Step 2: `voxel_and_sims/sample_voxels.py`

This script samples five datasets from the saved voxel-space file:
- `LF_train`
- `LF2_train`
- `HF_train`
- `CNP_val`
- `GP_val`

Example:

```bash
python voxel_and_sims/sample_voxels.py voxel_20cmBox_10mm_21102026.csv.gz \
  --outdir voxel_sets_20cmBox_10mm_test \
  --n-lf 20 --n-lf2 20 --n-hf 4 --n-cnp-val 8 --n-gp-val 8 \
  --prim-lf 500 --prim-hf 1000 --seed 12345 --sim-seed 12345
```

The script:
- writes one macro per simulated voxel,
- creates job lists / chunk files,
- writes a manifest describing each set,
- saves `events/lf2/*.npz` for LF2 source-event generation,
- writes QC figures (`qc.png`, `qc_3d.png`).

Notes:
- `--outdir` must be a new directory name, not the space-file name.
- Re-running with the same settings and larger counts extends a campaign; counts cannot go down.
- The sampling box and full space definition are fixed by the voxel-space file and checked against saved config metadata.
- `--seed` changes which voxels are selected.
- `--sim-seed` changes the per-voxel random simulation history while keeping the same voxel positions.

### Shared helper: `voxel_and_sims/voxel_grid.py`

The grid definition is centralized in `voxel_and_sims/voxel_grid.py`. This file defines:
- detector-frame coordinate conventions,
- voxel indexing and center/corner calculations,
- the `VoxelSpace` file format used by the sampling pipeline.

Example:

```bash
python voxel_and_sims/voxel_grid.py 10 --plot grid10.pdf
```

This script is not meant to be copied or edited per user; it is the shared geometry layer used by the rest of the project.

## Seed policy

- `--seed` controls which voxels are selected.
- `--sim-seed` controls the random numbers used inside each voxel simulation and during LF2 source-event generation.
- A fixed `--sim-seed` keeps the same per-voxel simulation seed for matching voxel positions, even when the campaign changes.
- For a new independent sample, use a new `--outdir` and a new `--seed`.

> [!NOTE]
> For one campaign, keep both fixed. A new `--seed` gives a new voxel sample. A new `--sim-seed` gives the same voxels but different random photon histories.

## Extension and rewrite behavior

Extending a campaign adds more voxels without changing the original fixed voxel choices.

If the same `--outdir`, `--seed`, and `--sim-seed` are reused with larger counts:
- macros are rewritten in place when needed,
- completed simulation outputs are skipped using `test -s <output> || ...`,
- LF2 event files are regenerated deterministically,
- job lists / chunk files may be rewritten with new boundaries.

Important caveat:
- `HF_train` depends on the exact LF2 overlap and HF ordering; changing `--n-lf2` or `--n-hf` can change the shared HF voxels and file ordering.

> [!WARNING]
> You cannot increase statistics per voxel within a single campaign. `--prim-lf` and `--prim-hf` are fixed in `config.json` once the campaign is created.

## Plotting and checks

### Voxel-space validation

```bash
python plotting/check_dead_voxels.py voxel_20cmBox_10mm_21102026.csv.gz \
  --gdml l1000.gdml --vox 10 \
  --dead voxel_20cmBox_10mm_21102026_dead_voxels.csv.gz \
  --zcut 1500
```

This script:
- plots voxel centres and dead voxels against the ULAr boundary,
- makes `(r, z)` and `(x, z)` views,
- writes a 3D view,
- counts string / level structure in the dead-voxel map.

### Sample-set validation

```bash
python plotting/plot_voxel_samples.py voxel_sets_20cmBox_10mm_test
```

This script:
- plots all five sample sets in 2D and 3D,
- shows occupancy maps and coordinate coverage,
- writes `overlap_audit.txt`,
- checks for forbidden overlaps between incompatible sets.

### Geometry / GDML-only sketches

```bash
python plotting/uar_and_ge.py
python plotting/voxel_borehole.py
```

These scripts visualize:
- the ULAr re-entrant tube,
- HPGe detector strings and support rods,
- a voxel near a borehole that passes the coarse 8-corner rule but still overlaps dead material.

## Post-simulation inspection

### Optical-step summary

```bash
python inspect_sims/plot_optical_steps.py voxel_sets_20cmBox_10mm_test
```

This script:
- reads the voxel manifests and available simulation outputs,
- matches files to their coordinate tags,
- counts optical-step entries per dataset,
- plots a combined histogram for all datasets.

This is useful for comparing the optical-step distributions across:
- `LF_train`
- `LF2_train`
- `HF_train`
- `CNP_val`
- `GP_val`

### Photon absorption sanity check

```bash
python plotting/photon_absorption.py
```

This helper contains a simple attenuation-length model and produces:
- the survival fraction versus z,
- the z positions where the survival drops to 10% and 1%.

## Notes

- The repo currently organizes scripts by stage (`voxel_and_sims/`, `plotting/`, `inspect_sims/`) rather than keeping everything at the project root.
- Older usage examples should include the relevant subdirectory prefix when invoking scripts.
- The generated campaign metadata (`config.json`, `manifest.json`) is important for reproducibility and safe extension.

## Typical end-to-end use

```bash
# 1) build the grid / admissible voxel space
python voxel_and_sims/make_voxel_space.py \
  --vox 10 --xrange -957 957 --yrange -957 957 --zrange -1350 1500 \
  --gdml l1000.gdml --dead-check --radius 957 \
  --out voxel_20cmBox_10mm_21102026.csv.gz

# 2) generate the LF/HF/CNP/GP samples
python voxel_and_sims/sample_voxels.py voxel_20cmBox_10mm_21102026.csv.gz \
  --outdir voxel_sets_20cmBox_10mm_test \
  --n-lf 20 --n-lf2 20 --n-hf 4 --n-cnp-val 8 --n-gp-val 8 \
  --prim-lf 500 --prim-hf 1000 --seed 12345 --sim-seed 12345

# 3) check the sampled sets
python plotting/plot_voxel_samples.py voxel_sets_20cmBox_10mm_test

# 4) inspect optical-step statistics
python inspect_sims/plot_optical_steps.py voxel_sets_20cmBox_10mm_test
```
