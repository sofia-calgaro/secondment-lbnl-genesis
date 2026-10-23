# Voxels sampling and optical photon simulations

Scripts to build a global voxel grid, remove voxels outside the ULAr volume or fully inside dead mass (HPGe diodes, Cu support rods), sample LF/HF training and validation voxels uniformly in (x, y, z), and generate macros / job lists for the simulations.

Each simulated voxel returns a `.lh5` file (it included momenta in output after modifying [remage](https://remage.readthedocs.io/en/stable/)):

```
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


## Requirements

`numpy`, `scipy`, `pandas`, `matplotlib`, `pyg4ometry` (GDML profiles), `geant4_pybind` (dead-volume check). A GDML file (eg `l1000.gdml` generated via [legend-pygeom-l1000](https://github.com/leg[...]) is needed for the geometry-aware checks.

## Workflow

1. **Step 1:** `make_voxel_space.py` defines the voxel space.
2. **Step 2:** `sample_voxels.py` samples the LF/HF training and validation sets.
3. Additional scripts for **plots/checks** are available. 

### Step 1: `make_voxel_space.py`

Builds all voxels of the grid inside a box, removes voxels not fully inside ULAr (analytic, from the GDML polycone) and voxels whose 8 corners are all in HPGe detectors and Cu supoort rods (Geant4-based dead-volume logic), and writes the shared voxel-space file.

```
python make_voxel_space.py --vox 10 
--xrange -957 957 --yrange -957 957 --zrange -1350 1500
--gdml l1000.gdml --dead-check --radius 957
--dead-bbox -850 850 -850 850 -1000 800
--out voxel_20cmBox_10mm_21102026.csv.gz
```


- `--dead-bbox` limits the (slow) check of dead volumes to this box (must contain all strings and Cu rods).
- Outputs (same prefix as `--out`): the voxel space itself, `_qc.pdf`, `_dead_voxels.csv.gz`, `_dead_voxels_by_volume.csv.gz` and `_removal_summary.json` (removed voxels per reason/volume).

### Step 2: `sample_voxels.py`

Samples 5 different datasets:
* LF_train
* LF2_train
* HF_train (a selectable fraction of these voxels can overlap with LF2_train ones)
* CNP_val (or LF_val; these voxels can randomly overlap with GP_val ones)
* GP_val (or HF_val; these voxels can randomly overlap with CNP_val ones) 

It automatically writes macros and job lists for all of them.

```
python sample_voxels.py voxel_20cmBox_10mm_21102026.csv.gz
--outdir voxel_sets_20cmBox_10mm_test
--n-lf 20 --n-lf2 20 --n-hf 4 --n-cnp-val 8 --n-gp-val 8
--prim-lf 500 --prim-hf 1000 --seed 12345 --sim-seed 12345
```


- `--outdir` must be a new name, **not** the space file name.
- Rerunning with the same settings and larger counts extends a campaign; counts cannot go down.
- Outputs in `--outdir`: `voxels_*.csv`, `manifest.json`, `config.json`, `macros/`, `jobs/`, `events/lf2/`, `qc.png`, `qc_3d.png`.
- `--scratch-dir` puts `results/`, `maps/`, `logs/` on scratch.

#### Seed policy

Short version:

- `--seed` changes which voxels are selected.
- `--sim-seed` changes the random numbers used inside each voxel simulation and in the LF2 source-event generation.

`--seed` seeds the random permutations that select the LF/LF2/HF/CNP/GP voxels and decides which LF2 voxels are shared with HF. It does not change the parity-based group assignment; that is fixed by the voxel indices, so the location groups remain disjoint for any seed.

`--sim-seed` fixes the per-voxel simulation seed as `hash(sim-seed, tier, voxel_size, ix, iy, iz)`. The same voxel always gets the same seed, independent of counts or ordering. This is what makes simulations reproducible voxel by voxel.

For one campaign, keep both fixed. A new `--seed` gives a new voxel sample; a new `--sim-seed` gives the same voxels but different random photon histories.

#### Extension and rewrite behavior

Extending a campaign adds more voxels, not more primaries per voxel. The script checks the fixed sampling settings (`--seed`, `--sim-seed`, counts, box, voxel size, space SHA, and `--prim-*` values) and refuses to extend if any of these change.

What gets rewritten and what does not:

- Macros are rewritten in place for all voxels in the set. Old macros that are unchanged come out identical because position, seed, and primaries are the same.
- Simulation outputs are not rewritten. Every job line starts with `test -s <output> || ...`, so completed `.stp.lh5` files are skipped.
- LF2 source events in `events/lf2/*.npz` are regenerated from deterministic seeds, so they remain identical.
- Job lists and chunk files are rewritten, and new chunk boundaries can appear as counts grow; this is harmless because finished jobs are skipped.

HF is the important exception: HF selection depends on shared LF2 voxels and on the order of the HF list. If you change `--n-lf2` or `--n-hf`, the shared HF voxels and the run ordering can change, so some HF voxels may be simulated again under new names. This is not a physics change, but it does mean some older HF outputs become orphaned.

For a new independent sample, use a new `--outdir` and a new `--seed`; keep `--sim-seed` fixed if you want the same voxel-level seeds for the matched voxel positions. For a new statistical realization of the same voxels, reuse the same `--seed` and use a new `--sim-seed`.

> [!WARNING]
> So far, no implementation for adding new sampled voxels to the ones previously generated, eg if we notice the first time we sampled the voxels and run the simulations we did not have enough statistics. The script only supports extending a campaign by increasing counts within the same fixed campaign.

> [!WARNING]
> So far, no implementation for enlarging the sampled box by extending the previous one and adding new sampled voxels. If done, it has to preserve the uniformity of sampled voxels on top of those already selected.

### Helper: `voxel_grid.py`

Shared grid definition (origin (0,0,0), detector frame, mm) and the `VoxelSpace` file format. Imported by the scripts above. Do not copy or edit it per user

To draw the grid (10mm edge voxel), use

```
python voxel_grid.py 10 --plot grid10.pdf 
```

## Plotting and checks

### Voxel space (after step 1)

```
python check_dead_voxels.py voxel_20cmBox_10mm_21102026.csv.gz --gdml l1000.gdml --vox 10 --dead voxel_20cmBox_10mm_21102026_dead_voxels.csv.gz --zcut 1500
```

- `check_dead_voxels.py`: plots voxel centres and dead voxels against the ULAr boundary (r-z, x-z, 3D), counts strings and detector levels (expect 42 and 8). Use the same `--vox` as in step 1.

### Sampled sets (after step 2)

```
python plot_voxel_samples.py voxel_sets_20cmBox_10mm_test
```


- `plot_voxel_samples.py`: plots all five sets (2D/3D, occupancy, coverage), writes `overlap_audit.txt` and, if found, `removal_summary.txt` with why voxels were removed. Output in `<sample_dir>/...`.

### Geometry (GDML only)

```
python uar_and_ge.py
python voxel_borehole.py
```


- `uar_and_ge.py`: 2D/3D scene of the ULAr re-entrant tube (cut at z=1500 mm), the HPGe detectors (336 in 42 strings) and Cu support rods. Reads `l1000.gdml` from the working directory.
- `voxel_borehole.py`: sketch of a V00101Z-like detector with a voxel that survives the 8-corner rule but partly overlaps Ge next to the borehole.

