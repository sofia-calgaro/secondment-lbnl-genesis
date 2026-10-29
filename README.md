# Voxels sampling and optical photon simulations

This repository contains tools to:
- build a global voxel grid,
- remove voxels outside the ULAr volume or fully inside dead mass (HPGe diodes / Cu support rods),
- sample LF/HF training and validation voxels uniformly in `(x, y, z)`,
- generate remage macros and job lists,
- inspect simulation outputs and optical-step statistics.

Each simulated voxel produces a `.lh5` output file. After the remage modification ([commit](https://github.com/Rashika-Gupta/remage-celeritas-int/commit/afa35e8dfcf44f575e7adc916e02b79499909994)), the files include photon momentum information and the optical step records needed for the analysis.

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

## Typical end-to-end use

- Build the grid/admissible voxel space via `voxel_and_sims/make_voxel_space.py`
- Check the volume to sample via `plotting/ular_and_ge.py`
- Generate the LF/HF/CNP/GP samples via `voxel_and_sims/sample_voxels.py`
- Check the sampled sets via `plotting/plot_voxel_samples.py`
- Run simulations via `submit_nersc.sh`
- Inspect optical-step statistics via `inspect_sims/plot_optical_steps.py`

This repository uses [Pixi](https://pixi.sh/) to define a reproducible local environment for the voxel-generation and analysis workflow.
This keeps the project dependencies isolated from the system Python and makes the workflow easier to reproduce across machines.The root `pixi.toml` sets up a workspace named `optmap_demo` and includes the main dependencies needed throughout the project. A GDML file is required for ULAr and dead-volume checks, for example `l1000.gdml` generated via `legend-pygeom-l1000`.

For large-scale production runs on NERSC Perlmutter, the repository also includes helper scripts to combine the Pixi environment with the locally built [momentum-enabled remage](https://github.com/Rashika-Gupta/remage-celeritas-int/commit/afa35e8dfcf44f575e7adc916e02b79499909994) installation:

- `setup_remage_momentum.sh` updates `PATH` and `LD_LIBRARY_PATH` so that the local remage binaries are used instead of the default system one, and exports `REMAGE_CPP_PATH` for the Python wrapper
- `submit_nersc.sh` creates a Slurm batch script that activates the standard Pixi environment, sources the local remage setup, and then distributes commands from a job list across a Slurm array.

A typical submission looks like:

```bash
./submit_nersc.sh \
  -j voxel_sets_x-800-600_y+50+250_z+400+600_5mm/jobs/lf/remage.txt \
  -N 4 -c 1 -p 32 -t 02:00:00 \
  -n box5mm_lf_remage
```

where `{...}/remage.txt` is a file containing one simulation command per line. The batch system distributes tasks across the requested array size. The script also supports `-D` to print the generated Slurm script without submitting.


## Workflow

1. **Step 1:** build the voxel space with `voxel_and_sims/make_voxel_space.py`
2. **Step 2:** sample the datasets in the defined grid volume generated above with `voxel_and_sims/sample_voxels.py`
3. Run the plotting/inspection utilities in `plotting/` and `inspect_sims/`

### Step 1: `voxel_and_sims/make_voxel_space.py`

This script builds the full voxel grid inside a selected box, removes voxels:
- outside the ULAr volume,
- partially or fully overlapping the dead mass,
- outside the allowed geometry as determined from the GDML profile and/or Geant4 checks.

Example:

```bash
pixi run python secondment-lbnl-genesis/voxel_and_sims/make_voxel_space.py \
  --vox 5 --xrange -957 957 --yrange -957 957 --zrange -1350 1500 \
  --gdml /dvs_ro/cfs/cdirs/m2676/users/calgaro/uzh/secondment/optmap_demo/l1000.gdm --dead-check --dead-bbox -850 850 -850 850 -1000 1500 \
  --out dead_mass_z1500mm_5mm.csv.gz --radius 957 --max-voxels 300000000
```

Important notes:
- `--dead-bbox` limits the expensive dead-volume check to a smaller box.
- Output files are written with the same prefix as `--out`:
  - `<out>` - voxel-space file
  - `<out stem>_qc.pdf`
  - `<out stem>_dead_voxels.csv.gz`
  - `<out stem>_partial_voxels.csv.gz`
  - additional removal metadata / summary files

For dead-mass rejection, the code uses the 8-corner rule: keep the voxel if at least one corner remains inside the ULAr volume; remove the voxel if all 8 corners fall in dead material / support structure (being more conservative by reducing the voxel edge by 0.05 mm per side)

<img src="https://github.com/user-attachments/assets/a39055ad-4e02-457b-8775-b0f123a0e7c0" alt="Voxel selection rule" width="400" style="max-width:100%; height:auto;">


### Step 2: `voxel_and_sims/sample_voxels.py`

This script samples five datasets from the saved voxel-space file:
- `LF_train`
- `LF2_train`
- `HF_train`
- `CNP_val`
- `GP_val`

Example:

```bash
pixi run python sample_voxels.py dead_mass_z1500mm_5mm.csv.gz \
  --outdir voxel_sets_x-800-600_y+50+250_z+400+600_5mm \
  --xrange -800 -600 --yrange 50 250 --zrange 400 600 \
  --n-lf 6400 --n-lf2 6400 --n-hf 640 --n-cnp-val 500 --n-gp-val 500 \
  --prim-lf 1500 --prim-hf 5000 --seed 12345 --sim-seed 12345 \
  --gdml /dvs_ro/cfs/cdirs/m2676/users/calgaro/uzh/secondment/optmap_demo/l1000.gdml --hf-overlap-lf2 0.5 \
  --scratch-dir /pscratch/sd/s/sofiacal/optmap/ \ 
  --map-settings map_setting_20cmBox_x-800-600_y50+250_z+400+600_5mm.yaml
```

where `map_setting_20cmBox_x-800-600_y50+250_z+400+600_5mm.yaml` is

```yaml
range_in_m:
  - [-0.8, -0.6] # see above
  - [0.05, 0.25] # see above
  - [0.4, 0.6] # see above
bins: [40, 40, 40] # 5mm edge
```

The script:
- writes one macro per simulated voxel,
- creates job lists / chunk files,
- writes a manifest describing each set,
- saves `events/lf2/*.npz` for LF2 source-event generation,
- generates plots with voxel distributed for each dataset in 2 (`qc.png`, `qc_3d.png`).

Notes:
- `--outdir` must be a new directory name, not the space-file name.
- Re-running with the same settings and larger counts extends a campaign; counts cannot go down.
- The sampling box and full space definition are fixed by the voxel-space file and checked against saved config metadata.
- `--seed` changes which voxels are selected.
- `--sim-seed` changes the per-voxel random simulation history while keeping the same voxel positions.

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

## Extension behavior

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
  --gdml /dvs_ro/cfs/cdirs/m2676/users/calgaro/uzh/secondment/optmap_demo/l1000.gdml --vox 10 \
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
python plotting/plot_voxel_samples.py voxel_sets_x-800-600_y+50+250_z+400+600_5mm
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
- the ULAr re-entrant tube, HPGe detector strings and support rods; add `--box -800 -600 50 250 400 600` if you want to visualize a volume with coordinates -800 < x < -600 mm, 50 < y < 250 mm, 400 < z < 600 mm.
- a voxel near a borehole that passes the coarse 8-corner rule but still overlaps dead material.

## Post-simulation inspection

### Optical-step summary & diagnostic plots

```bash
python inspect_sims/plot_optical_steps.py voxel_sets_x-800-600_y+50+250_z+400+600_5mm
```

This script inspects the simulation output files and summarizes how many optical hits each voxel gets. It reads the LH5 files, matches them to the voxel manifests, counts distinct SiPM channels, repeated hits, and primaries producing optical signals, then produces plots and summary CSVs across all datasets.

The output includes optical-step histograms, coverage maps, repeated-hit statistics, and primary-hit-vs-position diagnostics, which helps compare dataset performance and identify voxels with unusually low or high light yield.


### Simulation timing summary

```bash
python inspect_sims/timing.py voxel_sets_x-800-600_y+50+250_z+400+600_5mm
```

This script checks the timing logs from the voxel simulations and matches them back to the voxel manifests by run ID or coordinate tag. 
It reads the remage log files, extracts per-voxel timing metrics such as average event processing time, throughput, and vertex-sampling time.
It is useful for validating that the expected number of jobs ran, identifying missing/incomplete voxels, and comparing timing distributions across `LF_train`, `LF2_train`, `HF_train`, `CNP_val`, `GP_val`.

### Photon absorption sanity check

```bash
python plotting/photon_absorption.py
```

This contains a simple attenuation-length model and produces the survival fraction versus z together with the z positions where the survival drops to 10% and 1%.

### Check of existing simulations vs jobs

```bash
python inspect_sims/check_number_of_sims.py voxel_sets_x-800-600_y+50+250_z+400+600_5mm
```

Summary of existing and missing simulations after a comparison with the list of jobs.

🤖 AI tools (Claude Code and GitHub Copilot) were used to assist in generating portions of the scripts. All AI-assisted code was carefully reviewed and validated before being incorporated into the repository.
