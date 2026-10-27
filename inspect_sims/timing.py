#!/usr/bin/env python3
"""
Extract ReMaGe timing statistics from logs and plot distributions per dataset.

Matching logs to manifest voxels
--------------------------------
Log names look like  run00001_x-0.742_y+0.132_z+0.422.remage.log.
For every dataset the script builds two kinds of keys from the manifest,
using the manifest's own columns (coord_tag, tag, stp_file) first and the
x/y/z_mm columns only as a last resort:

    run key   : run00001_x-0.742_y+0.132_z+0.422   (exact, one per job)
    coord key : x-0.742_y+0.132_z+0.422            (shared voxels collide)

It then picks, per dataset, whichever key matches more of the logs and
prints the coverage of both, so a bad matching is visible immediately.

For each dataset an "N funnel" is printed:
    manifest voxels -> logs found -> unique voxels -> completed -> metric parsed
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = ["LF_train", "LF2_train", "HF_train", "CNP_val", "GP_val"]

# Fidelity (log directory) each dataset uses.
DATASET_FIDELITY = {
    "LF_train": "lf",
    "LF2_train": "lf",
    "HF_train": "hf",
    "CNP_val": "lf",
    "GP_val": "hf",
}

# Expected number of voxels per dataset (sanity check only).
EXPECTED_N = {
    "LF_train": 6400,
    "LF2_train": 6400,
    "HF_train": 640,
    "CNP_val": 500,
    "GP_val": 500,
}

# (column, axis label). Times are converted to ms in parse_log().
METRICS = [
    ("event_time_ms", "Average event processing time (ms/event)"),
    ("events_per_second", "Processing throughput (events/s)"),
    ("vertex_time_ms", "Average vertex sampling time (ms/event)"),
]

_C = r"x[+-]\d+\.\d+_y[+-]\d+\.\d+_z[+-]\d+\.\d+"
COORD_RE = re.compile(rf"({_C})")
RUN_RE = re.compile(rf"(run\d+_{_C})")
COORD_VALUES_RE = re.compile(
    r"x([+-]\d+\.\d+)_y([+-]\d+\.\d+)_z([+-]\d+\.\d+)"
)

PATTERNS = {
    "event_time_s": re.compile(
        r"average event processing time was\s*"
        r"([0-9.eE+-]+)\s*seconds/event"
    ),
    "events_per_second": re.compile(
        r"=\s*([0-9.eE+-]+)\s*events/second"
    ),
    "vertex_time_us": re.compile(
        r"average time to sample a vertex was\s*"
        r"([0-9.eE+-]+)\s*us/event"
    ),
    "events_requested": re.compile(
        r"Number of events to be processed:\s*(\d+)"
    ),
    "events_completed": re.compile(
        r"Run nr\.\s*\d+\s*completed\.\s*(\d+)\s*events simulated"
    ),
}


# ============================================================
# KEYS
# ============================================================

def extract_tag(value):
    """x/y/z coordinate tag contained in a string, or None."""
    match = COORD_RE.search(str(value))
    return match.group(1) if match else None


def extract_runkey(value):
    """runNNNNN_x.._y.._z.. key contained in a string, or None."""
    match = RUN_RE.search(str(value))
    return match.group(1) if match else None


def coords_mm_from_tag(tag):
    """(x, y, z) in mm from a coordinate tag (values in the tag are metres)."""
    match = COORD_VALUES_RE.search(str(tag))
    if not match:
        return (np.nan, np.nan, np.nan)
    return tuple(float(v) * 1000.0 for v in match.groups())


def coordinate_tag_from_manifest(row):
    """Last-resort tag built from x_mm/y_mm/z_mm (rounding may differ!)."""
    return "x{:+.3f}_y{:+.3f}_z{:+.3f}".format(
        float(row["x_mm"]) / 1000,
        float(row["y_mm"]) / 1000,
        float(row["z_mm"]) / 1000,
    )


# ============================================================
# MANIFESTS
# ============================================================

def load_manifests(root):
    """
    Returns dict[dataset] -> {
        n_rows, run_keys, coord_keys, coords (N x 3, mm), mode, keys
    }. mode/keys are finalised later by choose_key_modes().
    """
    manifest = {}

    for dataset in DATASETS:
        entry = {
            "n_rows": 0, "run_keys": set(), "coord_keys": set(),
            "coords": None, "coord_list": [], "mode": "coord",
            "keys": set(),
        }
        manifest[dataset] = entry

        path = root / f"voxels_{dataset}.csv"
        if not path.exists():
            print(f"WARNING: manifest not found: {path}")
            continue

        df = pd.read_csv(path)
        entry["n_rows"] = len(df)

        coord = pd.Series(np.nan, index=df.index, dtype=object)
        run = pd.Series(np.nan, index=df.index, dtype=object)
        sources = {}

        # Use the manifest's own name columns first.
        for col in ("coord_tag", "tag", "stp_file"):
            if col not in df.columns:
                continue
            c = df[col].map(extract_tag)
            r = df[col].map(extract_runkey)
            new_c = coord.isna() & c.notna()
            coord[new_c] = c[new_c]
            run[run.isna() & r.notna()] = r[run.isna() & r.notna()]
            sources[col] = int(new_c.sum())

        have_mm = {"x_mm", "y_mm", "z_mm"}.issubset(df.columns)
        missing = coord.isna()
        if missing.any() and have_mm:
            coord[missing] = df[missing].apply(
                coordinate_tag_from_manifest, axis=1
            )
            sources["x_mm/y_mm/z_mm (computed)"] = int(missing.sum())

        if coord.isna().any():
            raise ValueError(
                f"{path}: {int(coord.isna().sum())} rows have no usable "
                f"tag (need coord_tag, tag, stp_file or x_mm,y_mm,z_mm)."
            )

        print(f"{dataset}: {len(df)} manifest rows; tags from "
              + ", ".join(f"{k} ({v})" for k, v in sources.items() if v))
        if "x_mm/y_mm/z_mm (computed)" in sources:
            print("  NOTE: some tags were computed from x_mm/y_mm/z_mm; their "
                  "rounding can differ from the filenames.")
        if coord.duplicated().any():
            print(f"  NOTE: {int(coord.duplicated().sum())} rows share a "
                  f"coordinate with another row of this dataset")

        entry["coord_keys"] = set(coord)
        entry["coord_list"] = list(coord)
        if run.notna().all():
            entry["run_keys"] = set(run)

        if have_mm:
            entry["coords"] = df[["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        else:
            entry["coords"] = np.array(
                [coords_mm_from_tag(t) for t in coord], dtype=float
            )

    return manifest


# ============================================================
# LOG DISCOVERY AND KEY SELECTION
# ============================================================

def list_logs(root):
    paths = {}

    for fidelity in ("lf", "hf"):
        log_dir = root / "logs" / fidelity

        if not log_dir.exists():
            print(f"WARNING: log directory not found: {log_dir}")
            paths[fidelity] = []
            continue

        paths[fidelity] = sorted(log_dir.glob("*.remage.log"))
        print(f"Found {len(paths[fidelity])} logs in {log_dir}")

    return paths


def choose_key_modes(manifest, paths_by_fidelity):
    """Per dataset, pick the key type (run / coord) that matches more logs."""
    print("\nKEY MATCHING (fraction of manifest voxels found among the logs)")

    for dataset in DATASETS:
        m = manifest[dataset]
        names = paths_by_fidelity.get(DATASET_FIDELITY[dataset], [])
        log_runs = {extract_runkey(p.name) for p in names} - {None}
        log_coords = {extract_tag(p.name) for p in names} - {None}

        cov_run = (len(m["run_keys"] & log_runs) / len(m["run_keys"])
                   if m["run_keys"] else 0.0)
        cov_coord = (len(m["coord_keys"] & log_coords) / len(m["coord_keys"])
                     if m["coord_keys"] else 0.0)

        m["mode"] = ("run" if (m["run_keys"] and cov_run >= cov_coord)
                     else "coord")
        m["keys"] = m["run_keys"] if m["mode"] == "run" else m["coord_keys"]

        run_txt = f"{100 * cov_run:5.1f}%" if m["run_keys"] else "  n/a "
        print(f"  {dataset:10s} run-ID={run_txt}  coordinate={100 * cov_coord:5.1f}%"
              f"  -> matching by {m['mode'].upper()}")

        if max(cov_run, cov_coord) < 0.9 and m["n_rows"]:
            print(f"    WARNING: only {100 * max(cov_run, cov_coord):.1f}% of "
                  f"{dataset} voxels match any log name.")


def log_key(manifest, dataset, run_key, coord):
    return run_key if manifest[dataset]["mode"] == "run" else coord


# ============================================================
# LOG PARSING
# ============================================================

def read_metric(pattern, text):
    match = pattern.search(text)
    return float(match.group(1)) if match else np.nan


def parse_log(path):
    """Extract timing metrics and completion information from one log."""
    text = path.read_text(errors="replace")

    record = {
        "log_file": str(path),
        "mtime": path.stat().st_mtime,
        "run_key": extract_runkey(path.name),
        "coordinate": extract_tag(path.name),
        # seconds/event -> ms/event
        "event_time_ms": read_metric(PATTERNS["event_time_s"], text) * 1000,
        "events_per_second": read_metric(PATTERNS["events_per_second"], text),
        # us/event -> ms/event
        "vertex_time_ms": read_metric(PATTERNS["vertex_time_us"], text) / 1000,
        "events_requested": read_metric(PATTERNS["events_requested"], text),
        "events_completed": read_metric(PATTERNS["events_completed"], text),
    }

    record["completed"] = bool(
        np.isfinite(record["events_completed"])
        and (
            not np.isfinite(record["events_requested"])
            or record["events_completed"] == record["events_requested"]
        )
    )

    return record


def collect_logs(paths_by_fidelity, manifest, tol_mm=0.0):
    """One row per (log, matching dataset); unmatched logs get dataset=None."""
    records = []

    for fidelity, paths in paths_by_fidelity.items():
        allowed = [d for d in DATASETS if DATASET_FIDELITY[d] == fidelity]
        others = [d for d in DATASETS if d not in allowed]
        print(f"\nParsing {len(paths)} {fidelity.upper()} logs ...")

        for i, path in enumerate(paths, start=1):
            if i % 2000 == 0:
                print(f"  ... {i}/{len(paths)}")

            record = parse_log(path)
            record["fidelity"] = fidelity
            run_key, coord = record["run_key"], record["coordinate"]

            def keyed(datasets):
                out = []
                for d in datasets:
                    k = log_key(manifest, d, run_key, coord)
                    if k is not None and k in manifest[d]["keys"]:
                        out.append((d, k))
                return out

            matches = keyed(allowed)

            # Optional fallback: nearest manifest voxel within tol_mm.
            fuzzy = False
            if tol_mm > 0 and coord is not None:
                p = np.array(coords_mm_from_tag(coord))
                have = {d for d, _ in matches}
                for d in allowed:
                    m = manifest[d]
                    if d in have or m["mode"] != "coord" or m["coords"] is None:
                        continue
                    dist = np.linalg.norm(m["coords"] - p, axis=1)
                    j = int(dist.argmin())
                    if dist[j] <= tol_mm:
                        matches.append((d, m["coord_list"][j]))
                        fuzzy = True

            if run_key is None and coord is None:
                status = "no_coordinate_in_filename"
            elif matches:
                status = "matched"
            elif keyed(others):
                status = "wrong_fidelity_for_dataset"
            else:
                status = "no_manifest_match"

            record["mapping_status"] = status
            record["datasets"] = ";".join(d for d, _ in matches)
            record["shared_coordinate"] = len(matches) > 1
            record["fuzzy_match"] = fuzzy

            if matches:
                for dataset, key in matches:
                    copy = record.copy()
                    copy["dataset"] = dataset
                    copy["key"] = key
                    records.append(copy)
            else:
                record["dataset"] = None
                record["key"] = None
                records.append(record)

    df = pd.DataFrame(records)

    for col in ("dataset", "key"):
        if col not in df.columns:
            df[col] = None

    return df


def mark_used(df, keep_duplicates):
    """
    Add a boolean 'used' column.

    Used = matched to a dataset and, unless keep_duplicates, the newest
    completed log for that (dataset, voxel key).
    """
    matched = df["dataset"].notna()
    df["used"] = matched

    if keep_duplicates or not matched.any():
        return df

    order = df[matched].sort_values(["completed", "mtime"]).index
    dup = df.loc[order].duplicated(["dataset", "key"], keep="last")
    df.loc[dup[dup].index, "used"] = False

    return df


# ============================================================
# DIAGNOSTIC FOR LOGS THAT MATCH NO MANIFEST VOXEL
# ============================================================

def diagnose_unmatched(df, manifest, max_logs=2000):
    """How far is each unmatched log from the nearest manifest voxel?"""
    un = df[df["dataset"].isna() & df["coordinate"].notna()]
    un = un.drop_duplicates("log_file")

    if un.empty:
        return

    print("\nUNMATCHED LOGS: distance to nearest manifest voxel of the same fidelity")

    for fidelity in ("lf", "hf"):
        sub = un[un["fidelity"] == fidelity]
        pools = [
            manifest[d]["coords"] for d in DATASETS
            if DATASET_FIDELITY[d] == fidelity and manifest[d]["coords"] is not None
        ]
        if sub.empty or not pools:
            continue

        pool = np.vstack(pools)
        pts = np.array([coords_mm_from_tag(t) for t in sub["coordinate"]])[:max_logs]
        dist = np.empty(len(pts))
        off = np.empty((len(pts), 3))

        for start in range(0, len(pts), 100):
            chunk = pts[start:start + 100]
            diff = chunk[:, None, :] - pool[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            j = d.argmin(axis=1)
            dist[start:start + 100] = d[np.arange(len(chunk)), j]
            off[start:start + 100] = diff[np.arange(len(chunk)), j]

        print(f"  {fidelity.upper()}: {len(sub)} unmatched logs "
              f"(checked {len(pts)})")
        print(f"    nearest-voxel distance [mm]: median={np.median(dist):.2f}, "
              f"90%={np.percentile(dist, 90):.2f}, max={dist.max():.2f}")
        print(f"    within 1 mm: {100 * np.mean(dist <= 1.0):.0f}%, "
              f"within 2 mm: {100 * np.mean(dist <= 2.0):.0f}%")
        print(f"    median |offset| per axis [mm]: "
              f"x={np.median(np.abs(off[:, 0])):.2f}, "
              f"y={np.median(np.abs(off[:, 1])):.2f}, "
              f"z={np.median(np.abs(off[:, 2])):.2f}")

        if np.mean(dist <= 2.0) > 0.8:
            print("    -> logs sit within ~1-2 mm of manifest voxels: this is a "
                  "tag ROUNDING mismatch, not missing simulations.\n"
                  "       Fix: keep a tag/stp_file column in the manifests, or "
                  "re-run with --tolerance-mm 1.5")
        else:
            print("    -> logs are far from the manifest voxels: they probably "
                  "belong to a different (older/newer) voxel set.")


# ============================================================
# N FUNNEL
# ============================================================

def build_funnel(df, manifest):
    rows = []

    for dataset in DATASETS:
        m = manifest[dataset]
        sub = df[df["dataset"] == dataset]
        used = sub[sub["used"]]
        done = used[used["completed"]]

        row = {
            "dataset": dataset,
            "fidelity": DATASET_FIDELITY[dataset],
            "matching": m["mode"],
            "expected_voxels": EXPECTED_N.get(dataset),
            "manifest_rows": m["n_rows"],
            "manifest_voxels": len(m["keys"]),
            "logs_found": len(sub),
            "unique_voxels_with_log": sub["key"].nunique(),
            "voxels_without_log": len(m["keys"] - set(sub["key"])),
            "duplicate_logs_dropped": len(sub) - len(used),
            "completed": len(done),
            "incomplete": len(used) - len(done),
        }

        for metric, _ in METRICS:
            row[f"N_{metric}"] = int(done[metric].notna().sum())

        rows.append(row)

    return pd.DataFrame(rows)


def print_funnel(funnel, keep_duplicates):
    print("\nN FUNNEL PER DATASET")

    for _, r in funnel.iterrows():
        print(f"\n{r['dataset']} (uses {r['fidelity'].upper()} logs, "
              f"matched by {r['matching']}):")
        print(f"  expected voxels:             {r['expected_voxels']}")
        print(f"  manifest rows / voxels:      {r['manifest_rows']} / "
              f"{r['manifest_voxels']}")
        print(f"  logs found for this dataset: {r['logs_found']}")
        print(f"  unique voxels with a log:    {r['unique_voxels_with_log']}")
        print(f"  voxels without any log:      {r['voxels_without_log']}")
        print(f"  duplicate logs dropped:      {r['duplicate_logs_dropped']}"
              + ("  (disabled)" if keep_duplicates else ""))
        print(f"  completed:                   {r['completed']}")
        print(f"  incomplete:                  {r['incomplete']}")
        print("  N used in plots: "
              + ", ".join(f"{m}={r[f'N_{m}']}" for m, _ in METRICS))

        if r["expected_voxels"] and r["manifest_voxels"] != r["expected_voxels"]:
            print(f"  WARNING: manifest has {r['manifest_voxels']} voxels, "
                  f"expected {r['expected_voxels']}")
        if r["voxels_without_log"]:
            print(f"  WARNING: {r['voxels_without_log']} manifest voxels "
                  f"have no log")
        if r["duplicate_logs_dropped"] and not keep_duplicates:
            print("  NOTE: several logs per voxel; newest completed kept")
        if r["incomplete"]:
            print(f"  WARNING: {r['incomplete']} voxels have no completed log")


# ============================================================
# SUMMARY AND PLOTS
# ============================================================

def completed_used(df):
    return df[df["used"] & df["completed"]]


def write_summary(df, outdir):
    rows = []
    good = completed_used(df)

    for dataset in DATASETS:
        subset = good[good["dataset"] == dataset]

        for metric, _ in METRICS:
            values = subset[metric].dropna()
            has = len(values) > 0
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "n": len(values),
                "mean": values.mean() if has else np.nan,
                "std": values.std() if len(values) > 1 else np.nan,
                "median": values.median() if has else np.nan,
                "p05": values.quantile(0.05) if has else np.nan,
                "p95": values.quantile(0.95) if has else np.nan,
                "min": values.min() if has else np.nan,
                "max": values.max() if has else np.nan,
            })

    pd.DataFrame(rows).to_csv(outdir / "timing_summary.csv", index=False)


def plot_distributions(df, outdir):
    good = completed_used(df)

    for metric, label in METRICS:
        fig, axes = plt.subplots(
            1, len(DATASETS), figsize=(19, 4.2), squeeze=False
        )
        axes = axes.ravel()

        for ax, dataset in zip(axes, DATASETS):
            values = good.loc[good["dataset"] == dataset, metric].dropna()
            expected = EXPECTED_N.get(dataset)

            if len(values):
                bins = min(30, max(5, int(np.sqrt(len(values)))))
                ax.hist(values, bins=bins, color="dodgerblue")
                ax.axvline(values.median(), linestyle="--", color="black")
                ax.text(
                    0.97, 0.95,
                    f"N = {len(values)} (expected {expected})\n"
                    f"Median = {values.median():.4g}",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white",
                              alpha=0.8, edgecolor="0.7"),
                )
            else:
                ax.text(
                    0.5, 0.5, "No completed runs\nwith this metric",
                    transform=ax.transAxes, ha="center", va="center",
                )

            ax.set_title(f"{dataset} [{DATASET_FIDELITY[dataset].upper()}]",
                         fontsize=10)
            ax.set_xlabel(label)
            ax.set_ylabel("Number of simulations")

        fig.suptitle(label, fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}_by_dataset.pdf", bbox_inches="tight")
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "voxel_set",
        help="Path to voxel-set directory containing logs/ and manifests",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory (default: voxel-set/timing_analysis)",
    )
    parser.add_argument(
        "--tolerance-mm", type=float, default=0.0,
        help="If a log name matches no manifest tag exactly, match it to the\n"
             "nearest manifest voxel within this distance in mm (default: off)",
    )
    parser.add_argument(
        "--keep-duplicates", action="store_true",
        help="Count every log, even if a voxel has several",
    )
    args = parser.parse_args()

    root = Path(args.voxel_set).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else root / "timing_analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Voxel set: {root}\n")

    manifest = load_manifests(root)
    paths = list_logs(root)
    choose_key_modes(manifest, paths)

    df = collect_logs(paths, manifest, tol_mm=args.tolerance_mm)
    if df.empty:
        raise SystemExit("No ReMaGe logs found.")

    df = mark_used(df, args.keep_duplicates)
    df.to_csv(outdir / "timing_per_run.csv", index=False)

    funnel = build_funnel(df, manifest)
    funnel.to_csv(outdir / "n_funnel.csv", index=False)

    write_summary(df, outdir)
    plot_distributions(df, outdir)

    print(f"\nSaved timing analysis to: {outdir}")
    print(f"Parsed log files: {df['log_file'].nunique()}")

    print("\nLog mapping status (one entry per log file):")
    per_file = df.drop_duplicates("log_file")
    print(per_file["mapping_status"].value_counts().to_string())

    n_fuzzy = int(per_file["fuzzy_match"].sum())
    if n_fuzzy:
        print(f"\nNOTE: {n_fuzzy} logs were matched by distance "
              f"(<= {args.tolerance_mm} mm), not by exact name.")

    diagnose_unmatched(df, manifest)
    print_funnel(funnel, args.keep_duplicates)

    n_shared = int(per_file["shared_coordinate"].sum())
    if n_shared:
        print(f"\nNOTE: {n_shared} logs belong to voxels shared by several "
              f"datasets of the same fidelity; they are counted in each.")


if __name__ == "__main__":
    main()