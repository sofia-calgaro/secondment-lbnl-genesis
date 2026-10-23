
#!/usr/bin/env bash
set -euo pipefail

# Submit Remage jobs as a Slurm array on NERSC Perlmutter.
# Uses the local Momentum-enabled Remage while retaining the Pixi environment.

PROJECT="/global/cfs/cdirs/m2676/users/calgaro/uzh/secondment/optmap_demo"
SETUP="$PROJECT/setup_remage_momentum.sh"

usage() {
    cat <<'EOF'
Usage: submit_nersc.sh -j JOBS [options]

Required:
  -j FILE    Text file containing simulation commands

Options:
  -N N       Number of Slurm array tasks (default: 1)
  -c N       CPUs per array task / PROCS (default: 1)
  -p N       Maximum concurrent commands per array task (default: 1)
  -m N       Memory in GB (default: 256)
  -t TIME    Wall time (default: 04:00:00)
  -q QOS     Slurm QOS (default: regular)
  -A ACCOUNT Slurm account (default: m2676)
  -C CONSTR  Slurm constraint (default: cpu)
  -n NAME    Job name (default: remage)
  -d DEP     Slurm dependency specification, e.g. afterok:123456
  -e OPTS    Extra SBATCH options
  -D         Dry run; print the generated batch script without submitting
  -h         Show this help
EOF
}

JOBS=""
ARRAY_N=1
NODE_CPUS=1
PAR=1
MEM=256
TIME="04:00:00"
QOS="regular"
ACCOUNT="m2676"
CONSTRAINT="cpu"
NAME="remage"
DEPENDENCY=""
EXTRA=""
DRY_RUN=0

while getopts ":j:N:c:p:m:t:q:A:C:n:d:e:Dh" opt; do
    case "$opt" in
        j) JOBS="$OPTARG" ;;
        N) ARRAY_N="$OPTARG" ;;
        c) NODE_CPUS="$OPTARG" ;;
        p) PAR="$OPTARG" ;;
        m) MEM="$OPTARG" ;;
        t) TIME="$OPTARG" ;;
        q) QOS="$OPTARG" ;;
        A) ACCOUNT="$OPTARG" ;;
        C) CONSTRAINT="$OPTARG" ;;
        n) NAME="$OPTARG" ;;
        d) DEPENDENCY="$OPTARG" ;;
        e) EXTRA="$OPTARG" ;;
        D) DRY_RUN=1 ;;
        h) usage; exit 0 ;;
        :) echo "ERROR: -$OPTARG requires an argument" >&2; usage >&2; exit 2 ;;
        \?) echo "ERROR: Unknown option -$OPTARG" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$JOBS" ]] || { echo "ERROR: provide -j JOBS" >&2; exit 2; }
[[ -f "$JOBS" ]] || { echo "ERROR: job list not found: $JOBS" >&2; exit 2; }
[[ -f "$SETUP" ]] || { echo "ERROR: setup script not found: $SETUP" >&2; exit 2; }

for value in "$ARRAY_N" "$NODE_CPUS" "$PAR" "$MEM"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: array size, CPUs, concurrency, and memory must be positive integers" >&2
        exit 2
    }
done

JOBS="$(realpath "$JOBS")"
JOB_DIR="$(dirname "$JOBS")"
BATCH_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/submit_remage_XXXXXX.sh")"
trap 'rm -f "$BATCH_SCRIPT"' EXIT

cat > "$BATCH_SCRIPT" <<'SBATCH_SCRIPT'
#!/usr/bin/env bash
#SBATCH --nodes=1
SBATCH_SCRIPT

cat >> "$BATCH_SCRIPT" <<SBATCH_SCRIPT
#SBATCH --account=$ACCOUNT
#SBATCH --qos=$QOS
#SBATCH --constraint=$CONSTRAINT
#SBATCH --time=$TIME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$NODE_CPUS
#SBATCH --mem=${MEM}G
#SBATCH --exclusive
#SBATCH --array=0-$((ARRAY_N - 1))
#SBATCH --job-name=$NAME
#SBATCH --output=$JOB_DIR/slurm-%x-%A_%a.out
SBATCH_SCRIPT

if [[ -n "$DEPENDENCY" ]]; then
    echo "#SBATCH --dependency=$DEPENDENCY" >> "$BATCH_SCRIPT"
fi

if [[ -n "$EXTRA" ]]; then
    # EXTRA is expected to contain SBATCH option text, e.g. --mail-type=END
    echo "#SBATCH $EXTRA" >> "$BATCH_SCRIPT"
fi

cat >> "$BATCH_SCRIPT" <<'SBATCH_SCRIPT'

set -eo pipefail

PROJECT="/global/cfs/cdirs/m2676/users/calgaro/uzh/secondment/optmap_demo"
SETUP="$PROJECT/setup_remage_momentum.sh"
JOBS="__JOBS_FILE__"
ARRAY_N=__ARRAY_N__
PAR=__PAR__
NODE_CPUS=__NODE_CPUS__

cd "$PROJECT"

# Activate the normal Pixi environment for supporting dependencies.
eval "$(pixi shell-hook)"

# Override only Remage with the locally built Momentum-enabled version.
source "$SETUP"

echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Remage executable: $(command -v remage)"
echo "C++ executable: $REMAGE_CPP_PATH"
echo "CPUs per task: $NODE_CPUS"
echo "Concurrent commands: $PAR"
remage --version

export PROCS="$NODE_CPUS"
export OMP_NUM_THREADS="$NODE_CPUS"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Select this array task's share of the job list using round-robin assignment.
# Blank lines and comment lines are ignored.
mapfile -t TASK_JOBS < <(
    awk -v n="$ARRAY_N" -v task="$TASK_ID" '
        /^[[:space:]]*($|#)/ { next }
        (count++ % n) == task { print }
    ' "$JOBS"
)

if (( ${#TASK_JOBS[@]} == 0 )); then
    echo "No commands assigned to array task $TASK_ID."
    exit 0
fi

printf 'Commands assigned to this task: %s\n' "${#TASK_JOBS[@]}"

run_job() {
    local command="$1"

    # Replace the standard Pixi Remage entry point with the local wrapper.
    # The rest of each command, including its arguments and output/log paths,
    # is left unchanged.
    command="${command//pixi run remage/remage}"

    eval "$command"
}
export -f run_job

printf '%s\n' "${TASK_JOBS[@]}" |
    xargs -r -d '\n' -P "$PAR" -I '{}' \
        bash -c 'run_job "$1"' _ '{}'

echo "Array task $TASK_ID completed."
SBATCH_SCRIPT

# Insert values into the batch script without relying on the caller's cwd.
# Escape sed replacement characters in the job-list path.
sed -i \
    -e "s|__JOBS_FILE__|$JOBS|g" \
    -e "s|__ARRAY_N__|$ARRAY_N|g" \
    -e "s|__PAR__|$PAR|g" \
    -e "s|__NODE_CPUS__|$NODE_CPUS|g" \
    "$BATCH_SCRIPT"

if (( DRY_RUN )); then
    cat "$BATCH_SCRIPT"
    exit 0
fi

sbatch "$BATCH_SCRIPT"

