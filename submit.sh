#!/bin/bash
# ============================================================================
# HE-NAS -- DRAC submission script.
#
#   ONE-SHOT (submit, go to sleep, check next week):
#       sbatch submit.sh
#   ...runs the ENTIRE study: ge, random, both sequential baselines, expert,
#   the 2x2 ablation, and NSGA-II, at 30 seeds each, and writes everything to
#   results/. Nothing else to run afterwards except the analysis.
#
#   SINGLE EXPERIMENT (if you ever want just one):
#       sbatch submit.sh ge --dataset cifar10
#
# BEFORE THE FIRST SUBMISSION, on the LOGIN node (compute nodes have no
# internet, and the job will abort immediately if you skip this):
#       python main.py download --dataset cifar10
#
# The environment must already be prepared and on PATH -- this script does no
# module loading, no virtualenv creation, and no pip install.
#
# IF IT RUNS OUT OF TIME: just `sbatch submit.sh` again. Every experiment
# resumes at seed granularity (a seed whose results.json exists is skipped), so
# a resubmission continues where it stopped and costs nothing for finished work.
#
# WANT IT FASTER? Ask for a whole GPU node:
#       #SBATCH --gpus-per-node=h100:4
#       #SBATCH --cpus-per-task=48
# The script auto-detects the GPU count and pins workers round-robin across
# them. Be aware that on a def- allocation a 4-GPU job usually queues far
# longer than it saves, so 1 GPU with generous walltime often finishes sooner.
# ============================================================================

#SBATCH --account=def-yoursupervisor   # set to your DRAC allocation
#SBATCH --job-name=he-nas
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00
# Written to the submission directory, not logs/ -- SLURM opens this file
# before the script runs, so a non-existent directory kills the job instantly.
#SBATCH --output=he-nas_%j.out
#SBATCH --mail-user=you@example.com     # optional: job notifications
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

# Each worker is a Python dispatch loop feeding tiny kernels; this workload is
# launch-bound, not GPU-bound (measured: an H100 and an RTX 5070 Ti laptop GPU
# run it at the same 14.3 s/eval). Without pinning threads, every worker spawns
# OpenMP threads for all visible cores and they thrash -- the usual reason a
# multi-process run shows no speedup at all.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DATASET="${DATASET:-cifar10}"
N_SEEDS="${N_SEEDS:-30}"
SEED_BASE="${SEED_BASE:-42}"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[[ "$NGPU" -lt 1 ]] && NGPU=1
NCPU="${SLURM_CPUS_PER_TASK:-12}"
# One worker per core, but never more than 16 per GPU (past that the GPU's
# own scheduling overhead starts to dominate).
PARALLEL="${PARALLEL:-$(( NCPU < NGPU * 16 ? NCPU : NGPU * 16 ))}"
[[ "$PARALLEL" -lt 1 ]] && PARALLEL=1

STATUS="logs/STATUS.txt"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

say "================================================================"
say "HE-NAS study starting"
say "  dataset=$DATASET  seeds=$N_SEEDS from $SEED_BASE"
say "  GPUs=$NGPU  cores=$NCPU  concurrent workers=$PARALLEL"
say "  job=${SLURM_JOB_ID:-local}  node=$(hostname)"
say "================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$STATUS"

# Fail fast rather than burn the allocation. Compute nodes have no internet,
# but torchvision is called with download=True, so a missing or half-downloaded
# dataset makes every worker hang on a network call instead of erroring. Check
# by actually loading it with download disabled -- merely testing that
# data/$DATASET exists is not enough, since torchvision creates that directory
# even when the download fails.
if ! python -c "
from fitness import load_dataset
load_dataset('$DATASET', 1, 1, seed=0, download=False)
print('dataset OK: $DATASET')
" 2>&1 | tee -a "$STATUS"; then
    say "FATAL: dataset '$DATASET' is missing or incomplete on this node."
    say "       Compute nodes have no internet. On the LOGIN node, run:"
    say "         python main.py download --dataset $DATASET"
    say "       then resubmit."
    exit 1
fi

# Cheap guard on the search space itself -- asserts the depth model still
# matches the TenSEAL measurements and that every architecture the grammar can
# express has a secure chain. Seconds, no GPU.
if ! python main.py selftest; then
    say "FATAL: selftest failed. Not burning $N_SEEDS seeds on a broken model."
    exit 1
fi

# --------------------------------------------------------------------------- #
# run_experiment <label> <main.py args...>
#
# Fans the seeds out as independent single-seed processes, round-robin across
# whatever GPUs we were given, capped at $PARALLEL concurrent. Then makes one
# aggregation pass over the full seed range: that pass does no GPU work at all,
# it reloads every seed from disk and writes the correct summary JSON (the
# workers run with --no-summary so they never race on it).
# --------------------------------------------------------------------------- #
run_experiment() {
    local label="$1"; shift
    local t0 i seed gpu
    t0=$SECONDS
    say ">>> $label : starting ($N_SEEDS seeds, $PARALLEL at a time)"

    for i in $(seq 0 $((N_SEEDS - 1))); do
        seed=$((SEED_BASE + i))
        gpu=$((i % NGPU))
        CUDA_VISIBLE_DEVICES="$gpu" \
            python main.py "$@" --seeds 1 --seed-base "$seed" --no-summary \
            > "logs/${label}_seed${seed}.log" 2>&1 &
        while [[ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]]; do
            wait -n
        done
    done
    wait

    # Aggregation pass: no GPU work, everything resumes from disk.
    python main.py "$@" --seeds "$N_SEEDS" --seed-base "$SEED_BASE" \
        > "logs/${label}_summary.log" 2>&1
    say "<<< $label : done in $(( (SECONDS - t0) / 60 )) min"
}

# --------------------------------------------------------------------------- #
# Single-experiment mode: sbatch submit.sh ge --dataset cifar10
# --------------------------------------------------------------------------- #
if [[ $# -gt 0 ]]; then
    run_experiment "$1" "$@"
    say "Single experiment complete."
    exit 0
fi

# --------------------------------------------------------------------------- #
# Full study. Ordered cheapest-first so that if the wall clock runs out, the
# most important results are already on disk. Resubmit to continue.
# --------------------------------------------------------------------------- #
run_experiment expert     expert     --dataset "$DATASET"
run_experiment ge         ge         --dataset "$DATASET"
run_experiment random     random     --dataset "$DATASET"
run_experiment seq_arch   sequential --which arch_first --dataset "$DATASET"
run_experiment seq_ckks   sequential --which ckks_first --dataset "$DATASET"
run_experiment nsga2      nsga2      --dataset "$DATASET"
run_experiment ablation   ablation   --dataset "$DATASET"

say "================================================================"
say "ALL EXPERIMENTS COMPLETE"
say "================================================================"
ls -1 results/*_summary.json 2>/dev/null | tee -a "$STATUS"
say "Next: python champions.py   then   python main.py stats <A> <B>"
