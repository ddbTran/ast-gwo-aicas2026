#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh  –  Grid sweep over designs × seeds (parallel or serial)
#
# Usage:
#   bash scripts/run_sweep.sh [OPTIONS]
#
# Examples:
#   # Default sweep (all designs, seeds 0-4)
#   bash scripts/run_sweep.sh
#
#   # Custom designs and seeds, 4 parallel jobs
#   bash scripts/run_sweep.sh --designs "adder bar hyp" --seeds "0 1 2" --jobs 4
#
#   # Dry run (print commands, don't execute)
#   bash scripts/run_sweep.sh --dry-run
# =============================================================================

set -euo pipefail

# ── Sweep targets ─────────────────────────────────────────────────────────────
#DESIGNS="adder bar div hyp log2 max multiplier sin sqrt square"
DESIGNS="adder"
SEEDS="0 1 2 3 4"

# ── Shared GWO config ─────────────────────────────────────────────────────────
POP=50
DIM=80
ITERS=50
LUT_K=6
K_MIN=18.0
K_MAX=18.0
WORKERS=""            # ABC-level workers per job (empty = auto)
ABC_BIN="yosys-abc"
BENCHMARKS_DIR="./benchmarks/epfl/arithmetic"
RESULTS_DIR="./results/runs/ast_gwo/conference"

# ── Sweep control ─────────────────────────────────────────────────────────────
JOBS=1                # parallel sweep jobs (GNU parallel or xargs)
DRY_RUN=0
LOG_DIR="./results/sweep_logs"
RESUME=0              # skip combos whose iter.log already exists

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --designs)        DESIGNS="$2";         shift 2 ;;
        --seeds)          SEEDS="$2";           shift 2 ;;
        --pop)            POP="$2";             shift 2 ;;
        --dim)            DIM="$2";             shift 2 ;;
        --iters)          ITERS="$2";           shift 2 ;;
        --lut_k)          LUT_K="$2";           shift 2 ;;
        --k_min)          K_MIN="$2";           shift 2 ;;
        --k_max)          K_MAX="$2";           shift 2 ;;
        --workers)        WORKERS="$2";         shift 2 ;;
        --abc)            ABC_BIN="$2";         shift 2 ;;
        --benchmarks_dir) BENCHMARKS_DIR="$2";  shift 2 ;;
        --results_dir)    RESULTS_DIR="$2";     shift 2 ;;
        --jobs|-j)        JOBS="$2";            shift 2 ;;
        --log_dir)        LOG_DIR="$2";         shift 2 ;;
        --resume)         RESUME=1;             shift ;;
        --dry-run)        DRY_RUN=1;            shift ;;
        -h|--help)        usage ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Dependency checks ─────────────────────────────────────────────────────────
if ! command -v "$ABC_BIN" &>/dev/null; then
    echo "[ERROR] ABC binary not found: $ABC_BIN" >&2
    exit 1
fi

USE_PARALLEL=0
if [[ "$JOBS" -gt 1 ]]; then
    if command -v parallel &>/dev/null; then
        USE_PARALLEL=1
    else
        echo "[WARN] GNU parallel not found – falling back to xargs -P${JOBS}" >&2
    fi
fi

mkdir -p "$LOG_DIR"

# ── Build task list ───────────────────────────────────────────────────────────
TASK_FILE="$(mktemp /tmp/sweep_tasks_XXXXXX.txt)"
trap 'rm -f "$TASK_FILE"' EXIT

TOTAL=0
SKIPPED=0

for DESIGN in $DESIGNS; do
    BLIF="${BENCHMARKS_DIR}/${DESIGN}.blif"
    if [[ ! -f "$BLIF" ]]; then
        echo "[WARN] Design not found, skipping: $BLIF" >&2
        continue
    fi

    for SEED in $SEEDS; do
        # Build expected iter.log path for --resume check
        RUN_TAG="lut_k_${LUT_K}-pop_${POP}-dim_${DIM}-iters_${ITERS}"
        ITER_LOG="${RESULTS_DIR}/${RUN_TAG}/${DESIGN}/${SEED}/iter.log"

        if [[ "$RESUME" -eq 1 && -f "$ITER_LOG" ]]; then
            (( SKIPPED++ )) || true
            continue
        fi

        TASK="python -m algorithms.ast_gwo.ast_gwo"
        TASK+=" --design $DESIGN"
        TASK+=" --pop $POP --dim $DIM --iters $ITERS"
        TASK+=" --lut_k $LUT_K --seed $SEED"
        TASK+=" --k_min $K_MIN --k_max $K_MAX"
        TASK+=" --abc $ABC_BIN"
        TASK+=" --benchmarks_dir $BENCHMARKS_DIR"
        TASK+=" --results_dir $RESULTS_DIR"
        [[ -n "$WORKERS" ]] && TASK+=" --workers $WORKERS"

        echo "$TASK" >> "$TASK_FILE"
        (( TOTAL++ )) || true
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " AST-GWO – sweep"
echo "------------------------------------------------------------"
echo " designs : $(echo "$DESIGNS" | wc -w | tr -d ' ')"
echo " seeds   : $(echo "$SEEDS"   | wc -w | tr -d ' ')"
echo " tasks   : $TOTAL  (skipped: $SKIPPED)"
echo " jobs    : $JOBS"
echo " pop     : $POP  |  dim : $DIM  |  iters : $ITERS"
echo " results : $RESULTS_DIR"
echo " logs    : $LOG_DIR"
echo "============================================================"

if [[ "$TOTAL" -eq 0 ]]; then
    echo "No tasks to run."
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "[DRY RUN] Would execute:"
    cat "$TASK_FILE"
    exit 0
fi

# ── Execute ───────────────────────────────────────────────────────────────────
SWEEP_START=$(date +%s)
FAILED=0
DONE=0

run_task() {
    local cmd="$1"
    local design seed

    # Extract design and seed for log filename
    design=$(echo "$cmd" | grep -oP '(?<=--design )\S+')
    seed=$(echo "$cmd"   | grep -oP '(?<=--seed )\S+')
    local logfile="${LOG_DIR}/${design}_seed${seed}.log"

    if bash -c "$cmd" > "$logfile" 2>&1; then
        echo "[OK]   ${design} seed=${seed}"
    else
        echo "[FAIL] ${design} seed=${seed}  (see $logfile)"
        return 1
    fi
}
export -f run_task
export LOG_DIR

if [[ "$JOBS" -eq 1 ]]; then
    # ── Serial ────────────────────────────────────────────────────────────────
    while IFS= read -r cmd; do
        if run_task "$cmd"; then
            (( DONE++ ))  || true
        else
            (( FAILED++ )) || true
        fi
        echo "  Progress: $DONE / $TOTAL"
    done < "$TASK_FILE"

elif [[ "$USE_PARALLEL" -eq 1 ]]; then
    # ── GNU parallel ─────────────────────────────────────────────────────────
    parallel --jobs "$JOBS" --line-buffer run_task {} < "$TASK_FILE" || true
    DONE=$(grep -c '^\[OK\]'   /dev/stdin 2>/dev/null || echo "?")
    FAILED=$(grep -c '^\[FAIL\]' /dev/stdin 2>/dev/null || echo "?")

else
    # ── xargs fallback ────────────────────────────────────────────────────────
    xargs -P "$JOBS" -I{} bash -c 'run_task "$@"' _ {} < "$TASK_FILE" || true
fi

# ── Final report ──────────────────────────────────────────────────────────────
SWEEP_END=$(date +%s)
ELAPSED=$(( SWEEP_END - SWEEP_START ))

echo ""
echo "============================================================"
echo " Sweep complete"
echo " Elapsed : ${ELAPSED}s"
echo " Logs    : $LOG_DIR"
echo "============================================================"

if [[ "$FAILED" -gt 0 ]]; then
    echo "[WARN] $FAILED task(s) failed – check logs in $LOG_DIR" >&2
    exit 1
fi
