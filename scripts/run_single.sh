#!/usr/bin/env bash
# =============================================================================
# run_single.sh  –  Run one BinaryGWO experiment
#
# Usage:
#   bash scripts/run_single.sh [OPTIONS]
#
# Examples:
#   bash scripts/run_single.sh --design adder
#   bash scripts/run_single.sh --design bar --pop 50 --dim 80 --iters 50 --seed 0
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DESIGN="adder"
POP=50
DIM=80
ITERS=200
LUT_K=6
SEED=42
K_MIN=6.0
K_MAX=14.0
WORKERS=""          # empty = let Python decide
ABC_BIN="yosys-abc"
BENCHMARKS_DIR="./benchmarks/epfl/arithmetic"
#BENCHMARKS_DIR="./benchmarks/epfl/random_control"
RESULTS_DIR="./results/ast_gwo"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \{0,2\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --design)         DESIGN="$2";         shift 2 ;;
        --pop)            POP="$2";             shift 2 ;;
        --dim)            DIM="$2";             shift 2 ;;
        --iters)          ITERS="$2";           shift 2 ;;
        --lut_k)          LUT_K="$2";           shift 2 ;;
        --seed)           SEED="$2";            shift 2 ;;
        --k_min)          K_MIN="$2";           shift 2 ;;
        --k_max)          K_MAX="$2";           shift 2 ;;
        --workers)        WORKERS="$2";         shift 2 ;;
        --abc)            ABC_BIN="$2";         shift 2 ;;
        --benchmarks_dir) BENCHMARKS_DIR="$2";  shift 2 ;;
        --results_dir)    RESULTS_DIR="$2";     shift 2 ;;
        -h|--help)        usage ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
BLIF="${BENCHMARKS_DIR}/${DESIGN}.blif"
if [[ ! -f "$BLIF" ]]; then
    echo "[ERROR] Design file not found: $BLIF" >&2
    exit 1
fi

if ! command -v "$ABC_BIN" &>/dev/null; then
    echo "[ERROR] ABC binary not found: $ABC_BIN" >&2
    exit 1
fi

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
    python -m algorithms.ast_gwo.ast_gwo
    --design         "$DESIGN"
    --pop            "$POP"
    --dim            "$DIM"
    --iters          "$ITERS"
    --lut_k          "$LUT_K"
    --seed           "$SEED"
    --k_min          "$K_MIN"
    --k_max          "$K_MAX"
    --abc            "$ABC_BIN"
    --benchmarks_dir "$BENCHMARKS_DIR"
    --results_dir    "$RESULTS_DIR"
)

[[ -n "$WORKERS" ]] && CMD+=(--workers "$WORKERS")

# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " AST-GWO – single run"
echo "------------------------------------------------------------"
echo " design  : $DESIGN"
echo " pop     : $POP  |  dim : $DIM  |  iters : $ITERS"
echo " lut_k   : $LUT_K  |  seed : $SEED"
echo " k_min   : $K_MIN  |  k_max : $K_MAX"
echo " workers : ${WORKERS:-auto}"
echo " results : $RESULTS_DIR"
echo "============================================================"

START=$(date +%s)
"${CMD[@]}"
END=$(date +%s)

echo ""
echo "Done in $(( END - START ))s"
echo "Logs: $RESULTS_DIR/lut_k_${LUT_K}-pop_${POP}-dim_${DIM}-iters_${ITERS}/${DESIGN}/${SEED}/"
