#!/bin/bash
# =============================================================================
# EQNet Debug Training Script
# =============================================================================
#
# Three modes:
#   smoke   — Quick crash test (2 iters, 1 epoch, streams from cloud). Does it run?
#   overfit — Fixed-batch overfit (same data every step). Loss should -> 0.
#   overfit-random — Overfit with random crops of cached data. Tests generalization.
#
# For overfit mode, first download debug data:
#   python download_debug_data.py
#
# Usage:
#   ./train_debug.sh <model> [smoke|overfit|overfit-random|overfit-all]
#   ./train_debug.sh all [smoke|overfit|overfit-random|overfit-all]
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

DEVICE="mps"  # Use "mps" on Mac, "cuda" with GPU
OUTPUT_DIR="./output"
DEBUG_DATA="./data/debug"

# DAS GCS paths (for smoke tests)
DAS_DATA_PATH="gs://quakeflow_das"
DAS_LABEL_PATH="gs://quakeflow_das"
DAS_LABEL_LIST="gs://quakeflow_das/training/training_v1/labels_train.txt"

# =============================================================================
# Model Configs
# =============================================================================

# CEED smoke: stream from cloud, minimal iters
ceed_smoke() {
    local model=$1
    echo "[smoke] $model (CEED streaming)..."
    python train.py \
        --model "$model" \
        --dataset-type ceed \
        --streaming \
        --nx 1 \
        --buffer-size 1 \
        --batch-size 2 \
        --epochs 1 \
        --iters-per-epoch 2 \
        --workers 0 \
        --log-scale \
        --moving-norm \
        --device $DEVICE \
        --print-freq 1 \
        --output-dir "$OUTPUT_DIR/smoke_${model}"
}

# CEED overfit: local data, cached batch, many epochs
ceed_overfit() {
    local model=$1
    local overfit_mode=$2  # "fixed" or "random"
    local data_file="$DEBUG_DATA/ceed/001.parquet"
    if [ ! -f "$data_file" ]; then
        echo "Error: $data_file not found. Run: python download_debug_data.py"
        exit 1
    fi
    local iters=$( [ "$overfit_mode" = "fixed" ] && echo 10 || echo 100 )
    echo "[overfit-$overfit_mode] $model (CEED local, $iters iters/epoch)..."
    python train.py \
        --model "$model" \
        --dataset-type ceed \
        --ceed-data-files "$data_file" \
        --nx 1 \
        --buffer-size 1 \
        --batch-size 2 \
        --epochs 100 \
        --iters-per-epoch $iters \
        --workers 0 \
        --no-augment \
        --overfit "$overfit_mode" \
        --lr 0.0003 \
        --device $DEVICE \
        --print-freq 1 \
        --output-dir "$OUTPUT_DIR/overfit_${overfit_mode}_${model}"
}

# DAS smoke: stream from GCS, minimal iters
das_smoke() {
    local model=$1
    echo "[smoke] $model (DAS from GCS)..."
    python train.py \
        --model "$model" \
        --data-path $DAS_DATA_PATH \
        --label-path $DAS_LABEL_PATH \
        --label-list $DAS_LABEL_LIST \
        --nt 1024 \
        --nx 1024 \
        --batch-size 1 \
        --epochs 1 \
        --iters-per-epoch 2 \
        --workers 0 \
        --log-scale \
        --moving-norm \
        --device $DEVICE \
        --print-freq 1 \
        --output-dir "$OUTPUT_DIR/smoke_${model}"
}

# DAS overfit: local data, cached batch, many epochs
das_overfit() {
    local model=$1
    local overfit_mode=$2  # "fixed" or "random"
    local label_list="$DEBUG_DATA/das/labels_debug.txt"
    if [ ! -f "$label_list" ]; then
        echo "Error: $label_list not found. Run: python download_debug_data.py"
        exit 1
    fi
    local iters=$( [ "$overfit_mode" = "fixed" ] && echo 10 || echo 100 )
    echo "[overfit-$overfit_mode] $model (DAS local, $iters iters/epoch)..."
    python train.py \
        --model "$model" \
        --data-path "$DEBUG_DATA/das" \
        --label-path "$DEBUG_DATA/das" \
        --label-list "$label_list" \
        --nt 4096 \
        --nx 256 \
        --no-augment \
        --overfit "$overfit_mode" \
        --lr 0.0003 \
        --batch-size 1 \
        --epochs 100 \
        --iters-per-epoch $iters \
        --workers 0 \
        --device $DEVICE \
        --print-freq 1 \
        --output-dir "$OUTPUT_DIR/overfit_${overfit_mode}_${model}"
}

run_model() {
    local model=$1 mode=$2
    case "$model" in
        phasenet|phasenet_plus|phasenet_tf|phasenet_tf_plus)
            case "$mode" in
                smoke)          ceed_smoke "$model" ;;
                overfit)        ceed_overfit "$model" "fixed" ;;
                overfit-random) ceed_overfit "$model" "random" ;;
                overfit-all)    ceed_overfit "$model" "fixed"; ceed_overfit "$model" "random" ;;
            esac ;;
        phasenet_das|phasenet_das_plus)
            case "$mode" in
                smoke)          das_smoke "$model" ;;
                overfit)        das_overfit "$model" "fixed" ;;
                overfit-random) das_overfit "$model" "random" ;;
                overfit-all)    das_overfit "$model" "fixed"; das_overfit "$model" "random" ;;
            esac ;;
        *)
            echo "Unknown model: $model"; exit 1 ;;
    esac
}

ALL_MODELS="phasenet phasenet_plus phasenet_tf phasenet_tf_plus phasenet_das phasenet_das_plus"

# =============================================================================
# Main
# =============================================================================

show_help() {
    echo "EQNet Debug Training Script"
    echo ""
    echo "Usage: ./train_debug.sh <model> [smoke|overfit|overfit-random]"
    echo ""
    echo "Modes:"
    echo "  smoke          - Quick crash test: 2 iters, streams from cloud (default)"
    echo "  overfit        - Fixed-batch overfit: same crop every step (loss -> 0)"
    echo "  overfit-random - Random-crop overfit: different crops of cached data"
    echo "  overfit-all    - Run both overfit and overfit-random"
    echo "                   Requires: python download_debug_data.py"
    echo ""
    echo "Models:"
    echo "  phasenet, phasenet_plus, phasenet_tf, phasenet_tf_plus"
    echo "  phasenet_das, phasenet_das_plus"
    echo "  all     - Run all models"
    echo ""
    echo "Examples:"
    echo "  ./train_debug.sh phasenet smoke          # smoke test"
    echo "  ./train_debug.sh phasenet_das overfit     # fixed-batch overfit (loss -> 0)"
    echo "  ./train_debug.sh phasenet_das overfit-random  # random-crop overfit"
    echo "  ./train_debug.sh all overfit-all              # all models, both modes"
}

main() {
    local model="${1:-help}"
    local mode="${2:-smoke}"

    case "$model" in
        help|--help|-h) show_help ;;
        all)
            echo "Running $mode test for all models..."
            for m in $ALL_MODELS; do
                run_model "$m" "$mode"
            done
            echo "All $mode tests completed!"
            ;;
        *)
            run_model "$model" "$mode"
            ;;
    esac
}

main "$@"
