#!/bin/bash
# =============================================================================
# EQNet Debug Training Scripts
# =============================================================================
#
# Fast training scripts for debugging and testing.
# Uses minimal settings: small batch size, few iterations, buffer_size=1
# All data is read directly from cloud (GCS/HuggingFace).
#
# Usage:
#   ./train_debug.sh [model_name]
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

EPOCHS=1
BATCH_SIZE=2
ITERS_PER_EPOCH=2
BUFFER_SIZE=1
WORKERS=0
PRINT_FREQ=1
DEVICE="cpu"  # Use "mps" on Mac, "cuda" with GPU

OUTPUT_DIR="./output"

# DAS GCS paths
DAS_DATA_PATH="gs://quakeflow_das"
DAS_LABEL_PATH="gs://quakeflow_das"
DAS_LABEL_LIST="gs://quakeflow_das/training/training_v1/labels_train.txt"

# =============================================================================
# Debug Functions
# =============================================================================

debug_phasenet() {
    echo "Debug: PhaseNet (seismic, CEED streaming from HuggingFace)..."
    python train.py \
        --model phasenet \
        --dataset-type ceed \
        --streaming \
        --buffer-size $BUFFER_SIZE \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --iters-per-epoch $ITERS_PER_EPOCH \
        --workers $WORKERS \
        --device $DEVICE \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/debug_phasenet
}

debug_phasenet_plus() {
    echo "Debug: PhaseNet Plus (seismic, CEED streaming from HuggingFace)..."
    python train.py \
        --model phasenet_plus \
        --dataset-type ceed \
        --streaming \
        --buffer-size $BUFFER_SIZE \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --iters-per-epoch $ITERS_PER_EPOCH \
        --workers $WORKERS \
        --device $DEVICE \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/debug_phasenet_plus
}

debug_phasenet_das() {
    echo "Debug: PhaseNet DAS (DAS, streaming from GCS)..."
    python train.py \
        --model phasenet_das \
        --data-path $DAS_DATA_PATH \
        --label-path $DAS_LABEL_PATH \
        --label-list $DAS_LABEL_LIST \
        --nt 1024 \
        --nx 1024 \
        --batch-size 1 \
        --epochs $EPOCHS \
        --iters-per-epoch $ITERS_PER_EPOCH \
        --workers $WORKERS \
        --device $DEVICE \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/debug_phasenet_das
}

debug_phasenet_das_plus() {
    echo "Debug: PhaseNet DAS Plus (DAS + event, streaming from GCS)..."
    python train.py \
        --model phasenet_das_plus \
        --data-path $DAS_DATA_PATH \
        --label-path $DAS_LABEL_PATH \
        --label-list $DAS_LABEL_LIST \
        --nt 1024 \
        --nx 1024 \
        --batch-size 1 \
        --epochs $EPOCHS \
        --iters-per-epoch $ITERS_PER_EPOCH \
        --workers $WORKERS \
        --device $DEVICE \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/debug_phasenet_das_plus
}

debug_all() {
    echo "Running all debug tests..."
    debug_phasenet
    debug_phasenet_plus
    debug_phasenet_das
    debug_phasenet_das_plus
    echo ""
    echo "All debug tests completed!"
    echo "Figures saved to:"
    echo "  - $OUTPUT_DIR/debug_phasenet/figures/"
    echo "  - $OUTPUT_DIR/debug_phasenet_plus/figures/"
    echo "  - $OUTPUT_DIR/debug_phasenet_das/figures/"
    echo "  - $OUTPUT_DIR/debug_phasenet_das_plus/figures/"
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    echo "EQNet Debug Training Script"
    echo ""
    echo "Usage: ./train_debug.sh [model_name]"
    echo ""
    echo "Models:"
    echo "  phasenet          - Seismic phase picking (CEED from HuggingFace)"
    echo "  phasenet_plus     - Seismic phase + polarity + event (CEED)"
    echo "  phasenet_das      - DAS phase picking (GCS)"
    echo "  phasenet_das_plus - DAS phase + event detection (GCS)"
    echo "  all               - Run all debug tests"
    echo ""
    echo "Settings: device=$DEVICE, batch=$BATCH_SIZE, iters=$ITERS_PER_EPOCH, buffer=$BUFFER_SIZE"
}

main() {
    case "${1:-help}" in
        phasenet)          debug_phasenet ;;
        phasenet_plus)     debug_phasenet_plus ;;
        phasenet_das)      debug_phasenet_das ;;
        phasenet_das_plus) debug_phasenet_das_plus ;;
        all)               debug_all ;;
        help|--help|-h)    show_help ;;
        *)                 echo "Unknown: $1"; show_help; exit 1 ;;
    esac
}

main "$@"
