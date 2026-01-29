#!/bin/bash
# =============================================================================
# EQNet Training Scripts
# =============================================================================
#
# Usage:
#   ./train.sh [model_name]
#
# Models available:
#   - phasenet           : Basic phase picking (seismic)
#   - phasenet_plus      : Phase + polarity + event detection (seismic)
#   - phasenet_tf        : Phase picking with transformer (seismic)
#   - phasenet_tf_plus   : Transformer + polarity + event (seismic)
#   - phasenet_das       : Phase picking (DAS)
#   - phasenet_das_plus  : Phase + event detection (DAS)
#
# Example:
#   ./train.sh phasenet_plus
#   ./train.sh phasenet_das
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Common settings
EPOCHS=100
BATCH_SIZE=8
LR=3e-4
WEIGHT_DECAY=1e-4
WORKERS=4
PRINT_FREQ=10

# DAS-specific settings
DAS_BATCH_SIZE=4
DAS_EPOCHS=10

# Multi-GPU settings
NUM_GPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

# Paths
OUTPUT_DIR="./output"
WANDB_PROJECT="eqnet"

# DAS data paths (Google Cloud Storage)
DAS_DATA_PATH="gs://quakeflow_das"
DAS_LABEL_PATH="gs://quakeflow_das"
DAS_LABEL_LIST="gs://quakeflow_das/training/training_v1/labels_train.txt"
DAS_NOISE_LIST="gs://quakeflow_das/training/training_v1/noise_train.txt"
DAS_TEST_LABEL_LIST="gs://quakeflow_das/training/training_v1/labels_test.txt"
DAS_TEST_NOISE_LIST="gs://quakeflow_das/training/training_v1/noise_test.txt"

# CEED dataset settings
CEED_REGION="NC,SC"
CEED_MIN_SNR=2.0
CEED_ITERS_PER_EPOCH=1000

# =============================================================================
# Helper Functions
# =============================================================================

run_cmd() {
    local cmd="$1"
    echo "=================================================="
    echo "Running: $cmd"
    echo "=================================================="
    eval "$cmd"
}

get_train_cmd() {
    local base_cmd="$1"
    if [ "$NUM_GPU" -eq 0 ]; then
        echo "python $base_cmd --device cpu"
    elif [ "$NUM_GPU" -eq 1 ]; then
        echo "python $base_cmd"
    else
        echo "torchrun --standalone --nproc_per_node=$NUM_GPU $base_cmd"
    fi
}

# =============================================================================
# Seismic Models (3-component data)
# =============================================================================

train_phasenet() {
    echo "Training PhaseNet (basic phase picking)..."

    local base_cmd="train.py \
        --model phasenet \
        --backbone unet \
        --dataset-type ceed \
        --ceed-region $CEED_REGION \
        --streaming \
        --iters-per-epoch $CEED_ITERS_PER_EPOCH \
        --min-snr $CEED_MIN_SNR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --workers $WORKERS \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/phasenet \
        --stack-event \
        --stack-noise \
        --flip-polarity \
        --drop-channel \
        --sync-bn \
        --wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-name phasenet"

    run_cmd "$(get_train_cmd "$base_cmd")"
}

train_phasenet_plus() {
    echo "Training PhaseNet Plus (phase + polarity + event)..."

    local base_cmd="train.py \
        --model phasenet_plus \
        --backbone unet \
        --dataset-type ceed \
        --ceed-region $CEED_REGION \
        --streaming \
        --iters-per-epoch $CEED_ITERS_PER_EPOCH \
        --min-snr $CEED_MIN_SNR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --workers $WORKERS \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/phasenet_plus \
        --stack-event \
        --stack-noise \
        --flip-polarity \
        --drop-channel \
        --sync-bn \
        --wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-name phasenet_plus"

    run_cmd "$(get_train_cmd "$base_cmd")"
}

train_phasenet_tf() {
    echo "Training PhaseNet-TF (transformer backbone)..."

    local base_cmd="train.py \
        --model phasenet_tf \
        --backbone unet \
        --dataset-type ceed \
        --ceed-region $CEED_REGION \
        --streaming \
        --iters-per-epoch $CEED_ITERS_PER_EPOCH \
        --min-snr $CEED_MIN_SNR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --workers $WORKERS \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/phasenet_tf \
        --stack-event \
        --stack-noise \
        --flip-polarity \
        --drop-channel \
        --sync-bn \
        --wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-name phasenet_tf"

    run_cmd "$(get_train_cmd "$base_cmd")"
}

train_phasenet_tf_plus() {
    echo "Training PhaseNet-TF Plus (transformer + polarity + event)..."

    local base_cmd="train.py \
        --model phasenet_tf_plus \
        --backbone unet \
        --dataset-type ceed \
        --ceed-region $CEED_REGION \
        --streaming \
        --iters-per-epoch $CEED_ITERS_PER_EPOCH \
        --min-snr $CEED_MIN_SNR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --workers $WORKERS \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/phasenet_tf_plus \
        --stack-event \
        --stack-noise \
        --flip-polarity \
        --drop-channel \
        --sync-bn \
        --wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-name phasenet_tf_plus"

    run_cmd "$(get_train_cmd "$base_cmd")"
}

# =============================================================================
# DAS Models (single-channel strain rate data)
# =============================================================================

train_phasenet_das() {
    echo "Training PhaseNet-DAS (DAS phase picking)..."

    local base_cmd="train.py \
        --model phasenet_das \
        --backbone unet \
        --data-path $DAS_DATA_PATH \
        --label-path $DAS_LABEL_PATH \
        --label-list $DAS_LABEL_LIST \
        --noise-list $DAS_NOISE_LIST \
        --test-data-path $DAS_DATA_PATH \
        --test-label-path $DAS_LABEL_PATH \
        --test-label-list $DAS_TEST_LABEL_LIST \
        --test-noise-list $DAS_TEST_NOISE_LIST \
        --nt 3072 \
        --nx 5120 \
        --batch-size $DAS_BATCH_SIZE \
        --epochs $DAS_EPOCHS \
        --lr $LR \
        --weight-decay 1e-1 \
        --workers $WORKERS \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/phasenet_das \
        --stack-event \
        --stack-noise \
        --resample-space \
        --resample-time \
        --masking \
        --random-crop \
        --sync-bn \
        --wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-name phasenet_das"

    run_cmd "$(get_train_cmd "$base_cmd")"
}

train_phasenet_das_plus() {
    echo "Training PhaseNet-DAS Plus (DAS + event detection)..."

    local base_cmd="train.py \
        --model phasenet_das_plus \
        --backbone unet \
        --data-path $DAS_DATA_PATH \
        --label-path $DAS_LABEL_PATH \
        --label-list $DAS_LABEL_LIST \
        --noise-list $DAS_NOISE_LIST \
        --test-data-path $DAS_DATA_PATH \
        --test-label-path $DAS_LABEL_PATH \
        --test-label-list $DAS_TEST_LABEL_LIST \
        --test-noise-list $DAS_TEST_NOISE_LIST \
        --nt 3072 \
        --nx 5120 \
        --batch-size $DAS_BATCH_SIZE \
        --epochs $DAS_EPOCHS \
        --lr $LR \
        --weight-decay 1e-1 \
        --workers $WORKERS \
        --print-freq $PRINT_FREQ \
        --output-dir $OUTPUT_DIR/phasenet_das_plus \
        --stack-event \
        --stack-noise \
        --resample-space \
        --resample-time \
        --masking \
        --random-crop \
        --sync-bn \
        --wandb \
        --wandb-project $WANDB_PROJECT \
        --wandb-name phasenet_das_plus"

    run_cmd "$(get_train_cmd "$base_cmd")"
}

# =============================================================================
# Training All Models
# =============================================================================

train_all_seismic() {
    echo "Training all seismic models..."
    train_phasenet
    train_phasenet_plus
    train_phasenet_tf
    train_phasenet_tf_plus
}

train_all_das() {
    echo "Training all DAS models..."
    train_phasenet_das
    train_phasenet_das_plus
}

train_all() {
    echo "Training all models..."
    train_all_seismic
    train_all_das
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    echo "EQNet Training Script"
    echo ""
    echo "Usage: ./train.sh [model_name]"
    echo ""
    echo "Available models:"
    echo "  phasenet           - Basic phase picking (seismic)"
    echo "  phasenet_plus      - Phase + polarity + event (seismic)"
    echo "  phasenet_tf        - Transformer backbone (seismic)"
    echo "  phasenet_tf_plus   - Transformer + polarity + event (seismic)"
    echo "  phasenet_das       - Phase picking (DAS)"
    echo "  phasenet_das_plus  - Phase + event (DAS)"
    echo ""
    echo "Special commands:"
    echo "  all                - Train all models"
    echo "  all_seismic        - Train all seismic models"
    echo "  all_das            - Train all DAS models"
    echo ""
    echo "Environment:"
    echo "  Detected GPUs: $NUM_GPU"
}

main() {
    local model="${1:-help}"

    case "$model" in
        phasenet)
            train_phasenet
            ;;
        phasenet_plus)
            train_phasenet_plus
            ;;
        phasenet_tf)
            train_phasenet_tf
            ;;
        phasenet_tf_plus)
            train_phasenet_tf_plus
            ;;
        phasenet_das)
            train_phasenet_das
            ;;
        phasenet_das_plus)
            train_phasenet_das_plus
            ;;
        all)
            train_all
            ;;
        all_seismic)
            train_all_seismic
            ;;
        all_das)
            train_all_das
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown model: $model"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
