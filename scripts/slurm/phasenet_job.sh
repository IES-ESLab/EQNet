#!/bin/bash
#SBATCH --job-name=phasenet
#SBATCH --output=phasenet_%j.out
#SBATCH --error=phasenet_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --account=pc_montereydas
#SBATCH --qos=lr_normal

# Load required modules
# module load python/3.9

# Activate conda environment
source /global/home/users/zhuwq0/.local/miniconda3/etc/profile.d/conda.sh
conda activate phasenet


# Set paths
CWP="/global/home/users/zhuwq0/scratch/EQNet/scripts"
MODEL_PATH="/global/scratch/users/zhuwq0/PhaseNet"
FOLDER="$1"
CHUNK_ID="$2"
NUM_NODES="$3"

CHUNK_FILE="${CWP}/results/phasenet/${FOLDER}/splits/chunk_${CHUNK_ID}_${NUM_NODES}.txt"
RESULT_DIR="${CWP}/results/phasenet/${FOLDER}"


# Run PhaseNet
echo "python ${MODEL_PATH}/phasenet/predict.py \
    --model=${MODEL_PATH}/model/190703-214543 \
    --format das_event \
    --data_list ${CHUNK_FILE} \
    --batch_size 1 \
    --result_dir ${RESULT_DIR} \
    --subdir_level=0"
python ${MODEL_PATH}/phasenet/predict.py \
    --model=${MODEL_PATH}/model/190703-214543 \
    --format das_event \
    --data_list ${CHUNK_FILE} \
    --batch_size 1 \
    --result_dir ${RESULT_DIR} \
    --subdir_level=0 