#!/bin/bash
#SBATCH --account=pc_montereydas
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H100:4
#SBATCH --time=24:00:00
#SBATCH --job-name=phasenet
#SBATCH --output=phasenet_%j.out
#SBATCH --error=phasenet_%j.err

# Load any necessary modules here if needed
# module load ...

# Activate your conda environment if you're using one
# source activate your_env_name

# Run the training code
torchrun --nproc_per_node 4 train.py \
    --model phasenet_prompt \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --workers 16 \
    --batch-size 4 \
    --output-dir unet10
