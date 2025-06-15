#! /bin/bash
set -x

torchrun --master_port 29510  --nproc_per_node 4 train.py --model phasenet  --backbone xunet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_xunet --lr 0.0003 --wandb --wandb-project phasenet --wandb-name phasenet_xunet
torchrun --master_port 29511  --nproc_per_node 4 train.py --model phasenet  --backbone unet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_unet --lr 0.001 --wandb --wandb-project phasenet --wandb-name phasenet_unet

torchrun --master_port 29512  --nproc_per_node 4 train.py --model phasenet_tf  --backbone xunet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_tf_xunet --lr 0.0003 --wandb --wandb-project phasenet --wandb-name phasenet_tf_xunet
torchrun --master_port 29513  --nproc_per_node 4 train.py --model phasenet_tf  --backbone unet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_tf_unet --lr 0.001 --wandb --wandb-project phasenet --wandb-name phasenet_tf_unet

torchrun --master_port 29514  --nproc_per_node 4 train.py --model phasenet_plus  --backbone xunet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_plus_xunet --lr 0.0003 --wandb --wandb-project phasenet --wandb-name phasenet_plus_xunet
torchrun --master_port 29515  --nproc_per_node 4 train.py --model phasenet_plus  --backbone unet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_plus_unet --lr 0.001 --wandb --wandb-project phasenet --wandb-name phasenet_plus_unet

torchrun --master_port 29514  --nproc_per_node 4 train.py --model phasenet_plus_tf  --backbone xunet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_plus_xunet --lr 0.0003 --wandb --wandb-project phasenet --wandb-name phasenet_plus_tf_xunet
torchrun --master_port 29515  --nproc_per_node 4 train.py --model phasenet_plus_tf  --backbone unet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 128 --output-dir phasenet_plus_unet --lr 0.001 --wandb --wandb-project phasenet --wandb-name phasenet_plus_tf_unet

torchrun --master_port 29516  --nproc_per_node 4 train.py --model phasenet_prompt  --backbone xunet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 16 --output-dir phasenet_prompt_xunet --lr 0.0003 --wandb --wandb-project phasenet --wandb-name phasenet_prompt_xunet
torchrun --master_port 29517  --nproc_per_node 4 train.py --model phasenet_prompt  --backbone unet --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 --workers 16 --batch-size 16 --output-dir phasenet_prompt_unet --lr 0.001 --wandb --wandb-project phasenet --wandb-name phasenet_prompt_unet