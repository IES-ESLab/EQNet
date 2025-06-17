#! /bin/bash
set -x

torchrun --master_port 29510  --nproc_per_node 4 train.py --model phasenet  --backbone xunet \
    --batch-size 128 --workers 16  --lr 0.0003 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_xunet \
    --wandb --wandb-project phasenet --wandb-name phasenet_xunet

torchrun --master_port 29511  --nproc_per_node 4 train.py --model phasenet  --backbone unet \
    --batch-size 128 --workers 16  --lr 0.001 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_unet \
    --wandb --wandb-project phasenet --wandb-name phasenet_unet

torchrun --master_port 29512  --nproc_per_node 4 train.py --model phasenet_tf  --backbone xunet \
    --batch-size 128 --workers 16  --lr 0.0003 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_tf_xunet \
    --wandb --wandb-project phasenet --wandb-name phasenet_tf_xunet

torchrun --master_port 29513  --nproc_per_node 4 train.py --model phasenet_tf  --backbone unet \
    --batch-size 128 --workers 16  --lr 0.001 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_tf_unet \
    --wandb --wandb-project phasenet --wandb-name phasenet_tf_unet

torchrun --master_port 29514  --nproc_per_node 4 train.py --model phasenet_plus  --backbone xunet \
    --batch-size 128 --workers 16  --lr 0.0003 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_plus_xunet \
    --wandb --wandb-project phasenet --wandb-name phasenet_plus_xunet

torchrun --master_port 29515  --nproc_per_node 4 train.py --model phasenet_plus  --backbone unet \
    --batch-size 128 --workers 16  --lr 0.001 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_plus_unet \
    --wandb --wandb-project phasenet --wandb-name phasenet_plus_unet

torchrun --master_port 29517  --nproc_per_node 4 train.py --model phasenet_tf_plus  --backbone xunet \
    --batch-size 128 --workers 16  --lr 0.0003 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_tf_plus_xunet \
    --wandb --wandb-project phasenet --wandb-name phasenet_tf_plus_xunet

torchrun --master_port 29516  --nproc_per_node 4 train.py --model phasenet_tf_plus  --backbone unet \
    --batch-size 128 --workers 16  --lr 0.001 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_tf_plus_unet \
    --wandb --wandb-project phasenet --wandb-name phasenet_tf_plus_unet 

torchrun --master_port 29518  --nproc_per_node 4 train.py --model phasenet_prompt  --backbone xunet \
    --workers 16 --batch-size 16 --lr 0.0003 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_prompt_xunet \
    --wandb --wandb-project phasenet --wandb-name phasenet_prompt_xunet

torchrun --master_port 29519  --nproc_per_node 4 train.py --model phasenet_prompt  --backbone unet \
    --workers 16 --batch-size 16 --lr 0.001 \
    --hdf5-file /global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5 \
    --output-dir phasenet_prompt_unet \
    --wandb --wandb-project phasenet --wandb-name phasenet_prompt_unet

torchrun --master_port 29520 --nproc_per_node=4 ../train.py --model phasenet_das --backbone unet \
    --epochs=10 --batch-size=1 --workers=16 --sync-bn --lr 0.001 \
    # --wd=1e-1 --stack-event --stack-noise --resample-space --resample-time --masking --random-crop \
    --output=phasenet_das_unet \
    --data-path /global/scratch/users/zhuwq0/quakeflow_das \
    --data-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/data.txt \
    --label-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/labels_train.txt \
    --noise-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/noise_train.txt \
    --test-data-path /global/scratch/users/zhuwq0/quakeflow_das \
    --test-data-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/data.txt \
    --test-label-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/labels_test.txt \
    --test-noise-list results/training_v0/noise_test.txt

torchrun --master_port 29521 --nproc_per_node=4 train.py --model phasenet_das_plus --backbone unet \
    --epochs=10 --batch-size=1 --workers=16 --sync-bn --lr 0.001 \
    # --wd=1e-1 --stack-event --stack-noise --resample-space --resample-time --masking --random-crop \
    --output=phasenet_das_plus_unet \
    --data-path /global/scratch/users/zhuwq0/quakeflow_das \
    --data-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/data.txt \
    --label-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/labels_train.txt \
    --noise-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/noise_train.txt \
    --test-data-path /global/scratch/users/zhuwq0/quakeflow_das \
    --test-data-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/data.txt \
    --test-label-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/labels_test.txt \
    --test-noise-list /global/home/users/zhuwq0/scratch/EQNet/scripts/results/training_v0/noise_test.txt