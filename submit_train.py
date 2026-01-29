#!/usr/bin/env python
"""
SkyPilot Training Submission Script for EQNet.

Uses SkyPilot Python SDK for programmatic job submission.

Usage:
    # Direct launch (interactive)
    python submit_train.py --model phasenet_plus --launch

    # Managed jobs launch (fault-tolerant)
    python submit_train.py --model phasenet_das --jobs-launch

    # Launch all seismic models
    python submit_train.py --all-seismic --jobs-launch

Examples:
    # Train PhaseNet-DAS with 4 V100 GPUs
    python submit_train.py --model phasenet_das --gpus V100:4 --jobs-launch

    # Train on AWS with spot instances
    python submit_train.py --model phasenet --cloud aws --use-spot --jobs-launch
"""
import argparse
import os
import sys
from typing import Optional

import sky


# =============================================================================
# Configuration
# =============================================================================

MODEL_CONFIGS = {
    # Seismic models (3-component)
    "phasenet": {
        "type": "seismic",
        "dataset": "ceed",
        "batch_size": 256,
        "epochs": 100,
        "features": "phase picking",
    },
    "phasenet_plus": {
        "type": "seismic",
        "dataset": "ceed",
        "batch_size": 256,
        "epochs": 100,
        "features": "phase + polarity + event detection",
    },
    "phasenet_tf": {
        "type": "seismic",
        "dataset": "ceed",
        "batch_size": 128,
        "epochs": 100,
        "features": "phase picking with STFT (time-frequency)",
    },
    "phasenet_tf_plus": {
        "type": "seismic",
        "dataset": "ceed",
        "batch_size": 128,
        "epochs": 100,
        "features": "STFT + polarity + event detection",
    },
    # DAS models (single-channel strain rate)
    "phasenet_das": {
        "type": "das",
        "dataset": "quakeflow_das",
        "batch_size": 4,
        "epochs": 10,
        "features": "DAS phase picking",
    },
    "phasenet_das_plus": {
        "type": "das",
        "dataset": "quakeflow_das",
        "batch_size": 4,
        "epochs": 10,
        "features": "DAS phase + event detection",
    },
}

DEFAULT_GPUS = {"seismic": "V100:4", "das": "V100:4"}


# =============================================================================
# Task Builder
# =============================================================================


def build_task(
    model: str,
    cloud: Optional[str] = None,
    region: Optional[str] = None,
    gpus: Optional[str] = None,
    cpus: int = 64,
    disk_size: int = 300,
    use_spot: bool = True,
    num_nodes: int = 1,
    wandb_project: str = "eqnet",
    output_bucket: str = "gs://quakeflow_model",
    training_path: str = "results/training_v1",
) -> sky.Task:
    """Build a SkyPilot Task for training."""
    config = MODEL_CONFIGS.get(model)
    if not config:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_CONFIGS.keys())}")

    model_type = config["type"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    if gpus is None:
        gpus = DEFAULT_GPUS.get(model_type, "V100:4")

    # Build run command
    if model_type == "seismic":
        run_cmd = _build_seismic_run_cmd(model, batch_size, epochs)
    else:
        run_cmd = _build_das_run_cmd(model, batch_size, epochs)

    # Setup command - workdir is synced, so just install dependencies
    setup_cmd = """\
echo "Setting up EQNet training environment..."

pip install -r requirements.txt

echo "Setup complete."
"""

    # Create task
    task = sky.Task(
        name=f"eqnet-{model.replace('_', '-')}",
        setup=setup_cmd,
        run=run_cmd,
        workdir=".",
        num_nodes=num_nodes,
        envs={
            "MODEL_NAME": model,
            "WANDB_PROJECT": wandb_project,
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        },
    )

    # Set resources
    resources = sky.Resources(
        cloud=sky.clouds.CLOUD_REGISTRY.from_str(cloud) if cloud else None,
        region=region,
        accelerators=gpus,
        cpus=f"{cpus}+",
        disk_size=disk_size,
        disk_tier="high",
        use_spot=use_spot,
    )
    task.set_resources(resources)

    # Storage mounts (cloud storage -> remote paths)
    storage_mounts = {
        "/checkpoint": sky.Storage(source=output_bucket, mode=sky.StorageMode.MOUNT),
    }

    # CEED (seismic) loads directly from GCS via HuggingFace datasets - no mounts needed
    # DAS needs training list files (data.txt, labels_train.txt, etc.)
    if model_type == "das":
        storage_mounts["/training"] = sky.Storage(
            source=f"gs://quakeflow_das/{training_path}",
            mode=sky.StorageMode.COPY,
        )

    task.set_storage_mounts(storage_mounts)

    return task


def _build_seismic_run_cmd(model: str, batch_size: int, epochs: int) -> str:
    """Build run command for seismic models.

    Uses CEED dataset which loads directly from GCS via HuggingFace datasets.
    """
    return f"""\
echo "Starting training for $MODEL_NAME..."

num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
nproc_per_node=${{SKYPILOT_NUM_GPUS_PER_NODE}}

if [ "${{SKYPILOT_NODE_RANK}}" == "0" ]; then
    echo "Nodes: $num_nodes, GPUs per node: $nproc_per_node"
    nvidia-smi
fi

torchrun \\
    --nproc_per_node=${{nproc_per_node}} \\
    --node_rank=${{SKYPILOT_NODE_RANK}} \\
    --nnodes=$num_nodes \\
    --master_addr=$master_addr \\
    --master_port=8008 \\
    train.py \\
      --model $MODEL_NAME \\
      --dataset-type ceed \\
      --ceed-region NC,SC \\
      --streaming \\
      --iters-per-epoch 1000 \\
      --batch-size {batch_size} \\
      --epochs {epochs} \\
      --workers 12 \\
      --stack-event \\
      --stack-noise \\
      --flip-polarity \\
      --drop-channel \\
      --sync-bn \\
      --output-dir /checkpoint/$MODEL_NAME \\
      ${{WANDB_API_KEY:+--wandb --wandb-project $WANDB_PROJECT --wandb-name $MODEL_NAME}} \\
      --resume
"""


def _build_das_run_cmd(model: str, batch_size: int, epochs: int) -> str:
    """Build run command for DAS models."""
    return f"""\
echo "Starting training for $MODEL_NAME..."

num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`
master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
nproc_per_node=${{SKYPILOT_NUM_GPUS_PER_NODE}}

if [ "${{SKYPILOT_NODE_RANK}}" == "0" ]; then
    echo "Nodes: $num_nodes, GPUs per node: $nproc_per_node"
    nvidia-smi
fi

torchrun \\
    --nproc_per_node=${{nproc_per_node}} \\
    --node_rank=${{SKYPILOT_NODE_RANK}} \\
    --nnodes=$num_nodes \\
    --master_addr=$master_addr \\
    --master_port=8008 \\
    train.py \\
      --model $MODEL_NAME \\
      --batch-size {batch_size} \\
      --epochs {epochs} \\
      --data-path gs://quakeflow_das \\
      --label-path gs://quakeflow_das \\
      --label-list /training/labels_train.txt \\
      --noise-list /training/noise_train.txt \\
      --test-data-path gs://quakeflow_das \\
      --test-label-path gs://quakeflow_das \\
      --test-label-list /training/labels_test.txt \\
      --test-noise-list /training/noise_test.txt \\
      --nt 3072 \\
      --nx 5120 \\
      --weight-decay 1e-1 \\
      --workers 8 \\
      --stack-event \\
      --stack-noise \\
      --resample-space \\
      --resample-time \\
      --masking \\
      --random-crop \\
      --sync-bn \\
      --output-dir /checkpoint/$MODEL_NAME \\
      ${{WANDB_API_KEY:+--wandb --wandb-project $WANDB_PROJECT --wandb-name $MODEL_NAME}} \\
      --resume
"""


# =============================================================================
# CLI
# =============================================================================


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Submit EQNet training jobs to SkyPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()), help="Model to train")
    model_group.add_argument("--all-seismic", action="store_true", help="Train all seismic models")
    model_group.add_argument("--all-das", action="store_true", help="Train all DAS models")
    model_group.add_argument("--all", action="store_true", help="Train all models")

    # Launch mode
    launch_group = parser.add_mutually_exclusive_group()
    launch_group.add_argument("--launch", action="store_true", help="sky.launch() - direct launch")
    launch_group.add_argument("--jobs-launch", action="store_true", help="sky.jobs.launch() - managed jobs")
    launch_group.add_argument("--dry-run", action="store_true", help="Print task info without launching")

    # Cloud configuration
    parser.add_argument("--cloud", type=str, default=None, help="Cloud provider (gcp, aws, azure)")
    parser.add_argument("--region", type=str, default="us-west1", help="Cloud region (default: us-west1/Oregon)")
    parser.add_argument("--gpus", type=str, default=None, help="GPU type:count (e.g., V100:4, A100:8)")
    parser.add_argument("--cpus", type=int, default=32, help="Minimum CPUs (default: 32)")
    parser.add_argument("--disk-size", type=int, default=300, help="Disk size in GB (default: 300)")
    parser.add_argument("--use-spot", action="store_true", default=True, help="Use spot instances (default)")
    parser.add_argument("--no-spot", action="store_true", help="Don't use spot instances")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes (default: 1)")

    # Wandb configuration
    parser.add_argument("--wandb-project", type=str, default="eqnet", help="Wandb project name")

    # Output configuration
    parser.add_argument("--output-bucket", type=str, default="gs://quakeflow_model", help="GCS bucket for checkpoints")
    parser.add_argument("--training-path", type=str, default="results/training_v1", help="DAS training data path")

    # Launch options
    parser.add_argument("--cluster-name", type=str, default=None, help="Cluster name for direct launch")
    parser.add_argument("--job-name", type=str, default=None, help="Job name for jobs launch")
    parser.add_argument("--retry-until-up", action="store_true", help="Retry until cluster is up")
    parser.add_argument("--down", action="store_true", help="Tear down cluster after job finishes")

    return parser


def main():
    args = get_args_parser().parse_args()

    use_spot = args.use_spot and not args.no_spot

    # Get models to train
    if args.model:
        models = [args.model]
    elif args.all_seismic:
        models = [m for m, c in MODEL_CONFIGS.items() if c["type"] == "seismic"]
    elif args.all_das:
        models = [m for m, c in MODEL_CONFIGS.items() if c["type"] == "das"]
    elif args.all:
        models = list(MODEL_CONFIGS.keys())
    else:
        print("Error: Must specify --model, --all-seismic, --all-das, or --all")
        sys.exit(1)

    print(f"Models: {models}")
    print(f"Cloud: {args.cloud or 'auto'}, GPUs: {args.gpus or 'default'}, Spot: {use_spot}")
    wandb_key = os.environ.get("WANDB_API_KEY")
    print(f"Wandb: {'enabled' if wandb_key else 'disabled (WANDB_API_KEY not set)'}")

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model} ({MODEL_CONFIGS[model]['features']})")
        print(f"{'='*60}")

        task = build_task(
            model=model,
            cloud=args.cloud,
            region=args.region,
            gpus=args.gpus,
            cpus=args.cpus,
            disk_size=args.disk_size,
            use_spot=use_spot,
            num_nodes=args.num_nodes,
            wandb_project=args.wandb_project,
            output_bucket=args.output_bucket,
            training_path=args.training_path,
        )

        if args.dry_run:
            print(f"Task: {task.name}")
            print(f"Resources: {task.resources}")
            print(f"Num nodes: {task.num_nodes}")
            continue

        cluster_name = args.cluster_name or f"eqnet-{model.replace('_', '-')}"
        job_name = args.job_name or f"eqnet-{model.replace('_', '-')}"

        if args.jobs_launch:
            print(f"Launching managed job: {job_name}")
            request_id = sky.jobs.launch(task, name=job_name)
            print(f"Request ID: {request_id}")
            # Stream logs until job finishes
            sky.stream_and_get(request_id)
            sky.jobs.tail_logs(name=job_name, follow=True)
        elif args.launch:
            print(f"Launching cluster: {cluster_name}")
            request_id = sky.launch(
                task,
                cluster_name=cluster_name,
                retry_until_up=args.retry_until_up,
                down=args.down,
            )
            print(f"Request ID: {request_id}")
            # Stream logs until job finishes
            job_id, handle = sky.stream_and_get(request_id)
            sky.tail_logs(cluster_name, job_id, follow=True)
        else:
            print("No launch mode specified. Use --launch or --jobs-launch")

    print("\nDone!")


if __name__ == "__main__":
    main()
