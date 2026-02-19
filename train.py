"""
Training script for EQNet models.

Follows best practices from:
- Karpathy's nanoGPT: Clean, minimal, educational
- timm: Registry patterns, good defaults
- torchvision: Reference training scripts, distributed utilities

Usage:
    python train.py --model phasenet --hf-dataset AI4EPS/quakeflow_nc
    python train.py --model phasenet_plus --data-path /path/to/data
"""
import datetime
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib
matplotlib.use("agg")

import eqnet
import eqnet.utils as eqnet_utils
import utils
try:
    import wandb
except ImportError:
    wandb = None
from eqnet.data import SeismicTraceIterableDataset, CEEDDataset, CEEDIterableDataset, DASIterableDataset
from eqnet.data.ceed import default_train_transforms as ceed_train_transforms, default_eval_transforms as ceed_eval_transforms
from eqnet.data.das import default_train_transforms as das_train_transforms, default_eval_transforms as das_eval_transforms
from eqnet.models.unet import moving_normalize

# DAS model names for dataset selection
DAS_MODELS = {"phasenet_das", "phasenet_das_plus"}

logger = logging.getLogger("EQNet")

# =============================================================================
# Configuration
# =============================================================================

# Model feature registry - which losses each model uses
MODEL_FEATURES = {
    "phasenet": {"phase"},
    "phasenet_tf": {"phase"},
    "phasenet_plus": {"phase", "event", "polarity"},
    "phasenet_tf_plus": {"phase", "event", "polarity"},
    "phasenet_prompt": {"phase", "event", "polarity", "prompt"},
    "phasenet_das": {"phase"},
    "phasenet_das_plus": {"phase", "event"},
}


def get_model_features(model_name: str) -> set:
    """Get features supported by a model."""
    return MODEL_FEATURES.get(model_name, {"phase"})


# =============================================================================
# Learning Rate Schedule (Karpathy style)
# =============================================================================

def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return (max_lr - min_lr) * step / max(1, warmup_steps) + min_lr
    if step >= max_steps or max_steps <= warmup_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: torch.utils.data.DataLoader,
    model_ema: Optional[utils.ExponentialMovingAverage],
    scaler: Optional[torch.cuda.amp.GradScaler],
    args,
    epoch: int,
    total_samples: int,
):
    """Train for one epoch."""
    features = get_model_features(args.model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))

    # Add loss meters based on model features
    for feat in features:
        if feat == "phase":
            metric_logger.add_meter("loss_phase", utils.SmoothedValue(window_size=1, fmt="{value}"))
        elif feat == "event":
            metric_logger.add_meter("loss_event_center", utils.SmoothedValue(window_size=1, fmt="{value}"))
            metric_logger.add_meter("loss_event_time", utils.SmoothedValue(window_size=1, fmt="{value}"))
        elif feat == "polarity":
            metric_logger.add_meter("loss_polarity", utils.SmoothedValue(window_size=1, fmt="{value}"))
        elif feat == "prompt":
            metric_logger.add_meter("loss_prompt", utils.SmoothedValue(window_size=1, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    ctx = (
        nullcontext()
        if args.device in ["cpu", "mps"]
        else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    )

    model.train()
    processed_samples = 0
    last_batch, last_output = None, None

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if args.iters_per_epoch and i >= args.iters_per_epoch:
            break
        optimizer.zero_grad()

        with ctx:
            output = model(batch)
        loss = output["loss"]
        last_batch, last_output = batch, output

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        lr_scheduler.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        batch_size = batch["data"].shape[0]
        processed_samples += batch_size

        # Update metrics
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        for feat in features:
            if feat == "phase" and "loss_phase" in output:
                metric_logger.update(loss_phase=output["loss_phase"].item())
            elif feat == "event":
                if "loss_event_center" in output:
                    metric_logger.update(loss_event_center=output["loss_event_center"].item())
                if "loss_event_time" in output:
                    metric_logger.update(loss_event_time=output["loss_event_time"].item())
            elif feat == "polarity" and "loss_polarity" in output:
                metric_logger.update(loss_polarity=output["loss_polarity"].item())
            elif feat == "prompt" and "loss_prompt" in output:
                metric_logger.update(loss_prompt=output["loss_prompt"].item())

        # Wandb logging
        if args.wandb and utils.is_main_process():
            log = {
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/step": i,
            }
            for key in ["loss_phase", "loss_event_center", "loss_event_time", "loss_polarity", "loss_prompt"]:
                if key in output:
                    log[f"train/{key}"] = output[key].item()
            wandb.log(log)

        if processed_samples >= total_samples:
            break

        # Periodic checkpoint
        if (i + 1) % 1000 == 0:
            utils.save_on_master(model.state_dict(), os.path.join(args.output_dir, "model_tmp.pth"))

    # Plot training results
    plot_results(last_batch, last_output, args, epoch, "train")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    args,
    epoch: int = 0,
    total_samples: int = 1,
):
    """Evaluate the model."""
    features = get_model_features(args.model)
    metric_logger = utils.MetricLogger(delimiter="  ")

    for feat in features:
        if feat == "phase":
            metric_logger.add_meter("loss_phase", utils.SmoothedValue(window_size=1, fmt="{value}"))
        elif feat == "event":
            metric_logger.add_meter("loss_event_center", utils.SmoothedValue(window_size=1, fmt="{value}"))
            metric_logger.add_meter("loss_event_time", utils.SmoothedValue(window_size=1, fmt="{value}"))
        elif feat == "polarity":
            metric_logger.add_meter("loss_polarity", utils.SmoothedValue(window_size=1, fmt="{value}"))
        elif feat == "prompt":
            metric_logger.add_meter("loss_prompt", utils.SmoothedValue(window_size=1, fmt="{value}"))

    model.eval()
    processed_samples = 0
    last_batch, last_output = None, None

    for batch in metric_logger.log_every(data_loader, args.print_freq, "Test:"):
        output = model(batch)
        last_batch, last_output = batch, output
        batch_size = batch["data"].shape[0]

        metric_logger.meters["loss"].update(output["loss"].item(), n=batch_size)
        for feat in features:
            if feat == "phase" and "loss_phase" in output:
                metric_logger.meters["loss_phase"].update(output["loss_phase"].item(), n=batch_size)
            elif feat == "event":
                if "loss_event_center" in output:
                    metric_logger.meters["loss_event_center"].update(output["loss_event_center"].item(), n=batch_size)
                if "loss_event_time" in output:
                    metric_logger.meters["loss_event_time"].update(output["loss_event_time"].item(), n=batch_size)
            elif feat == "polarity" and "loss_polarity" in output:
                metric_logger.meters["loss_polarity"].update(output["loss_polarity"].item(), n=batch_size)
            elif feat == "prompt" and "loss_prompt" in output:
                metric_logger.meters["loss_prompt"].update(output["loss_prompt"].item(), n=batch_size)

        processed_samples += batch_size
        if processed_samples > total_samples:
            break

    metric_logger.synchronize_between_processes()
    print(f"Test loss = {metric_logger.loss.global_avg:.3e}")

    if args.wandb and utils.is_main_process():
        log = {"test/loss": metric_logger.loss.global_avg, "test/epoch": epoch}
        for key in ["loss_phase", "loss_event_center", "loss_event_time", "loss_polarity", "loss_prompt"]:
            if hasattr(metric_logger, key):
                log[f"test/{key}"] = getattr(metric_logger, key).global_avg
        wandb.log(log)

    plot_results(last_batch, last_output, args, epoch, "test")
    return metric_logger


def plot_results(batch, output, args, epoch, prefix=""):
    """Plot training/evaluation results."""
    if batch is None or output is None:
        return

    with torch.inference_mode():
        if "phase" not in output:
            return

        phase = torch.softmax(output["phase"], dim=1).cpu().float()
        batch["raw_data"] = batch["data"]
        batch["data"] = moving_normalize(batch["data"])

        # DAS models
        if args.model == "phasenet_das":
            eqnet_utils.plot_phasenet_das_train(
                batch, phase, epoch=epoch, figure_dir=args.figure_dir, prefix=prefix,
            )
        elif args.model == "phasenet_das_plus":
            event_center = torch.sigmoid(output["event_center"]).cpu().float() if "event_center" in output else None
            event_time = output["event_time"].cpu().float() if "event_time" in output else None
            eqnet_utils.plot_phasenet_das_plus_train(
                batch, phase, event_center=event_center, event_time=event_time,
                epoch=epoch, figure_dir=args.figure_dir, prefix=prefix,
            )
        # Seismic models with polarity
        elif args.model in ("phasenet_plus", "phasenet_tf_plus"):
            polarity = torch.sigmoid(output["polarity"]).cpu().float() if "polarity" in output else None
            event_center = torch.sigmoid(output["event_center"]).cpu().float() if "event_center" in output else None
            event_time = output["event_time"].cpu().float() if "event_time" in output else None
            eqnet_utils.plot_phasenet_plus_train(
                batch, phase, polarity=polarity, event_center=event_center,
                event_time=event_time, epoch=epoch, figure_dir=args.figure_dir, prefix=prefix,
            )
        # Seismic models with spectrogram
        elif args.model in ("phasenet_tf",):
            batch["spectrogram"] = output["spectrogram"].cpu().float() if "spectrogram" in output else None
            eqnet_utils.plot_phasenet_tf_train(
                batch, phase, event_center=None, event_time=None,
                epoch=epoch, figure_dir=args.figure_dir, prefix=prefix,
            )
        # Base phasenet
        else:
            eqnet_utils.plot_phasenet_train(
                batch, phase, epoch=epoch, figure_dir=args.figure_dir, prefix=prefix,
            )
        print("Plotting...")


# =============================================================================
# Dataset Factory
# =============================================================================

def get_dataset_type(args) -> str:
    """Determine dataset type based on args and model."""
    if args.dataset_type != "auto":
        return args.dataset_type
    # Auto-detect: DAS models use DAS dataset
    if args.model in DAS_MODELS:
        return "das"
    # Default to seismic_trace for other models
    return "seismic_trace"


def create_dataset(args, training: bool = True, rank: int = 0, world_size: int = 1):
    """Create dataset based on configuration."""
    dataset_type = get_dataset_type(args)

    # CEED dataset (seismic, 3-component)
    if dataset_type == "ceed":
        transforms = ceed_train_transforms(crop_length=4096, augment=not args.no_augment) if training else ceed_eval_transforms(crop_length=4096)
        # group_by controls data reading: "station" (nx=1) or "event" (nx=N)
        # target_nx only used with group_by="event" for consistent batching
        group_by = "station" if args.nx == 1 else "event"
        target_nx = args.nx if group_by == "event" else None
        overfit = args.overfit if training else None
        if args.streaming or overfit:
            return CEEDIterableDataset(
                region=args.ceed_region,
                years=args.ceed_years,
                days=args.ceed_days,
                data_files=args.ceed_data_files,
                transforms=transforms,
                min_snr=args.min_snr,
                buffer_size=args.buffer_size,
                shuffle_buffer_size=args.buffer_size,
                group_by=group_by,
                target_nx=target_nx,
                overfit=overfit,
            )
        else:
            return CEEDDataset(
                region=args.ceed_region,
                years=args.ceed_years,
                days=args.ceed_days,
                data_files=args.ceed_data_files,
                transforms=transforms,
                min_snr=args.min_snr,
                group_by=group_by,
                target_nx=target_nx,
            )

    # DAS dataset (single-channel strain rate)
    if dataset_type == "das":
        if training:
            transforms = das_train_transforms(
                nt=args.nt, nx=args.nx,
                enable_stacking=args.stack_event,
                enable_noise_stacking=args.stack_noise,
                enable_resample_time=args.resample_time,
                enable_resample_space=args.resample_space,
                enable_masking=args.masking,
                augment=not args.no_augment,
            )
        else:
            transforms = das_eval_transforms()

        return DASIterableDataset(
            data_path=args.data_path if training else (args.test_data_path or args.data_path),
            data_list=args.data_list if training else args.test_data_list,
            label_path=args.label_path if training else (args.test_label_path or args.label_path),
            label_list=args.label_list if training else args.test_label_list,
            noise_list=args.noise_list if training else args.test_noise_list,
            phases=args.phases,
            nt=args.nt,
            nx=args.nx,
            min_nt=min(args.nt, 256),
            min_nx=min(args.nx, 256),
            format=args.format,
            training=training,
            transforms=transforms,
            overfit=args.overfit if training else None,
            rank=rank,
            world_size=world_size,
        )

    # Default: SeismicTraceIterableDataset (seismic, 3-component)
    return SeismicTraceIterableDataset(
        data_path=args.data_path if training else args.test_data_path,
        data_list=args.data_list if training else args.test_data_list,
        hdf5_file=args.hdf5_file if training else args.test_hdf5_file,
        hf_dataset=args.hf_dataset,
        format=args.format,
        training=training,
        stack_event=args.stack_event if training else False,
        picks_dict=args.picks_dict,
        events_dict=args.events_dict,
        stack_noise=args.stack_noise if training else False,
        flip_polarity=args.flip_polarity if training else False,
        drop_channel=args.drop_channel if training else False,
        rank=rank,
        world_size=world_size,
    )


# =============================================================================
# Main Training Loop
# =============================================================================

def main(args):
    # Setup output directory
    if args.output_dir:
        utils.mkdir(args.output_dir)
        args.figure_dir = os.path.join(args.output_dir, "figures")
        utils.mkdir(args.figure_dir)

    utils.init_distributed_mode(args)
    print(args)

    # Distributed setup
    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1

    # Set seeds for reproducibility
    seed = 1337 + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(args.device)

    # Mixed precision setup
    dtype = "bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16" and torch.cuda.is_available()))
    args.dtype, args.ptdtype = dtype, ptdtype

    # CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not args.use_deterministic_algorithms
    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Create datasets
    dataset = create_dataset(args, training=True, rank=rank, world_size=world_size)
    # Test dataset: check for test data path/list depending on dataset type
    has_test_data = (
        args.test_hdf5_file
        or args.test_data_path
        or args.test_data_list
        or args.test_label_list
    )
    dataset_test = create_dataset(args, training=False, rank=rank, world_size=world_size) if has_test_data else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False,
    ) if dataset_test else None

    # Build model
    model_kwargs = {
        "log_scale": args.log_scale,
        "moving_norm": (1024, 256) if args.moving_norm else None,
    }
    model = eqnet.models.__dict__[args.model].build_model(backbone=args.backbone, **model_kwargs)
    logger.info(f"Model:\n{model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.to(device)

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wandb logging
    if args.wandb and utils.is_main_process():
        wandb.init(
            project=args.wandb_project or args.model,
            name=args.wandb_name,
            entity=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
        )

    # Optimizer and scheduler
    parameters = utils.set_weight_decay(
        model, args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
    )
    optimizer = torch.optim.AdamW(parameters, lr=1.0, weight_decay=args.weight_decay)

    try:
        iters_per_epoch = max(1, len(data_loader))
        # Allow --iters-per-epoch to override (useful for debugging)
        if args.iters_per_epoch:
            iters_per_epoch = args.iters_per_epoch
    except TypeError:
        # IterableDataset doesn't have __len__, use fallback
        iters_per_epoch = args.iters_per_epoch
    warmup_steps = args.lr_warmup_epochs * iters_per_epoch
    max_steps = args.epochs * iters_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda it: get_lr(it, args.lr, args.lr * args.lr_min_ratio, warmup_steps, max_steps)
    )

    # Distributed wrapper
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # EMA
    model_ema = None
    if args.model_ema:
        adjust = world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = min(1.0, (1.0 - args.model_ema_decay) * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    # Resume from checkpoint
    if args.resume:
        checkpoint_path = args.checkpoint or os.path.join(args.output_dir, "checkpoint.pth")
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if model_ema and "model_ema" in checkpoint:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

    # Training loop
    start_time = time.time()
    best_loss = float("inf")

    # Get dataset sizes (handle IterableDataset which has no len)
    if args.iters_per_epoch:
        dataset_size = args.iters_per_epoch * args.batch_size
    else:
        try:
            dataset_size = len(dataset)
        except TypeError:
            dataset_size = 1000 * args.batch_size  # fallback
    try:
        dataset_test_size = len(dataset_test) if dataset_test else 0
    except TypeError:
        dataset_test_size = args.iters_per_epoch * args.batch_size if args.iters_per_epoch else 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            model, optimizer, lr_scheduler, data_loader, model_ema,
            scaler, args, epoch, dataset_size,
        )

        metric = None
        if data_loader_test:
            metric = evaluate(model, data_loader_test, args, epoch, dataset_test_size)
            if model_ema:
                evaluate(model_ema, data_loader_test, args, epoch, dataset_test_size)

        # Save checkpoint
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if model_ema:
            checkpoint["model_ema"] = model_ema.state_dict()
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()

        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # Save best model
        if utils.is_main_process():
            current_loss = metric.loss.global_avg if metric else 0
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(checkpoint, os.path.join(args.output_dir, "model_best.pth"))

                if args.wandb:
                    artifact = wandb.Artifact(args.wandb_name or "model", type="model", metadata={"epoch": epoch, "loss": best_loss})
                    artifact.add_file(os.path.join(args.output_dir, "model_best.pth"))
                    wandb.log_artifact(artifact)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training completed in {total_time}")

    if args.wandb and utils.is_main_process():
        wandb.finish()


# =============================================================================
# Argument Parser
# =============================================================================

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="EQNet Training", add_help=add_help)

    # Data
    parser.add_argument("--data-path", default="./", type=str)
    parser.add_argument("--data-list", nargs="+", default=None)
    parser.add_argument("--hdf5-file", default=None, type=str)
    parser.add_argument("--hf-dataset", default=None, type=str)
    parser.add_argument("--format", default="h5", type=str, choices=["h5", "hf"])
    parser.add_argument("--test-data-path", default=None, type=str)
    parser.add_argument("--test-data-list", nargs="+", default=None)
    parser.add_argument("--test-hdf5-file", default=None, type=str)
    parser.add_argument("--picks-dict", default=None, type=str)
    parser.add_argument("--events-dict", default=None, type=str)

    # Dataset type (auto-detected for DAS models)
    parser.add_argument("--dataset-type", default="auto", choices=["auto", "seismic_trace", "ceed", "das"])

    # CEED dataset
    parser.add_argument("--ceed-region", default="SC", type=str)
    parser.add_argument("--ceed-years", nargs="+", type=int, default=None)
    parser.add_argument("--ceed-days", nargs="+", type=int, default=None)
    parser.add_argument("--ceed-data-files", nargs="+", default=None, help="Local parquet files (overrides region/years/days)")
    parser.add_argument("--min-snr", type=float, default=0.0)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--iters-per-epoch", type=int, default=1000, help="Iterations per epoch for streaming datasets")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Buffer size for sample stacking (use 1 for fast debugging)")

    # DAS dataset
    parser.add_argument("--label-path", default="./", type=str)
    parser.add_argument("--label-list", nargs="+", default=None)
    parser.add_argument("--noise-list", nargs="+", default=None)
    parser.add_argument("--test-label-path", default=None, type=str)
    parser.add_argument("--test-label-list", nargs="+", default=None)
    parser.add_argument("--test-noise-list", nargs="+", default=None)
    parser.add_argument("--phases", nargs="+", default=["P", "S"], type=str)
    parser.add_argument("--nt", default=3072, type=int, help="DAS time samples")
    parser.add_argument("--nx", default=5120, type=int, help="DAS space samples")
    parser.add_argument("--resample-space", action="store_true")
    parser.add_argument("--resample-time", action="store_true")
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--random-crop", action="store_true")

    # Model
    parser.add_argument("--model", default="phasenet", type=str)
    parser.add_argument("--backbone", default="unet", type=str)

    # Training
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)

    # Optimizer
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--norm-weight-decay", default=None, type=float)
    parser.add_argument("--clip-grad-norm", default=None, type=float)
    parser.add_argument("--lr-warmup-epochs", default=1, type=int)
    parser.add_argument("--lr-min-ratio", default=0.1, type=float)

    # Checkpointing
    parser.add_argument("--output-dir", default="./output", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--start-epoch", default=0, type=int)

    # EMA
    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--model-ema-steps", type=int, default=32)
    parser.add_argument("--model-ema-decay", type=float, default=0.99998)

    # Distributed
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://", type=str)
    parser.add_argument("--sync-bn", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true", help="Disable all augmentations (for overfit testing)")
    parser.add_argument("--overfit", default=None, choices=["fixed", "random"], help="Overfit mode: 'fixed' (same crop every step) or 'random' (different crops of cached data)")
    parser.add_argument("--log-scale", action="store_true", help="Enable log transform in model")
    parser.add_argument("--moving-norm", action="store_true", help="Enable moving normalization in model")
    parser.add_argument("--stack-noise", action="store_true")
    parser.add_argument("--stack-event", action="store_true")
    parser.add_argument("--flip-polarity", action="store_true")
    parser.add_argument("--drop-channel", action="store_true")

    # Logging
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-name", default=None, type=str)
    parser.add_argument("--wandb-group", default=None, type=str)
    parser.add_argument("--wandb-dir", default="./", type=str)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
