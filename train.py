import datetime
import logging
import os
import random
import time
from contextlib import nullcontext

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import wandb
import datasets

import eqnet
import utils
from eqnet.utils.station_sampler import StationSampler, cut_reorder_keys, create_groups
from eqnet.data import (
    AutoEncoderIterableDataset,
    DASDataset,
    DASIterableDataset,
    SeismicNetworkIterableDataset,
    SeismicTraceIterableDataset,
)
from eqnet.models.unet import moving_normalize

matplotlib.use("agg")
logger = logging.getLogger("EQNet")


def evaluate(model, data_loader, scaler, args, epoch=0, total_sample=1):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: "

    num_processed_samples = 0
    with torch.inference_mode():
        for meta in metric_logger.log_every(data_loader, args.print_freq, header):
            output = model(meta)
            loss = output["loss"]
            batch_size = meta["data"].shape[0]
            metric_logger.meters["loss"].update(loss.item(), n=batch_size)
            num_processed_samples += batch_size
            if num_processed_samples > total_sample:
                break

    plot_results(meta, model, output, args, epoch, "test_")
    del meta, output, loss

    metric_logger.synchronize_between_processes()
    print(f"Test loss = {metric_logger.loss.global_avg:.3e}")
    if args.wandb and utils.is_main_process():
        wandb.log({"test/test_loss": metric_logger.loss.global_avg, "test/epoch": epoch})

    return metric_logger


def train_one_epoch(
    model,
    optimizer,
    lr_scheduler,
    data_loader,
    model_ema,
    scaler,
    args,
    epoch,
    total_sample,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    if args.model == "phasenet":
        metric_logger.add_meter("loss_phase", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_event", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("loss_polarity", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    # ctx = nullcontext() if scaler is None else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    model.train()
    num_processed_samples = 0
    for i, meta in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        with ctx:
            output = model(meta)

        loss = output["loss"]
        optimizer.zero_grad()
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

        batch_size = meta["data"].shape[0]
        num_processed_samples += batch_size
        if num_processed_samples >= total_sample:
            break
        # break

        metric_logger.update(lr=optimizer.param_groups[0]["lr"], loss=loss.item())

        if args.wandb and utils.is_main_process():
            wandb.log(
                {
                    "train/train_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                    "train/batch": i,
                }
            )

    model.eval()
    plot_results(meta, model, output, args, epoch, "train_")
    del meta, output, loss


def plot_results(meta, model, output, args, epoch, prefix=""):
    with torch.inference_mode():
        if args.model == "phasenet":
            phase = torch.softmax(output["phase"], dim=1).cpu().float()
            event = torch.sigmoid(output["event"]).cpu().float()
            polarity = torch.sigmoid(output["polarity"]).cpu().float()
            # meta["raw"] = meta["data"].clone()
            meta["data"] = moving_normalize(meta["data"])
            print("Plotting...")
            eqnet.utils.visualize_phasenet_train(meta, phase, event, polarity, epoch=epoch, figure_dir=args.figure_dir)
            del output, phase, event, polarity

        if args.model == "deepdenoiser":
            pass

        elif args.model == "phasenet_das":
            phase = torch.softmax(output["phase"], dim=1).cpu().float()
            meta["data"] = moving_normalize(meta["data"], filter=2048, stride=256)
            print("Plotting...")
            eqnet.utils.visualize_das_train(meta, phase, epoch=epoch, figure_dir=args.figure_dir, prefix=prefix)
            del output, phase

        elif args.model == "autoencoder":
            preds = model(meta)
            print("Plotting...")
            eqnet.utils.visualize_autoencoder_das_train(meta, preds, epoch=epoch, figure_dir=args.figure_dir)
            del preds

        elif args.model == "eqnet":
            output = model(meta)
            phase = F.softmax(output["phase"], dim=1).cpu().float()
            event = torch.sigmoid(output["event"]).cpu().float()
            print("Plotting...")
            eqnet.utils.visualize_eqnet_train(meta, phase, event, epoch=epoch, figure_dir=args.figure_dir)
            del output, phase, event


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
        figure_dir = os.path.join(args.output_dir, "figures")
        args.figure_dir = figure_dir
        utils.mkdir(figure_dir)

    utils.init_distributed_mode(args)
    print(args)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        rank, world_size = 0, 1

    torch.manual_seed(1337 + rank)
    random.seed(1337 + rank)
    np.random.seed(1337 + rank)

    device = torch.device(args.device)
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    args.dtype, args.ptdtype = dtype, ptdtype
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.model == "phasenet":
        dataset = SeismicTraceIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            hdf5_file=args.hdf5_file,
            format="h5",
            training=True,
            stack_event=args.stack_event,
            flip_polarity=args.flip_polarity,
            drop_channel=args.drop_channel,
            rank=rank,
            world_size=world_size,
        )
        train_sampler = None
        dataset_test = dataset
        test_sampler = None
    elif args.model == "phasenet_das":
        dataset = DASIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            label_path=args.label_path,
            label_list=args.label_list,
            noise_list=args.noise_list,
            phases=args.phases,
            nt=args.nt,
            nx=args.nx,
            format="h5",
            training=True,
            stack_noise=args.stack_noise,
            stack_event=args.stack_event,
            resample_space=args.resample_space,
            resample_time=args.resample_time,
            masking=args.masking,
            rank=rank,
            world_size=world_size,
        )
        train_sampler = None
        dataset_test = DASIterableDataset(
            data_path=args.test_data_path,
            data_list=args.test_data_list,
            label_path=args.test_label_path,
            label_list=args.test_label_list,
            noise_list=args.test_noise_list,
            phases=args.phases,
            nt=args.nt,
            nx=args.nx,
            format="h5",
            training=True,
            stack_noise=False,
            stack_event=False,
            resample_space=False,
            resample_time=False,
            masking=False,
            rank=rank,
            world_size=world_size,
        )
        test_sampler = None
    elif args.model == "autoencoder":
        dataset = AutoEncoderIterableDataset(
            data_path=args.data_path,
            format="h5",
            training=True,
        )
        train_sampler = None
        dataset_test = dataset
        test_sampler = None
    elif args.model == "eqnet":
        if args.huggingface_dataset:
            try:
                # Get the directory of the train.py
                code_dir = os.path.dirname(os.path.abspath(__file__))
                script_dir = os.path.join(code_dir, "eqnet/data/quakeflow_nc.py")
                dataset = datasets.load_dataset(script_dir, split="train", name="NCEDC_full_size")
            except:
                dataset = datasets.load_dataset("AI4EPS/quakeflow_nc", split="train", name="NCEDC_full_size")

            dataset = dataset.with_format("torch")
            dataset_dict = dataset.train_test_split(test_size=0.2, shuffle=False)
            dataset = dataset_dict["train"]
            dataset_test = dataset_dict["test"]

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
            else:
                train_sampler = torch.utils.data.RandomSampler(dataset)
                test_sampler = torch.utils.data.SequentialSampler(dataset_test)

            group_ids = create_groups(dataset, args.num_stations_list)
            dataset = dataset.map(lambda x: cut_reorder_keys(x, num_stations_list=args.num_stations_list))
            train_batch_sampler = StationSampler(train_sampler, group_ids, args.batch_size, args.num_stations_list)

        else:
            dataset = SeismicNetworkIterableDataset(args.dataset)
            train_sampler = None
            test_sampler = None
            dataset_test = dataset
    else:
        raise ("Unknown model")

    if args.huggingface_dataset:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.workers,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            sampler=test_sampler,
            num_workers=args.workers,
            collate_fn=None,
            drop_last=False,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            # collate_fn=utils.collate_fn,
            collate_fn=None,
            # shuffle=True,
            drop_last=True,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            collate_fn=None,
            drop_last=False,
        )

    model = eqnet.models.__dict__[args.model].build_model(
        backbone=args.backbone,
        in_channels=1,
        out_channels=(len(args.phases) + 1),
        ## phasenet-das
        reg=args.reg,
        ## phasenet
        polarity_loss_weight=args.polarity_loss_weight,
    )
    logger.info("Model:\n{}".format(model))

    print("Model:\n{}".format(model))

    model.to(device)
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)  # requires PyTorch 2.0

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    # scaler = torch.cuda.amp.GradScaler() if args.amp else None
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    iters_per_epoch = len(data_loader)
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size * iters_per_epoch, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.epochs - args.lr_warmup_epochs) * iters_per_epoch, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "polynomiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and PolynomialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs * iters_per_epoch
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs * iters_per_epoch
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs * iters_per_epoch],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    # logging
    if args.wandb and utils.is_main_process():
        wandb.init(
            project=args.wandb_project, name=args.wandb_name, entity=args.wandb_group, dir=args.wandb_dir, config=args
        )
        if args.wandb_watch:
            wandb.watch(model, log="all", log_freq=args.print_freq)

    start_time = time.time()
    best_loss = float("inf")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            optimizer,
            lr_scheduler,
            data_loader,
            model_ema,
            scaler,
            args,
            epoch,
            len(dataset),
        )

        metric = evaluate(model, data_loader_test, scaler, args, epoch, len(dataset_test))
        if model_ema:
            metric = evaluate(model_ema, data_loader_test, scaler, args, epoch, len(dataset_test))

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
        if utils.is_main_process() and metric.loss.global_avg < best_loss:
            best_loss = metric.loss.global_avg
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
            if args.wandb:
                best_model = wandb.Artifact(
                    f"{args.wandb_name}",
                    type="model",
                    metadata=dict(epoch=epoch, loss=best_loss),
                )
                best_model.add_file(os.path.join(args.output_dir, "checkpoint.pth"))
                wandb.log_artifact(best_model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if args.wandb and utils.is_main_process():
        wandb.finish()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="./", type=str, help="dataset path")
    parser.add_argument("--data-list", nargs="+", default=None, type=str, help="dataset list")
    parser.add_argument("--label-path", nargs="+", default=None, type=str, help="label path")
    parser.add_argument("--label-list", nargs="+", default=None, type=str, help="label path")
    parser.add_argument("--noise-list", nargs="+", default=None, type=str, help="noise list")
    parser.add_argument("--hdf5-file", default=None, type=str, help="hdf5 file for training")
    parser.add_argument("--test-data-path", default="./", type=str, help="test dataset path")
    parser.add_argument("--test-data-list", default="+", type=None, help="test dataset list")
    parser.add_argument("--test-label-path", default="+", type=None, help="test label path")
    parser.add_argument("--test-label-list", default="+", type=None, help="test label path")
    parser.add_argument("--test-noise-list", default="+", type=None, help="test noise list")
    parser.add_argument("--dataset", default="", type=str, help="dataset name")
    parser.add_argument("--model", default="phasenet_das", type=str, help="model name")
    parser.add_argument("--backbone", default="unet", type=str, help="model backbone")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile model to torchscript",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=8,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )

    ## training hyper params
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=1,
        type=int,
        help="the number of epochs to warmup (default: 1)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="linear",
        type=str,
        help="the warmup method (default: linear)",
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument(
        "--lr-scheduler", default="cosineannealinglr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # model moving average
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    ## Data Augmentation
    parser.add_argument("--phases", default=["P", "S"], type=str, nargs="+", help="phases to use")
    parser.add_argument("--reg", default=0.0, type=float, help="laplace regularization for phasenet-das")
    parser.add_argument("--nt", default=1024 * 3, type=int, help="number of time samples")
    parser.add_argument("--nx", default=1024 * 5, type=int, help="number of space samples")
    parser.add_argument("--stack-noise", action="store_true", help="Stack noise")
    parser.add_argument("--stack-event", action="store_true", help="Stack event")
    parser.add_argument("--flip-polarity", action="store_true", help="Flip polarity")
    parser.add_argument("--drop-channel", action="store_true", help="Drop channel")
    parser.add_argument("--resample-space", action="store_true", help="Resample space resolution")
    parser.add_argument("--resample-time", action="store_true", help="Resample time  resolution")
    parser.add_argument("--masking", action="store_true", help="Masking of the input data")
    parser.add_argument("--polarity-loss-weight", default=1.0, type=float, help="Polarity loss weight")
    parser.add_argument("--random-crop", action="store_true", help="Random size")
    parser.add_argument("--crop-nt", default=1024, type=int, help="Crop time samples")
    parser.add_argument("--crop-nx", default=1024, type=int, help="Crop space samples")

    # wandb
    parser.add_argument("--wandb", action="store_true", help="use wandb")
    parser.add_argument("--wandb-project", default="phasenet-das", type=str, help="wandb project name")
    parser.add_argument("--wandb-name", default=None, type=str, help="wandb run name")
    parser.add_argument("--wandb-group", default=None, type=str, help="wandb group name")
    parser.add_argument("--wandb-dir", default="./", type=str, help="wandb dir")
    parser.add_argument("--wandb-watch", action="store_true", help="wandb watch model")

    # huggingface dataset
    parser.add_argument("--huggingface-dataset", action="store_true", help="use huggingface dataset")
    parser.add_argument(
        "--num-stations-list", default=[5, 10, 20], type=int, nargs="+", help="possible stations number of the dataset"
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
