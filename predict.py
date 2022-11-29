import logging
import multiprocessing
import os
import warnings

import pandas as pd
import torch
import torch.utils.data

import eqnet
import utils
from eqnet.data import DASIterableDataset, SeismicTraceIterableDataset
from eqnet.utils import detect_peaks, extract_picks, merge_picks, plot_das, plot_phasenet
from eqnet.models.unet import normalize_local
from eqnet.models.phasenet_das import normalize_local as normalize_local_das

warnings.filterwarnings("ignore", ".*Length of IterableDataset.*")
logger = logging.getLogger("EQNet")


def pred_phasenet(model, data_loader, pick_path, figure_path, args):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Predicting:"
    with torch.inference_mode():
        for meta in metric_logger.log_every(data_loader, 1, header):

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(meta)
                if "phase" in output:
                    phase_scores = torch.softmax(output["phase"], dim=1)  # [batch, nch, nt, nsta]
                    if "polarity" in output:
                        polarity_scores = (torch.sigmoid(output["polarity"]) - 0.5) * 2.0
                    else:
                        polarity_scores = None
                    topk_phase_scores, topk_phase_inds = detect_peaks(phase_scores, vmin=args.min_prob, kernel=21)
                    phase_picks_ = extract_picks(
                        topk_phase_inds,
                        topk_phase_scores,
                        file_name=meta["file_name"],
                        station_name=meta["station_name"],
                        begin_time=meta["begin_time"] if "begin_time" in meta else None,
                        dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                        vmin=args.min_prob,
                        phases=args.phases,
                        polarity_score=polarity_scores,
                    )

                if "event" in output:
                    event_scores = torch.sigmoid(output["event"])
                    topk_event_scores, topk_event_inds = detect_peaks(event_scores, vmin=args.min_prob, kernel=21)
                    event_picks_ = extract_picks(
                        topk_event_inds,
                        topk_event_scores,
                        file_name=meta["file_name"],
                        begin_time=meta["begin_time"] if "begin_time" in meta else None,
                        dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                        vmin=args.min_prob,
                        phases=["event"],
                    )

            for i in range(len(meta["file_name"])):

                if len(phase_picks_[i]) == 0:
                    ## keep an empty file for the file with no picks to make it easier to track processed files
                    with open(os.path.join(pick_path, meta["file_name"][i].replace("/", "_") + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(phase_picks_[i])
                picks_df.sort_values(by=["phase_index"], inplace=True)
                picks_df.to_csv(
                    os.path.join(pick_path, meta["file_name"][i].replace("/", "_") + ".csv"),
                    columns=[
                        "station_name",
                        "phase_index",
                        "phase_time",
                        "phase_score",
                        "phase_type",
                        "phase_polarity",
                    ],
                    index=False,
                )

            if args.plot_figure:
                meta["waveform_raw"] = meta["waveform"].clone()
                meta["waveform"] = normalize_local(meta["waveform"])
                plot_phasenet(
                    meta,
                    phase_scores.cpu(),
                    event_scores.cpu(),
                    polarity=polarity_scores.cpu() if polarity_scores is not None else None,
                    picks=phase_picks_,
                    phases=args.phases,
                    file_name=meta["file_name"],
                    dt=meta["dt_s"] if "dt_s" in meta else torch.tensor(0.01),
                    figure_dir=figure_path,
                )
                print("saving:", meta["file_name"])

    return 0


def pred_phasenet_das(model, data_loader, pick_path, figure_path, args):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Predicting:"
    with torch.inference_mode():
        for meta in metric_logger.log_every(data_loader, 1, header):

            with torch.cuda.amp.autocast(enabled=args.amp):
                scores = torch.softmax(model(meta), dim=1)  # [batch, nch, nt, nsta]
                topk_scores, topk_inds = detect_peaks(scores, vmin=args.min_prob, kernel=21)

                picks_ = extract_picks(
                    topk_inds,
                    topk_scores,
                    file_name=meta["file_name"],
                    begin_time=meta["begin_time"] if "begin_time" in meta else None,
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    begin_channel_index=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
                    dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                    vmin=args.min_prob,
                    phases=args.phases,
                )

            for i in range(len(meta["file_name"])):
                if len(picks_[i]) == 0:
                    ## keep an empty file for the file with no picks to make it easier to track processed files
                    with open(os.path.join(pick_path, meta["file_name"][i] + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(picks_[i])
                picks_df["channel_index"] = picks_df["station_name"].apply(lambda x: int(x))
                picks_df.sort_values(by=["channel_index", "phase_index"], inplace=True)
                picks_df.to_csv(
                    os.path.join(pick_path, meta["file_name"][i] + ".csv"),
                    columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type"],
                    index=False,
                )

            if args.plot_figure:
                plot_das(
                    meta["data"].cpu().numpy(),
                    scores.cpu().numpy(),
                    picks=picks_,
                    phases=args.phases,
                    file_name=meta["file_name"],
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    begin_channel_index=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
                    dt=meta["dt_s"] if "dt_s" in meta else torch.tensor(0.01),
                    dx=meta["dx_m"] if "dx_m" in meta else torch.tensor(10.0),
                    figure_dir=figure_path,
                )

    if args.distributed:
        torch.distributed.barrier()
        if args.cut_patch and utils.is_main_process():
            merge_picks(pick_path)
    else:
        if args.cut_patch:
            merge_picks(pick_path)

    return 0


def main(args):

    result_path = args.result_path
    pick_path = os.path.join(result_path, f"picks_{args.model}_raw")
    figure_path = os.path.join(result_path, f"figures_{args.model}_raw")
    if not os.path.exists(result_path):
        utils.mkdir(result_path)
    if not os.path.exists(pick_path):
        utils.mkdir(pick_path)
    if not os.path.exists(figure_path):
        utils.mkdir(figure_path)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    model = eqnet.models.__dict__[args.model](
        backbone=args.backbone, in_channels=1, out_channels=(len(args.phases) + 1), use_polarity=args.use_polarity
    )

    logger.info("Model:\n{}".format(model))

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
    else:
        if args.model == "phasenet" and (not args.use_polarity):
            raise ("No pretrained model for phasenet, please use phasenet_polarity instead")
        elif (args.model == "phasenet") and (args.use_polarity):
            model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-Polarity-v1/model_99.pth"
        elif args.model == "phasenet_das":
            model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v3/model_10.pth"
        else:
            raise
        state_dict = torch.hub.load_state_dict_from_url(
            model_url, model_dir="./", progress=True, check_hash=True, map_location="cpu"
        )
        model_without_ddp.load_state_dict(state_dict["model"], strict=False)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        rank, world_size = 0, 1

    if args.model == "phasenet":
        dataset = SeismicTraceIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            format="mseed",
            training=False,
            rank=rank,
            world_size=world_size,
        )
        sampler = None
    elif args.model == "phasenet_das":
        dataset = DASIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            format=args.format,
            rank=rank,
            world_size=world_size,
            nx=args.nx,
            nt=args.nt,
            training=False,
            dataset=args.dataset,
            cut_patch=args.cut_patch,
            filtering=args.filtering,
            skip_files=args.skip_files,
        )
        sampler = None
    else:
        raise ("Unknown model")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(args.workers, multiprocessing.cpu_count()),
        collate_fn=None,
        prefetch_factor=2,
        drop_last=False,
    )

    if args.model == "phasenet":
        pred_phasenet(model, data_loader, pick_path, figure_path, args)

    if args.model == "phasenet_das":
        pred_phasenet_das(model, data_loader, pick_path, figure_path, args)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="EQNet Model", add_help=add_help)

    # model
    parser.add_argument("--model", default="phasenet_das", type=str, help="model name")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--backbone", default="unet", type=str, help="model backbone")
    parser.add_argument("--phases", default=["P", "S"], type=str, nargs="+", help="phases to use")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--data_path", type=str, default="./", help="path to data directory")
    parser.add_argument("--data_list", type=str, default=None, help="selectecd data list")
    parser.add_argument("--result_path", type=str, default="results", help="path to result directory")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--format", type=str, default="h5", help="data format")
    parser.add_argument("--dataset", type=str, default=None, help="The name of dataset: mammoth, eqnet, or None")
    parser.add_argument("--filtering", action="store_true", help="If filter the data")
    parser.add_argument("--cut_patch", action="store_true", help="If cut patch for continuous data")
    parser.add_argument("--nt", default=1024 * 3, type=int, help="number of time samples for each patch")
    parser.add_argument("--nx", default=1024 * 3, type=int, help="number of spatial samples for each patch")
    parser.add_argument("--skip_files", default=None, help="If skip the files that have been processed")
    parser.add_argument("--min_prob", default=0.6, type=float, help="minimum probability for picking")
    parser.add_argument("--use_polarity", action="store_true", help="If use polarity information")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
