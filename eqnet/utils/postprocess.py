import os
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def detect_peaks(scores, vmin=0.3, kernel=101, stride=1, K=0):

    nb, nc, nt, nx = scores.shape
    pad = kernel // 2
    smax = F.max_pool2d(scores, (kernel, 1), stride=(stride, 1), padding=(pad, 0))[:, :, :nt, :]
    keep = (smax == scores).float()
    scores = scores * keep

    batch, chn, nt, ns = scores.size()
    scores = torch.transpose(scores, 2, 3)
    if K == 0:
        K = max(round(nt * 10.0 / 3000.0), 3)
    if chn == 1:
        topk_scores, topk_inds = torch.topk(scores, K)
    else:
        topk_scores, topk_inds = torch.topk(scores[:, 1:, :, :].view(batch, chn - 1, ns, -1), K)
    topk_inds = topk_inds % nt

    return topk_scores.detach().cpu(), topk_inds.detach().cpu()


def extract_picks(
    topk_index,
    topk_score,
    file_name=None,
    begin_time=None,
    station_id=None,
    phases=["P", "S"],
    vmin=0.3,
    dt=0.01,
    polarity_score=None,
    waveform=None,
    window_amp=[10, 5],
    **kwargs,
):
    """Extract picks from prediction results.
    Args:
        topk_scores ([type]): [Nb, Nc, Ns, Ntopk] "batch, channel, station, topk"
        file_names ([type], optional): [Nb]. Defaults to None.
        station_ids ([type], optional): [Ns]. Defaults to None.
        t0 ([type], optional): [Nb]. Defaults to None.
        config ([type], optional): [description]. Defaults to None.

    Returns:
        picks [type]: {file_name, station_id, pick_time, pick_prob, pick_type}
    """

    batch, nch, nst, ntopk = topk_score.shape
    # assert nch == len(phases)

    picks = []
    if isinstance(dt, float):
        dt = [dt for i in range(batch)]
    else:
        dt = [dt[i].item() for i in range(batch)]
    if ("begin_channel_index" in kwargs) and (kwargs["begin_channel_index"] is not None):
        begin_channel_index = [x.item() for x in kwargs["begin_channel_index"]]
    else:
        begin_channel_index = [0 for i in range(batch)]
    if ("begin_time_index" in kwargs) and (kwargs["begin_time_index"] is not None):
        begin_time_index = [x.item() for x in kwargs["begin_time_index"]]
    else:
        begin_time_index = [0 for i in range(batch)]

    if waveform is not None:
        waveform_amp = torch.max(torch.abs(waveform), dim=1)[0]
        # waveform_amp = torch.sqrt(torch.mean(waveform ** 2, dim=1))

        if len(window_amp) == 1:
            window_amp = [window_amp[0] for i in range(len(phases))]

    for i in range(batch):
        picks_per_file = []
        if file_name is None:
            file_i = f"{i:04d}"
        else:
            file_i = file_name[i]

        if begin_time is None:
            begin_i = "1970-01-01T00:00:00.000"
        else:
            begin_i = begin_time[i] 
            if len(begin_i) == 0:
                begin_i = "1970-01-01T00:00:00.000"
        begin_i = datetime.fromisoformat(begin_i.rstrip("Z"))

        for j in range(nch):
            if waveform is not None:
                window_amp_i = int(window_amp[j] / dt[i])

            for k in range(nst):
                if station_id is None:
                    station_i = f"{k + begin_channel_index[i]:04d}"
                else:
                    station_i = station_id[k][i]

                topk_index_ijk, ii = torch.sort(topk_index[i, j, k])
                topk_score_ijk = topk_score[i, j, k][ii]

                # for ii, (index, score) in enumerate(zip(topk_index[i, j, k], topk_score[i, j, k])):
                for ii, (index, score) in enumerate(zip(topk_index_ijk, topk_score_ijk)):
                    if score > vmin:
                        pick_index = index.item() + begin_time_index[i]
                        pick_time = (begin_i + timedelta(seconds=index.item() * dt[i])).isoformat(
                            timespec="milliseconds"
                        )
                        pick_dict = {
                                # "file_name": file_i,
                                "station_id": station_i,
                                "phase_index": pick_index,
                                "phase_time": pick_time,
                                "phase_score": f"{score.item():.3f}",
                                "phase_type": phases[j],
                                # "dt": dt[i],
                            }
                        
                        if polarity_score is not None:
                            pick_dict["phase_polarity"] = f"{polarity_score[i, 0, index, k].item():.3f}"
                        
                        if waveform is not None:
                            j1 = topk_index_ijk[ii]
                            j2 = min(j1 + window_amp_i, topk_index_ijk[ii + 1]) if ii < len(topk_index_ijk) - 1 else j1 + window_amp_i
                            pick_dict["phase_amplitude"] = f"{torch.max(waveform_amp[i, j1:j2, k]).item():.3e}"

                        picks_per_file.append(pick_dict)

        picks.append(picks_per_file)
    return picks


def merge_das_picks(raw_folder="picks_phasenet_das", merged_folder=None, min_picks=10):

    in_path = Path(raw_folder)

    if merged_folder is None:
        out_path = Path(raw_folder + "_merged")
    else:
        out_path = Path(merged_folder)

    if not out_path.exists():
        out_path.mkdir()

    files = in_path.glob("*_*_*.csv")

    file_group = defaultdict(list)
    for file in files:
        file_group[file.stem.split("_")[0]].append(file)  ## event_id

    num_picks = 0
    for k in tqdm(file_group, desc=f"{out_path}"):
        picks = []
        header = None
        for i, file in enumerate(sorted(file_group[k])):
            with open(file, "r") as f:
                tmp = f.readlines()
                if (len(tmp) > 0) and (header == None):
                    header = tmp[0]
                    picks.append(header)
                picks.extend(tmp[1:])  ## without header

        if len(picks) > min_picks:
            with open(out_path.joinpath(f"{k}.csv"), "w") as f:
                f.writelines(picks)

        num_picks += len(picks)

    print(f"Number of picks: {num_picks}")
    return 0


def merge_seismic_picks(pick_path):
    csv_files = sorted(glob(os.path.join(pick_path, "*.csv")))
    num_picks = 0
    with open(pick_path.rstrip("/")+".csv", "w") as fp_out:
        first_non_empty = True
        for i, file in enumerate(tqdm(csv_files, desc="Merging picks")):
            with open(file, "r") as fp_in:
                lines = fp_in.readlines()
                if first_non_empty and (len(lines) > 0):
                    fp_out.writelines(lines[0])
                    first_non_empty = False
                fp_out.writelines(lines[1:])
                num_picks += max(0, len(lines) - 1)
    print("Total number of picks: {}".format(num_picks))