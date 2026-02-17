import os
import random
from datetime import datetime, timedelta
from glob import glob

import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, IterableDataset
import os
import time

# mp.set_start_method("spawn", force=True)

def normalize(data: torch.Tensor):
    """channel-wise normalization

    Args:
        data (tensor): [nch, nt, nx]

    Returns:
        tensor: [nch, nt, nx]
    """
    nch, nt, nx = data.shape
    data = data.double()
    mean = torch.mean(data, dim=(1), keepdims=True)
    std = torch.std(data, dim=(1), keepdims=True)
    std[std == 0.0] = 1.0
    data = data / std
    return data.float()


def generate_phase_label(
    data: torch.Tensor,
    phase_list: list,
    label_width: list = [150],
    mask_width: list = None,
    label_shape: str = "gaussian",
):
    """generate gaussian-shape label for phase picks

    Args:
        data (tensor): [nch, nt, nx]
        phase_list (list): [[p_channel, p_index], [s_channel, s_index], [other phases]]
        label_width (list, optional): [150, 150] samples.
        label_shape (str, optional): Defaults to "gaussian".
        space_mask (tensor, optional): [nch, nt, nx], 1 for valid, 0 for invalid.
        return_time_mask (bool, optional): Use to prevent stacking phases too closely in in time. Defaults to True.

    Returns:
        phase label: [nch, nt, nx]
    """
    nch, nt, nx = data.shape

    if mask_width is None:
        mask_width = [label_width] * len(phase_list)

    target = np.zeros([len(phase_list) + 1, nt, nx], dtype=np.float32)
    ## mask for window near the phase arrival
    time_mask = np.zeros([nt, nx], dtype=np.float32)

    if len(label_width) == 1:
        label_width = label_width * len(phase_list)

    space_mask = np.zeros((len(phase_list), nx), dtype=bool)
    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for trace, phase_time in picks:
            trace = int(trace)
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :, trace] += gaussian
            space_mask[i, trace] = True
            # time_mask[int(phase_time) - w : int(phase_time) + w, trace] = 1
            time_mask[:, trace] = 1

    space_mask = np.all(space_mask, axis=0)  ## traces with all picks
    target[0:1, :, space_mask] = np.maximum(0, 1 - np.sum(target[1:, :, space_mask], axis=0, keepdims=True))
    target[:, :, ~space_mask] = 0

    # plt.figure(figsize=(12, 12))
    # plt.subplot(3, 1, 1)
    # plt.imshow((data[0, :, :] - torch.mean(data[0, :, :], dim=0, keepdims=True))/torch.std(data[0, :, :], dim=0, keepdims=True), vmin=-2, vmax=2, aspect="auto", cmap="seismic")
    # plt.subplot(3, 1, 2)
    # target_ = np.zeros([max(3, len(phase_list)+1), nt, nx], dtype=np.float32)
    # for i in range(target.shape[0]):
    #     target_[i, :, :] = target[i, :, :]
    # if target_.shape[0] == 4:
    #     # target_ =  target_[1:, :, :]
    #     target_ = target_[[1, 2, 3], :, :]
    # print(target_.shape, target.shape)
    # plt.imshow(target_.transpose(1, 2, 0), aspect="auto")
    # plt.subplot(3, 1, 3)
    # plt.imshow(time_mask[0,:,:], aspect="auto")
    # plt.savefig("test_label.png")
    # raise
    time_mask = time_mask[np.newaxis, :, :]

    return target, time_mask

def generate_event_label(
    data,
    center,
    duration,
    label_width=150,
    mask_width=None,
):

    nch, nt, nx = data.shape
    target_center = np.zeros([nt, nx], dtype=np.float32)
    target_time = np.zeros([nt, nx], dtype=np.float32)
    center_mask = np.zeros([nt, nx], dtype=np.float32)
    time_mask = np.zeros([nt, nx], dtype=np.float32)

    if mask_width is None:
        mask_width = int(label_width * 1.5)

    for c0, d0 in zip(center, duration):
        ich = c0[0]
        assert ich == d0[0]
        c0 = c0[1]
        d0 = d0[1]
        t = np.arange(nt) - c0
        gaussian = np.exp(-(t**2) / (2 * (label_width / 6) ** 2))
        gaussian[gaussian < 0.05] = 0.0
        target_center[:, ich] += gaussian
        target_time[:, ich] = t + d0
        time_mask[int(c0) - mask_width : int(c0) + mask_width, ich] = 1.0
        center_mask[:, ich] = 1.0

    target_center = target_center[np.newaxis, :, :]
    target_time = target_time[np.newaxis, :, :]
    center_mask = center_mask[np.newaxis, :, :]
    time_mask = time_mask[np.newaxis, :, :]
    return target_center, target_time, center_mask, time_mask

def stack_event(
    data1,
    targets1,
    masks1,
    snrs1,
    data2,
    targets2,
    masks2,
    snrs2,
    min_shift=0,
    max_shift=1024 * 2,
):
    """targets[0] is the phase label"""
    tries = 0
    max_tries = 100
    nch, nt, nx = data2.shape
    success = False
    while tries < max_tries:
        # shift = random.randint(-nt, nt)
        shift = random.randint(-max_shift, max_shift)
        # if masks2 is not None:
        #     masks2_ = {k: torch.clone(v) for k, v in masks2.items()}
        #     masks2_ = {k: torch.roll(v, shift, dims=-2) for k, v in masks2_.items()}
        #     if any(torch.max(masks1[k] + masks2_[k]) >= 2 for k in ["event_time_mask"]):
        #         tries += 1
        #         continue

        data2_ = torch.clone(data2)
        data2_ = torch.roll(data2_, shift, dims=-2)
        masks2_ = {k: torch.clone(v) for k, v in masks2.items()}
        targets2_ = {k: torch.clone(v) for k, v in targets2.items()}
        for k, v in masks2_.items():
            masks2_[k] = torch.roll(v, shift, dims=-2)
        for k, v in targets2_.items():
            targets2_[k] = torch.roll(v, shift, dims=-2)

        ## approximately after normalization, noise=1, signal=snr, so signal ~ noise * snr
        # data = data1 + data2_ * (1 + max(0, snr1 - 1.0) * torch.rand(1) * 0.5)
        data = data1 * (1 + torch.rand(1) * 2) + data2_ * (1 + torch.rand(1) * 2)

        targets = {k: torch.zeros_like(v) for k, v in targets1.items()}
        for k, v in targets.items():
            if k in ["phase_pick"]:
                targets[k][1:, :, :] = targets1[k][1:, :, :] + targets2_[k][1:, :, :]
                targets[k][0, :, :] = torch.maximum(torch.tensor(0.0), 1.0 - torch.sum(targets[k][1:, :, :], axis=0))
            else:
                targets[k][:, :, :] = torch.maximum(targets1[k][:, :, :], targets2_[k][:, :, :])

        masks = {k: torch.zeros_like(v) for k, v in masks1.items()}
        for k, v in masks.items():
            masks[k][:, :, :] = masks1[k][:, :, :] + masks2_[k][:, :, :]
            if k in ["phase_mask", "event_center_mask"]:
                masks[k][masks[k] < 2.0] = 0.0
                masks[k][masks[k] >= 2.0] = 1.0
            else:
                masks[k][masks[k] >= 1.0] = 1.0

        targets["phase_pick"][:, :, (masks["phase_mask"] == 0).all(dim=(0, 1))] = 0
        targets["event_center"][:, :, (masks["event_center_mask"] == 0).all(dim=(0, 1))] = 0
        targets["event_time"][:, :, (masks["event_time_mask"] == 0).all(dim=(0, 1))] = 0

        success = True
        break

    if tries >= max_tries:
        data = data1
        targets = targets1
        masks = masks1
        print(f"stack event failed, tries={tries}")

    return data, targets, masks, success


def pad_data(data, targets, masks, nt=1024 * 4, nx=1024 * 6):
    """pad data to the same size as required nt and nx"""
    nch, w, h = data.shape
    if h < nx:
        with torch.no_grad():
            data_ = data.unsqueeze(0)
            targets_ = {k: v.unsqueeze(0) for k, v in targets.items()}
            masks_ = {k: v.unsqueeze(0) for k, v in masks.items()}
            if (nx // h - 1) > 0:
                for i in range(nx // h - 1):
                    data_ = F.pad(data_, (0, h - 1, 0, 0), mode="reflect")
                    targets_ = {k: F.pad(v, (0, h - 1, 0, 0), mode="reflect") for k, v in targets_.items()}
                    masks_ = {k: F.pad(v, (0, h - 1, 0, 0), mode="reflect") for k, v in masks_.items()}
                data_ = F.pad(data_, (0, nx // h - 1, 0, 0), mode="reflect")
                targets_ = {k: F.pad(v, (0, nx // h - 1, 0, 0), mode="reflect") for k, v in targets_.items()}
                masks_ = {k: F.pad(v, (0, nx // h - 1, 0, 0), mode="reflect") for k, v in masks_.items()}
            data_ = F.pad(data_, (0, nx % h, 0, 0), mode="reflect").squeeze(0)
            targets_ = {k: F.pad(v, (0, nx % h, 0, 0), mode="reflect").squeeze(0) for k, v in targets_.items()}
            masks_ = {k: F.pad(v, (0, nx % h, 0, 0), mode="reflect").squeeze(0) for k, v in masks_.items()}
    else:
        data_ = data
        targets_ = targets
        masks_ = masks
    return data_, targets_, masks_

def cut_data(
    data: torch.Tensor,
    targets = [],
    masks = [],
    label_width = 150,
    nt: int = 1024 * 3,
    nx: int = 1024 * 5,
):
    """cut data window for training"""

    tmp_sum = 0
    max_sum = 0
    tmp_tries = 0
    max_tries = 100
    max_w0 = 0
    max_h0 = 0
    w, h = data.shape[-2:]
    while tmp_sum < label_width / 2 * nx * 0.1:
        w0 = np.random.randint(0, max(1, w - nt))
        h0 = np.random.randint(0, max(1, h - nx))
        if len(targets) > 0:
            tmp_sum = torch.sum(targets["phase_pick"][1:, w0 : w0 + nt, h0 : h0 + nx])  # nch, nt, nx
        else:
            tmp_sum = nx * nt
        if tmp_sum > max_sum:
            max_sum = tmp_sum
            max_w0 = w0
            max_h0 = h0
        tmp_tries += 1
        if tmp_tries >= max_tries:
            break
    w0 = max_w0
    h0 = max_h0

    data_ = data[:, w0 : w0 + nt, h0 : h0 + nx].clone()
    targets_ = {k: v[..., w0 : w0 + nt, h0 : h0 + nx].clone() for k, v in targets.items()}
    masks_ = {k: v[..., w0 : w0 + nt, h0 : h0 + nx].clone() for k, v in masks.items()}
    return data_, targets_, masks_


def cut_noise(noise: torch.Tensor, nt: int = 1024 * 3, nx: int = 1024 * 5):
    nch, w, h = noise.shape
    w0 = np.random.randint(0, max(1, w - nt))
    h0 = np.random.randint(0, max(1, h - nx))
    return noise[:, w0 : w0 + nt, h0 : h0 + nx]


def pad_noise(noise: torch.Tensor, nt: int = 1024 * 3, nx: int = 1024 * 5):
    """pad noise to the same size as required nt and nx"""

    nch, w, h = noise.shape
    if w < nt:
        with torch.no_grad():
            noise = noise.unsqueeze(0)
            if (nt // w - 1) > 0:
                for i in range(nt // w - 1):
                    noise = F.pad(noise, (0, 0, 0, w - 1), mode="reflect")
                noise = F.pad(noise, (0, 0, 0, nt // w - 1), mode="reflect")
            noise = F.pad(noise, (0, 0, 0, nt % w), mode="reflect").squeeze(0)
    if h < nx:
        with torch.no_grad():
            noise = noise.unsqueeze(0)
            if (nx // h - 1) > 0:
                for i in range(nx // h - 1):
                    noise = F.pad(noise, (0, h - 1, 0, 0), mode="reflect")
                noise = F.pad(noise, (0, nx // h - 1, 0, 0), mode="reflect")
            noise = F.pad(noise, (0, nx % h, 0, 0), mode="reflect").squeeze(0)
    return noise


def calc_snr(data: torch.Tensor, picks: list, noise_window: int = 200, signal_window: int = 200):
    SNR = []
    S = []
    N = []
    for trace, phase_time in picks:
        trace = int(trace)
        phase_time = int(phase_time)
        noise = torch.std(data[:, max(0, phase_time - noise_window) : phase_time, trace])
        signal = torch.std(data[:, phase_time : phase_time + signal_window, trace])
        S.append(signal)
        N.append(noise)
        SNR.append(signal / noise)

    return np.median(SNR), np.median(S), np.median(N)


def stack_noise(data, noise, snr):
    ## approximately after normalization, noise=1, signal=snr, so signal ~ noise * snr
    return data + noise * max(0, snr - 2) * torch.rand(1)

def flip_lr(data, targets=[], masks=[]):
    data = data.flip(-1)
    targets = {k: v.flip(-1) for k, v in targets.items()}
    masks = {k: v.flip(-1) for k, v in masks.items()}
    return data, targets, masks

def masking(data, targets, masks,nt=256, nx=256):
    nc0, nt0, nx0 = data.shape
    nt_ = random.randint(32, nt)
    nt0_ = random.randint(0, nt0 - nt_)
    data_ = data.clone()
    targets_ = {k: v.clone() for k, v in targets.items()}
    masks_ = {k: v.clone() for k, v in masks.items()}

    data_[:, nt0_ : nt0_ + nt_, :] = 0.0
    for k, v in targets_.items():
        if k == "phase_pick":
            targets_[k][0, nt0_ : nt0_ + nt_, :] = 1.0
            targets_[k][1:, nt0_ : nt0_ + nt_, :] = 0.0
        # else:
        #     targets_[k][:, nt0_ : nt0_ + nt_, :] = 0.0 # event center could still be predicted 

    return data_, targets_, masks_


def masking_edge(data, targets, masks, nt=1024, nx=1024):
    """masking edges to prevent edge effects"""

    crop_nt = random.randint(1, nt)
    crop_nx = random.randint(1, nx)

    data_ = data.clone()
    targets_ = {k: v.clone() for k, v in targets.items()}
    masks_ = {k: v.clone() for k, v in masks.items()}

    data_[:, -crop_nt:, :] = 0.0

    for k, v in targets_.items():
        if k in ["phase_pick"]:
            targets_[k][0, -crop_nt:, :] = 1.0
            targets_[k][1:, -crop_nt:, :] = 0.0
        # if k in ["event_center"]:
        else:
            targets_[k][:, -crop_nt:, :] = 0.0
    for k, v in masks_.items():
        if k in ["event_time_mask"]:
            masks_[k][:,-crop_nt:, :] = 0.0

    return data_, targets_, masks_


def resample_space(data, targets, masks, noise=None, factor=1):
    """resample space by factor to adjust the spatial resolution"""
    nch, nt, nx = data.shape
    scale_factor = random.uniform(min(1, factor), max(1, factor))
    with torch.no_grad():
        data_ = F.interpolate(data, scale_factor=scale_factor, mode="nearest")
        targets_ = {k: F.interpolate(v, scale_factor=scale_factor, mode="nearest") for k, v in targets.items()}
        masks_ = {k: F.interpolate(v, scale_factor=scale_factor, mode="nearest") for k, v in masks.items()}
        if noise is not None:
            noise_ = F.interpolate(noise, scale_factor=scale_factor, mode="nearest")
        else:
            noise_ = None
    return data_, targets_, masks_, noise_


def resample_time(data, picks, noise=None, factor=1):
    """resample time by factor to adjust the temporal resolution

    Args:
        picks (list): [[[channel_index, time_index], ..], [[channel_index, time_index], ], ...]
    """
    nch, nt, nx = data.shape
    scale_factor = random.uniform(min(1, factor), max(1, factor))
    with torch.no_grad():
        data_ = F.interpolate(data.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
        if noise is not None:
            noise_ = F.interpolate(noise.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
        else:
            noise_ = None
    picks_ = []
    for phase in picks:
        tmp = []
        for p in phase:
            tmp.append([p[0], p[1] * scale_factor])
        picks_.append(tmp)
    return data_, picks_, noise_


def filter_channels(picks):
    """filter channels with all pick types"""

    commmon_channel_index = []
    for event_index in picks["event_index"].unique():
        for phase_type in picks["phase_type"].unique():
            commmon_channel_index.append(
                picks[(picks["event_index"] == event_index) & (picks["phase_type"] == phase_type)][
                    "channel_index"
                ].tolist()
            )
    commmon_channel_index = set.intersection(*map(set, commmon_channel_index))
    picks = picks[picks["channel_index"].isin(commmon_channel_index)]
    return picks



def read_PASSCAL_segy(fid, nTraces=1250, nSample=900000, TraceOff=0, strain_rate=True):
    """Function to read PASSCAL segy raw data
    For Ridgecrest data, there are 1250 channels in total,
    Sampling rate is 250 Hz so for one hour data: 250 * 3600 samples
    author: Jiuxun Yin
    source: https://github.com/SCEDC/cloud/blob/master/pds_ridgecrest_das.ipynb
    """
    fs = nSample / 3600  # sampling rate
    data = np.zeros((nTraces, nSample), dtype=np.float32)

    fid.seek(3600)
    # Skipping traces if necessary
    fid.seek(TraceOff * (240 + nSample * 4), 1)
    # Looping over traces
    for ii in range(nTraces):
        fid.seek(240, 1)
        bytes = fid.read(nSample * 4)
        data[ii, :] = np.frombuffer(bytes, dtype=np.float32)

    fid.close()

    # Convert the phase-shift to strain (in nanostrain)
    Ridgecrest_conversion_factor = 1550.12 / (0.78 * 4 * np.pi * 1.46 * 8)
    data = data * Ridgecrest_conversion_factor

    if strain_rate:
        data = np.gradient(data, axis=1) * fs

    return data


def roll_by_gather(data, dim, shifts: torch.LongTensor):
    nch, h, w = data.shape

    if dim == 0:
        arange1 = torch.arange(h).view((1, h, 1)).repeat((nch, 1, w))
        arange2 = (arange1 - shifts) % h
        return torch.gather(data, 1, arange2)
    elif dim == 1:
        arange1 = torch.arange(w).view((1, 1, w)).repeat((nch, h, 1))
        arange2 = (arange1 - shifts.unsqueeze(0)) % w
        return torch.gather(data, 2, arange2)
    else:
        raise ValueError("dim must be 0 or 1")


class DASIterableDataset(IterableDataset):
    def __init__(
        self,
        data_path="./",
        data_list=None,
        format="h5",
        prefix="",
        suffix="",
        nt=1024 * 3,
        nx=1024 * 5,
        min_nt=1024,
        min_nx=1024,
        ## training
        training=False,
        phases=["P", "S"],
        label_path="./",
        subdir=3,
        label_list=None,
        noise_list=None,
        stack_noise=False,
        stack_event=False,
        resample_time=False,
        resample_space=False,
        skip_existing=False,
        pick_path="./",
        folder_depth=1,  # parent folder depth of pick_path
        num_patch=2,
        masking=False,
        highpass_filter=0.0,
        filter_params={
            "freqmin": 0.1,
            "freqmax": 10.0,
            "corners": 4,
            "zerophase": True,
        },
        ## continuous data
        system=None,  # "eqnet" or "optasense" or None
        cut_patch=False,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.data_path = data_path
        self.format = format
        self.prefix = prefix
        self.suffix = suffix
        self.subdir = subdir

        self.data_path = data_path
        if data_list is not None:
            if type(data_list) == list:
                self.data_list = []
                for data_list_ in data_list:
                    with open(data_list_, "r") as f:
                        # read lines without \n
                        self.data_list += f.read().rstrip("\n").split("\n")
            else:
                with open(data_list, "r") as f:
                    self.data_list = f.read().rstrip("\n").split("\n")
        else:
            self.data_list = glob(os.path.join(self.data_path, f"{prefix}*{suffix}.{format}"))

        if not training:
            self.data_list = self.data_list[rank::world_size]

        ## continuous data
        self.system = system
        self.cut_patch = cut_patch
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.01  # s
        self.dx = kwargs["dx"] if "dx" in kwargs else 10.0  # m
        self.nt = nt
        self.nx = nx
        self.min_nt = min_nt
        self.min_nx = min_nx
        assert self.nt % self.min_nt == 0
        assert self.nx % self.min_nx == 0

        ## training and data augmentation
        self.training = training
        self.phases = phases
        self.label_path = label_path
        if label_list is not None:
            if type(label_list) is list:
                self.label_list = []
                for label_list_ in label_list:
                    with open(label_list_, "r") as f:
                        self.label_list += f.read().rstrip("\n").split("\n")
            else:
                with open(label_list, "r") as f:
                    self.label_list = f.read().rstrip("\n").split("\n")
            if training:
                self.label_list = self.label_list[: len(self.label_list) // world_size * world_size]
            self.label_list = self.label_list[rank::world_size]
        else:
            self.label_list = glob(self.label_path + f"/*.csv")
        self.min_picks = kwargs["min_picks"] if "min_picks" in kwargs else 500
        if noise_list is not None:
            if type(noise_list) is list:
                self.noise_list = []
                for noise_list_ in noise_list:
                    with open(noise_list_, "r") as f:
                        self.noise_list += f.read().rstrip("\n").split("\n")
            else:
                with open(noise_list, "r") as f:
                    self.noise_list = f.read().rstrip("\n").split("\n")
        self.stack_noise = stack_noise
        self.stack_event = stack_event
        self.resample_space = resample_space
        self.resample_time = resample_time
        self.skip_existing = skip_existing
        self.pick_path = pick_path
        self.folder_depth = folder_depth
        self.num_patch = num_patch
        self.masking = masking
        self.highpass_filter = highpass_filter

        if self.training:
            print(f"Total samples: {len(self.label_list)} files")
        else:
            print(f"Total samples: {len(self.data_list)} files")

        ## pre-calcuate length
        self._data_len = self._count()

    def __len__(self):
        return self._data_len

    def _count(self):
        if self.training:
            return len(self.label_list) * self.num_patch

        if not self.cut_patch:
            return len(self.data_list)
        else:
            if self.format == "h5":
                with fsspec.open(self.data_list[0], "rb") as fs:
                    with h5py.File(fs, "r") as meta:
                        if self.system == "optasense":
                            attrs = {}
                            if "Data" in meta:
                                nx, nt = meta["Data"].shape
                                attrs["dt_s"] = meta["Data"].attrs["dt"]
                                attrs["dx_m"] = meta["Data"].attrs["dCh"]
                            else:
                                nx, nt = meta["Acquisition/Raw[0]/RawData"].shape
                                dx = meta["Acquisition"].attrs["SpatialSamplingInterval"]
                                fs = meta["Acquisition/Raw[0]"].attrs["OutputDataRate"]
                                attrs["dx_m"] = dx
                                attrs["dt_s"] = 1.0 / fs
                        else:
                            nx, nt = meta["data"].shape
                            attrs = dict(meta["data"].attrs)
                if self.resample_time and ("dt_s" in attrs):
                    if (attrs["dt_s"] != 0.01) and (int(round(1.0 / attrs["dt_s"])) % 100 == 0):
                        nt = int(nt / round(0.01 / attrs["dt_s"]))

            elif self.format == "segy":
                print("Start reading segy file")
                with fsspec.open(self.data_list[0], "rb") as fs:
                    nx, nt = read_PASSCAL_segy(fs).shape
                print("End reading segy file")
            else:
                raise ValueError("Unknown dataset")

            return len(self.data_list) * ((nt - 1) // self.nt + 1) * ((nx - 1) // self.nx + 1)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if self.training:
            return iter(self.sample_training(self.label_list[worker_id::num_workers]))
        else:
            return iter(self.sample(self.data_list[worker_id::num_workers]))

    def sample_training(self, file_list):
        while True:
            ## load picks
            file_list = np.random.permutation(file_list)
            for label_file in file_list:
                picks = pd.read_csv(self.label_path + "/" + label_file)
                if "channel_index" not in picks.columns:
                    picks = picks.rename(columns={"station_id": "channel_index"})

                meta = {}
                for pick_type in self.phases:
                    meta[pick_type] = picks[picks["phase_type"] == pick_type][
                        ["channel_index", "phase_index"]
                    ].to_numpy()

                ## load waveform data
                data_file = "/".join(
                    label_file.replace("labels", "data").replace(".csv", ".h5").split("/")[-self.subdir:]
                )  # folder/data/event_id

                try:
                    with fsspec.open(self.data_path + "/" + data_file, "rb") as f:
                        with h5py.File(f, "r") as fp:
                            data = fp["data"][:, :].T
                            dt = fp["data"].attrs["dt_s"]
                            dx = fp["data"].attrs["dx_m"]
                        data = data[np.newaxis, :, :]  # nchn, nt, nx
                        data = data / np.std(data)
                        data = torch.from_numpy(data.astype(np.float32))

                except:
                    print(f"Error reading {data_file}")
                    continue

                ## basic normalize
                data = data - torch.mean(data, dim=1, keepdim=True)

                # load noise
                noise = None
                if self.stack_noise and (self.noise_list is not None):
                    tmp = self.noise_list[np.random.randint(0, len(self.noise_list))]
                    try:
                        with fsspec.open(os.path.join(self.data_path, tmp), "rb") as f:
                            with h5py.File(f, "r") as fp:
                                noise = fp["data"][:, :].T
                            ## The first 30s are noise in the training data
                            noise = np.roll(noise, max(0, self.nt - 3000), axis=0)  # nt, nx
                            noise = noise[np.newaxis, : self.nt, :]  # nchn, nt, nx
                            noise = noise / np.std(noise)
                            noise = torch.from_numpy(noise.astype(np.float32))

                        noise = noise - torch.mean(noise, dim=1, keepdim=True)
                    except:
                        print(f"Error reading noise file {tmp}")
                        noise = torch.zeros([1, self.nt, self.nx], dtype=torch.float32)

                ## snr
                if "P" in meta:
                    snr, S, N = calc_snr(data, meta["P"])
                else:
                    snr, S, N = 0, 0, 0


                ## generate training labels
                picks = [meta[x] for x in self.phases]

                ## augmentation
                rand_i = np.random.rand()
                if self.resample_time:
                    if rand_i < 0.2:
                        data, picks, noise = resample_time(data, picks, noise, 3)
                    elif rand_i < 0.4:
                        data, picks, noise = resample_time(data, picks, noise, 0.5)

                ## generate training labels
                phase_pick, phase_mask = generate_phase_label(data, picks)
                phase_pick = torch.from_numpy(phase_pick)
                phase_mask = torch.from_numpy(phase_mask)

                c0 = [[x1[0], (x1[1] + x2[1]) / 2] for x1, x2 in zip(picks[0], picks[1])]
                t0 = [[x1[0], x2[1] - x1[1]] for x1, x2 in zip(picks[0], picks[1])]

                event_center, event_time, event_center_mask, event_time_mask = generate_event_label(data, c0, t0)
                event_center = torch.from_numpy(event_center)
                event_time = torch.from_numpy(event_time)
                event_center_mask = torch.from_numpy(event_center_mask)
                event_time_mask = torch.from_numpy(event_time_mask)


                targets = {
                    "phase_pick": phase_pick,
                    "event_center": event_center,
                    "event_time": event_time,
                }
                masks = {
                    "phase_mask": phase_mask,
                    "event_center_mask": event_center_mask,
                    "event_time_mask": event_time_mask,
                }

                ## augmentation
                status_stack_event = False
                if self.stack_event and (snr > 10) and (np.random.rand() < 0.3):
                    data, targets, masks, status_stack_event = stack_event(
                        data, targets, masks, snr, data, targets, masks, snr
                    )

                ## augmentation
                if self.resample_space:
                    if rand_i < 0.2:
                        data, targets, masks, noise = resample_space(data, targets, masks, noise, 5)
                    elif (rand_i < 0.4) and (data.shape[-1] > 2000):
                        data, targets, masks, noise = resample_space(data, targets, masks, noise, 0.5)

                ## pad data
                data, targets, masks = pad_data(data, targets, masks, nx=self.nx + self.nx // 2)
                if self.stack_noise:
                    noise = pad_noise(noise, self.nt, self.nx + self.nx // 2)

                for ii in range(self.num_patch):
                    data_, targets_, masks_ = cut_data(data, targets, masks, nt=self.nt, nx=self.nx)

                    ## augmentation
                    if self.stack_noise and (not status_stack_event) and (np.random.rand() < 0.5):
                        noise_ = cut_noise(noise, self.nt, self.nx)
                        data_ = stack_noise(data_, noise_, snr)

                    ## augmentation
                    if np.random.rand() < 0.5:
                        data_, targets_, masks_ = flip_lr(data_, targets_, masks_)

                    ## augmentation
                    if self.masking and (np.random.rand() < 0.2):
                        data_, targets_, masks_ = masking(data_, targets_, masks_)


                    # ## prevent edge effect on the right and bottom
                    # if np.random.rand() < 0.05:
                    #     data_, targets_, masks_ = masking_edge(data_, targets_, masks_)


                    # data_ = normalize(data_)
                    if np.random.rand() < 0.5:
                        data_ = data_ - torch.median(data_, dim=-2, keepdims=True)[0]

                    phase_pick_, event_center_, event_time_ = targets_["phase_pick"], targets_["event_center"], targets_["event_time"]
                    phase_mask_, event_center_mask_, event_time_mask_ = masks_["phase_mask"], masks_["event_center_mask"], masks_["event_time_mask"]

                    ## FIXME: shift (nt, nx) to (nx, nt)
                    data_ = data_.permute(0, 2, 1)
                    phase_pick_ = phase_pick_.permute(0, 2, 1)
                    phase_mask_ = phase_mask_.permute(0, 2, 1)
                    event_center_ = event_center_.permute(0, 2, 1)
                    event_time_ = event_time_.permute(0, 2, 1)
                    event_center_mask_ = event_center_mask_.permute(0, 2, 1)
                    event_time_mask_ = event_time_mask_.permute(0, 2, 1)
                    event_feature_scale = 16
                    event_center_ = event_center_[:, ::, ::event_feature_scale]
                    event_time_ = event_time_[:, ::, ::event_feature_scale]
                    event_center_mask_ = event_center_mask_[:, ::, ::event_feature_scale]
                    event_time_mask_ = event_time_mask_[:, ::, ::event_feature_scale]

                    yield {
                        "data": torch.nan_to_num(data_),
                        "phase_pick": phase_pick_,
                        "phase_mask": phase_mask_,
                        "event_center": event_center_,
                        "event_time": event_time_,
                        "event_time_mask": event_time_mask_,
                        "event_center_mask": event_center_mask_,
                        "file_name": os.path.splitext(label_file.split("/")[-1])[0] + f"_{ii:02d}",
                        "height": data_.shape[-2],
                        "width": data_.shape[-1],
                        "dt_s": dt,
                        "dx_m": dx,
                    }

    def sample(self, file_list):
        for file in file_list:
            if not self.cut_patch:
                existing = self.check_existing(file)
                if self.skip_existing and existing:
                    print(f"Skip existing file {file}")
                    continue

            sample = {}

            if self.format == "npz":
                data = np.load(file)["data"]

            elif self.format == "npy":
                data = np.load(file)  # (nx, nt)
                sample["begin_time"] = datetime.fromisoformat("1970-01-01 00:00:00")
                sample["dt_s"] = 0.01
                sample["dx_m"] = 10.0

            elif self.format == "h5" and (self.system is None):
                with fsspec.open(file, "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        dataset = fp["data"]  # nt x nx
                        data = dataset[()]
                        if "begin_time" in dataset.attrs:
                            sample["begin_time"] = datetime.fromisoformat(dataset.attrs["begin_time"].rstrip("Z"))
                        if "dt_s" in dataset.attrs:
                            sample["dt_s"] = dataset.attrs["dt_s"]
                        else:
                            sample["dt_s"] = self.dt
                        if "dx_m" in dataset.attrs:
                            sample["dx_m"] = dataset.attrs["dx_m"]
                        else:
                            sample["dx_m"] = self.dx
            elif (self.format == "h5") and (self.system == "optasense"):
                with fsspec.open(file, "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        # dataset = fp["Data"]
                        if "Data" in fp:  # converted format by Ettore Biondi
                            dataset = fp["Data"]
                            sample["begin_time"] = datetime.fromisoformat(dataset.attrs["startTime"].rstrip("Z"))
                            sample["dt_s"] = dataset.attrs["dt"]
                            sample["dx_m"] = dataset.attrs["dCh"]
                        else:
                            dataset = fp["Acquisition/Raw[0]/RawData"]
                            dx = fp["Acquisition"].attrs["SpatialSamplingInterval"]
                            fs = fp["Acquisition/Raw[0]"].attrs["OutputDataRate"]
                            begin_time = dataset.attrs["PartStartTime"].decode()

                            sample["dx_m"] = dx
                            sample["dt_s"] = 1.0 / fs
                            sample["begin_time"] = datetime.fromisoformat(begin_time.rstrip("Z"))

                        nx, nt = dataset.shape
                        sample["nx"] = nx
                        sample["nt"] = nt

                        ## check existing
                        existing = self.check_existing(file, sample)
                        if self.skip_existing and existing:
                            print(f"Skip existing file {file}")
                            continue

                        data = dataset[()]  # (nx, nt)
                        data = np.gradient(data, axis=-1, edge_order=2) / sample["dt_s"]

            elif self.format == "segy":
                meta = {}
                with fsspec.open(file, "rb") as fs:
                    data = read_PASSCAL_segy(fs)

                ## FIXME: hard code for Ridgecrest DAS
                sample["begin_time"] = datetime.strptime(file.split("/")[-1].rstrip(".segy"), "%Y%m%d%H")
                sample["dt_s"] = 1.0 / 250.0
                sample["dx_m"] = 8.0
            else:
                raise (f"Unsupported format: {self.format}")

            if self.resample_time:
                if (sample["dt_s"] != 0.01) and (int(round(1.0 / sample["dt_s"])) % 100 == 0):
                    print(f"Resample {file} from time interval {sample['dt_s']} to 0.01")
                    data = data[..., :: int(0.01 / sample["dt_s"])]
                    sample["dt_s"] = 0.01

            data = data - np.mean(data, axis=-1, keepdims=True)  # (nx, nt)
            data = data - np.median(data, axis=-2, keepdims=True)
            if (self.highpass_filter is not None):
                b, a = scipy.signal.butter(2, self.highpass_filter, "hp", fs=100)
                data = scipy.signal.filtfilt(b, a, data, axis=-1)  # (nt, nx)

            data = data.T  # (nx, nt) -> (nt, nx)
            data = data[np.newaxis, :, :]  # (nchn, nt, nx)
            data = torch.from_numpy(data.astype(np.float32))

            # data = torch.from_numpy(data).float()
            # data = data - torch.mean(data, axis=-1, keepdims=True)  # (nx, nt)
            # data = data - torch.median(data, axis=-2, keepdims=True).values
            # data = data.T  # (nx, nt) -> (nt, nx)
            # data = data.unsqueeze(0)  # (nchn, nt, nx)

            if not self.cut_patch:
                nt, nx = data.shape[1:]
                data = padding(data, self.min_nt, self.min_nx)

                ## FIXME: (nt, nx) -> (nx, nt)
                data = data.permute(0, 2, 1)

                yield {
                    "data": data,
                    "nt": nt,
                    "nx": nx,
                    # "file_name": os.path.splitext(file.split("/")[-1])[0],
                    "file_name": file,
                    "begin_time": sample["begin_time"].isoformat(timespec="milliseconds"),
                    "begin_time_index": 0,
                    "begin_channel_index": 0,
                    "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                    "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                }
            else:
                _, nt, nx = data.shape
                for i in list(range(0, nt, self.nt)):
                    for j in list(range(0, nx, self.nx)):
                        if self.skip_existing:
                            if os.path.exists(
                                os.path.join(
                                    self.pick_path, os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv"
                                )
                            ):
                                print(
                                    f"Skip existing file",
                                    os.path.join(
                                        self.pick_path,
                                        os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv",
                                    ),
                                )
                                continue
                        data_patch = data[:, i : i + self.nt, j : j + self.nx]
                        _, nt_, nx_ = data_patch.shape
                        data_patch = padding(data_patch, self.min_nt, self.min_nx)

                        ## FIXME: (nt, nx) -> (nx, nt)
                        data_patch = data_patch.permute(0, 2, 1)
                        yield {
                            "data": data_patch,
                            "nt": nt_,
                            "nx": nx_,
                            # "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}",
                            "file_name": os.path.splitext(file)[0] + f"_{i:04d}_{j:04d}",
                            "begin_time": (sample["begin_time"] + timedelta(seconds=i * float(sample["dt_s"]))).isoformat(
                                timespec="milliseconds"
                            ),
                            "begin_time_index": i,
                            "begin_channel_index": j,
                            "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                            "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                        }

    def check_existing(self, file, sample=None):
        parent_dir = "/".join(file.split("/")[-self.folder_depth : -1])
        existing = True
        if not self.cut_patch:
            if not os.path.exists(
                os.path.join(
                    os.path.join(self.pick_path, parent_dir, os.path.splitext(file.split("/")[-1])[0] + ".csv")
                )
            ):
                existing = False
        else:
            nx, nt = sample["nx"], sample["nt"]
            if self.resample_time:
                if (sample["dt_s"] != 0.01) and (int(round(1.0 / sample["dt_s"])) % 100 == 0):
                    nt = int(nt / round(0.01 / sample["dt_s"]))
            for i in list(range(0, nt, self.nt)):
                for j in list(range(0, nx, self.nx)):
                    if not os.path.exists(
                        os.path.join(
                            self.pick_path,
                            parent_dir,
                            os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv",
                        )
                    ):
                        existing = False

        return existing


class AutoEncoderIterableDataset(DASIterableDataset):
    def __init__(
        self,
        data_path="./",
        noise_path=None,
        format="npz",
        prefix="",
        suffix="",
        training=False,
        stack_noise=False,
        highpass_filter=0.0,
        **kwargs,
    ):
        super().__init__(data_path, noise_path, format=format, training=training)

    def sample(self, file_list):
        sample = {}
        # for file in file_list:
        idx = 0
        while True:
            if self.training:
                file = file_list[np.random.randint(0, len(file_list))]
            else:
                if idx >= len(file_list):
                    break
                file = file_list[idx]
                idx += 1

            if self.training and (self.format == "h5"):
                with h5py.File(file, "r") as f:
                    data = f["data"][()]
                    data = data[np.newaxis, :, :]  # nchn, nt, nx
                    data = torch.from_numpy(data.astype(np.float32))
            else:
                raise (f"Unsupported format: {self.format}")

            data = data - np.median(data, axis=2, keepdims=True)
            data = normalize(data)  # nch, nt, nx

            if self.training:
                for ii in range(10):
                    pre_nt = 255
                    data_ = cut_data(data, pre_nt=pre_nt)
                    if data_ is None:
                        continue
                    if np.random.rand() < 0.5:
                        data_ = add_moveout(data_)
                    data_ = data_[:, pre_nt:, :]
                    if np.random.rand() < 0.5:
                        data_ = flip_lr(data_)
                    data_ = data_ - np.median(data_, axis=2, keepdims=True)

                    yield {
                        "data": data_,
                        "phase_pick": data_,
                        "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{ii:02d}",
                        "height": data_.shape[-2],
                        "width": data_.shape[-1],
                    }
            else:
                sample["data"] = data
                if self.nt is None:
                    self.nt = data.shape[1]
                if self.nx is None:
                    self.nx = data.shape[2]
                for i in list(range(0, data.shape[1], self.nt)):
                    if self.nt + i + 512 >= data.shape[1]:
                        tn = data.shape[1]
                    else:
                        tn = i + self.nt
                    for j in list(range(0, data.shape[2], self.nx)):
                        if self.nx + j + 512 >= data.shape[2]:
                            xn = data.shape[2]
                        else:
                            xn = j + self.nx
                        yield {
                            "data": data[:, i:tn, j:xn],
                            "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}",
                            "begin_time": (sample["begin_time"] + timedelta(i * sample["dt_s"])).isoformat(
                                timespec="milliseconds"
                            ),
                            "begin_time_index": i,
                            "begin_channel_index": j,
                            "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                            "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                        }
                        if xn == data.shape[2]:
                            break
                    if tn == data.shape[1]:
                        break


class DASDataset(Dataset):
    def __init__(
        self,
        data_path="./",
        noise_path=None,
        label_path=None,
        format="npz",
        prefix="",
        suffix="",
        training=True,
        stack_noise=True,
        phases=["P", "S"],
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.label_path = label_path
        self.format = format
        self.training = training
        self.prefix = prefix
        self.suffix = suffix
        self.phases = phases
        self.data_list = sorted(glob(os.path.join(data_path, f"{prefix}*{suffix}.{format}")))
        if label_path is not None:
            if type(label_path) is list:
                self.label_list = []
                for i in range(len(label_path)):
                    self.label_list += list(sorted(glob(os.path.join(label_path[i], f"{prefix}*{suffix}.csv"))))
            else:
                self.label_list = sorted(glob(os.path.join(label_path, f"{prefix}*{suffix}.csv")))
        print(os.path.join(data_path, f"{prefix}*{suffix}.{format}"), len(self.data_list))
        if self.noise_path is not None:
            self.noise_list = glob(os.path.join(noise_path, f"*.{format}"))
        self.num_data = len(self.data_list)
        self.min_picks = kwargs["min_picks"] if "min_picks" in kwargs else 500
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.01
        self.dx = kwargs["dx"] if "dx" in kwargs else 10.0  # m

    def __len__(self):
        if self.label_path is not None:
            return len(self.label_list)
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = {}
        if self.training and (self.format == "npz"):
            meta = np.load(self.data_list[idx])
            data = meta["data"]
            data = data[np.newaxis, :, :]
            data = torch.from_numpy(data.astype(np.float32))

        elif self.training and (self.format == "h5"):
            file = self.label_list[idx]
            picks = pd.read_csv(file)
            meta = {}
            for pick_type in self.phases:
                meta[pick_type] = picks[picks["phase_type"] == pick_type][["channel_index", "phase_index"]].to_numpy()
            # if (len(meta["p_picks"]) < 500) or (len(meta["s_picks"]) < 500):
            #     continue
            tmp = file.split("/")
            tmp[-2] = "data"
            tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
            with h5py.File("/".join(tmp), "r") as f:
                data = f["data"][()]
                data = data[np.newaxis, :, :]  # nchn, nt, nx
                data = torch.from_numpy(data.astype(np.float32))

            if self.stack_noise and (not self.noise_path):
                tries = 0
                max_tries = 10
                while tries < max_tries:
                    tmp_file = self.label_list[np.random.randint(0, len(self.label_list))]
                    tmp_picks = pd.read_csv(tmp_file)
                    if tmp_picks["phase_index"].min() < 3000:
                        tries += 1
                        continue
                    tmp = tmp_file.split("/")
                    tmp[-2] = "data"
                    tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
                    with h5py.File("/".join(tmp), "r") as f:
                        noise = f["data"][()]
                        noise = noise[np.newaxis, :, :]  # nchn, nt, nx
                        noise = torch.from_numpy(noise.astype(np.float32))
                    break
                if tries >= max_tries:
                    print(f"Failed to find noise file for {file}")
                    noise = torch.zeros_like(data)

        elif self.format == "npz":
            meta = np.load(self.data_list[idx])
            data = meta["data"]
            data = data[np.newaxis, :, :]
            # data = np.diff(data, axis=-2)
            # b, a = scipy.signal.butter(2, 4, 'hp', fs=100)
            # b, a = scipy.signal.butter(2, [0.5, 2.5], 'bandpass', fs=100)
            # data = scipy.signal.filtfilt(b, a, data, axis=-2)
            data = torch.from_numpy(data.astype(np.float32))

        elif self.format == "h5":
            begin_time_index = 0
            begin_channel_index = 0
            with h5py.File(self.data_list[idx], "r") as f:
                data = f["data"][()]
                # data = data[np.newaxis, :, :]
                data = data[np.newaxis, begin_time_index:, begin_channel_index:]
                if "begin_time" in f["data"].attrs:
                    if begin_time_index == 0:
                        sample["begin_time"] = datetime.fromisoformat(
                            f["data"].attrs["begin_time"].rstrip("Z")
                        ).isoformat(timespec="milliseconds")
                    else:
                        sample["begin_time_index"] = begin_time_index
                        sample["begin_time"] = (
                            datetime.fromisoformat(f["data"].attrs["begin_time"].rstrip("Z"))
                            + timedelta(seconds=begin_time_index * f["data"].attrs["dt_s"])
                        ).isoformat(timespec="milliseconds")
                if "dt_s" in f["data"].attrs:
                    sample["dt_s"] = f["data"].attrs["dt_s"]
                if "dx_m" in f["data"].attrs:
                    sample["dx_m"] = f["data"].attrs["dx_m"]
                data = torch.from_numpy(data.astype(np.float32))

        elif self.format == "segy":
            data = load_segy(os.path.join(self.data_path, self.data_list[idx]), nTrace=self.nTrace)
            data = torch.from_numpy(data)
            with torch.no_grad():
                data = torch.diff(data, n=1, dim=-1)
                data = F.interpolate(
                    data.unsqueeze(dim=0),
                    scale_factor=self.raw_dt / self.dt,
                    mode="linear",
                    align_corners=False,
                )
                data = data.permute(0, 2, 1)
        else:
            raise (f"Unsupported format: {self.format}")

        # data = normalize_local_1d(data)
        data = data - np.median(data, axis=2, keepdims=True)
        data = normalize(data)

        if self.training:
            if self.stack_noise:
                if torch.max(torch.abs(noise)) > 0:
                    noise = normalize(noise)
            picks = [meta[x] for x in self.phases]
            phase_pick = generate_phase_label(data, picks)
            phase_pick = torch.from_numpy(phase_pick)
            snr = calc_snr(data, meta["p_picks"])
            with_event = False
            if (snr > 3) and (np.random.rand() < 0.3):
                data, phase_pick = stack_event(data, phase_pick, data, phase_pick, snr)
                with_event = True
            pre_nt = 255
            data, phase_pick = cut_data(data, phase_pick, pre_nt=pre_nt)
            if np.random.rand() < 0.5:
                data, phase_pick = add_moveout(data, phase_pick)
            data = data[:, pre_nt:, :]
            phase_pick_ = phase_pick[:, pre_nt:, :]
            # if (snr > 10) and (np.random.rand() < 0.5):
            if not with_event:
                noise = cut_noise(noise)
                data = stack_noise(data, noise, snr)
            if np.random.rand() < 0.5:
                data, phase_pick = flip_lr(data, phase_pick)

            data = data - np.median(data, axis=2, keepdims=True)
            sample["phase_pick"] = phase_pick

        sample["data"] = data
        sample["file_name"] = os.path.splitext(self.data_list[idx].split("/")[-1])[0]
        sample["height"], sample["width"] = sample["data"].shape[-2:]

        return sample
