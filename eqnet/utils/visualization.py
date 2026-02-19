import os
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x):
    """x: [batch, channel, time, station]"""
    x = x - torch.mean(x, dim=2, keepdim=True)
    std = torch.std(x, dim=2, keepdim=True)
    std[std == 0] == 1
    x = x / std / 6
    return x


def plot_autoencoder_das_train(meta, preds, epoch, figure_dir="figures"):
    meta_data = meta["data"]
    raw_data = meta_data.clone().permute(0, 2, 3, 1).numpy()
    # data = normalize_local(meta_data.clone()).permute(0, 2, 3, 1).numpy()
    targets = meta["targets"].permute(0, 2, 3, 1).numpy()

    y = preds.permute(0, 2, 3, 1).numpy()

    for i in range(len(raw_data)):
        raw_vmax = np.std(raw_data[i]) * 2
        raw_vmin = -raw_vmax

        vmax = np.std(raw_data[i]) * 2
        vmin = -vmax

        fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=False, sharey=False)
        im = ax[0, 0].imshow(
            raw_data[i], vmin=raw_vmin, vmax=raw_vmax, interpolation="none", cmap="seismic", aspect="auto"
        )
        fig.colorbar(im, ax=ax[0, 0])
        im = ax[0, 1].imshow(
            targets[i], vmin=raw_vmin, vmax=raw_vmax, interpolation="none", cmap="seismic", aspect="auto"
        )
        fig.colorbar(im, ax=ax[0, 1])
        im = ax[1, 0].imshow(y[i], vmin=raw_vmin, vmax=raw_vmax, interpolation="none", cmap="seismic", aspect="auto")
        fig.colorbar(im, ax=ax[1, 0])
        im = ax[1, 1].imshow(y[i], interpolation="none", cmap="seismic", aspect="auto")
        fig.colorbar(im, ax=ax[1, 1])

        # ax[0, 1].imshow(y[i], vmin=0, vmax=1, interpolation='none', aspect='auto')
        # ax[1, 1].imshow(targets[i],  vmin=0, vmax=1, interpolation='none', aspect='auto')
        # ax[0, 1].imshow(y[i], interpolation='none', aspect='auto')
        # ax[1, 1].imshow(targets[i], interpolation='none', aspect='auto')

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}.png", dpi=300)

        plt.close(fig)


def _das_phase_rgb(phase_pick, mask=None):
    """Convert (nx, nt, 3) phase labels to RGB image (nt, nx, 3).

    Channels: 0=noise, 1=P, 2=S. Display: P=red, S=blue, mask=green tint.
    Transposes (nx, nt) -> (nt, nx) for display (time on y-axis).
    """
    p = phase_pick[:, :, 1].T  # (nt, nx)
    s = phase_pick[:, :, 2].T
    rgb = np.ones((*p.shape, 3))
    rgb[:, :, 1] = np.clip(1.0 - p * 0.7, 0, 1)  # P -> red
    rgb[:, :, 2] = np.clip(1.0 - p * 0.7, 0, 1)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] - s * 0.7, 0, 1)  # S -> blue
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] - s * 0.7, 0, 1)
    if mask is not None:
        m = mask[:, :, 0].T if mask.ndim == 3 else mask.T
        rgb[:, :, 0] = np.where(m > 0, rgb[:, :, 0] * 0.85, rgb[:, :, 0])
        rgb[:, :, 2] = np.where(m > 0, rgb[:, :, 2] * 0.85, rgb[:, :, 2])
    return rgb


def plot_phasenet_das_train(meta, phase_pick, epoch=0, figure_dir="figures", prefix="", **kwargs):

    norm_data = meta["data"].cpu().numpy().squeeze(1)  # (nb, nx, nt)
    nb, nx, nt = norm_data.shape
    raw = meta["raw_data"].cpu().numpy().squeeze(1) if "raw_data" in meta else norm_data

    if phase_pick is not None:
        phase_pick = phase_pick.cpu().numpy().transpose(0, 2, 3, 1)  # (nb, nx, nt, 3)

    y_phase_pick = meta["phase_pick"].cpu().numpy().transpose(0, 2, 3, 1) if "phase_pick" in meta else None
    y_phase_mask = meta["phase_mask"].cpu().numpy().transpose(0, 2, 3, 1) if "phase_mask" in meta else None

    imshow_kwargs = dict(aspect="auto", interpolation="nearest")

    for i in range(nb):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Row 1: Raw Data | Phase Label
        raw_i = raw[i].T
        vmax_raw = np.percentile(np.abs(raw_i), 95) or 1.0
        axes[0, 0].imshow(raw_i, cmap="seismic", vmin=-vmax_raw, vmax=vmax_raw, **imshow_kwargs)
        axes[0, 0].set_title(f"Raw ({meta['file_name'][i]})")
        axes[0, 0].set_ylabel("Time")

        mask_i = y_phase_mask[i] if y_phase_mask is not None else None
        if y_phase_pick is not None:
            axes[0, 1].imshow(_das_phase_rgb(y_phase_pick[i], mask=mask_i), **imshow_kwargs)
        axes[0, 1].set_title("Phase Label")

        # Row 2: Normalized Data | Phase Prediction
        norm_i = norm_data[i].T
        vmax_norm = np.percentile(np.abs(norm_i), 95) or 1.0
        axes[1, 0].imshow(norm_i, cmap="seismic", vmin=-vmax_norm, vmax=vmax_norm, **imshow_kwargs)
        axes[1, 0].set_title("Normalized")
        axes[1, 0].set_ylabel("Time")

        if phase_pick is not None:
            axes[1, 1].imshow(_das_phase_rgb(phase_pick[i]), **imshow_kwargs)
        axes[1, 1].set_title("Phase Pred")

        for ax in axes[1]:
            ax.set_xlabel("Channel")

        fig.tight_layout()
        rank_str = f"_{int(os.environ['RANK']):02d}" if "RANK" in os.environ else ""
        fig.savefig(f"{figure_dir}/{epoch:02d}{rank_str}_{i:02d}_{prefix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

def _plot_event_center(ax, fig, event_center, center_mask, imshow_kwargs):
    """Plot event center: white bg, green mask tint, red overlay for Gaussian."""
    ec = event_center.T  # (nt, nx)
    rgb = np.ones((*ec.shape, 3))
    # Green tint for center_mask
    if center_mask is not None:
        m = center_mask.T
        rgb[:, :, 0] = np.where(m > 0, rgb[:, :, 0] * 0.85, rgb[:, :, 0])
        rgb[:, :, 2] = np.where(m > 0, rgb[:, :, 2] * 0.85, rgb[:, :, 2])
    ax.imshow(rgb, **imshow_kwargs)
    # Red overlay for event center
    event_rgba = np.zeros((*ec.shape, 4))
    event_rgba[:, :, 0] = 1.0  # red
    event_rgba[:, :, 3] = ec * 0.9
    ax.imshow(event_rgba, **imshow_kwargs)


def _plot_event_time(ax, fig, event_time, time_mask, imshow_kwargs, vlim=None):
    """Plot event time masked to time_mask region with colorbar."""
    nt, nx = event_time.shape
    disp = np.where(time_mask > 0, event_time, np.nan)
    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin = np.nanmin(disp) if np.any(time_mask > 0) else 0.0
        vmax = np.nanmax(disp) if np.any(time_mask > 0) else 1.0
    im = ax.imshow(disp, cmap="seismic", vmin=vmin, vmax=vmax, **imshow_kwargs)
    fig.colorbar(im, ax=ax, shrink=0.8)


def plot_phasenet_das_plus_train(
    meta, phase_pick, event_center=None, event_time=None,
    epoch=0, figure_dir="figures", prefix="", **kwargs,
):
    norm_data = meta["data"].cpu().numpy().squeeze(1)  # (nb, nx, nt)
    nb, nx, nt = norm_data.shape
    raw = meta["raw_data"].cpu().numpy().squeeze(1) if "raw_data" in meta else norm_data

    # Predictions to numpy
    if phase_pick is not None:
        phase_pick = phase_pick.cpu().numpy().transpose(0, 2, 3, 1)  # (nb, nx, nt, 3)
    if event_center is not None:
        event_center = event_center.cpu().numpy().squeeze(1)  # (nb, nx_e, nt_e)
    if event_time is not None:
        event_time = event_time.cpu().numpy().squeeze(1)

    # Labels
    y_phase_pick = meta["phase_pick"].cpu().numpy().transpose(0, 2, 3, 1) if "phase_pick" in meta else None
    y_phase_mask = meta["phase_mask"].cpu().numpy().transpose(0, 2, 3, 1) if "phase_mask" in meta else None
    y_event_center = meta["event_center"].cpu().numpy().squeeze(1) if "event_center" in meta else None
    y_event_time = meta["event_time"].cpu().numpy().squeeze(1) if "event_time" in meta else None
    y_center_mask = meta["event_center_mask"].cpu().numpy().squeeze(1) if "event_center_mask" in meta else None
    y_time_mask = meta["event_time_mask"].cpu().numpy().squeeze(1) if "event_time_mask" in meta else None

    imshow_kwargs = dict(aspect="auto", interpolation="nearest")

    for i in range(nb):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        # Row 1: Raw Data | Phase Label | Event Center Label | Event Time Label
        raw_i = raw[i].T
        vmax_raw = np.percentile(np.abs(raw_i), 95) or 1.0
        axes[0, 0].imshow(raw_i, cmap="seismic", vmin=-vmax_raw, vmax=vmax_raw, **imshow_kwargs)
        axes[0, 0].set_title(f"Raw ({meta['file_name'][i]})")
        axes[0, 0].set_ylabel("Time")

        mask_i = y_phase_mask[i] if y_phase_mask is not None else None
        if y_phase_pick is not None:
            axes[0, 1].imshow(_das_phase_rgb(y_phase_pick[i], mask=mask_i), **imshow_kwargs)
        axes[0, 1].set_title("Phase Label")

        cm = y_center_mask[i] if y_center_mask is not None else None
        tm = y_time_mask[i] if y_time_mask is not None else None

        if y_event_center is not None:
            _plot_event_center(axes[0, 2], fig, y_event_center[i], center_mask=cm, imshow_kwargs=imshow_kwargs)
        axes[0, 2].set_title("Event Center Label")

        # Compute shared event time range from time_mask region
        et_vlim = None
        if y_event_time is not None and tm is not None:
            vals = [y_event_time[i][tm > 0]]
            if event_time is not None:
                vals.append(event_time[i][tm > 0])
            vals = np.concatenate(vals)
            if len(vals) > 0:
                et_vlim = (vals.min(), vals.max())

        if y_event_time is not None and tm is not None:
            _plot_event_time(axes[0, 3], fig, y_event_time[i].T, tm.T, imshow_kwargs, vlim=et_vlim)
        axes[0, 3].set_title("Event Time Label")

        # Row 2: Normalized Data | Phase Pred | Event Center Pred | Event Time Pred
        norm_i = norm_data[i].T
        vmax_norm = np.percentile(np.abs(norm_i), 95) or 1.0
        axes[1, 0].imshow(norm_i, cmap="seismic", vmin=-vmax_norm, vmax=vmax_norm, **imshow_kwargs)
        axes[1, 0].set_title("Normalized")
        axes[1, 0].set_ylabel("Time")

        if phase_pick is not None:
            axes[1, 1].imshow(_das_phase_rgb(phase_pick[i]), **imshow_kwargs)
        axes[1, 1].set_title("Phase Pred")

        if event_center is not None:
            _plot_event_center(axes[1, 2], fig, event_center[i], center_mask=None, imshow_kwargs=imshow_kwargs)
        axes[1, 2].set_title("Event Center Pred")

        if event_time is not None and tm is not None:
            _plot_event_time(axes[1, 3], fig, event_time[i].T, tm.T, imshow_kwargs, vlim=et_vlim)
        axes[1, 3].set_title("Event Time Pred")

        for ax in axes[1]:
            ax.set_xlabel("Channel")

        fig.tight_layout()
        rank_str = f"_{int(os.environ['RANK']):02d}" if "RANK" in os.environ else ""
        fig.savefig(f"{figure_dir}/{epoch:02d}{rank_str}_{i:02d}_{prefix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)



def plot_phasenet_train(meta, phase, epoch=0, figure_dir="figures", prefix=""):

    nb, nc, nx, nt = meta["data"].shape
    dt = 0.01
    t = torch.arange(nt) * dt

    for i in range(meta["data"].shape[0]):
        plt.close("all")

        chn_name = ["E", "N", "Z"] if nc >= 3 else [f"Ch{j}" for j in range(nc)]
        num_data_plots = min(nc, 3)

        if "raw_data" in meta:
            shift = num_data_plots
            fig, axes = plt.subplots(num_data_plots * 2 + 1, 1, figsize=(10, 6))
            for j in range(num_data_plots):
                axes[j].plot(t, meta["raw_data"][i, j, 0, :], linewidth=0.5, color="k", label=f"{chn_name[j]}")
                axes[j].set_xticklabels([])
                axes[j].set_xlim(t[0], t[-1])
                axes[j].grid("on")
        else:
            fig, axes = plt.subplots(num_data_plots + 1, 1, figsize=(10, 6))
            shift = 0

        for j in range(num_data_plots):
            axes[j + shift].plot(t, meta["data"][i, j, 0, :], linewidth=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + shift].set_xticklabels([])
            axes[j + shift].set_xlim(t[0], t[-1])
            axes[j + shift].grid("on")

        k = num_data_plots + shift
        axes[k].plot(t, phase[i, 1, 0, :], "b")
        axes[k].plot(t, phase[i, 2, 0, :], "r")
        axes[k].plot(t, meta["phase_pick"][i, 1, 0, :], "--C0")
        axes[k].plot(t, meta["phase_pick"][i, 2, 0, :], "--C3")
        axes[k].plot(t, meta["phase_mask"][i, 0, 0, :], ":", color="gray")
        axes[k].set_ylim(-0.05, 1.05)
        axes[k].set_xlim(t[0], t[-1])
        # axes[k].set_xticklabels([])
        axes[k].grid("on")

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}_{prefix}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{prefix}.png", dpi=300)

        if i >= 20:
            break


def plot_phasenet_tf_train(meta, phase, event_center, event_time, epoch=0, figure_dir="figures", prefix=""):

    for i in range(meta["data"].shape[0]):
        plt.close("all")
        chn_name = ["E", "N", "Z"]
        fig, axes = plt.subplots(8, 1, figsize=(10, 10))

        idx = 0
        for j in range(3):
            axes[idx + j].plot(meta["data"][i, j, 0, :], linewidth=0.5, color="k", label=f"{chn_name[j]}")
            axes[idx + j].set_xticklabels([])
            axes[idx + j].grid("on")
            axes[idx + j].autoscale(enable=True, axis="x", tight=True)

        idx = 3
        if meta["spectrogram"].shape[1] == 6:
            meta["spectrogram"] = meta["spectrogram"][:, ::2, :, :]
        for j in range(3):
            # vmax = meta["spectrogram"][i, j, :, :].abs().max().item()
            vmax = 6
            # axes[idx + j].pcolormesh(meta["spectrogram"][i, j, :, :], cmap="seismic", vmin=-vmax, vmax=vmax)
            vmax = torch.std(meta["spectrogram"][i, j, :, :]) * 3
            axes[idx + j].pcolormesh(meta["spectrogram"][i, j, :, :], cmap="viridis", vmin=0, vmax=vmax)  ## norm
            axes[idx + j].set_xticklabels([])

        idx = 6
        axes[idx].plot(phase[i, 1, 0, :], "b")
        axes[idx].plot(phase[i, 2, 0, :], "r")
        axes[idx].plot(meta["phase_pick"][i, 1, 0, :], "--C0")
        axes[idx].plot(meta["phase_pick"][i, 2, 0, :], "--C3")
        axes[idx].plot(meta["phase_mask"][i, 0, 0, :], ":", color="gray")
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].autoscale(enable=True, axis="x", tight=True)
        axes[idx].set_xticklabels([])
        axes[idx].grid("on")

        idx = 7
        if event_center is not None:
            axes[idx].plot(event_center[i, 0, 0, :], "b")
            axes[idx].plot(meta["event_center"][i, 0, 0, :], "--C0")
            axes[idx].plot(meta["event_mask"][i, 0, 0, :], ":", color="gray")
            axes[idx].set_ylim(-0.05, 1.05)
            axes[idx].autoscale(enable=True, axis="x", tight=True)
            axes[idx].set_xticklabels([])
            axes[idx].grid("on")

            axes2 = axes[idx].twinx()
            axes2.plot(event_time[i, 0, 0, :], "--C1")
            axes2.plot(meta["event_time"][i, 0, 0, :], ":C3")

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}_{prefix}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{prefix}.png", dpi=300)

        if i >= 20:
            break


def plot_phasenet_plus_train(
    meta, phase, polarity=None, event_center=None, event_time=None, epoch=0, figure_dir="figures", prefix=""
):
    nb, nc, nx, nt = meta["data"].shape
    dt = 0.01
    nt_event = event_center.shape[-1]
    dt_event = dt * nt / nt_event
    nt_polarity = polarity.shape[-1]
    dt_polarity = dt * nt / nt_polarity

    for i in range(meta["data"].shape[0]):
        plt.close("all")

        chn_name = ["E", "N", "Z"]

        if "raw_data" in meta:
            shift = 3
            fig, axes = plt.subplots(9, 1, figsize=(10, 10))
            t = torch.arange(nt) * dt
            for j in range(3):
                axes[j].plot(t, meta["raw_data"][i, j, 0, :], linewidth=0.5, color="k", label=f"{chn_name[j]}")
                axes[j].set_xlim(t[0], t[-1])
                axes[j].set_xticklabels([])
                axes[j].grid("on")
        else:
            fig, axes = plt.subplots(6, 1, figsize=(10, 10))
            shift = 0

        for j in range(3):
            t = torch.arange(nt) * dt
            axes[j + shift].plot(t, meta["data"][i, j, 0, :], linewidth=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + shift].set_xlim(t[0], t[-1])
            axes[j + shift].set_xticklabels([])
            axes[j + shift].grid("on")

        k = 3 + shift
        t = torch.arange(nt) * dt
        axes[k].plot(t, phase[i, 1, 0, :], "b")
        axes[k].plot(t, phase[i, 2, 0, :], "r")
        axes[k].plot(t, meta["phase_pick"][i, 1, 0, :], "--C0")
        axes[k].plot(t, meta["phase_pick"][i, 2, 0, :], "--C3")
        axes[k].plot(t, meta["phase_mask"][i, 0, 0, :], ":", color="gray")
        axes[k].set_xlim(t[0], t[-1])
        axes[k].set_ylim(-0.05, 1.05)
        axes[k].set_xticklabels([])
        axes[k].grid("on")

        t = torch.arange(nt_polarity) * dt_polarity

        # 1 channel for polarity
        axes[k + 1].plot(t, polarity[i, 0, 0, :], "b")
        axes[k + 1].plot(t, meta["polarity"][i, 0, 0, :], "--C0")
        axes[k + 1].plot(t, meta["polarity_mask"][i, 0, 0, :], ":", color="gray")

        # # 3 channels for polarity
        # axes[k + 1].plot(t, (1.0 + polarity[i, 1, 0, :]) / 2.0, "g", alpha=0.3, linewidth=0.5)
        # axes[k + 1].plot(t, (1.0 - polarity[i, 2, 0, :]) / 2.0, "g", alpha=0.3, linewidth=0.5)
        # axes[k + 1].plot(t, ((polarity[i, 1, 0, :] - polarity[i, 2, 0, :]) + 1.0) / 2.0, "b", alpha=1.0, linewidth=1.0)
        # axes[k + 1].plot(t, (1.0 + meta["polarity"][i, 1, 0, :]) / 2.0, "--C0", alpha=0.5, linewidth=1.0)
        # axes[k + 1].plot(t, (1.0 - meta["polarity"][i, 2, 0, :]) / 2.0, "--C0", alpha=0.5, linewidth=1.0)
        # axes[k + 1].plot(t, meta["polarity_mask"][i, 0, 0, :], ":", color="gray")

        axes[k + 1].set_xlim(t[0], t[-1])
        axes[k + 1].set_ylim(-0.05, 1.05)
        axes[k + 1].set_xticklabels([])
        axes[k + 1].grid("on")

        t = torch.arange(nt_event) * dt_event
        t_label = torch.arange(meta["event_center"].shape[-1]) * dt_event
        axes[k + 2].plot(t, event_center[i, 0, 0, :], "b")
        axes[k + 2].plot(t_label, meta["event_center"][i, 0, 0, :], "--C0")
        axes[k + 2].plot(t_label, meta["event_center_mask"][i, 0, 0, :], ":", color="gray")
        axes[k + 2].set_xlim(t[0], t[-1])
        axes[k + 2].set_ylim(-0.05, 1.05)
        axes[k + 2].grid("on")

        axes2 = axes[k + 2].twinx()
        axes2.plot(t, event_time[i, 0, 0, :], "--C1")
        axes2.plot(t_label, meta["event_time"][i, 0, 0, :], ":C3")

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}_{prefix}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{prefix}.png", dpi=300)

        if i >= 20:
            break


def plot_phasenet_prompt_train(
    meta,
    phase,
    polarity=None,
    event_center=None,
    event_time=None,
    prompt_center=None,
    epoch=0,
    figure_dir="figures",
    prefix="",
):
    nb, nc, nx, nt = meta["data"].shape
    dt = 0.01
    nt_event = event_center.shape[-1]
    dt_event = dt * nt / nt_event
    nt_polarity = polarity.shape[-1]
    dt_polarity = dt * nt / nt_polarity
    nt_prompt = prompt_center.shape[-1]
    dt_prompt = dt * nt / nt_prompt

    normalize = lambda x: x / x.abs().max()

    for i in range(meta["data"].shape[0]):
        plt.close("all")

        fig, axes = plt.subplots(3, 2, figsize=(15, 8))

        t = torch.arange(nt) * dt
        for j in range(nx):
            axes[0, 0].plot(t, normalize(meta["data"][i, -1, j, :]) + j, linewidth=0.5, color="k")
        axes[0, 0].set_xlim(t[0], t[-1])
        # axes[0, 0].set_xticklabels([])
        axes[0, 0].grid("on")

        # t = torch.arange(nt_prompt) * dt_prompt
        t = meta["position"][i, 0, :, 0]
        t_origin = t.clone()
        ## HARD CODE
        dt = t[1] - t[0]
        t = t[:, None] * 16 + torch.arange(16) * dt
        t = t.flatten()

        # scale = meta["position"][i, -1, 0, 0] - meta["position"][i, 0, 0, 0]  # B, T, S, 3
        # t_prompt = (meta["prompt"][i, 0] - meta["position"][i, 0, 0, 0]) * (t[-1]-t[0]) / scale + t[0]
        t_prompt = meta["prompt"][i, 0] * 16

        for j in range(nx):
            axes[0, 1].plot(t_origin * 16, torch.zeros_like(t_origin) + j, ".", color="r")

        for j in range(nx):
            axes[0, 1].plot(t, prompt_center[i, 0, j, :] + j, "b")
            axes[0, 1].plot(t, meta["prompt_center"][i, 0, j, :] + j, "--C0")
            axes[0, 1].plot(t, meta["prompt_mask"][i, 0, j, :] + j, ":", color="gray")
        axes[0, 1].plot([t_prompt, t_prompt], [-0.05, 1.05], "r", linewidth=1.0)
        axes[0, 1].set_xlim(t[0], t[-1])
        # axes[0, 1].set_xticklabels([])
        axes[0, 1].grid("on")

        t = torch.arange(nt) * dt
        for j in range(nx):
            axes[1, 0].plot(t, phase[i, 1, j, :] + j, "b")
            axes[1, 0].plot(t, phase[i, 2, j, :] + j, "r")
            axes[1, 0].plot(t, meta["phase_pick"][i, 1, j, :] + j, "--C0")
            axes[1, 0].plot(t, meta["phase_pick"][i, 2, j, :] + j, "--C3")
            axes[1, 0].plot(t, meta["phase_mask"][i, 0, j, :] + j, ":", color="gray")
        axes[1, 0].set_xlim(t[0], t[-1])
        # axes[1, 0].set_xticklabels([])
        axes[1, 0].grid("on")

        t = torch.arange(nt_polarity) * dt_polarity
        for j in range(nx):
            axes[2, 0].plot(t, polarity[i, 0, j, :] + j, "b")
            axes[2, 0].plot(t, meta["polarity"][i, 0, j, :] + j, "--C0")
            axes[2, 0].plot(t, meta["polarity_mask"][i, 0, j, :] + j, ":", color="gray")
        axes[2, 0].set_xlim(t[0], t[-1])
        # axes[2, 0].set_xticklabels([])
        axes[2, 0].grid("on")

        t = torch.arange(nt_event) * dt_event
        for j in range(nx):
            axes[1, 1].plot(t, event_center[i, 0, j, :] + j, "b")
            axes[1, 1].plot(t, meta["event_center"][i, 0, j, :] + j, "--C0")
            axes[1, 1].plot(t, meta["event_mask"][i, 0, j, :] + j, ":", color="gray")
        axes[1, 1].set_xlim(t[0], t[-1])
        axes[1, 1].grid("on")

        # axes2 = axes[1, 1].twinx()
        # axes2.plot(t, event_time[i, 0, :, 0], "--C1")
        # axes2.plot(t, meta["event_time"][i, 0, :, 0], ":C3")

        axes[2, 1].scatter(meta["position"][i, :, 0, 1], meta["position"][i, :, 0, 2], color="b", s=10)
        axes[2, 1].scatter(meta["prompt"][i, 1], meta["prompt"][i, 2], color="r", s=10)

        # axes[2, 1].scatter(meta["station_location"][i, 0, :], meta["station_location"][i, 1, :], color="b", s=10)
        # axes[2, 1].scatter(meta["station_location"][i, 0, 0], meta["station_location"][i, 1, 0], color="r", s=10)

        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{rank:02d}_{i:02d}_{prefix}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{prefix}.png", dpi=300)

        if i >= 20:
            break


def plot_phasenet(
    meta,
    phase,
    phase_picks=None,
    dt=0.01,
    nt=6000 * 10,
    file_name=None,
    figure_dir="figures",
    **kwargs,
):
    nb, nc, ns, nt = phase.shape
    chn_name = ["E", "N", "Z"]
    if isinstance(dt, torch.Tensor):
        dt = dt.item()

    if "begin_time" in meta:
        begin_time = meta["begin_time"]
    else:
        begin_time = [0] * nb

    for i in range(nb):

        plt.close("all")

        chn_name = ["E", "N", "Z"]

        fig, axes = plt.subplots(4, 1, figsize=(10, 5))
        idx = 0
        for j in range(3):
            t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
            axes[j + idx].plot(t, meta["data"][i, j, 0, :], lw=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + idx].set_xlim(t[0], t[-1])
            axes[j + idx].set_xticklabels([])
            axes[j + idx].grid("on")
            axes[j + idx].legend(loc="upper right")

        idx = 3
        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
        axes[idx].plot(t, phase[i, 2, 0, :], "r", lw=1.0)
        axes[idx].plot(t, phase[i, 1, 0, :], "b", lw=1.0)
        color = {"P": "b", "S": "r"}
        for ii, pick in enumerate(phase_picks[i]):
            tt = pd.to_datetime(pick["phase_time"])
            axes[idx].plot([tt, tt], [-0.05, 1.05], f"--{color[pick['phase_type']]}", linewidth=0.8)

        axes[idx].plot([], [], "-b", label="P-phase")
        axes[idx].plot([], [], "-r", label="S-phase")
        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].set_xticklabels([])
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")

        fig.savefig(
            os.path.join(figure_dir, file_name[i].replace("/", "_") + f"_{ii:06d}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def plot_phasenet_tf(
    meta,
    phase,
    polarity=None,
    event_center=None,
    event_time=None,
    phase_picks=None,
    event_detects=None,
    dt=0.01,
    nt=6000 * 10,
    file_name=None,
    figure_dir="figures",
):
    # nb, nc, nt, ns = meta["data"].shape
    nb, nc, ns, nt = meta["data"].shape
    if isinstance(dt, torch.Tensor):
        dt = dt.item()
    nt_event = event_center.shape[-1]
    dt_event = dt * nt / nt_event
    # nt_polarity = polarity.shape[-1]
    # dt_polarity = dt * nt / nt_polarity

    if "begin_time" in meta:
        begin_time = meta["begin_time"]
    else:
        begin_time = [0] * nb

    for i in range(meta["data"].shape[0]):
        plt.close("all")

        chn_name = ["E", "N", "Z"]

        fig, axes = plt.subplots(8, 1, figsize=(10, 10))
        idx = 0

        for j in range(3):
            t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
            axes[idx + j].plot(t, meta["data"][i, j, 0, :], lw=0.5, color="k", label=f"{chn_name[j]}")
            axes[idx + j].set_xlim(t[0], t[-1])
            axes[idx + j].set_xticklabels([])
            axes[idx + j].grid("on")
            axes[idx + j].legend(loc="upper right")

        idx = 3
        if meta["spectrogram"].shape[1] == 6:
            meta["spectrogram"] = meta["spectrogram"][:, ::2, :, :]
        for j in range(3):
            # vmax = meta["spectrogram"][i, j, :, :].abs().max().item()
            # vmax = 6
            vmax = torch.std(meta["spectrogram"][i, j, :, :]) * 3
            # axes[idx + j].pcolormesh(meta["spectrogram"][i, j, :, :], cmap="seismic", vmin=-vmax, vmax=vmax)
            axes[idx + j].pcolormesh(meta["spectrogram"][i, j, :, :], vmin=0, vmax=vmax)
            axes[idx + j].set_xticklabels([])

        idx = 6
        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
        axes[idx].plot(t, phase[i, 2, 0, :], "r", lw=1.0)
        axes[idx].plot(t, phase[i, 1, 0, :], "b", lw=1.0)
        color = {"P": "b", "S": "r"}
        for ii, pick in enumerate(phase_picks[i]):
            tt = pd.to_datetime(pick["phase_time"])
            axes[idx].plot([tt, tt], [-0.05, 1.05], f"--{color[pick['phase_type']]}", linewidth=0.8)

        axes[idx].plot([], [], "-b", label="P-phase")
        axes[idx].plot([], [], "-r", label="S-phase")
        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].set_xticklabels([])
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")

        # t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt_polarity, freq=pd.Timedelta(seconds=dt_polarity))
        # # axes[k + 1].plot(t, (polarity[i, 0, :, 0] - 0.5) * 2.0, "b", label="polarity")
        # # axes[k + 1].plot(t, polarity[i, 1, :, 0] - polarity[i, 2, :, 0], "b", label="Polarity")
        # for ii, pick in enumerate(phase_picks[i]):
        #     tt = pd.to_datetime(pick["phase_time"])
        #     amp = pick["phase_polarity"]
        #     if abs(amp) > 0.15:
        #         axes[k + 1].annotate(
        #             "",
        #             xy=(tt, -0.03 * np.sign(amp)),
        #             xytext=(tt, amp),
        #             arrowprops=dict(arrowstyle="<-", color=f"{color[pick['phase_type']]}", lw=1.5),
        #         )
        # axes[k + 1].plot([], [], "-b", label="P-polarity")
        # axes[k + 1].plot([], [], "-r", label="S-polarity")
        # axes[k + 1].plot([t[0], t[-1]], [0.0, 0.0], "-", color="blue", lw=1.0)
        # axes[k + 1].set_xlim(t[0], t[-1])
        # axes[k + 1].set_ylim(-1.05, 1.05)
        # axes[k + 1].set_xticklabels([])
        # axes[k + 1].grid("on")
        # axes[k + 1].legend(loc="upper right")

        idx = 7
        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt_event, freq=pd.Timedelta(seconds=dt_event))
        axes[idx].plot(t, event_center[i, 0, 0, :], "b", label="Event")
        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")
        axes[idx].set_xlabel("Time (s)")

        # axes2 = axes[k + 2].twinx()
        # axes2.plot(t, event_time[i, 0, :, 0], "--C1")
        # axes2.set_ylabel("Time (s)")

        for ii, event in enumerate(event_detects[i]):
            ot = pd.to_datetime(event["event_time"])
            axes[idx].plot([ot, ot], [-0.05, 1.05], "--r", linewidth=2.0)
            at = pd.to_datetime(event["center_time"])
            axes[idx].plot([at, at], [-0.05, 1.05], "--b", linewidth=1.0)
            axes[idx].annotate(
                "",
                xy=(max(t[0], at), 0.3),
                xytext=(max(t[0], ot), 0.3),
                arrowprops=dict(arrowstyle="<-", color="C1", lw=2),
            )
        axes[idx].plot([], [], "--C3", label="Origin time")

        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")
        axes[idx].set_xlabel("Time (s)")

        fig.tight_layout()

        fig.savefig(
            os.path.join(figure_dir, file_name[i].replace("/", "_") + ".png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def plot_phasenet_plus(
    meta,
    phase,
    polarity=None,
    event_center=None,
    event_time=None,
    phase_picks=None,
    event_detects=None,
    dt=0.01,
    nt=6000 * 10,
    file_name=None,
    figure_dir="figures",
    **kwargs,
):
    # nb, nc, nt, ns = meta["data"].shape
    nb, nc, ns, nt = meta["data"].shape
    if isinstance(dt, torch.Tensor):
        dt = dt.item()
    nt_event = event_center.shape[-1]
    dt_event = dt * nt / nt_event
    nt_polarity = polarity.shape[-1]
    dt_polarity = dt * nt / nt_polarity

    if "begin_time" in meta:
        begin_time = meta["begin_time"]
    else:
        begin_time = [0] * nb

    for i in range(meta["data"].shape[0]):
        plt.close("all")

        chn_name = ["E", "N", "Z"]

        # if "raw_data" in meta:
        #     fig, axes = plt.subplots(9, 1, figsize=(10, 10))
        #     t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
        #     for j in range(3):
        #         axes[j].plot(t, meta["raw_data"][i, j, :, 0], lw=0.5, color="k", label=f"{chn_name[j]}")
        #         axes[j].set_xlim(t[0], t[-1])
        #         axes[j].set_xticklabels([])
        #         axes[j].grid("on")
        #         axes[j + idx].legend(loc="upper right")
        # else:
        fig, axes = plt.subplots(6, 1, figsize=(10, 10))
        idx = 0
        for j in range(3):
            t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
            axes[j + idx].plot(t, meta["data"][i, j, 0, :], lw=0.5, color="k", label=f"{chn_name[j]}")
            axes[j + idx].set_xlim(t[0], t[-1])
            axes[j + idx].set_xticklabels([])
            axes[j + idx].grid("on")
            axes[j + idx].legend(loc="upper right")

        idx = 3
        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt, freq=pd.Timedelta(seconds=dt))
        axes[idx].plot(t, phase[i, 2, 0, :], "r", lw=1.0)
        axes[idx].plot(t, phase[i, 1, 0, :], "b", lw=1.0)
        color = {"P": "b", "S": "r"}
        for ii, pick in enumerate(phase_picks[i]):
            tt = pd.to_datetime(pick["phase_time"])
            axes[idx].plot([tt, tt], [-0.05, 1.05], f"--{color[pick['phase_type']]}", linewidth=0.8)

        axes[idx].plot([], [], "-b", label="P-phase")
        axes[idx].plot([], [], "-r", label="S-phase")
        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].set_xticklabels([])
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")

        idx = 4
        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt_polarity, freq=pd.Timedelta(seconds=dt_polarity))
        # axes[idx].plot(t, (polarity[i, 0, :, 0] - 0.5) * 2.0, "b", label="polarity")
        # axes[idx].plot(t, polarity[i, 1, :, 0] - polarity[i, 2, :, 0], "b", label="Polarity")
        for ii, pick in enumerate(phase_picks[i]):
            tt = pd.to_datetime(pick["phase_time"])
            amp = pick["phase_polarity"]
            if abs(amp) > 0.15:
                axes[idx].annotate(
                    "",
                    xy=(tt, -0.03 * np.sign(amp)),
                    xytext=(tt, amp),
                    arrowprops=dict(arrowstyle="<-", color=f"{color[pick['phase_type']]}", lw=1.5),
                )
        axes[idx].plot([], [], "-b", label="P-polarity")
        axes[idx].plot([], [], "-r", label="S-polarity")
        axes[idx].plot([t[0], t[-1]], [0.0, 0.0], "-", color="blue", lw=1.0)
        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-1.05, 1.05)
        axes[idx].set_xticklabels([])
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")

        idx = 5
        t = pd.date_range(pd.Timestamp(begin_time[i]), periods=nt_event, freq=pd.Timedelta(seconds=dt_event))
        axes[idx].plot(t, event_center[i, 0, 0, :], "b", label="Event")
        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")
        axes[idx].set_xlabel("Time (s)")

        # axes2 = axes[idx].twinx()
        # axes2.plot(t, event_time[i, 0, :, 0], "--C1")
        # axes2.set_ylabel("Time (s)")

        for ii, event in enumerate(event_detects[i]):
            ot = pd.to_datetime(event["event_time"])
            axes[idx].plot([ot, ot], [-0.05, 1.05], "--r", linewidth=2.0)
            at = pd.to_datetime(event["center_time"])
            axes[idx].plot([at, at], [-0.05, 1.05], "--b", linewidth=1.0)
            axes[idx].annotate(
                "",
                xy=(max(t[0], at), 0.3),
                xytext=(max(t[0], ot), 0.3),
                arrowprops=dict(arrowstyle="<-", color="C1", lw=2),
            )
        axes[idx].plot([], [], "--C3", label="Origin time")

        axes[idx].set_xlim(t[0], t[-1])
        axes[idx].set_ylim(-0.05, 1.05)
        axes[idx].grid("on")
        axes[idx].legend(loc="upper right")
        axes[idx].set_xlabel("Time (s)")

        fig.tight_layout()

        fig.savefig(
            os.path.join(figure_dir, file_name[i].replace("/", "_") + ".png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def plot_phasenet_prompt(
    meta,
    phase,
    polarity=None,
    event_center=None,
    event_time=None,
    prompt_center=None,
    phase_picks=None,
    event_detects=None,
    dt=0.01,
    nt=6000 * 10,
    file_name=None,
    figure_dir="figures",
    **kwargs,
):
    # nb, nc, nt, ns = meta["data"].shape
    nb, nc, ns, nt = meta["data"].shape
    print(f"{nb = }{nc = }{ns = }{nt = }")
    dt = 0.01
    nt_event = event_center.shape[-1]
    dt_event = dt * nt / nt_event
    nt_polarity = polarity.shape[-1]
    dt_polarity = dt * nt / nt_polarity
    nt_prompt = prompt_center.shape[-1]
    dt_prompt = dt * nt / nt_prompt

    normalize = lambda x: x / x.abs().max()

    for i in range(meta["data"].shape[0]):

        plt.close("all")

        fig, axes = plt.subplots(3, 2, figsize=(15, 3 + ns * 0.8))

        t = torch.arange(nt) * dt
        for j in range(ns):
            axes[0, 0].plot(t, normalize(meta["data"][i, -1, j, :]) + j, linewidth=0.5, color="k")
        axes[0, 0].set_xlim(t[0], t[-1])
        # axes[0, 0].set_xticklabels([])
        axes[0, 0].grid("on")

        # t = torch.arange(nt_prompt) * dt_prompt
        t = meta["position"][i, 0, :, 0]
        t_origin = t.clone()
        ## HARD CODE
        dt = t[1] - t[0]
        t = t[:, None] * 16 + torch.arange(16) * dt
        t = t.flatten()

        # scale = meta["position"][i, -1, 0, 0] - meta["position"][i, 0, 0, 0]  # B, T, S, 3
        # # t_prompt = (meta["prompt"][i, 0] - meta["position"][i, 0, 0, 0]) * (t[-1]-t[0]) / scale + t[0]
        t_prompt = meta["prompt"][i, 0] * 16

        for j in range(ns):
            axes[0, 1].plot(t_origin * 16, torch.zeros_like(t_origin) + j, ".", color="r")

        for j in range(ns):
            axes[0, 1].plot(t, prompt_center[i, 0, j, :] + j, "b")
            axes[0, 1].plot(t, meta["prompt_center"][i, 0, j, :] + j, "--C0")
            axes[0, 1].plot(t, meta["prompt_mask"][i, 0, j, :] + j, ":", color="gray")
        # axes[0, 1].axvline(t_prompt, color="r")
        axes[0, 1].plot([t_prompt, t_prompt], [-0.05, 1.05], "r", linewidth=1.0)
        axes[0, 1].set_xlim(t[0], t[-1])
        # axes[0, 1].set_xticklabels([])
        axes[0, 1].grid("on")

        t = torch.arange(nt) * dt

        for j in range(ns):
            axes[1, 0].plot(t, phase[i, 1, j, :] + j, "b")
            axes[1, 0].plot(t, phase[i, 2, j, :] + j, "r")
            axes[1, 0].plot(t, meta["phase_pick"][i, 1, j, :] + j, "--C0")
            axes[1, 0].plot(t, meta["phase_pick"][i, 2, j, :] + j, "--C3")
            axes[1, 0].plot(t, meta["phase_mask"][i, 0, j, :] + j, ":", color="gray")
        axes[1, 0].set_xlim(t[0], t[-1])
        # axes[1, 0].set_xticklabels([])
        axes[1, 0].grid("on")

        t = torch.arange(nt_polarity) * dt_polarity
        for j in range(ns):
            axes[2, 0].plot(t, polarity[i, 0, j, :] + j, "b")
            axes[2, 0].plot(t, meta["polarity"][i, 0, j, :] + j, "--C0")
            axes[2, 0].plot(t, meta["polarity_mask"][i, 0, j, :] + j, ":", color="gray")
        axes[2, 0].set_xlim(t[0], t[-1])
        # axes[2, 0].set_xticklabels([])
        axes[2, 0].grid("on")

        t = torch.arange(nt_event) * dt_event
        for j in range(ns):
            axes[1, 1].plot(t, event_center[i, 0, j, :] + j, "b")
            axes[1, 1].plot(t, meta["event_center"][i, 0, j, :] + j, "--C0")
            axes[1, 1].plot(t, meta["event_mask"][i, 0, j, :] + j, ":", color="gray")
        axes[1, 1].set_xlim(t[0], t[-1])
        axes[1, 1].grid("on")

        # axes2 = axes[1, 1].twinx()
        # axes2.plot(t, event_time[i, 0, :, 0], "--C1")
        # axes2.plot(t, meta["event_time"][i, 0, :, 0], ":C3")

        fig.savefig(
            os.path.join(figure_dir, file_name[i].replace("/", "_") + ".png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def plot_eqnet_train(meta, phase, event, epoch, figure_dir="figures"):
    for i in range(meta["data"].shape[0]):
        plt.close("all")
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        for j in range(phase.shape[-1]):
            axes[0].plot((meta["data"][i, -1, :, j]) / torch.std(meta["data"][i, -1, :, j]) / 8 + j)

            axes[1].plot(phase[i, 1, :, j] + j, "r")
            axes[1].plot(phase[i, 2, :, j] + j, "b")
            axes[1].plot(meta["phase_pick"][i, 1, :, j] + j, "--C3")
            axes[1].plot(meta["phase_pick"][i, 2, :, j] + j, "--C0")

            axes[2].plot(event[i, :, j] + j, "b")
            axes[2].plot(meta["event_center"][i, :, j] + j, "--C0")

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}_{local_rank}.png", dpi=300)
        else:
            fig.savefig(f"{figure_dir}/{epoch:02d}_{i:02d}.png", dpi=300)


def plot_das(data, pred, picks=None, phases=["P", "S"], file_name=None, figure_dir="./figures", epoch=0, **kwargs):
    ## pytorch BCHW => BHWC
    data = normalize(data)
    data = np.transpose(data, [0, 2, 3, 1])
    pred = np.transpose(pred, [0, 2, 3, 1])
    if pred.shape[-1] < 3:
        pred_ = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 3))
        pred_[:, :, :, : pred.shape[-1]] = pred
        pred = pred_
    if pred.shape[-1] == 4:
        pred = pred[:, :, :, 1:]

    if file_name is None:
        file_name = [f"{epoch:03d}_{i:03d}" for i in range(len(data))]
    file_name = [x if isinstance(x, str) else x.decode() for x in file_name]

    if "dx" in kwargs:
        if type(kwargs["dx"]) is list:
            dx = [kwargs["dx"][i].item() for i in range(len(data))]
        else:
            dx = [kwargs["dx"].item() for i in range(len(data))]
    else:
        dx = [10.0 for i in range(len(data))]
    if "dt" in kwargs:
        if type(kwargs["dt"]) is list:
            dt = [kwargs["dt"][i].item() for i in range(len(data))]
        else:
            dt = [kwargs["dt"].item() for i in range(len(data))]
    else:
        dt = [0.01 for i in range(len(data))]

    if ("begin_channel_index" in kwargs) and (kwargs["begin_channel_index"] is not None):
        begin_channel_index = [x.item() for x in kwargs["begin_channel_index"]]
    else:
        begin_channel_index = [0 for i in range(len(data))]
    if ("begin_time_index" in kwargs) and (kwargs["begin_time_index"] is not None):
        begin_time_index = [x.item() for x in kwargs["begin_time_index"]]
    else:
        begin_time_index = [0 for i in range(len(data))]

    nt, nx = data.shape[1], data.shape[2]
    # x = np.arange(nx) * dx
    # t = np.arange(nt) * dt

    for i in range(len(data)):
        if (picks is not None) and (len(picks[i]) > 0):
            picks_ = pd.DataFrame(picks[i])  # picks per file

        # std = np.std(data[i, :, :, 0])
        std = torch.std(data[i, :, :, 0]).item()

        # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
        # fig, axs = plt.subplots(1, 1)
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        # im = axs[0].pcolormesh(
        #     (np.arange(nx) + begin_channel_index[i]) * dx[i] / 1e3,  # km
        #     (np.arange(nt) + begin_time_index[i]) * dt[i],
        #     data[i, :, :, 0],
        #     vmin=-std,
        #     vmax=std,
        #     cmap="seismic",
        #     shading="auto",
        #     rasterized=True,
        # )
        im = axs[0].imshow(
            data[i, :, :, 0],
            extent=[
                begin_channel_index[i] * dx[i] / 1e3,
                (begin_channel_index[i] + nx) * dx[i] / 1e3,
                (begin_time_index[i] + nt) * dt[i],
                begin_time_index[i] * dt[i],
            ],
            vmin=-std,
            vmax=std,
            cmap="seismic",
            aspect="auto",
            interpolation="none",
            origin="upper",
        )
        axs[0].set_xlabel("Distance (km)")
        axs[0].set_ylabel("Time (s)")
        # axs[0].invert_yaxis()
        axs[0].xaxis.tick_top()
        axs[0].xaxis.set_label_position("top")

        # im = axs[0, 0].imshow(
        #     data[i, :, :, 0],
        #     vmin=-std,
        #     vmax=std,
        #     cmap="seismic",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im0, ax=axs[0])
        # axs[0].set_title("DAS data")

        # im2 = axs[1].pcolormesh(x, t, pred[i, :, :, 2],  cmap="seismic", vmin=-1, vmax=1, alpha=0.5, shading='auto', rasterized=True)
        # im1 = axs[1].pcolormesh(x, t, -pred[i, :, :, 1],  cmap="seismic", vmin=-1, vmax=1, alpha=0.5, shading='auto', rasterized=True)

        # axs[1].invert_yaxis()
        # # plt.colorbar(im1, ax=axs[1])
        # axs[1].set_title("Prediction")

        # im = axs[0, 1].imshow(
        #     pred[i, :, :, 1],
        #     vmin=0,
        #     # vmax=0.5,
        #     cmap="hot",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im, ax=axs[0, 1])
        # axs[0, 1].set_title("P-phase")

        # im = axs[1, 0].imshow(
        #     pred[i, :, :, 2],
        #     vmin=0,
        #     # vmax=0.5,
        #     cmap="hot",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im, ax=axs[1, 0])
        # axs[1, 0].set_title("S-phase")

        # # axs[1].pcolormesh(1-pred[i, :, :, 0], vmin=0, vmax=1, cmap="hot", rasterized=True)
        # # axs[1].invert_yaxis()
        # im = axs[1, 1].imshow(
        #     1 - pred[i, :, :, 0],
        #     vmin=0,
        #     # vmax=0.5,
        #     cmap="hot",
        #     interpolation='none',
        #     aspect="auto",
        # )
        # plt.colorbar(im, ax=axs[1, 1])
        # axs[1, 1].set_title("(P+S)")

        if (picks is not None) and (len(picks[i]) > 0):
            for phase in phases:
                tmp_picks = picks_[picks_["phase_type"] == phase]
                axs[0].plot(
                    tmp_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
                    tmp_picks["phase_index"] * dt[i],
                    # ".C0",
                    # ".C2",
                    # linewidth=5,
                    # color=
                    ".",
                    linewidth=0.0,
                    markersize=0.2,
                    alpha=0.7,
                    label=f"{phase}-phase",
                )

            # p_picks = picks_[picks_["phase_type"] == "P"]
            # s_picks = picks_[picks_["phase_type"] == "S"]
            # ps_picks = picks_[picks_["phase_type"] == "PS"]
            # sp_picks = picks_[picks_["phase_type"] == "SP"]
            # axs[0].plot(
            #     p_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     p_picks["phase_index"] * dt[i],
            #     # ".C0",
            #     ".C2",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.5,
            #     alpha=1.0,
            #     label="P-phase",
            # )
            # axs[0].plot(
            #     s_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     s_picks["phase_index"] * dt[i],
            #     # "-C3",
            #     # ".C2",
            #     ".C0",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.5,
            #     alpha=1.0,
            #     label="S-phase",
            # )

            # axs[0].plot(
            #     sp_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     sp_picks["phase_index"] * dt[i],
            #     # ".C0",
            #     ".C1",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.05,
            #     alpha=0.5,
            #     label="SP-phase",
            # )
            # axs[0].plot(
            #     ps_picks["station_id"].astype("int") * dx[i] / 1e3,  # km
            #     ps_picks["phase_index"] * dt[i],
            #     # "-C3",
            #     # ".C2",
            #     ".C3",
            #     # linewidth=5,
            #     linewidth=0.0,
            #     markersize=0.05,
            #     alpha=0.5,
            #     label="PS-phase",
            # )

            axs[0].legend(markerscale=20.0)
            # axs[1].plot(p_picks["station_id"], p_picks["phase_index"], "r,", linewidth=0)
            # axs[1].plot(s_picks["station_id"], s_picks["phase_index"], "b,", linewidth=0)

        # im = axs[1].pcolormesh(
        #     (np.arange(nx) + begin_channel_index[i]) * dx[i] / 1e3,  # km
        #     (np.arange(nt) + begin_time_index[i]) * dt[i],
        #     pred[i, :, :, 0],
        #     vmin=0,
        #     vmax=1,
        #     cmap="hot",
        #     shading="auto",
        #     rasterized=True,
        # )
        im = axs[1].imshow(
            pred[i, :, :, :],
            extent=[
                begin_channel_index[i] * dx[i] / 1e3,
                (begin_channel_index[i] + nx) * dx[i] / 1e3,
                (begin_time_index[i] + nt) * dt[i],
                begin_time_index[i] * dt[i],
            ],
            aspect="auto",
            interpolation="none",
            origin="upper",
        )
        # axs[1].set_xlabel("Distance (km)")
        axs[1].set_ylabel("Time (s)")
        # axs[1].invert_yaxis()
        axs[1].xaxis.tick_top()
        # axs[1].xaxis.set_label_position("top")

        fig.tight_layout()
        try:
            fig.savefig(
                os.path.join(figure_dir, file_name[i] + ".png"),
                bbox_inches="tight",
                dpi=300,
            )
        except FileNotFoundError:
            os.makedirs(os.path.dirname(os.path.join(figure_dir, file_name[i])), exist_ok=True)
            fig.savefig(
                os.path.join(figure_dir, file_name[i] + ".png"),
                bbox_inches="tight",
                dpi=300,
            )

        plt.close(fig)

    return 0
