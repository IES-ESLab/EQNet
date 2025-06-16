# %%
import fsspec
import h5py
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# %%
protocol = "file://"
bucket = "/global/scratch/users/zhuwq0/quakeflow_das"

# %%
fs = fsspec.filesystem(protocol.replace("://", ""))
# folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south"]
folders = ["mammoth_north_100km", "mammoth_south_100km"]


# %%
def maping_csv_h5(csv, h5_folder):
    tmp = csv.split("/")
    folder = tmp[-3]
    fname = tmp[-1]
    h5_path = f"{h5_folder}/{folder}/data/{fname.replace('.csv', '.h5')}"

    return h5_path



# %%
for folder in folders:
    for i, csv in enumerate(glob(f"./results/phasenet/{folder}/picks/*.csv")):
        picks = pd.read_csv(csv)
        # print(f"1C {len(picks)} picks")
        if len(picks) < 5000:
            continue
        h5_path = maping_csv_h5(csv, bucket)
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:, :]
        nx, nt = data.shape

        plt.figure(figsize=(8, 16))
        data -= np.mean(data, axis=-1, keepdims=True)
        data /= np.std(data, axis=-1, keepdims=True)
        vmax = np.std(data) * 5
        plt.pcolormesh(data, vmin=-vmax, vmax=vmax, cmap="gray")
        color = ["C0" if phase_type == "P" else "C3" for phase_type in picks["phase_type"]]
        plt.scatter(picks["phase_index"], picks["station_id"], color=color, s=10, marker=".")
        plt.show()

        # ## compare with 3C
        # csv = csv.replace("picks", "picks_3c")
        # picks = pd.read_csv(csv)
        # print(f"3C {len(picks)} picks")
        # # if len(picks) < 2000:
        # #     continue
        # h5_path = maping_csv_h5(csv, bucket)
        # with h5py.File(h5_path, "r") as f:
        #     data = f["data"][:, :]
        # nx, nt = data.shape

        # plt.figure(figsize=(8, 16))
        # data -= np.mean(data, axis=-1, keepdims=True)
        # data /= np.std(data, axis=-1, keepdims=True)
        # vmax = np.std(data) * 5
        # plt.pcolormesh(data, vmin=-vmax, vmax=vmax, cmap="gray")
        # color = ["C0" if phase_type == "P" else "C3" for phase_type in picks["phase_type"]]
        # plt.scatter(picks["phase_index"], picks["station_id"], color=color, s=10, marker=".")
        # plt.show()

        
        if i > 10:
            break

        # plt.figure()
        # begin = -500
        # end = 500
        # for i, row in picks.iterrows():
        #     station_id = row["station_id"]
        #     phase_index = row["phase_index"]
        #     phase_type = row["phase_type"]  # in P and S
        #     phase_score = row["phase_score"]
        #     color = "C0" if phase_type == "P" else "C1"
        #     vmin = data[station_id, max(0, phase_index + begin) : min(nt, phase_index + end)].min()
        #     vmax = data[station_id, max(0, phase_index + begin) : min(nt, phase_index + end)].max()
        #     plt.clf()
        #     plt.plot(data[station_id, :], alpha=0.5, color="k")
        #     plt.plot([phase_index, phase_index], [vmin * 2, vmax * 2], "-", color=color)
        #     plt.legend(f"{phase_index}: {phase_type}: {phase_score:.2f}")
        #     phase_indices = picks[
        #         (picks["station_id"] == station_id)
        #         & (picks["phase_index"] > max(0, phase_index + begin))
        #         & (picks["phase_index"] < min(nt, phase_index + end))
        #     ][["phase_index", "phase_type"]].values

        #     for phase_index, phase_type in phase_indices:
        #         color = "C0" if phase_type == "P" else "C1"
        #         plt.plot([phase_index, phase_index], [vmin * 2, vmax * 2], "--", color=color)
        #     plt.xlim(phase_index + begin, phase_index + end)

        #     plt.show()
        #     # plt.savefig(f"check_phasenet.png")
        #     # time.sleep(5)

# %%
