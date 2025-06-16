import os
from pathlib import Path
from glob import glob

# %%
data_path = "/global/scratch/users/zhuwq0/quakeflow_das"
CWP = "/global/home/users/zhuwq0/scratch/EQNet/scripts"

# %%
folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south", "mammoth_north_100km", "mammoth_south_100km"]
for folder in folders:
    h5_list = glob(f"{data_path}/{folder}/data/*h5")
    print(f"Processing {folder}: {len(h5_list)} h5 files")

    # %%
    result_path = Path(f"{CWP}/results/phasenet/{folder}")
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # %%
    with open(result_path / f"h5_list.txt", "w") as f:
        for i, h5 in enumerate(h5_list):
            f.write(h5 + "\n")