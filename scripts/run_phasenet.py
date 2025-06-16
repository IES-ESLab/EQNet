# %%
import os
from pathlib import Path

import fsspec

# %%
# protocol = "gs://"
# bucket = "quakeflow_das"
protocol = "file://"
bucket = "/global/scratch/users/zhuwq0/quakeflow_das"

# %%
fs = fsspec.filesystem(protocol.replace("://", ""))
folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south", "mammoth_north_100km", "mammoth_south_100km"]
for folder in folders:
    print(f"Processing {folder}:")
    h5_list = fs.glob(f"{bucket}/{folder}/data/*h5")

    # %%
    result_path = Path(f"results/phasenet/{folder}")
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # %%
    with open(result_path / f"h5_list.txt", "w") as f:
        for i, h5 in enumerate(h5_list):
            f.write(f"{protocol}" + h5 + "\n")

    # %%

# %%
for folder in folders:

    result_path = Path(f"results/phasenet/{folder}")
    
    MODEL_PATH = "/global/scratch/users/zhuwq0/PhaseNet"
    os.system("conda activate phasenet")
    cmd = f"python {MODEL_PATH}/phasenet/predict.py --model={MODEL_PATH}/model/190703-214543 --format das_event --data_list {result_path}/h5_list.txt --batch_size 1 --result_dir {result_path} --subdir_level=0"
    print(cmd)
    # os.system(cmd)

    # %%
    # cmd = f"gsutil -m cp -r {result_path}/picks gs://quakeflow_das/{folder}/phasenet/picks"
    # print(cmd)
    # os.system(cmd)
