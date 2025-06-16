#!/usr/bin/env python3
import os
from pathlib import Path
import fsspec
import subprocess

def check_and_upload():
    # Setup filesystem
    protocol = "file://"
    bucket = "/global/scratch/users/zhuwq0/quakeflow_das"
    PWD = "/global/home/users/zhuwq0/scratch/EQNet/scripts"
    fs = fsspec.filesystem(protocol.replace("://", ""))
    
    folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", 
              "ridgecrest_south", "mammoth_north_100km", "mammoth_south_100km"]
    
    for folder in folders:
        print(f"\nProcessing {folder}:")
        result_path = Path(f"{PWD}/results/phasenet/{folder}")
        
        # Check if picks directory exists
        picks_dir = result_path / "picks"
        if not picks_dir.exists():
            print(f"Warning: No picks directory found for {folder}")
            continue
            
        # Get all h5 files
        # h5_list = fs.glob(f"{bucket}/{folder}/data/*h5")
        # h5_files = {Path(h5).stem for h5 in h5_list}
        h5_dir = Path(f"{bucket}/{folder}/data")
        h5_files = {Path(f).stem for f in h5_dir.glob("*.h5")}
        
        # Get all CSV files
        csv_files = {f.stem for f in picks_dir.glob("*.csv")}
        
        # Check for missing CSVs
        missing_csvs = h5_files - csv_files
        if missing_csvs:
            print(f"Warning: Missing CSV files for {len(missing_csvs)}/{len(h5_files)} h5 files:")
            # for h5 in missing_csvs:
            #     print(f"  - {h5}")
        else:
            print(f"All {len(h5_files)} h5 files have corresponding CSV files")
            
        # Upload to Google Cloud
        print(f"\nUploading picks to Google Cloud for {folder}...")
        cmd = f"gsutil -m cp -r {picks_dir} gs://quakeflow_das/{folder}/phasenet/picks"
        print(f"{cmd}")
        # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # if result.returncode == 0:
        #     print(f"Successfully uploaded picks for {folder}")
        # else:
        #     print(f"Failed to upload picks for {folder}:")
        #     print(result.stderr)

if __name__ == "__main__":
    check_and_upload()