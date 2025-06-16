#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import argparse


def submit_jobs(num_nodes):
    # Process each folder
    folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", 
              "ridgecrest_south", "mammoth_north_100km", "mammoth_south_100km"]

    # partitions = ["lr6", "lr7", "lr8"]
    # partitions = ["lr6"]
    partitions = ["lr7"]


    for folder in folders:
        CWP = "/global/home/users/zhuwq0/scratch/EQNet/scripts"
        split_dir = Path(f"{CWP}/results/phasenet/{folder}/splits")
        if not split_dir.exists():
            print(f"Skipping {folder} - split lists not found")
            continue
            
        print(f"\nSubmitting jobs for {folder}:")
        for i in range(num_nodes):
            chunk_file = split_dir / f"chunk_{i:03d}_{num_nodes:03d}.txt"
            if not chunk_file.exists():
                continue
            if len(open(chunk_file).readlines()) == 0:
                continue
                
            # Submit SLURM job

            # cmd = ["sbatch", f"{CWP}/slurm/phasenet_job.sh", folder, f"{i:03d}", f"{num_nodes:03d}"]
            ## rotate between partitions
            cmd = ["sbatch", f"--partition={partitions[i % len(partitions)]}", f"{CWP}/slurm/phasenet_job.sh", folder, f"{i:03d}", f"{num_nodes:03d}"]
            print(" ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"Submitted job {job_id} for chunk {i:03d}")
            else:
                print(f"Failed to submit job for chunk {i:03d}: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description='Submit SLURM jobs for PhaseNet processing')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes to process the data')
    args = parser.parse_args()
    
    submit_jobs(args.num_nodes)

if __name__ == "__main__":
    main() 