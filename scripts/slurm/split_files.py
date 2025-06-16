#!/usr/bin/env python3
import os
from pathlib import Path
import math
import argparse
from glob import glob

def split_file(input_file, num_nodes):
    """Split a file into roughly equal chunks."""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    chunk_size = math.ceil(total_lines / num_nodes)

    # existing csvs
    existing_csvs = glob(f"{input_file.parent}/picks/*.csv")
    processed = [x.replace("\n", "").split("/")[-1].replace(".csv", ".h5") for x in existing_csvs]
    print(f"Found {len(processed)} processed csv files")

    # Create output directory
    output_dir = Path(input_file).parent / "splits"
    output_dir.mkdir(exist_ok=True)
    
    # Split and write chunks
    for i in range(num_nodes):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_lines)
        chunk = lines[start_idx:end_idx]

        # filter out processed files
        chunk = [x for x in chunk if x.replace("\n", "").split("/")[-1] not in processed]
        
        if chunk:  # Only write if there are lines
            output_file = output_dir / f"chunk_{i:03d}_{num_nodes:03d}.txt"
            with open(output_file, 'w') as f:
                f.writelines(chunk)
            print(f"Created {output_file} with {len(chunk)} lines")

        else:
            output_file = output_dir / f"chunk_{i:03d}_{num_nodes:03d}.txt"
            if output_file.exists():
                output_file.unlink()
                print(f"Deleted {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Split h5_list.txt files for parallel processing')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes to split the data into')
    args = parser.parse_args()
    
    # Process each folder
    folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", 
              "ridgecrest_south", "mammoth_north_100km", "mammoth_south_100km"]

    CWP = "/global/home/users/zhuwq0/scratch/EQNet/scripts"
    
    for folder in folders:
        input_file = Path(f"{CWP}/results/phasenet/{folder}/h5_list.txt")
        if input_file.exists():
            print(f"\nProcessing {folder}:")
            split_file(input_file, args.num_nodes)

if __name__ == "__main__":
    main() 