"""Download small debug datasets for overfit testing.

Downloads:
  - CEED: Records with polarity labels from GCS (scans multiple days)
  - DAS: 1 event H5 file + label CSV from GCS

Usage:
    python download_debug_data.py
"""
import os

DATA_DIR = "data/debug"


def download_ceed():
    """Download CEED data with polarity labels.

    Scans multiple days to find events with P-phase polarity ('U'/'D').
    Saves only records from events that have at least one polarity label.
    """
    from eqnet.data.ceed import load_quakeflow_dataset

    out_dir = os.path.join(DATA_DIR, "ceed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "001.parquet")

    # Scan days to find records with polarity (every record must have U/D)
    min_records = 50
    all_records = []

    for day in range(1, 31):
        print(f"  Scanning day {day:03d}...")
        try:
            ds = load_quakeflow_dataset(region="SC", years=[2025], days=[day])
        except Exception as e:
            print(f"    Skip: {e}")
            continue

        for record in ds:
            if record.get("p_phase_polarity") in ("U", "D") and record.get("p_phase_index") is not None:
                all_records.append(record)

        print(f"    Found {len(all_records)} records with polarity so far")
        if len(all_records) >= min_records:
            break

    if not all_records:
        print("  WARNING: No records with polarity found. Downloading raw day 001.")
        ds = load_quakeflow_dataset(region="SC", years=[2025], days=[1])
        all_records = list(ds)

    # Save as parquet via HuggingFace datasets (imported inside eqnet.data.ceed)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(all_records)
    pq.write_table(table, out_path)
    print(f"  Saved {len(all_records)} records (all with polarity) to {out_path}")


def download_das():
    """Download 1 DAS event (H5 + label CSV).

    Uses ci37329188 from mammoth_north — a well-labeled event with
    2746 P + 2746 S picks across 4670 channels.
    """
    import fsspec
    from eqnet.data.das import get_gcs_storage_options

    out_dir = os.path.join(DATA_DIR, "das")
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    storage_options = get_gcs_storage_options()

    # Hardcoded: good event with dense P/S picks
    event_id = "ci37329188"
    label_src = f"gs://quakeflow_das/training/training_v1/mammoth_north/labels/{event_id}.csv"
    data_src = f"gs://quakeflow_das/mammoth_north/data/{event_id}.h5"
    label_dst = os.path.join(out_dir, "labels", f"{event_id}.csv")
    data_dst = os.path.join(out_dir, "data", f"{event_id}.h5")

    print(f"Downloading DAS event: {event_id}")
    for src, dst, name in [(label_src, label_dst, "label"), (data_src, data_dst, "data")]:
        if os.path.exists(dst):
            print(f"  {name} already exists: {dst}")
            continue
        with fsspec.open(src, "rb", **storage_options) as f_in:
            with open(dst, "wb") as f_out:
                f_out.write(f_in.read())
        print(f"  Saved {name}: {dst}")

    # Write local label list
    label_list_dst = os.path.join(out_dir, "labels_debug.txt")
    with open(label_list_dst, "w") as f:
        f.write(f"labels/{event_id}.csv\n")
    print(f"  Label list: {label_list_dst}")


if __name__ == "__main__":
    print("Downloading debug datasets...")
    print()
    download_ceed()
    print()
    download_das()
    print()
    print(f"Done! Debug data saved to {DATA_DIR}/")
    print()
    print("Usage:")
    print("  ./train_debug.sh phasenet overfit")
    print("  ./train_debug.sh phasenet_das overfit")
