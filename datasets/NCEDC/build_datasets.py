# %%
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import multiprocessing as mp

plt.rcParams["figure.facecolor"] = "white"

# %%
h5_file = "ncedc.h5"
event_csv = pd.read_hdf(h5_file, "events")
event_csv["time_id"] = event_csv["time"].apply(lambda x: x[0:4] + x[5:7] + x[8:10] + x[11:13] + x[14:16] + x[17:19] + x[20:22])
phase_csv = pd.read_hdf(h5_file, "catalog")
phase_csv.set_index("event_index", inplace=True)
with open("event_id.json", "r") as f:
    time_to_event_id = json.load(f)

# %%
pre_window = 3000
post_window = 6000
sampling_rate = 100
# plt.figure()

with h5py.File(h5_file, "r") as fp_in:
    # with h5py.File(output_path.joinpath(f"{event['index']:06}.h5"), "w") as fp_out:
    with h5py.File("ncedc_seismic_dataset.h5", "w") as fp_out:

        for i, (_, event) in tqdm(enumerate(event_csv.iterrows()), total=len(event_csv)):

            first_p_arrival = datetime.fromisoformat(phase_csv.loc[[event["index"]]]["p_time"].min())
            anchor_time = first_p_arrival  ## put anchor at 30s of the window
            # anchor_time = event_time

            # event_id = event["index"]
            event_id = "nc" + time_to_event_id[event["time_id"]]
            event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f")
            event_time_index = int((event_time - anchor_time).total_seconds() * sampling_rate) + pre_window
            event_latitude = event["latitude"]
            event_longitude = event["longitude"]
            event_depth_km = event["depth_km"]

            group = fp_out.create_group(f"{event_id}")
            group.attrs["event_id"] = event_id
            group.attrs["event_time"] = event_time.isoformat(timespec="milliseconds")
            group.attrs["event_time_index"] = event_time_index
            group.attrs["begin_time"] = (anchor_time - timedelta(pre_window/sampling_rate)).isoformat(timespec="milliseconds")
            group.attrs["end_time"] = (anchor_time + timedelta(post_window/sampling_rate)).isoformat(timespec="milliseconds")
            # group.attrs["time_reference"] = anchor_time.isoformat(timespec="milliseconds")
            # group.attrs["time_before"] = pre_window/sampling_rate
            # group.attrs["time_after"] = post_window/sampling_rate
            group.attrs["latitude"] = event_latitude
            group.attrs["longitude"] = event_longitude
            group.attrs["depth_km"] = event_depth_km
            group.attrs["magnitude"] = event["magnitude"]
            group.attrs["magnitude_type"] = event["magnitude_type"]
 
            for j, (_, phase) in enumerate(phase_csv.loc[[event["index"]]].iterrows()):

                trace = fp_in[f"data/{phase['fname']}"]
                p_time = datetime.fromisoformat(trace.attrs["p_time"])
                s_time = datetime.fromisoformat(trace.attrs["s_time"])
                p_arrival_index = int((p_time - anchor_time).total_seconds() * sampling_rate) + pre_window
                s_arrival_index = int((s_time - anchor_time).total_seconds() * sampling_rate) + pre_window

                begin_index = trace.attrs["p_idx"] - p_arrival_index
                end_index = begin_index + pre_window + post_window
                if begin_index < 0:
                    print(trace.attrs["p_idx"], p_arrival_index, begin_index, end_index)
                # if trace[begin_index:end_index,:].shape != (9000, 3):
                #     print(trace.shape, p_arrival_index, begin_index, end_index)

                waveform = np.zeros([pre_window + post_window, 3])
                waveform[: trace[max(0, begin_index) : max(0, end_index), :].shape[0], :] = trace[
                    max(0, begin_index) : max(0, end_index), :
                ] * 1e6
                network = trace.attrs["network"]
                station = trace.attrs["station"]
                location = trace.attrs["location_code"] if trace.attrs["location_code"] != "--" else ""
                channels = trace.attrs["channels"].split(",")

                begin_time = anchor_time - timedelta(seconds=pre_window / sampling_rate)
                end_time = anchor_time + timedelta(seconds=post_window / sampling_rate)
                distance_km = trace.attrs["distance_km"]
                azimuth = trace.attrs["azimuth"]
                first_motion = trace.attrs["first_motion"]
                if first_motion not in ["U", "D"]:
                    first_motion = "N"
                emergence_angle = trace.attrs["emergence_angle"]
                station_latitude = trace.attrs["station_latitude"]
                station_longitude = trace.attrs["station_longitude"]
                station_elevation_m = trace.attrs["station_elevation_m"]
                dt_s = trace.attrs["dt"]
                unit = trace.attrs["unit"]
                snr = trace.attrs["snr"]
                phase_type = ["P", "S"]
                phase_index = [p_arrival_index, s_arrival_index]
                phase_time = [p_time.isoformat(timespec="milliseconds"), s_time.isoformat(timespec="milliseconds")]
                phase_score = [phase["p_weight"], phase["s_weight"]]
                phase_remark = [phase["p_remark"], phase["s_remark"]]
                phase_polarity = [first_motion, 'N']
                event_ids = [event_id, event_id]
                assert dt_s == 1.0 / sampling_rate

                station_id = f"{network}.{station}.{location}.{channels[0][:-1]}"
                fp_out[f"{event_id}/{station_id}"] = waveform * 1e6
                attrs = fp_out[f"{event_id}/{station_id}"].attrs
                attrs["network"] = network
                attrs["station"] = station
                attrs["location"] = location
                attrs["component"] = [x[-1] for x in channels]
                attrs["distance_km"] = distance_km
                attrs["azimuth"] = azimuth
                attrs["emergence_angle"] = emergence_angle
                attrs["latitude"] = station_latitude
                attrs["longitude"] = station_longitude
                attrs["elevation_m"] = station_elevation_m
                attrs["dt_s"] = dt_s
                attrs["unit"] = "1e-6" + unit
                attrs["snr"] = snr
                attrs["phase_type"] = phase_type
                attrs["phase_index"] = phase_index
                attrs["phase_time"] = phase_time
                attrs["phase_score"] = phase_score
                attrs["phase_remark"] = phase_remark
                attrs["phase_polarity"] = phase_polarity
                attrs["event_id"] = event_ids

                # print(begin_time, end_time)
                # plt.plot(trace[begin_index:end_index,-1]/np.std(trace[begin_index:end_index,-1])/10 + j)
                # plt.plot([(p_time - begin_time).total_seconds()*sampling_rate, (p_time - begin_time).total_seconds()*sampling_rate], [j-2, j+2], '--r')
                # plt.plot([(s_time - begin_time).total_seconds()*sampling_rate, (s_time - begin_time).total_seconds()*sampling_rate], [j-2, j+2], '--b')

            # if i > 1000:
            #     break

# plt.show()

# %%
