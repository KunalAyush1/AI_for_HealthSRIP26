import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import pickle


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vis import load_signal, load_events




def bandpass_filter(signal, fs, low=0.17, high=0.4, order=4):
    """
    Apply a Butterworth bandpass filter to retain breathing-related frequencies.
    """
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, signal)




def convert_to_seconds(df, start_time):
    """
    Convert timestamps to seconds relative to start_time.
    """
    df = df.copy()
    df["time_sec"] = (df["timestamp"] - start_time).dt.total_seconds()
    return df




def resample_spo2(spo2_df, target_time):
    """
    Resample SpO₂ to match airflow timeline.
    """
    f = interp1d(
        spo2_df["time_sec"],
        spo2_df["value"],
        kind="linear",
        fill_value="extrapolate"
    )
    return f(target_time)




def create_windows(signal_len, fs=32, window_sec=30, overlap=0.5):
    """
    Create overlapping windows.
    """
    window_size = int(window_sec * fs)
    step_size = int(window_size * (1 - overlap))

    windows = []
    for start in range(0, signal_len - window_size + 1, step_size):
        end = start + window_size
        windows.append((start, end))

    return windows




def convert_events_to_seconds(events_df, start_time):
    """
    Convert event timestamps to seconds.
    """
    events = []

    for _, row in events_df.iterrows():
        start = (row["start"] - start_time).total_seconds()
        end = (row["end"] - start_time).total_seconds()

        events.append({
            "start": start,
            "end": end,
            "label": row["label"]
        })

    return events




def map_label(label):
    """
    Map raw labels to simplified classes.
    """
    if label == "Normal":
        return "Normal"
    elif "Hypopnea" in label:
        return "Hypopnea"
    elif "Apnea" in label:
        return "Apnea"
    else:
        return "Ignore"


def get_label(start_idx, end_idx, events, fs=32):
    """
    Assign label based on overlap with events.
    """
    start_sec = start_idx / fs
    end_sec = end_idx / fs

    best_label = "Normal"
    max_overlap = 0

    for event in events:
        overlap = max(0, min(end_sec, event["end"]) - max(start_sec, event["start"]))
        window_length = end_sec - start_sec

        if overlap / window_length >= 0.5 and overlap > max_overlap:
            max_overlap = overlap
            best_label = event["label"]

    return best_label




def process_participant(folder_path):

    files = os.listdir(folder_path)

    signal_files = [f for f in files if f.endswith(".txt") and "event" not in f.lower()]

    airflow_file = [f for f in signal_files if "flow" in f.lower()][0]
    thorac_file = [f for f in signal_files if "thorac" in f.lower()][0]
    spo2_file = [f for f in signal_files if "spo2" in f.lower()][0]
    event_file = [f for f in files if "event" in f.lower()][0]

    airflow = load_signal(os.path.join(folder_path, airflow_file))
    thorac = load_signal(os.path.join(folder_path, thorac_file))
    spo2 = load_signal(os.path.join(folder_path, spo2_file))
    events_df = load_events(os.path.join(folder_path, event_file))

    
    airflow = airflow.sort_values(by="timestamp")
    thorac = thorac.sort_values(by="timestamp")
    spo2 = spo2.sort_values(by="timestamp")

    start_time = airflow["timestamp"].iloc[0]

    airflow = convert_to_seconds(airflow, start_time)
    thorac = convert_to_seconds(thorac, start_time)
    spo2 = convert_to_seconds(spo2, start_time)

    
    airflow["value"] = bandpass_filter(airflow["value"].values, 32)
    thorac["value"] = bandpass_filter(thorac["value"].values, 32)

    
    target_time = airflow["time_sec"].values

    thorac_interp = np.interp(target_time, thorac["time_sec"], thorac["value"])
    spo2_resampled = resample_spo2(spo2, target_time)

    signal = np.stack([
        airflow["value"].values,
        thorac_interp,
        spo2_resampled
    ], axis=1)

    signal = np.nan_to_num(signal)

    windows = create_windows(len(signal))
    events = convert_events_to_seconds(events_df, start_time)

    X, y = [], []

    for start, end in windows:
        window_data = signal[start:end]
        label = get_label(start, end, events)

        X.append(window_data)
        y.append(label)

    

    participant_name = os.path.basename(folder_path)

    X_clean = []
    y_clean = []
    participant_ids = []

    for i in range(len(X)):
        mapped_label = map_label(y[i])

        if mapped_label == "Ignore":
            continue

        X_clean.append(X[i])
        y_clean.append(mapped_label)
        participant_ids.append(participant_name)

    return X_clean, y_clean, participant_ids




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", type=str, required=True)
    parser.add_argument("-out_dir", type=str, required=True)
    args = parser.parse_args()

    X_all = []
    y_all = []
    participant_all = []

    participants = sorted([
        p for p in os.listdir(args.in_dir)
        if os.path.isdir(os.path.join(args.in_dir, p))
    ])

    for p in participants:
        print(f"Processing {p}...")
        folder_path = os.path.join(args.in_dir, p)

        X, y, p_ids = process_participant(folder_path)

        X_all.extend(X)
        y_all.extend(y)
        participant_all.extend(p_ids)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    participant_all = np.array(participant_all)

    from collections import Counter

    print("\nLabel Distribution:")
    print(Counter(y_all))

    print("\nParticipant Distribution:")
    print(Counter(participant_all))

    os.makedirs(args.out_dir, exist_ok=True)

    save_path = os.path.join(args.out_dir, "dataset.pkl")

    with open(save_path, "wb") as f:
        pickle.dump((X_all, y_all, participant_all), f)

    print(f"\nDataset saved to: {save_path}")
    print(f"Shape: {X_all.shape}")


if __name__ == "__main__":
    main()