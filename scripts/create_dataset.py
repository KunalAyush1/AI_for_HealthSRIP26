import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import pickle

# Fix import path (important if running from root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vis import load_signal, load_events




def bandpass_filter(signal, fs, low=0.17, high=0.4, order=4):
    """
    Apply a butterworth bandpass filter to retain breathing related frequencies
    Args:
        signal (np.ndarray): Input 1D Signal
        fs(int): Sampling frequency in Hz
        low(float): Lower cutoff frequency in Hz
        high(float): upper cutoff frequecy in Hz
        order(int): Filter order
        
    Returns:
        np.ndarray: Filtered Signal
    """
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, signal)




def convert_to_seconds(df, start_time):
    """
    Convert timestamps in a DataFrame to relative seconds from a reference start time.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' column.
        start_time (pd.Timestamp): Reference start time.

    Returns:
        pd.DataFrame: DataFrame with added 'time_sec' column.
    """
    df = df.copy()
    df["time_sec"] = (df["timestamp"] - start_time).dt.total_seconds()
    return df




def resample_spo2(spo2_df, target_time):
    """
    Resample SpO₂ signal to match a target time axis using interpolation.

    Args:
        spo2_df (pd.DataFrame): SpO₂ data with 'time_sec' and 'value'.
        target_time (np.ndarray): Target time points (in seconds).

    Returns:
        np.ndarray: Resampled SpO₂ values aligned to target_time.
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
    Generate overlapping window indices for segmenting the signal.

    Args:
        signal_len (int): Total number of samples in the signal.
        fs (int): Sampling frequency in Hz.
        window_sec (int): Window duration in seconds.
        overlap (float): Fractional overlap between windows (0–1).

    Returns:
        list: List of (start_idx, end_idx) tuples.
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
    Convert event timestamps to seconds relative to a reference start time.

    Args:
        events_df (pd.DataFrame): DataFrame with 'start', 'end', and 'label'.
        start_time (pd.Timestamp): Reference start time.

    Returns:
        list: List of dictionaries with 'start', 'end', and 'label' in seconds.
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




def get_label(start_idx, end_idx, events, fs=32):
    """
    Assign a label to a signal window based on overlap with annotated events.

    Args:
        start_idx (int): Start index of the window.
        end_idx (int): End index of the window.
        events (list): List of event dictionaries with 'start', 'end', 'label'.
        fs (int): Sampling frequency in Hz.

    Returns:
        str: Assigned label ('Normal' or event type).
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
    """
    Process all signals and annotations for a participant to create labeled windows.

    Steps:
    - Load signals and event data
    - Convert timestamps to seconds
    - Apply filtering to respiratory signals
    - Align and resample signals to a common timeline
    - Segment into overlapping windows
    - Assign labels based on event overlap

    Args:
        folder_path (str): Path to participant data folder.

    Returns:
        tuple: (X, y) where X is a list of windowed signals and y is list of labels.
    """
    files = os.listdir(folder_path)
    
    signal_files = [f for f in files if f.endswith(".txt") and "event" not in f.lower() ]

    airflow_file = [f for f in signal_files if "flow" in f.lower()][0]
    thorac_file = [f for f in signal_files if "thorac" in f.lower()][0]
    spo2_file = [f for f in signal_files if "spo2" in f.lower()][0]
    event_file = [f for f in files if "event" in f.lower()][0]

    airflow = load_signal(os.path.join(folder_path, airflow_file))
    thorac = load_signal(os.path.join(folder_path, thorac_file))
    spo2 = load_signal(os.path.join(folder_path, spo2_file))
    events_df = load_events(os.path.join(folder_path, event_file))

    start_time = airflow["timestamp"].iloc[0]

    airflow = convert_to_seconds(airflow, start_time)
    thorac = convert_to_seconds(thorac, start_time)
    spo2 = convert_to_seconds(spo2, start_time)

    # Filtering
    airflow["value"] = bandpass_filter(airflow["value"].values, 32)
    thorac["value"] = bandpass_filter(thorac["value"].values, 32)

    # Align signals
    target_time = airflow["time_sec"].values

    thorac_interp = np.interp(target_time, thorac["time_sec"], thorac["value"])
    spo2_resampled = resample_spo2(spo2, target_time)

    signal = np.stack([
        airflow["value"].values,
        thorac_interp,
        spo2_resampled
    ], axis=1)

    windows = create_windows(len(signal))
    events = convert_events_to_seconds(events_df, start_time)

    X, y = [], []

    for start, end in windows:
        window_data = signal[start:end]
        label = get_label(start, end, events)

        X.append(window_data)
        y.append(label)

    return X, y




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", type=str, required=True)
    parser.add_argument("-out_dir", type=str, required=True)
    args = parser.parse_args()

    X_all = []
    y_all = []

    participants = sorted([
        p for p in os.listdir(args.in_dir)
        if os.path.isdir(os.path.join(args.in_dir, p))
    ])

    for p in participants:
        print(f"Processing {p}...")
        folder_path = os.path.join(args.in_dir, p)

        X, y = process_participant(folder_path)

        X_all.extend(X)
        y_all.extend(y)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    os.makedirs(args.out_dir, exist_ok=True)

    save_path = os.path.join(args.out_dir, "dataset.pkl")

    with open(save_path, "wb") as f:
        pickle.dump((X_all, y_all), f)

    print(f"\nDataset saved to: {save_path}")
    print(f"Shape: {X_all.shape}, Labels: {len(y_all)}")


if __name__ == "__main__":
    main()