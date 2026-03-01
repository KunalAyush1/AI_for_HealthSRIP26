import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_signal(file_path):
    """
    Loads a text file and returns a pd.DataFrame
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    
    data_starting_index = None
    for i, line in enumerate(lines):
        if "Data:" in line:
            data_starting_index = i + 1
            break

    if data_starting_index is None:
        raise ValueError(f"No Data section found in {file_path}")

    timestamps = []
    values = []

    for line in lines[data_starting_index:]:
        line = line.strip()

        if not line or ";" not in line:
            continue

        parts = line.split(";")
        timestamp_str = parts[0].strip()
        value_str = parts[1].strip().replace(",", ".")

        try:
            timestamps.append(timestamp_str)
            values.append(float(value_str))
        except:
            continue

    timestamps = pd.to_datetime(
        timestamps,
        format="%d.%m.%Y %H:%M:%S,%f"
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })

    return df


def load_events(file_path):
    """
    Loads event annotation file and returns a pd.DataFrame
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_lines = [line.strip() for line in lines if ";" in line]

    event_starts = []
    event_ends = []
    labels = []

    for line in data_lines:
        parts = line.split(";")

        if len(parts) < 3:
            continue

        time_range = parts[0].strip()
        label = parts[2].strip()

        if "-" not in time_range:
            continue

        start_str, end_time_str = time_range.split("-")

        start_dt = pd.to_datetime(
            start_str,
            format="%d.%m.%Y %H:%M:%S,%f"
        )

        end_dt = pd.to_datetime(
            start_dt.strftime("%d.%m.%Y ") + end_time_str,
            format="%d.%m.%Y %H:%M:%S,%f"
        )

        event_starts.append(start_dt)
        event_ends.append(end_dt)
        labels.append(label)

    df_events = pd.DataFrame({
        "start": event_starts,
        "end": event_ends,
        "label": labels
    })

    return df_events


def create_visualization(participant_path):
    """
    Plots signals and annotated breathing events
    """

    files = os.listdir(participant_path)

    
    signal_files = [
        f for f in files
        if f.endswith(".txt") and "event" not in f.lower()
    ]

    event_files = [
        f for f in files
        if "event" in f.lower()
    ]

    if not signal_files or not event_files:
        raise ValueError("Signal files or event file missing in participant folder.")

    
    airflow_file = [f for f in signal_files if "flow" in f.lower()][0]
    thorac_file = [f for f in signal_files if "thorac" in f.lower()][0]
    spo2_file = [f for f in signal_files if "spo2" in f.lower()][0]
    event_file = event_files[0]

    
    airflow = load_signal(os.path.join(participant_path, airflow_file))
    thorac = load_signal(os.path.join(participant_path, thorac_file))
    spo2 = load_signal(os.path.join(participant_path, spo2_file))
    events = load_events(os.path.join(participant_path, event_file))

    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    
    axes[0].plot(airflow["timestamp"], airflow["value"])
    axes[0].set_title("Nasal Airflow")
    axes[0].set_ylabel("Amplitude")

    
    axes[1].plot(thorac["timestamp"], thorac["value"])
    axes[1].set_title("Thoracic Movement")
    axes[1].set_ylabel("Amplitude")

   
    axes[2].plot(spo2["timestamp"], spo2["value"])
    axes[2].set_title("SpO₂")
    axes[2].set_ylabel("Oxygen (%)")
    axes[2].set_xlabel("Time")

    
    for _, row in events.iterrows():
        for ax in axes:
            ax.axvspan(row["start"], row["end"], alpha=0.3)

    plt.tight_layout()

    
    os.makedirs("Visualizations", exist_ok=True)

    participant_name = os.path.basename(participant_path)
    save_path = os.path.join(
        "Visualizations",
        f"{participant_name}_visualization.pdf"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-name",
        type=str,
        help="Path to participant folder (e.g., Data/AP01)"
    )

    args = parser.parse_args()

    # Direct mode
    if args.name:
        create_visualization(args.name)

    
    else:
        data_folder = "Data"

        if not os.path.exists(data_folder):
            print("Data folder not found.")
            exit()

        patients = sorted([
            folder for folder in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, folder))
        ])

        if not patients:
            print("No participant folders found.")
            exit()

        print("\nAvailable Participants:")
        for i, patient in enumerate(patients):
            print(f"{i+1}. {patient}")

        try:
            choice = int(input("\nSelect participant number: ")) - 1
            if choice < 0 or choice >= len(patients):
                print("Invalid selection.")
                exit()
        except:
            print("Invalid input.")
            exit()

        selected_path = os.path.join(data_folder, patients[choice])
        create_visualization(selected_path)