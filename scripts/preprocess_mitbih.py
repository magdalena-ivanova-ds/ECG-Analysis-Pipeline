import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from config import mitbih_dir, model1_dir, mitbih_fs, mitbih_lead_index, window_sec, stride_sec
from utils_ecg import make_dirs, zscore_signal, bandpass_filter, extract_fixed_windows

non_beat_symbols = {"+", "~", "|", "!", "[", "]", '"', "x", "(", ")", "p", "t", "u", "`", "^", "="}

def get_record_names():
    # use the .dat files to find all available MIT-BIH records
    dat_files = sorted(mitbih_dir.glob("*.dat"))

    if not dat_files:
        raise FileNotFoundError(
            f"No .dat files were found in {mitbih_dir}. the MIT-BIH data may be missing."
        )

    return [file.stem for file in dat_files]


def record_files_available(record_name):
    # each MIT-BIH record needs the signal file, header file, and annotation file
    required_files = [
        mitbih_dir / f"{record_name}.dat",
        mitbih_dir / f"{record_name}.hea",
        mitbih_dir / f"{record_name}.atr",
    ]

    return all(path.exists() for path in required_files)


def get_valid_peak_samples(annotation):
    # keep only annotations that correspond to actual heartbeat positions
    peaks = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in non_beat_symbols:
            peaks.append(sample)

    return np.array(peaks, dtype=int)


def preprocess_mitbih():
    make_dirs([model1_dir])

    record_names = get_record_names()

    all_windows = []
    all_labels = []
    metadata_rows = []

    print("Preprocessing MIT-BIH data for model 1.")

    for record_name in tqdm(record_names):
        if not record_files_available(record_name):
            print(f"Skipping incomplete record: {record_name}")
            continue

        record_path = str(mitbih_dir / record_name)

        signal, _ = wfdb.rdsamp(record_path)
        ecg = signal[:, mitbih_lead_index]

        ecg = zscore_signal(ecg)
        ecg = bandpass_filter(ecg, fs=mitbih_fs)

        annotation = wfdb.rdann(record_path, "atr")
        peaks = get_valid_peak_samples(annotation)

        record_windows, record_labels, record_metadata = extract_fixed_windows(
            signal=ecg,
            peaks=peaks,
            fs=mitbih_fs,
            window_sec=window_sec,
            stride_sec=stride_sec,
        )

        all_windows.append(record_windows)
        all_labels.append(record_labels)

        for start, end, num_peaks in record_metadata:
            metadata_rows.append({
                "record_name": record_name,
                "start_idx": start,
                "end_idx": end,
                "num_peaks": num_peaks,
            })

    x_model1 = np.concatenate(all_windows, axis=0)
    y_model1 = np.concatenate(all_labels, axis=0)
    metadata_df = pd.DataFrame(metadata_rows)

    np.save(model1_dir / "X_model1.npy", x_model1)
    np.save(model1_dir / "y_model1.npy", y_model1)
    metadata_df.to_csv(model1_dir / "model1_metadata.csv", index=False)

    print(f"Saved X_model1.npy with shape: {x_model1.shape}")
    print(f"Saved y_model1.npy with shape: {y_model1.shape}")
    print("MIT-BIH preprocessing completed.")


if __name__ == "__main__":
    preprocess_mitbih()