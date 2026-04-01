from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from config import (
    MITBIH_DIR,
    MODEL1_DIR,
    MITBIH_FS,
    MITBIH_LEAD_INDEX,
    WINDOW_SEC,
    STRIDE_SEC
)
from utils_ecg import (
    make_dirs,
    zscore_signal,
    bandpass_filter,
    extract_fixed_windows
)

NON_BEAT_SYMBOLS = {
    "+", "~", "|", "!", "[", "]", '"', "x", "(", ")", "p", "t", "u", "`", "^", "="
}


def get_record_names():
    """
    Find MIT-BIH record names by scanning .dat files.
    This avoids depending on the RECORDS file.
    """
    dat_files = sorted(MITBIH_DIR.glob("*.dat"))

    if not dat_files:
        raise FileNotFoundError(
            f"No .dat files found in {MITBIH_DIR}. MIT-BIH download is probably incomplete."
        )

    records = [file.stem for file in dat_files]
    return records


def validate_record_files(record_name):
    """
    Check that .dat, .hea, and .atr all exist for a record.
    """
    needed = [
        MITBIH_DIR / f"{record_name}.dat",
        MITBIH_DIR / f"{record_name}.hea",
        MITBIH_DIR / f"{record_name}.atr"
    ]
    return all(path.exists() for path in needed)


def get_valid_peak_samples(annotation):
    """
    Keep only beat-related annotation samples.
    """
    peaks = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in NON_BEAT_SYMBOLS:
            peaks.append(sample)

    return np.array(peaks, dtype=int)


def preprocess_mitbih():
    make_dirs([MODEL1_DIR])

    records = get_record_names()

    all_X = []
    all_Y = []
    all_meta_rows = []

    print("Preprocessing MIT-BIH for Model 1...")

    for record_name in tqdm(records):
        if not validate_record_files(record_name):
            print(f"Skipping incomplete record: {record_name}")
            continue

        record_path = str(MITBIH_DIR / record_name)

        signal, fields = wfdb.rdsamp(record_path)
        ecg = signal[:, MITBIH_LEAD_INDEX]

        ecg = zscore_signal(ecg)
        ecg = bandpass_filter(ecg, fs=MITBIH_FS)

        ann = wfdb.rdann(record_path, "atr")
        peaks = get_valid_peak_samples(ann)

        X_record, Y_record, meta_record = extract_fixed_windows(
            signal=ecg,
            peaks=peaks,
            fs=MITBIH_FS,
            window_sec=WINDOW_SEC,
            stride_sec=STRIDE_SEC
        )

        all_X.append(X_record)
        all_Y.append(Y_record)

        for start, end, num_peaks in meta_record:
            all_meta_rows.append({
                "record_name": record_name,
                "start_idx": start,
                "end_idx": end,
                "num_peaks": num_peaks
            })

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    meta_df = pd.DataFrame(all_meta_rows)

    # Save files
    np.save(MODEL1_DIR / "X_model1.npy", X)
    np.save(MODEL1_DIR / "y_model1.npy", Y)
    meta_df.to_csv(MODEL1_DIR / "model1_metadata.csv", index=False)

    print(f"Saved X_model1.npy with shape: {X.shape}")
    print(f"Saved y_model1.npy with shape: {Y.shape}")
    print("MIT-BIH preprocessing finished.")


if __name__ == "__main__":
    preprocess_mitbih()