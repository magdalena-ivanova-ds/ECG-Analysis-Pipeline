import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from config import (
    PTBXL_DIR,
    MODEL2_DIR,
    PTBXL_FS,
    PTBXL_LEAD_INDEX,
    BEAT_BEFORE_SEC,
    BEAT_AFTER_SEC,
    TARGET_SUPERCLASSES,
    CLASS_NAME_MAP
)
from utils_ecg import (
    make_dirs,
    zscore_signal,
    bandpass_filter,
    simple_peak_detector,
    parse_scp_codes,
    choose_single_superclass,
    extract_beat_windows
)


def load_metadata():
    ptbxl_df = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")
    scp_df = pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)
    return ptbxl_df, scp_df


def preprocess_ptbxl():
    make_dirs([MODEL2_DIR])

    print("Loading PTB-XL metadata...")
    ptbxl_df, scp_df = load_metadata()

    # Parse dictionary-like diagnosis column
    ptbxl_df["scp_codes_dict"] = ptbxl_df["scp_codes"].apply(parse_scp_codes)

    # Keep only one broad superclass to make classification simpler
    ptbxl_df["target_superclass"] = ptbxl_df["scp_codes_dict"].apply(
        lambda x: choose_single_superclass(x, scp_df, TARGET_SUPERCLASSES)
    )

    # Drop rows that map to none or multiple groups
    clean_df = ptbxl_df.dropna(subset=["target_superclass"]).copy()

    # Use records100, so use filename_lr
    clean_df["signal_path"] = clean_df["filename_lr"].apply(lambda x: PTBXL_DIR / x)

    print(f"Records after label filtering: {len(clean_df)}")

    all_records = []
    all_record_labels = []
    record_meta = []

    all_beats = []
    all_beat_labels = []
    beat_meta = []

    print("Preprocessing PTB-XL for Model 2...")

    for _, row in tqdm(clean_df.iterrows(), total=len(clean_df)):
        record_path = str(row["signal_path"])

        # Read ECG
        signal, fields = wfdb.rdsamp(record_path)
        ecg = signal[:, PTBXL_LEAD_INDEX]

        # Normalize + filter
        ecg = zscore_signal(ecg)
        ecg = bandpass_filter(ecg, fs=PTBXL_FS)

        label_code = row["target_superclass"]
        label_name = CLASS_NAME_MAP[label_code]

        # Save full normalized record
        all_records.append(ecg)
        all_record_labels.append(label_code)

        record_meta.append({
            "ecg_id": row["ecg_id"],
            "label_code": label_code,
            "label_name": label_name,
            "signal_path": str(row["signal_path"])
        })

        # Create preliminary heartbeat windows using simple peak detection
        peaks = simple_peak_detector(ecg, fs=PTBXL_FS)
        beat_windows = extract_beat_windows(
            signal=ecg,
            peaks=peaks,
            fs=PTBXL_FS,
            before_sec=BEAT_BEFORE_SEC,
            after_sec=BEAT_AFTER_SEC
        )

        for i, beat in enumerate(beat_windows):
            all_beats.append(beat)
            all_beat_labels.append(label_code)
            beat_meta.append({
                "ecg_id": row["ecg_id"],
                "beat_index": i,
                "label_code": label_code,
                "label_name": label_name
            })

    # Convert to arrays
    X_records = np.array(all_records, dtype=np.float32)
    y_records = np.array(all_record_labels)

    X_beats = np.array(all_beats, dtype=np.float32)
    y_beats = np.array(all_beat_labels)

    record_meta_df = pd.DataFrame(record_meta)
    beat_meta_df = pd.DataFrame(beat_meta)

    # Save files
    np.save(MODEL2_DIR / "X_model2_records.npy", X_records)
    np.save(MODEL2_DIR / "y_model2_records.npy", y_records)

    np.save(MODEL2_DIR / "X_model2_beats.npy", X_beats)
    np.save(MODEL2_DIR / "y_model2_beats.npy", y_beats)

    record_meta_df.to_csv(MODEL2_DIR / "model2_record_metadata.csv", index=False)
    beat_meta_df.to_csv(MODEL2_DIR / "model2_beat_metadata.csv", index=False)

    print(f"Saved X_model2_records.npy with shape: {X_records.shape}")
    print(f"Saved X_model2_beats.npy with shape: {X_beats.shape}")
    print("PTB-XL preprocessing finished.")


if __name__ == "__main__":
    preprocess_ptbxl()