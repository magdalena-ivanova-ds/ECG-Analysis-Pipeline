import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from config import ptbxl_dir, model2_dir, ptbxl_fs, ptbxl_lead_index, beat_before_sec, beat_after_sec, target_superclasses, class_name_map
from utils_ecg import make_dirs, zscore_signal, bandpass_filter, simple_peak_detector, parse_scp_codes, choose_single_superclass, extract_beat_windows


def load_metadata():
    ptbxl_df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")
    scp_df = pd.read_csv(ptbxl_dir / "scp_statements.csv", index_col=0)

    return ptbxl_df, scp_df


def preprocess_ptbxl():
    make_dirs([model2_dir])

    print("loading PTB-XL metadata...")
    ptbxl_df, scp_df = load_metadata()

    # PTB-XL stores the diagnostic labels as text dictionaries.
    # These are parsed first so they can be mapped to broader classes.
    ptbxl_df["scp_codes_dict"] = ptbxl_df["scp_codes"].apply(parse_scp_codes)

    ptbxl_df["target_superclass"] = ptbxl_df["scp_codes_dict"].apply(
        lambda codes: choose_single_superclass(
            codes,
            scp_df,
            target_superclasses,
        )
    )

    # Keep only records that clearly map to one of the selected classes.
    clean_df = ptbxl_df.dropna(subset=["target_superclass"]).copy()

    # The project uses the low-resolution 100 Hz version of PTB-XL.
    clean_df["signal_path"] = clean_df["filename_lr"].apply(
        lambda path: ptbxl_dir / path
    )

    print(f"Records kept after label filtering: {len(clean_df)}")

    all_records = []
    all_record_labels = []
    record_metadata = []

    all_beats = []
    all_beat_labels = []
    beat_metadata = []

    print("Preprocessing PTB-XL data for model 2.")

    for _, row in tqdm(clean_df.iterrows(), total=len(clean_df)):
        record_path = str(row["signal_path"])

        signal, _ = wfdb.rdsamp(record_path)
        ecg = signal[:, ptbxl_lead_index]

        ecg = zscore_signal(ecg)
        ecg = bandpass_filter(ecg, fs=ptbxl_fs)

        label_code = row["target_superclass"]
        label_name = class_name_map[label_code]

        # Keep the full ECG record as the record-level version.
        all_records.append(ecg)
        all_record_labels.append(label_code)

        record_metadata.append({
            "ecg_id": row["ecg_id"],
            "label_code": label_code,
            "label_name": label_name,
            "signal_path": str(row["signal_path"]),
        })

        # Create beat-level samples from the same ECG record.
        # This uses a simple peak detector only for preprocessing.
        peaks = simple_peak_detector(ecg, fs=ptbxl_fs)

        beat_windows = extract_beat_windows(
            signal=ecg,
            peaks=peaks,
            fs=ptbxl_fs,
            before_sec=beat_before_sec,
            after_sec=beat_after_sec,
        )

        for beat_index, beat in enumerate(beat_windows):
            all_beats.append(beat)
            all_beat_labels.append(label_code)

            beat_metadata.append({
                "ecg_id": row["ecg_id"],
                "beat_index": beat_index,
                "label_code": label_code,
                "label_name": label_name,
            })

    x_records = np.array(all_records, dtype=np.float32)
    y_records = np.array(all_record_labels)

    x_beats = np.array(all_beats, dtype=np.float32)
    y_beats = np.array(all_beat_labels)

    record_metadata_df = pd.DataFrame(record_metadata)
    beat_metadata_df = pd.DataFrame(beat_metadata)

    np.save(model2_dir / "X_model2_records.npy", x_records)
    np.save(model2_dir / "y_model2_records.npy", y_records)

    np.save(model2_dir / "X_model2_beats.npy", x_beats)
    np.save(model2_dir / "y_model2_beats.npy", y_beats)

    record_metadata_df.to_csv(
        model2_dir / "model2_record_metadata.csv",
        index=False,
    )
    beat_metadata_df.to_csv(
        model2_dir / "model2_beat_metadata.csv",
        index=False,
    )

    print(f"Saved X_model2_records.npy with shape: {x_records.shape}")
    print(f"Saved X_model2_beats.npy with shape: {x_beats.shape}")
    print("PTB-XL preprocessing completed.")


if __name__ == "__main__":
    preprocess_ptbxl()