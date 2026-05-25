import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import model1_dir, model2_dir, ptbxl_dir


def save_split_indices(save_path, train_idx, val_idx, test_idx):
    # The split arrays do not all have the same length, so the shorter ones are padded before saving them into one csv file.
    max_len = max(len(train_idx), len(val_idx), len(test_idx))

    split_df = pd.DataFrame({
        "train_idx": list(train_idx) + [None] * (max_len - len(train_idx)),
        "val_idx": list(val_idx) + [None] * (max_len - len(val_idx)),
        "test_idx": list(test_idx) + [None] * (max_len - len(test_idx)),
    })

    split_df.to_csv(save_path, index=False)


def create_model1_splits(random_state=42):
    # Model 1 uses the MIT-BIH windows.
    # The split is 70% training, 15% validation, and 15% testing.
    metadata_path = model1_dir / "model1_metadata.csv"
    meta_df = pd.read_csv(metadata_path)

    indices = np.arange(len(meta_df))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        shuffle=True
    )

    save_split_indices(
        model1_dir / "model1_splits.csv",
        train_idx,
        val_idx,
        test_idx
    )

    print("Model 1 split file created.")
    print(f"Training samples:   {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples:       {len(test_idx)}")


def create_model2_splits():
    # Model 2 uses the official PTB-XL folds:
    # folds 1 to 8 for training, fold 9 for validation, and fold 10 for testing.
    metadata_path = model2_dir / "model2_record_metadata.csv"
    record_meta_df = pd.read_csv(metadata_path)

    ptbxl_df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv")

    merged_df = record_meta_df.merge(
        ptbxl_df[["ecg_id", "strat_fold"]],
        on="ecg_id",
        how="left"
    )

    train_records = merged_df[merged_df["strat_fold"].between(1, 8)].copy()
    val_records = merged_df[merged_df["strat_fold"] == 9].copy()
    test_records = merged_df[merged_df["strat_fold"] == 10].copy()

    train_record_ids = set(train_records["ecg_id"])
    val_record_ids = set(val_records["ecg_id"])
    test_record_ids = set(test_records["ecg_id"])

    save_split_indices(
        model2_dir / "model2_record_splits.csv",
        train_records.index.to_numpy(),
        val_records.index.to_numpy(),
        test_records.index.to_numpy()
    )

    # The beat-level split follows the record-level split.
    # This avoids beats from the same ECG record ending up in different sets.
    beat_meta_path = model2_dir / "model2_beat_metadata.csv"
    beat_meta_df = pd.read_csv(beat_meta_path)

    train_beats = beat_meta_df[beat_meta_df["ecg_id"].isin(train_record_ids)].copy()
    val_beats = beat_meta_df[beat_meta_df["ecg_id"].isin(val_record_ids)].copy()
    test_beats = beat_meta_df[beat_meta_df["ecg_id"].isin(test_record_ids)].copy()

    save_split_indices(
        model2_dir / "model2_beat_splits.csv",
        train_beats.index.to_numpy(),
        val_beats.index.to_numpy(),
        test_beats.index.to_numpy()
    )

    print("Model 2 split files created.")
    print(f"Training records:   {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Test records:       {len(test_records)}")
    print(f"Training beats:     {len(train_beats)}")
    print(f"Validation beats:   {len(val_beats)}")
    print(f"Test beats:         {len(test_beats)}")


def main():
    create_model1_splits()
    create_model2_splits()
    print("Split creation finished.")


if __name__ == "__main__":
    main()