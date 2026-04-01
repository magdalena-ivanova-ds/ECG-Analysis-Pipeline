import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import MODEL1_DIR, MODEL2_DIR, PTBXL_DIR


def save_split_indices(save_path, train_idx, val_idx, test_idx):
    """
    Save split indices into one csv file.
    """
    max_len = max(len(train_idx), len(val_idx), len(test_idx))

    df = pd.DataFrame({
        "train_idx": list(train_idx) + [None] * (max_len - len(train_idx)),
        "val_idx": list(val_idx) + [None] * (max_len - len(val_idx)),
        "test_idx": list(test_idx) + [None] * (max_len - len(test_idx)),
    })

    df.to_csv(save_path, index=False)


def create_model1_splits(random_state=42):
    """
    Create train/val/test split for MIT-BIH windows.

    Simple and understandable:
    - 70% train
    - 15% validation
    - 15% test
    """
    metadata_path = MODEL1_DIR / "model1_metadata.csv"
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
        MODEL1_DIR / "model1_splits.csv",
        train_idx,
        val_idx,
        test_idx
    )

    print("Model 1 splits saved.")
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")


def create_model2_splits():
    """
    Create train/val/test split for PTB-XL using official strat_fold.

    PTB-XL standard:
    - folds 1-8 = train
    - fold 9 = validation
    - fold 10 = test
    """
    metadata_path = MODEL2_DIR / "model2_record_metadata.csv"
    record_meta_df = pd.read_csv(metadata_path)

    # Need original PTB-XL metadata again because strat_fold is there

    ptbxl_df = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv")

    merged = record_meta_df.merge(
        ptbxl_df[["ecg_id", "strat_fold"]],
        on="ecg_id",
        how="left"
    )

    train_records = merged[merged["strat_fold"].between(1, 8)].copy()
    val_records = merged[merged["strat_fold"] == 9].copy()
    test_records = merged[merged["strat_fold"] == 10].copy()

    train_record_ids = set(train_records["ecg_id"].tolist())
    val_record_ids = set(val_records["ecg_id"].tolist())
    test_record_ids = set(test_records["ecg_id"].tolist())

    # Save record-level splits
    save_split_indices(
        MODEL2_DIR / "model2_record_splits.csv",
        train_records.index.to_numpy(),
        val_records.index.to_numpy(),
        test_records.index.to_numpy()
    )

    # Beat-level splits
    beat_meta_path = MODEL2_DIR / "model2_beat_metadata.csv"
    beat_meta_df = pd.read_csv(beat_meta_path)

    train_beats = beat_meta_df[beat_meta_df["ecg_id"].isin(train_record_ids)].copy()
    val_beats = beat_meta_df[beat_meta_df["ecg_id"].isin(val_record_ids)].copy()
    test_beats = beat_meta_df[beat_meta_df["ecg_id"].isin(test_record_ids)].copy()

    save_split_indices(
        MODEL2_DIR / "model2_beat_splits.csv",
        train_beats.index.to_numpy(),
        val_beats.index.to_numpy(),
        test_beats.index.to_numpy()
    )

    print("Model 2 splits saved.")
    print(f"Record Train: {len(train_records)}")
    print(f"Record Val:   {len(val_records)}")
    print(f"Record Test:  {len(test_records)}")
    print(f"Beat Train:   {len(train_beats)}")
    print(f"Beat Val:     {len(val_beats)}")
    print(f"Beat Test:    {len(test_beats)}")


def main():
    create_model1_splits()
    create_model2_splits()
    print("All split files created.")


if __name__ == "__main__":
    main()