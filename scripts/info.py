import numpy as np
import pandas as pd
from pathlib import Path

from config import MODEL1_DIR, MODEL2_DIR

# MODEL 1: MIT-BIH exact counts
X1 = np.load(MODEL1_DIR / "X_model1.npy")
y1 = np.load(MODEL1_DIR / "y_model1.npy")
meta1 = pd.read_csv(MODEL1_DIR / "model1_metadata.csv")
splits1 = pd.read_csv(MODEL1_DIR / "model1_splits.csv")

train1 = splits1["train_idx"].dropna().astype(int)
val1 = splits1["val_idx"].dropna().astype(int)
test1 = splits1["test_idx"].dropna().astype(int)

print("MODEL 1: MIT-BIH")
print(f"X_model1 shape: {X1.shape}")
print(f"y_model1 shape: {y1.shape}")
print(f"Total windows/samples: {len(X1)}")
print(f"Window length: {X1.shape[1]} samples")
print(f"Number of original records used: {meta1['record_name'].nunique()}")
print(f"Train samples: {len(train1)}")
print(f"Val samples:   {len(val1)}")
print(f"Test samples:  {len(test1)}")
print()

windows_per_record = meta1.groupby("record_name").size().sort_values(ascending=False)
print("Windows per record:")
print(windows_per_record)
print()

# MODEL 2: PTB-XL exact counts
X2_records = np.load(MODEL2_DIR / "X_model2_records.npy", allow_pickle=True)
y2_records = np.load(MODEL2_DIR / "y_model2_records.npy", allow_pickle=True)

X2_beats = np.load(MODEL2_DIR / "X_model2_beats.npy")
y2_beats = np.load(MODEL2_DIR / "y_model2_beats.npy", allow_pickle=True)

record_meta2 = pd.read_csv(MODEL2_DIR / "model2_record_metadata.csv")
beat_meta2 = pd.read_csv(MODEL2_DIR / "model2_beat_metadata.csv")

record_splits2 = pd.read_csv(MODEL2_DIR / "model2_record_splits.csv")
beat_splits2 = pd.read_csv(MODEL2_DIR / "model2_beat_splits.csv")

train2_records = record_splits2["train_idx"].dropna().astype(int)
val2_records = record_splits2["val_idx"].dropna().astype(int)
test2_records = record_splits2["test_idx"].dropna().astype(int)

train2_beats = beat_splits2["train_idx"].dropna().astype(int)
val2_beats = beat_splits2["val_idx"].dropna().astype(int)
test2_beats = beat_splits2["test_idx"].dropna().astype(int)

print("MODEL 2: PTB-XL")
print(f"X_model2_records shape: {X2_records.shape}")
print(f"y_model2_records shape: {y2_records.shape}")
print(f"Total record-level samples: {len(X2_records)}")
print()

print(f"X_model2_beats shape: {X2_beats.shape}")
print(f"y_model2_beats shape: {y2_beats.shape}")
print(f"Total beat-level samples: {len(X2_beats)}")
print(f"Beat window length: {X2_beats.shape[1]} samples")
print()

print(f"Train records: {len(train2_records)}")
print(f"Val records:   {len(val2_records)}")
print(f"Test records:  {len(test2_records)}")
print()

print(f"Train beats: {len(train2_beats)}")
print(f"Val beats:   {len(val2_beats)}")
print(f"Test beats:  {len(test2_beats)}")
print()

print("Record-level class counts:")
print(record_meta2["label_code"].value_counts())
print()

print("Beat-level class counts:")
print(beat_meta2["label_code"].value_counts())
print()

beats_per_record = beat_meta2.groupby("ecg_id").size()
print("Beats per ECG record:")
print(beats_per_record.describe())