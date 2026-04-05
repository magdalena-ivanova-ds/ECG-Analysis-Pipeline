from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============== IMPORT DATA: INPUT (X) AND LABELS (y) ===============


data_dir = Path(__file__).resolve().parent.parent / "data/processed/model1"

# ----- input data model 1 -----
X = np.load(data_dir / "X_model1.npy")
# ----- labels model 1 -----
y = np.load(data_dir / "y_model1.npy")


# =============== DATA INVESTIGATION ===============

# print(type(X))
print("X.shape:", X.shape)
# print(X)

# print(type(y))
print("y.shape:", y.shape)
# print(y)

print("Unique labels found:", np.unique(y))
print("Check if input appears normalised:", X[0][:5])

# ----- how many peaks expected per window -----

peak_counts = y.sum(axis=1)
print("Number of peaks in first 20 windows: ",peak_counts[:20],"\n")


# =============== VISUAL DATA INVESTIGATION ===============


# ----- plot the first ECG window with dots where R peaks are based on labels -----

plt.plot(X[0])
peak_indices = np.where(y[0] == 1)[0]
plt.scatter(peak_indices, X[0][peak_indices], color='red', s=30)
# plt.show()

# =============== IMPORT DATA: DATA SPLIT ===============

splits = pd.read_csv(data_dir / "model1_splits.csv")
print("Structure of data split file:\n", splits.head(), "\n")

# ----- get the rows for training and their labels

train_indices = splits["train_idx"].dropna().astype(int)
X_train = X[train_indices]
y_train = y[train_indices]

print("X training shape:", X_train.shape)
print("y training shape:", y_train.shape)