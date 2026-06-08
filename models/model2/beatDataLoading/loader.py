"""Load beat-level Model 2 data and official PTB-XL stratified splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.config import (
    BEAT_SPLITS_PATH,
    CLASS_LABELS,
    NORM_KEEP_RATIO,
    NORM_LABEL,
    RANDOM_SEED,
    X_BEATS_PATH,
    Y_BEATS_PATH,
)


def load_beat_arrays():
    X = np.load(X_BEATS_PATH)
    y = np.load(Y_BEATS_PATH)
    return X, y


def load_official_split():
    """
    Official folds via model2_beat_splits.csv:
    folds 1-8 -> train, fold 9 -> val, fold 10 -> test.
    """
    X, y = load_beat_arrays()
    splits = pd.read_csv(BEAT_SPLITS_PATH)

    train_idx = splits["train_idx"].dropna().astype(int).to_numpy()
    val_idx = splits["val_idx"].dropna().astype(int).to_numpy()
    test_idx = splits["test_idx"].dropna().astype(int).to_numpy()

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def encode_labels(y: np.ndarray) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_LABELS)}
    return np.array([label_to_idx[label] for label in y], dtype=np.int64)


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """weight_i = total / (num_classes * count_i) from training labels only."""
    y_enc = encode_labels(y_train)
    num_classes = len(CLASS_LABELS)
    total = len(y_enc)
    counts = np.bincount(y_enc, minlength=num_classes).astype(np.float64)
    return total / (num_classes * counts)


def class_distribution(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {label: int(count) for label, count in zip(labels, counts)}


def mild_undersample_norm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    keep_ratio: float = NORM_KEEP_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only a fraction of NORM beats in the training set; leave minorities untouched."""
    norm_idx = np.where(y_train == NORM_LABEL)[0]
    other_idx = np.where(y_train != NORM_LABEL)[0]
    n_keep = max(1, int(len(norm_idx) * keep_ratio))
    rng = np.random.default_rng(seed)
    kept_norm = rng.choice(norm_idx, size=n_keep, replace=False)
    keep_idx = np.sort(np.concatenate([kept_norm, other_idx]))
    return X_train[keep_idx], y_train[keep_idx]
