"""
Train Random Forest baselines for Model 2 beat-level disease classification.

Run from project root:
    python models/model2/training/train_rf.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL2_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL2_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL2_ROOT))

from beatDataLoading.loader import (
    class_distribution,
    encode_labels,
    load_official_split,
    mild_undersample_norm,
)
from training.logging_utils import compute_all_metrics, format_classification_report
from training.config import (
    APPLY_NORM_UNDERSAMPLING,
    METRICS_SUMMARY_PATH,
    NORM_KEEP_RATIO,
    RANDOM_SEED,
    REPORTS_DIR,
    RF_BASELINE_CLASS_WEIGHT,
    RF_BASELINE_ESTIMATORS,
    RF_BASELINE_PATH,
    RF_IMPROVED_CLASS_WEIGHT,
    RF_IMPROVED_ESTIMATORS,
    RF_IMPROVED_PATH,
    SAVED_WEIGHTS_DIR,
)
from training.logging_utils import (
    log_class_distribution,
    log_confusion_summary,
    log_header,
    log_metrics_block,
    log_model_params,
    log_stage_timing,
)


def train_and_evaluate_rf(
    name: str,
    X_train: np.ndarray,
    y_train_enc: np.ndarray,
    X_val: np.ndarray,
    y_val_enc: np.ndarray,
    n_estimators: int,
    class_weight,
) -> tuple[RandomForestClassifier, dict]:
    start = time.time()
    log_model_params(name, {
        "n_estimators": n_estimators,
        "class_weight": class_weight,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "train_samples": len(y_train_enc),
    })

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    print(f"\n[{name}] Fitting on {len(y_train_enc):,} training beats...")
    clf.fit(X_train, y_train_enc)
    log_stage_timing(f"{name} fit", start)

    val_preds = clf.predict(X_val)
    metrics = compute_all_metrics(y_val_enc, val_preds)
    log_metrics_block(metrics, split="validation")
    log_confusion_summary(y_val_enc, val_preds, title=f"{name} confusion matrix")
    print("\n" + format_classification_report(y_val_enc, val_preds))
    return clf, metrics


def main():
    log_header("MODEL 2 — RANDOM FOREST TRAINING")
    SAVED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test = load_official_split()
    y_train_enc = encode_labels(y_train)
    y_val_enc = encode_labels(y_val)
    y_test_enc = encode_labels(y_test)

    log_class_distribution(class_distribution(y_train), "Full training set distribution")
    log_stage_timing("Data loading", t0)

    # --- Baseline RF (no class weights) ---
    baseline_clf, baseline_val = train_and_evaluate_rf(
        "RF Baseline",
        X_train, y_train_enc, X_val, y_val_enc,
        n_estimators=RF_BASELINE_ESTIMATORS,
        class_weight=RF_BASELINE_CLASS_WEIGHT,
    )
    joblib.dump(baseline_clf, RF_BASELINE_PATH)
    print(f"Saved baseline RF -> {RF_BASELINE_PATH}")

    # --- Improved RF (balanced + mild NORM undersampling on train only) ---
    X_train_bal = X_train
    y_train_bal = y_train
    if APPLY_NORM_UNDERSAMPLING:
        print(f"\nApplying mild NORM undersampling (keep_ratio={NORM_KEEP_RATIO}) on TRAIN only...")
        X_train_bal, y_train_bal = mild_undersample_norm(X_train, y_train)
        log_class_distribution(class_distribution(y_train_bal), "Undersampled training set")

    improved_clf, improved_val = train_and_evaluate_rf(
        "RF Improved",
        X_train_bal, encode_labels(y_train_bal), X_val, y_val_enc,
        n_estimators=RF_IMPROVED_ESTIMATORS,
        class_weight=RF_IMPROVED_CLASS_WEIGHT,
    )
    joblib.dump(improved_clf, RF_IMPROVED_PATH)
    print(f"Saved improved RF -> {RF_IMPROVED_PATH}")

    # Test evaluation (improved model only — selected by val macro F1)
    test_preds = improved_clf.predict(X_test)
    improved_test = compute_all_metrics(y_test_enc, test_preds)
    log_metrics_block(improved_test, split="test (improved RF)")

    summary = {
        "rf_baseline_validation": baseline_val,
        "rf_improved_validation": improved_val,
        "rf_improved_test": improved_test,
        "norm_undersample_ratio": NORM_KEEP_RATIO if APPLY_NORM_UNDERSAMPLING else None,
    }

    existing = {}
    if METRICS_SUMMARY_PATH.exists():
        with METRICS_SUMMARY_PATH.open(encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(summary)
    with METRICS_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"\nMetrics summary updated -> {METRICS_SUMMARY_PATH}")
    log_stage_timing("Total RF training pipeline", t0)


if __name__ == "__main__":
    main()
