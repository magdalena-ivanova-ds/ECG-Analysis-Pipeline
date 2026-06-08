"""Verbose training logs for Model 2."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from training.config import CLASS_LABELS


def compute_all_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_LABELS))), average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_LABELS))), zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": float(precision),
        "recall": float(recall),
        "per_class": {
            CLASS_LABELS[i]: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_LABELS))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASS_LABELS)))
        ).tolist(),
    }


def format_classification_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, target_names=CLASS_LABELS, digits=4, zero_division=0
    )


def predict_cnn(model, X: np.ndarray, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def log_header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def log_class_distribution(dist: dict[str, int], label: str = "Class distribution"):
    print(f"\n{label}:")
    total = sum(dist.values())
    for cls in CLASS_LABELS:
        count = dist.get(cls, 0)
        pct = 100.0 * count / total if total else 0
        print(f"  {cls:5s}: {count:7,d}  ({pct:5.2f}%)")


def log_model_params(model_name: str, params: dict):
    print(f"\n{model_name} parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")


def log_stage_timing(stage: str, start_time: float):
    elapsed = time.time() - start_time
    print(f"\n[{stage}] finished in {elapsed:.1f}s ({elapsed / 60:.2f} min)")


def per_class_f1_dict(y_true, y_pred) -> dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_LABELS))), zero_division=0
    )
    return {CLASS_LABELS[i]: float(f1[i]) for i in range(len(CLASS_LABELS))}


def log_epoch_summary(
    epoch: int,
    total_epochs: int,
    loss: float,
    accuracy: float,
    macro_f1: float,
    per_class: dict[str, float],
    weighted_f1: float | None = None,
    extra: str = "",
):
    print(f"\nEpoch {epoch}/{total_epochs}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    if weighted_f1 is not None:
        print(f"Weighted F1: {weighted_f1:.4f}")
    print("Class F1:")
    for cls in CLASS_LABELS:
        print(f"  {cls}: {per_class.get(cls, 0.0):.4f}")
    if extra:
        print(extra)


def log_confusion_summary(y_true, y_pred, title: str = "Confusion matrix summary"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS))))
    print(f"\n{title}:")
    header = "          " + "  ".join(f"{c:>6s}" for c in CLASS_LABELS)
    print(header)
    for i, row_label in enumerate(CLASS_LABELS):
        row = "  ".join(f"{cm[i, j]:6d}" for j in range(len(CLASS_LABELS)))
        print(f"  {row_label:5s}  {row}")


def log_metrics_block(metrics: dict, split: str = "validation"):
    print(f"\n--- {split.upper()} METRICS ---")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    for cls in CLASS_LABELS:
        pc = metrics["per_class"][cls]
        print(f"  {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")
