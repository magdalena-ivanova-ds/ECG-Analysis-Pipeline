"""
Train lightweight CNN for Model 2 beat-level disease classification.

Run from project root:
    python models/model2/training/train_cnn.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

MODEL2_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL2_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL2_ROOT))

from beatDataLoading.loader import (
    class_distribution,
    compute_class_weights,
    encode_labels,
    load_official_split,
    mild_undersample_norm,
)
from training.logging_utils import compute_all_metrics, predict_cnn
from neuralNetworkArchitecture.beat_cnn import BeatCNNClassifier
from training.config import (
    APPLY_NORM_UNDERSAMPLING,
    CNN_BATCH_SIZE,
    CNN_DROPOUT,
    CNN_LEARNING_RATE,
    CNN_LOG_EVERY_N_BATCHES,
    CNN_BASELINE_EPOCHS,
    CNN_NUM_EPOCHS,
    CNN_PATIENCE,
    CNN_USE_CLASS_WEIGHTS,
    CNN_WEIGHTS_PATH,
    METRICS_SUMMARY_PATH,
    NORM_KEEP_RATIO,
    RANDOM_SEED,
    REPORTS_DIR,
    SAVED_WEIGHTS_DIR,
)
from training.logging_utils import (
    log_class_distribution,
    log_confusion_summary,
    log_epoch_summary,
    log_header,
    log_metrics_block,
    log_model_params,
    log_stage_timing,
    per_class_f1_dict,
)


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_tensors(X: np.ndarray, y_enc: np.ndarray):
    return (
        torch.tensor(X[:, np.newaxis, :], dtype=torch.float32),
        torch.tensor(y_enc, dtype=torch.long),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_cnn(
    X_train: np.ndarray,
    y_train_enc: np.ndarray,
    X_val: np.ndarray,
    y_val_enc: np.ndarray,
    class_weights: np.ndarray,
    use_class_weights: bool,
    device: str,
    num_epochs: int | None = None,
) -> tuple[BeatCNNClassifier, dict, list]:
    num_epochs = num_epochs or CNN_NUM_EPOCHS
    set_seed()
    X_train_t, y_train_t = prepare_tensors(X_train, y_train_enc)
    X_val_t, y_val_t = prepare_tensors(X_val, y_val_enc)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=CNN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=CNN_BATCH_SIZE, shuffle=False
    )

    model = BeatCNNClassifier(num_classes=5, dropout=CNN_DROPOUT).to(device)
    log_model_params("BeatCNNClassifier", {
        "trainable_parameters": count_parameters(model),
        "epochs": num_epochs,
        "batch_size": CNN_BATCH_SIZE,
        "learning_rate": CNN_LEARNING_RATE,
        "dropout": CNN_DROPOUT,
        "class_weights_in_loss": use_class_weights,
        "early_stopping_patience": CNN_PATIENCE,
        "device": device,
    })

    if use_class_weights:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("\nClass weights in loss:", dict(zip(["NORM", "MI", "HYP", "CD", "STTC"], class_weights.round(3))))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_macro_f1 = -1.0
    best_state = None
    history = []
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        print(f"\n{'=' * 72}\nEpoch {epoch}/{num_epochs} START")
        model.train()
        running_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch} batches", unit="batch", leave=False)
        for batch_idx, (xb, yb) in enumerate(batch_bar, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if batch_idx % CNN_LOG_EVERY_N_BATCHES == 0:
                batch_bar.write(f"  batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}")

        avg_loss = running_loss / max(n_batches, 1)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_acc = (val_preds == val_targets).mean()
        val_macro = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_weighted = f1_score(val_targets, val_preds, average="weighted", zero_division=0)
        per_class = per_class_f1_dict(val_targets, val_preds)
        scheduler.step(val_macro)

        log_epoch_summary(
            epoch, num_epochs, avg_loss, val_acc, val_macro, per_class,
            weighted_f1=val_weighted,
            extra=f"LR={optimizer.param_groups[0]['lr']:.6f} | epoch_time={time.time()-epoch_start:.1f}s",
        )

        if epoch % 5 == 0 or epoch == num_epochs:
            log_confusion_summary(val_targets, val_preds, title=f"Checkpoint confusion matrix (epoch {epoch})")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro),
            "val_weighted_f1": float(val_weighted),
            "per_class_f1": per_class,
        })

        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  >> NEW BEST validation macro F1: {best_macro_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  >> No improvement ({epochs_no_improve}/{CNN_PATIENCE})")

        if epochs_no_improve >= CNN_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    log_stage_timing("CNN training", train_start)

    X_val_cnn = X_val[:, np.newaxis, :].astype(np.float32)
    final_preds = predict_cnn(model, X_val_cnn, device=device)
    val_metrics = compute_all_metrics(y_val_enc, final_preds)
    return model, val_metrics, history


def main():
    log_header("MODEL 2 — CNN TRAINING")
    SAVED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    t0 = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test = load_official_split()
    y_val_enc = encode_labels(y_val)
    y_test_enc = encode_labels(y_test)

    log_class_distribution(class_distribution(y_train), "Full training set")

    if APPLY_NORM_UNDERSAMPLING:
        print(f"\nMild NORM undersampling on TRAIN (keep_ratio={NORM_KEEP_RATIO})...")
        X_train, y_train = mild_undersample_norm(X_train, y_train)
        log_class_distribution(class_distribution(y_train), "Undersampled training set")

    y_train_enc = encode_labels(y_train)
    class_weights = compute_class_weights(y_train)

    # Train without weights (baseline comparison)
    print("\n" + "#" * 72)
    print("EXPERIMENT A: CNN without class weights")
    print("#" * 72)
    _, cnn_no_w_val, _ = train_cnn(
        X_train, y_train_enc, X_val, y_val_enc, class_weights,
        use_class_weights=False, device=device, num_epochs=CNN_BASELINE_EPOCHS,
    )

    # Train with weights (primary model)
    print("\n" + "#" * 72)
    print("EXPERIMENT B: CNN with class weights (PRIMARY)")
    print("#" * 72)
    model, cnn_w_val, history = train_cnn(
        X_train, y_train_enc, X_val, y_val_enc, class_weights,
        use_class_weights=CNN_USE_CLASS_WEIGHTS, device=device,
    )

    torch.save(model.state_dict(), CNN_WEIGHTS_PATH)
    print(f"\nSaved CNN weights -> {CNN_WEIGHTS_PATH}")

    X_test_cnn = X_test[:, np.newaxis, :].astype(np.float32)
    test_preds = predict_cnn(model, X_test_cnn, device=device)
    cnn_test = compute_all_metrics(y_test_enc, test_preds)
    log_metrics_block(cnn_test, split="test (CNN + class weights)")

    history_path = SAVED_WEIGHTS_DIR / "cnn_training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    existing = {}
    if METRICS_SUMMARY_PATH.exists():
        with METRICS_SUMMARY_PATH.open(encoding="utf-8") as f:
            existing = json.load(f)
    existing.update({
        "cnn_no_weights_validation": cnn_no_w_val,
        "cnn_with_weights_validation": cnn_w_val,
        "cnn_with_weights_test": cnn_test,
    })
    with METRICS_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"\nMetrics summary -> {METRICS_SUMMARY_PATH}")
    log_stage_timing("Total CNN pipeline", t0)


if __name__ == "__main__":
    main()
