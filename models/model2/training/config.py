"""Central configuration for Model 2 beat-level disease classification."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL2_ROOT = Path(__file__).resolve().parents[1]

BEAT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "model2"
X_BEATS_PATH = BEAT_DATA_DIR / "X_model2_beats.npy"
Y_BEATS_PATH = BEAT_DATA_DIR / "y_model2_beats.npy"
BEAT_SPLITS_PATH = BEAT_DATA_DIR / "model2_beat_splits.csv"

SAVED_WEIGHTS_DIR = MODEL2_ROOT / "savedTrainedWeights"
REPORTS_DIR = MODEL2_ROOT / "reports"

CLASS_LABELS = ["NORM", "MI", "HYP", "CD", "STTC"]
NORM_LABEL = "NORM"
RANDOM_SEED = 42

# Random Forest
RF_BASELINE_ESTIMATORS = 200
RF_IMPROVED_ESTIMATORS = 300
RF_BASELINE_CLASS_WEIGHT = None
RF_IMPROVED_CLASS_WEIGHT = "balanced_subsample"

# Mild majority-class undersampling (training set only)
APPLY_NORM_UNDERSAMPLING = True
NORM_KEEP_RATIO = 0.5

# CNN (lightweight baseline)
CNN_NUM_EPOCHS = 40
CNN_BASELINE_EPOCHS = 20
CNN_BATCH_SIZE = 256
CNN_LEARNING_RATE = 1e-3
CNN_DROPOUT = 0.5
CNN_PATIENCE = 10
CNN_LOG_EVERY_N_BATCHES = 200
CNN_USE_CLASS_WEIGHTS = True

# Artifact filenames
RF_BASELINE_PATH = SAVED_WEIGHTS_DIR / "rf_baseline.joblib"
RF_IMPROVED_PATH = SAVED_WEIGHTS_DIR / "rf_improved.joblib"
CNN_WEIGHTS_PATH = SAVED_WEIGHTS_DIR / "cnn_beat_classifier.pt"
METRICS_SUMMARY_PATH = SAVED_WEIGHTS_DIR / "training_metrics_summary.json"
