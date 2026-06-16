# ECG Analysis Pipeline

## Overview

This project implements a **two-stage ECG analysis system**:

### Model 1 – R-peak Detection

* **Input:** raw ECG signal
* **Output:** positions of R-peaks (heartbeats)
* Goal: detect when each heartbeat occurs

---

### Model 2 – Disease Classification

* **Input:** ECG segments around R-peaks
* **Output:** predicted heart condition
* Goal: classify cardiac conditions based on heartbeat patterns

---

## Full Pipeline Concept

```text
raw ECG → Model 1 → R-peaks → segment extraction → Model 2 → diagnosis
```

* Model 1 extracts structure (heartbeat timing)
* Model 2 analyzes structure (heartbeat shape and patterns)

---

# What Was Done (Data + Preprocessing)

All data handling and preprocessing is fully completed.

## 1. Dataset Integration

Two datasets were used:

### MIT-BIH (for Model 1)

* raw ECG signals
* annotated R-peaks

### PTB-XL (for Model 2)

* ECG recordings
* diagnostic labels (SCP codes)

---

## 2. Signal Preprocessing

Applied to both datasets:

* noise filtering (bandpass filtering)
* normalization (standardization of signals)

Goal:

* remove noise
* make signals consistent
* improve model performance

---

## 3. Model-Specific Data Preparation

### Model 1 (MIT-BIH)

Process:

* ECG signals split into fixed windows (720 samples)
* R-peak annotations converted into binary sequences

Result:

* each sample = ECG window
* label = sequence showing peak positions

---

### Model 2 (PTB-XL)

Two versions created:

## Beat-Level (recommended)

* segments extracted around R-peaks (70 samples)
* each segment labeled with a disease

This focuses on:

* QRS shape
* heartbeat-level patterns

---

## Record-Level

* full ECG signals used (1000 samples)
* one label per signal

---

## 4. Label Processing (Important)

PTB-XL provides labels as **SCP code dictionaries**, for example:

```text
{'IMI': 35.0, 'ABQRS': 0.0}
```

These are:

* detailed medical codes
* multi-label

### Mapping Step

Using `scp_statements.csv`, all SCP codes were mapped into 5 main classes:

* NORM → normal ECG
* MI → myocardial infarction
* HYP → hypertrophy
* CD → conduction disturbance
* STTC → ST/T changes

Each ECG (or beat) is assigned **one final label**.

---

## 5. Train / Validation / Test Splits

### Model 1

* 70% train
* 15% validation
* 15% test

### Model 2 (PTB-XL official split)

* folds 1–8 → train
* fold 9 → validation
* fold 10 → test

---

## 6. Final Output Files

All processed files are in:

```text
data/processed/
```

---

# Model 1 Files (R-peak Detection)

Folder:

```text
data/processed/model1/
```

### Required Files

* `X_model1.npy`
  → ECG windows (input)

* `y_model1.npy`
  → binary peak labels (output)

* `model1_splits.csv`
  → train / val / test indices

---

## What Model 1 should do

Train a model that:

* takes ECG signal as input
* predicts positions of R-peaks

---

# Model 2 Files (Disease Classification)

Folder:

```text
data/processed/model2/
```

---

## Beat-Level (recommended)

* `X_model2_beats.npy`
  → ECG segments around peaks

* `y_model2_beats.npy`
  → disease labels

* `model2_beat_splits.csv`
  → train / val / test

---

## Record-Level (alternative)

* `X_model2_records.npy`
* `y_model2_records.npy`
* `model2_record_splits.csv`

---

## What Model 2 should do

Train a model that:

* takes ECG segments (or full signals)
* predicts heart condition

---

# How the Two Models Connect

## Training Phase (current)

* Model 2 uses **ground truth R-peaks** (from dataset)
* ensures clean and stable learning

---

## Final Pipeline (integration)

Once Model 1 is trained:

1. Model 1 predicts R-peaks
2. segments are extracted around predicted peaks
3. Model 2 uses those segments

```text
raw ECG → Model 1 → predicted peaks → segments → Model 2
```

---

# Workflow Between Team Members

## Person 1 (Data + Preprocessing)

Completed:

* dataset handling
* cleaning and normalization
* preprocessing
* label mapping
* train/val/test splits
* creation of all final files

---

## Person 2 (Model 1 – R-peak Detection)

Use:

* `X_model1.npy`
* `y_model1.npy`
* `model1_splits.csv`

Task:

* train model to detect R-peaks

---

## Person 3 (Model 2 – Classification)

### Phase 1 (now)

Use:

* `X_model2_beats.npy`
* `y_model2_beats.npy`
* `model2_beat_splits.csv`

Task:

* train classifier using clean data

---

### Phase 2 (later)

* replace ground truth peaks with Model 1 predictions
* evaluate full pipeline

---

# Important Note

Everything is already processed and uploaded.

## GitHub Status

* All datasets are already preprocessed
* All `.npy` and `.csv` files are included
* No one needs to run preprocessing scripts

You can directly:

* load the files
* start training models

---

# Key Design Decisions

* separated detection and classification
* used clean labels for training
* simplified SCP codes into 5 classes
* prepared both beat-level and record-level data
* used proper dataset splits


---

# Model 1 Status Update (Person 2)

## Model 1 Implementation

Model 1 has been implemented as a 1D CNN for sequence-based R-peak detection.

### Input
- ECG window of shape `(720,)`

### Output
- probability sequence of shape `(720,)`
- post-processed into predicted R-peak indices using `find_peaks(...)`

### Final Model
- architecture: `SimpleCNN`
- framework: PyTorch
- post-processing:
  - `height = 0.25`
  - `distance = 30`
- evaluation tolerance:
  - `tolerance = 10`

---

## Model 1 Final Performance

### Validation Set
- Precision: `0.9844`
- Recall: `0.9203`
- F1 Score: `0.9513`

### Test Set
- Precision: `0.9834`
- Recall: `0.9251`
- F1 Score: `0.9534`

These results show strong and stable R-peak detection performance with good generalization from validation to test data.

---

## Saved Model

The trained Model 1 weights are saved as:

```text
models/model1_simplecnn.pth
```

---

## Model 1 → Model 2 Handoff

A pipeline script was implemented to connect Model 1 outputs with Model 2 inputs:

```text
scripts/pipeline_model1_to_model2.py
```

---

## Exported Data for Model 2 (Ready to Use)

To avoid running the pipeline manually, example outputs from Model 1 are already generated and saved.

### Files

```text
data/processed/model2/model1_predicted_beats_example.npy
data/processed/model2/model1_predicted_beats_example_metadata.csv
```

---

### What is inside?

#### `model1_predicted_beats_example.npy`

- shape: `(n_beats, 70)`
- each row = one ECG beat centered around a predicted R-peak
- directly usable as input for Model 2

---

#### `model1_predicted_beats_example_metadata.csv`

- columns:
  - `signal_idx` → original ECG window index
  - `peak_idx` → detected R-peak position

- each row corresponds to one beat in the `.npy` file

---

## How to Use (Person 3)

### Recommended (training)
Use clean dataset:
```text
X_model2_beats.npy
y_model2_beats.npy
```

---

### ### Pipeline Simulation (using Model 1 output instead of ground truth)

For testing the full pipeline:

```text
model1_predicted_beats_example.npy
```

---

## Important Notes

- Model 1 predictions are highly accurate but not perfect
- recall < 1.0 → some beats may be missing
- edge peaks are removed if no full 70-sample window exists
- labels are NOT included → must be handled separately

---

## Status

Model 1 is fully complete:
- trained
- evaluated (validation + test)
- saved
- integrated into pipeline
- producing Model 2-ready input

No further work required.



