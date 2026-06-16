# Run Guide

This file gives short instructions for running or reproducing the ECG analysis project.

The preprocessing and generated files are already included. For normal review, the project can be used directly from the prepared files, without rerunning the raw dataset download or preprocessing.

---

## Already Generated Files

The final processed data is already stored in:

```text
data/processed/
```

### Model 1

```text
data/processed/model1/X_model1.npy
data/processed/model1/y_model1.npy
data/processed/model1/model1_metadata.csv
data/processed/model1/model1_splits.csv
```

### Model 2

```text
data/processed/model2/X_model2_beats.npy
data/processed/model2/y_model2_beats.npy
data/processed/model2/model2_beat_metadata.csv
data/processed/model2/model2_beat_splits.csv
data/processed/model2/X_model2_records.npy
data/processed/model2/y_model2_records.npy
data/processed/model2/model2_record_metadata.csv
data/processed/model2/model2_record_splits.csv
```

### Model 1 to Model 2 Handoff Output

The output from Model 1 that is used to create Model 2-ready beat segments is also already generated:

```text
data/processed/model2/X_model2_beats_from_model1.npy
data/processed/model2/model2_beats_from_model1_metadata.csv
```

This handoff is part of the final pipeline. It doesn't need to be rerun unless you want to reproduce that step.

---

## Setup

Create and activate a virtual environment.

Install the requirements:

```powershell
pip install -r requirements.txt
```

---

## Train Model 1

Model 1 uses the prepared MIT-BIH files in:

```text
data/processed/model1/
```

Run from the `scripts/` folder:

```powershell
cd scripts
python train_model1.py
```

The trained weights are saved as:

```text
models/model1_simplecnn.pth
```

---

## Reproduce the Model 1 to Model 2 Handoff

This step uses the trained Model 1 to predict R-peaks and extract beat segments for Model 2.

Run from the project root:

```powershell
python scripts\pipeline_model1_to_model2.py
```

Generated outputs:

```text
data/processed/model2/X_model2_beats_from_model1.npy
data/processed/model2/model2_beats_from_model1_metadata.csv
```

These files are already included, so this is only needed for reproduction.

---

## Train Model 2

Model 2 uses the prepared PTB-XL beat-level files in:

```text
data/processed/model2/
```

### Random Forest baseline

Run from the project root:

```powershell
python models\model2\training\train_rf.py
```

### CNN classifier

Run from the project root:

```powershell
python models\model2\training\train_cnn.py
```

Model 2 outputs are saved mainly in:

```text
models/model2/savedTrainedWeights/
models/model2/reports/
```

---

## Regenerate Preprocessing

The preprocessing scripts are included for reproducibility, but they are not needed for normal use.

The full preprocessing workflow is controlled by:

```text
scripts/run_all.py
```

By default, the heavy steps should stay disabled (they are commented).

Only uncomment them if you want to rebuild the processed data from scratch.

