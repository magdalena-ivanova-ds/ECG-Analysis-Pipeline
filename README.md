# ECG Analysis Pipeline

A two-stage ECG analysis project for **R-peak detection** and **ECG disease classification**.

The project combines two connected tasks:

1. **Model 1: R-peak detection**
   - detects heartbeat positions in ECG signals
   - trained on MIT-BIH ECG windows with R-peak labels

2. **Model 2: disease classification**
   - classifies ECG beat segments into diagnostic classes
   - trained on PTB-XL beat-level ECG data

The full pipeline is:

```text
raw ECG signal -> Model 1 -> predicted R-peaks -> beat segmentation -> Model 2 -> disease class
```

Model 2 does **not** classify the direct output of Model 1. Model 1 only gives R-peak positions. These positions are then used to cut ECG beat segments, and those beat segments are used as input for Model 2.

---

## Project Overview

This repository contains:

- ECG preprocessing scripts
- processed training files
- exploratory data analysis
- Model 1 training and evaluation
- Model 2 training scripts
- trained model outputs
- Model 1 to Model 2 handoff script
- generated beat segments from Model 1 predictions

The preprocessing and model pipeline are already completed. The processed `.npy` and `.csv` files are included, so the raw preprocessing doesn't need to be rerun for normal use.

---

## Datasets

Two datasets were used because the two models require different labels.

### MIT-BIH Arrhythmia Database

MIT-BIH was used for **Model 1** because it contains ECG signals with annotated R-peak positions.

In this project:

- sampling rate: `360 Hz`
- selected lead: one ECG lead
- window length: `720` samples
- window duration: `2 seconds`
- label type: binary R-peak mask

Model 1 input and output:

```text
input:  ECG window with 720 samples
output: binary sequence with 720 values
```

Each output value indicates whether the corresponding signal position is an R-peak or not.

### PTB-XL ECG Dataset

PTB-XL was used for **Model 2** because it contains diagnostic ECG labels.

In this project:

- sampling rate: `100 Hz`
- selected lead: one ECG lead
- record length: `1000` samples
- beat segment length: `70` samples
- label type: diagnostic class

Model 2 input and output:

```text
input:  ECG beat segment with 70 samples
output: one disease class
```

---

## PTB-XL Label Mapping

PTB-XL labels are stored as SCP code dictionaries, for example:

```text
{'IMI': 35.0, 'ABQRS': 0.0}
```

These SCP codes are detailed diagnostic ECG codes. Since one ECG record can contain multiple codes, the raw labels were mapped into five broader classes to make the classification task clearer.

| Code | Meaning |
|---|---|
| NORM | Normal ECG |
| MI | Myocardial infarction |
| HYP | Hypertrophy |
| CD | Conduction disturbance |
| STTC | ST/T wave changes |

The final Model 2 task is a five-class classification problem:

```text
NORM, MI, HYP, CD, STTC
```

---

## Preprocessing

The preprocessing was completed before training. The main steps were:

1. load ECG signals
2. select one ECG lead
3. apply z-score normalization
4. apply bandpass filtering
5. create model-specific input and label files
6. create train / validation / test splits
7. save the final arrays and metadata files

A Butterworth bandpass filter was used to reduce baseline drift and high-frequency noise while preserving the relevant ECG signal shape.

### Model 1 Preprocessing

For MIT-BIH:

- ECG records were split into fixed 2-second windows
- each window contains `720` samples
- R-peak annotations were converted into binary label sequences
- each label sequence has the same length as the input window

### Model 2 Preprocessing

For PTB-XL, two versions were created:

1. **Record-level data**
   - full ECG records
   - `1000` samples per record
   - one label per record

2. **Beat-level data**
   - short ECG beat segments
   - `70` samples per beat
   - one label per beat

The beat-level version is the main version for Model 2, because the final pipeline also classifies beat segments extracted around R-peaks.

---

## Final Processed Files

All processed data is stored in:

```text
data/processed/
```

### Model 1 Files

Folder:

```text
data/processed/model1/
```

Main files:

```text
X_model1.npy
y_model1.npy
model1_metadata.csv
model1_splits.csv
```

Final shapes:

```text
X_model1.npy -> (86592, 720)
y_model1.npy -> (86592, 720)
```

Meaning:

- `X_model1.npy` contains the ECG input windows
- `y_model1.npy` contains the binary R-peak labels
- `model1_splits.csv` contains the train / validation / test split information
- `model1_metadata.csv` contains metadata for the created windows

### Model 2 Beat-Level Files

Folder:

```text
data/processed/model2/
```

Main beat-level files:

```text
X_model2_beats.npy
y_model2_beats.npy
model2_beat_metadata.csv
model2_beat_splits.csv
```

Final shapes:

```text
X_model2_beats.npy -> (320871, 70)
y_model2_beats.npy -> (320871,)
```

Meaning:

- `X_model2_beats.npy` contains ECG beat segments
- `y_model2_beats.npy` contains the disease labels
- `model2_beat_splits.csv` contains the train / validation / test split information
- `model2_beat_metadata.csv` contains beat-level metadata

### Model 2 Record-Level Files

Record-level files are also available:

```text
X_model2_records.npy
y_model2_records.npy
model2_record_metadata.csv
model2_record_splits.csv
```

Final shapes:

```text
X_model2_records.npy -> (16244, 1000)
y_model2_records.npy -> (16244,)
```

The record-level data is available as an alternative representation, but the beat-level data is the main input for Model 2.

---

## Data Splits

### Model 1

Model 1 uses a 70 / 15 / 15 split:

```text
train:      70%
validation: 15%
test:       15%
```

### Model 2

Model 2 follows the PTB-XL fold-based split:

```text
folds 1-8 -> train
fold 9    -> validation
fold 10   -> test
```

For the beat-level data, beats from the same ECG record are kept in the same split to avoid information leakage.

---

## Model 1: R-Peak Detection

Model 1 is a 1D CNN for sequence-based R-peak detection.

### Input and Output

```text
input:  ECG window of shape (720,)
output: probability sequence of shape (720,)
```

The output probability sequence is post-processed into predicted R-peak positions using peak detection.

### Architecture

The implemented Model 1 architecture is `SimpleCNN`.

It uses 1D convolutional layers to process the ECG window and produce one prediction value for each position in the signal.

### Training Details

Main setup:

```text
framework: PyTorch
loss: binary cross-entropy
optimizer: Adam
post-processing: scipy.signal.find_peaks
peak height threshold: 0.25
minimum peak distance: 30
evaluation tolerance: 10 samples
```

### Model 1 Performance

Validation set:

```text
Precision: 0.9844
Recall:    0.9203
F1 Score:  0.9513
```

Test set:

```text
Precision: 0.9834
Recall:    0.9251
F1 Score:  0.9534
```

The results indicate good R-peak detection performance, with similar validation and test scores.

The trained Model 1 weights are saved as:

```text
models/model1_simplecnn.pth
```

Relevant files:

```text
scripts/train_model1.py
scripts/pipeline_model1_to_model2.py
models/model1_simplecnn.pth
```

---

## Model 2: Disease Classification

Model 2 classifies ECG beat segments into one of five diagnostic classes:

```text
NORM, MI, HYP, CD, STTC
```

The main Model 2 input data is:

```text
data/processed/model2/X_model2_beats.npy
data/processed/model2/y_model2_beats.npy
data/processed/model2/model2_beat_splits.csv
```

### Model 2 Approaches

The repository contains two Model 2 training approaches:

1. **Random Forest baseline**
2. **Beat-level CNN classifier**

### Random Forest Baseline

Relevant file:

```text
models/model2/training/train_rf.py
```

The Random Forest baseline uses the beat segments as feature vectors. It provides a simpler baseline model for comparison with the CNN.

Saved outputs include:

```text
models/model2/savedTrainedWeights/rf_baseline.joblib
models/model2/savedTrainedWeights/rf_improved.joblib
```

### Beat-Level CNN

Relevant file:

```text
models/model2/training/train_cnn.py
```

The CNN classifier uses 1D convolutional layers to learn patterns from the beat-level ECG segments.

Relevant architecture file:

```text
models/model2/neuralNetworkArchitecture/beat_cnn.py
```

Saved output:

```text
models/model2/savedTrainedWeights/cnn_beat_classifier.pt
```

### Model 2 Reports and Outputs

Model 2 outputs, metrics, and reports are saved in:

```text
models/model2/savedTrainedWeights/
models/model2/reports/
```

Because the Model 2 dataset is imbalanced, evaluation should not rely only on accuracy. Precision, recall, F1-score, and the confusion matrix are more useful for understanding class-specific performance.

---

### Model 2 Results

The best Model 2 CNN with class weights reached a test Macro F1 of `0.411` and test accuracy of `0.577`.

The improved Random Forest baseline reached a higher test accuracy of `0.632`, but a lower Macro F1 of `0.395`.

Since the dataset is imbalanced, Macro F1 was treated as more important than accuracy. Accuracy alone can be misleading because a model can perform well on the majority class while still missing smaller disease classes.

---

## Model 1 to Model 2 Handoff

The Model 1 to Model 2 handoff is part of the final pipeline.

The process is:

1. Model 1 predicts R-peak positions.
2. Beat segments are extracted around the predicted peaks.
3. The extracted beat segments are passed to Model 2.

The handoff script is:

```text
scripts/pipeline_model1_to_model2.py
```

Generated handoff output files are already included:

```text
data/processed/model2/X_model2_beats_from_model1.npy
data/processed/model2/model2_beats_from_model1_metadata.csv
```

These files contain beat segments generated using Model 1 predictions. They can be used to test the full pipeline connection between Model 1 and Model 2.

The handoff script does not need to be rerun for normal submission because the output files are already generated. It is included for reproducibility.

---

## Exploratory Data Analysis

The EDA was used to check whether the preprocessing output was correct and usable for training.

### Model 1 EDA

The Model 1 EDA checked:

- input and label shapes
- train / validation / test split sizes
- number of windows per ECG record
- number of R-peaks per ECG window
- ECG windows plotted together with their R-peak labels

The main visual check was whether the R-peak labels aligned with the visible heartbeat peaks.

### Model 2 EDA

The Model 2 EDA checked:

- record-level and beat-level shapes
- class distributions
- beat-level class imbalance
- beats per ECG record
- average beat shape by disease class

The main finding was that the beat-level Model 2 dataset is imbalanced. NORM is the largest class, while HYP is the smallest class.

---

## Project Structure

Important folders:

```text
data/processed/                 final processed data files
data/processed/model1/           Model 1 input, labels, metadata, splits
data/processed/model2/           Model 2 input, labels, metadata, splits
scripts/                         preprocessing, Model 1 training, handoff scripts
models/model2/                   Model 2 architecture, loaders, training scripts, outputs
models/model2/reports/           Model 2 reports and plots
models/model2/savedTrainedWeights/ saved Model 2 weights and metrics
notebook_eda/                    EDA notebook
```

Main documentation files:

```text
README.md       project explanation
RUN_GUIDE.md    short reproduction / run instructions
```

---

## Important Notes and Limitations

- MIT-BIH and PTB-XL use different sampling rates.
- MIT-BIH is sampled at `360 Hz`, while PTB-XL is sampled at `100 Hz`.
- Model 1 peak sample indices cannot be directly reused for PTB-XL-style Model 2 input without converting them based on time or handling the sampling-rate difference.
- Model 2 beat-level data was created during preprocessing and is the main training input.
- The Model 1 to Model 2 handoff output is already generated and included.
- Model 2 has class imbalance, so accuracy alone is not enough for evaluation.
- The project separates R-peak detection and disease classification so that both parts can be trained and evaluated more clearly.

---

## Dataset Links

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- PTB-XL ECG Dataset: https://physionet.org/content/ptb-xl/1.0.3/
