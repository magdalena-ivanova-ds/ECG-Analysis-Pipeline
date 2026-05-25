# ECG analysis pipeline

With this, I prepare the data for a two-stage ECG analysis system.

The idea is simple:

```text
raw ECG signal -> model 1 -> predicted R-peaks -> beat segments -> model 2 -> disease class
```

Model 1 finds the heartbeat positions in an ECG signal. These heartbeat positions are called R-peaks. Model 2 then uses short ECG segments around those peaks to classify the heart condition.

At this stage, the data preparation part is finished. The processed files are already available in `data/processed/`, so the model training parts can start directly from those files.


## Datasets used

Two datasets are used because the two models need different types of labels:

1. [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)  
   Used for Model 1 because it provides beat-level arrhythmia labels.

2. [PTB-XL ECG Dataset](https://physionet.org/content/ptb-xl/1.0.3/)  
   Used for Model 2 because it provides diagnostic ECG-level disease labels.

## MIT-BIH for model 1

MIT-BIH is used for the R-peak detection model.

This dataset is useful for model 1 because it contains ECG signals together with heartbeat annotations. These annotations show where the R-peaks are located in the signal.

In this project:

```text
sampling rate: 360 Hz
selected lead: lead index 0
window length: 2 seconds
window size: 720 samples
stride: 1 second
```

Each long ECG recording is split into smaller overlapping windows. For each window, a binary label array is created:

```text
0 = no R-peak at this position
1 = R-peak at this position
```

So model 1 learns this task:

```text
ECG window -> R-peak positions inside that window
```

## PTB-XL for model 2

PTB-XL is used for the disease classification model.

This dataset is useful for model 2 because it contains ECG recordings with diagnostic labels. The labels are stored in a column called `scp_codes`.

The raw labels are not simple class names. They are stored as dictionary-like values, for example:

```text
{'NORM': 100.0, 'SR': 0.0}
{'IMI': 35.0, 'ABQRS': 0.0}
```

Each key is a diagnostic code, and each value gives the confidence or relevance of that code. One ECG can therefore have more than one diagnosis.

To keep the project manageable, these detailed SCP codes are mapped into five broader diagnostic groups:

| final class | meaning |
| --- | --- |
| NORM | normal ECG |
| MI | myocardial infarction |
| HYP | hypertrophy |
| CD | conduction disturbance |
| STTC | ST/T changes |

Only records that clearly map to one of these five classes are kept.

In this project:

```text
sampling rate: 100 Hz
selected lead: lead index 0
record length: 1000 samples
record duration: 10 seconds
```

## Record-level and beat-level data

For model 2, two versions of the PTB-XL data were prepared.

The record-level version keeps the full ECG recording as one sample:

```text
one 10-second ECG record -> one disease label
```

The beat-level version cuts each ECG record into smaller heartbeat segments:

```text
one heartbeat segment -> one disease label
```

Each beat segment gets the same label as the ECG record it came from. For example, if one ECG record is labeled `MI`, then all beat segments extracted from that record are also labeled `MI`.

The beat-level version is the main version for model 2 because the final pipeline will use R-peaks from model 1 to extract heartbeat segments.

## Important note about the beat segments

The current PTB-XL beat segments were created during preprocessing with a simple peak detection method. This was done so that model 2 can already be trained on beat-level input.

This simple detector is not the final R-peak detection model.

The final version of the pipeline should work like this:

```text
raw ECG -> model 1 predicts R-peaks -> extract beat segments -> model 2 predicts disease class
```

So model 2 is trained first with prepared beat segments. Later, when model 1 is ready, the beat segments can be extracted from model 1 predictions and the full system can be tested again.

There is also a sampling-rate difference between the two datasets:

```text
MIT-BIH: 360 Hz
PTB-XL: 100 Hz
```

Because of that, the final integration must handle sampling rates carefully. Peak positions should either be converted by time or extracted from signals with a consistent sampling rate.

## Preprocessing steps

The preprocessing does the following:

1. Loads the ECG signals.
2. Selects one ECG lead.
3. Normalizes the signal with z-score normalization.
4. Applies a bandpass filter to reduce noise.
5. Creates model-specific samples and labels.
6. Saves the final arrays and metadata files.

## Model 1 data

Model 1 uses MIT-BIH.

Folder:

```text
data/processed/model1/
```

Files:

| file | description |
| --- | --- |
| `X_model1.npy` | ECG windows used as input |
| `y_model1.npy` | binary R-peak labels for each window |
| `model1_metadata.csv` | record name, window start, window end, number of peaks |
| `model1_splits.csv` | train, validation, and test indices |

Current data summary:

```text
X_model1 shape: (86592, 720)
y_model1 shape: (86592, 720)
total windows: 86592
window length: 720 samples
original records used: 48
train samples: 60614
validation samples: 12989
test samples: 12989
```

Each sample is a 2-second ECG window. Each label is a binary sequence of the same length, where the value `1` marks an R-peak.

## Model 2 data

Model 2 uses PTB-XL.

Folder:

```text
data/processed/model2/
```

Beat-level files:

| file | description |
| --- | --- |
| `X_model2_beats.npy` | heartbeat segments used as input |
| `y_model2_beats.npy` | disease labels for the heartbeat segments |
| `model2_beat_metadata.csv` | ECG id, beat index, and label information |
| `model2_beat_splits.csv` | train, validation, and test indices |

Record-level files:

| file | description |
| --- | --- |
| `X_model2_records.npy` | full 10-second ECG records |
| `y_model2_records.npy` | disease labels for the full records |
| `model2_record_metadata.csv` | ECG id, label, and signal path |
| `model2_record_splits.csv` | train, validation, and test indices |

Current data summary:

```text
X_model2_records shape: (16244, 1000)
y_model2_records shape: (16244,)
total record-level samples: 16244

X_model2_beats shape: (320871, 70)
y_model2_beats shape: (320871,)
total beat-level samples: 320871
beat window length: 70 samples
```

The beat window has 70 samples. Since PTB-XL is sampled at 100 Hz, this is 0.7 seconds of ECG data.

The current beat window uses:

```text
0.25 seconds before the detected peak -> 25 samples
0.45 seconds after the detected peak -> 45 samples
```

This keeps the R-peak and also includes more of the post-peak ECG shape, which can be useful for disease-related patterns such as ST/T changes.

## Model 2 class distribution

Record-level class counts:

```text
NORM    9069
MI      2532
STTC    2400
CD      1708
HYP      535
```

Beat-level class counts:

```text
NORM    179381
MI       50228
STTC     48659
CD       32618
HYP       9985
```

The dataset is imbalanced. `NORM` is the largest class, while `HYP` is the smallest class. This should be considered later during model training and evaluation.

## Splits

Model 1 uses a random 70/15/15 split of the MIT-BIH windows:

```text
70% training
15% validation
15% testing
```

Model 2 uses the official PTB-XL folds:

```text
folds 1 to 8 -> training
fold 9 -> validation
fold 10 -> testing
```

The beat-level split follows the record-level split. This means beats from the same ECG record stay in the same split and are not mixed between training, validation, and testing.

## Exploratory data analysis

The EDA notebook is included in the `notebook_eda/` folder.

It shows the summaries and plots directly inside the notebook.

The notebook covers:
- Model 1 shapes
- Model 1 split sizes
- Windows per MIT-BIH record
- R-peaks per window
- Example ECG windows with R-peak labels
- Model 1 signal value distribution

- Model 2 shapes
- Model 2 split sizes
- Record-level class distribution
- Beat-level class distribution
- Beats per ECG record
- Class distribution by split
- Record-level vs beat-level comparison
- Average beat shape per class
- Example beat segment per class
- Model 2 signal value distribution

## How to use the processed files

The data is already processed, so the model training parts don't need to run the preprocessing scripts again.

Anna (who will build Model 1) should use:

```text
data/processed/model1/X_model1.npy
data/processed/model1/y_model1.npy
data/processed/model1/model1_splits.csv
```

Dimitar (who will build Model 2) should use the beat-level model 2 files:

```text
data/processed/model2/X_model2_beats.npy
data/processed/model2/y_model2_beats.npy
data/processed/model2/model2_beat_splits.csv
```

The record-level model 2 files are kept as an alternative version, but the beat-level version is the main one for the planned final pipeline.

## How to run

Install the requirements:

```bash
pip install -r requirements.txt
```

To view the full EDA:

```text
open the notebook in notebooks/ and run all cells
```

The full preprocessing scripts are kept in the repository for reproducibility. They don't need to be run again unless the processed data has to be recreated.

## Optional full pipeline script

`scripts/run_all.py` contains switches for controlling what should run.

By default, the heavy steps should stay turned off if the data is already processed:

```python
run_downloads = False
run_preprocessing = False
run_splits = False
```

Set these values to `True` only if the raw data or processed files need to be recreated.

## Responsibilities

I finished the data and preprocessing part:

```text
dataset handling
signal cleaning
normalization
window creation for model 1
beat extraction for model 2
label mapping for PTB-XL
train/validation/test splits
EDA notebook
processed output files
```

Anna should train and evaluate model 1:

```text
input: ECG windows
output: predicted R-peak locations
```

Dimitar should train and evaluate model 2:

```text
input: beat-level ECG segments
output: disease class
```

After both models are ready, we should connect them and evaluate the full pipeline:

```text
raw ECG -> model 1 -> predicted peaks -> extracted beat segments -> model 2 -> diagnosis
```

## Important limitations

- The two models are prepared separately at this stage.

- Model 2 is currently trained on beat segments created during preprocessing, not on segments created from model 1 predictions.

- The final connected pipeline should therefore be evaluated again after model 1 predictions are used.

- The two datasets use different sampling rates, so this must be handled.

## Main design decisions

- The project uses two separate models because R-peak detection and disease classification are different tasks.

- MIT-BIH is used for model 1 because it provides clear R-peak annotations.

- PTB-XL is used for model 2 because it provides diagnostic ECG labels.

- PTB-XL labels are simplified into five broad classes to make the classification task more manageable.

- Beat-level PTB-XL data is the main input for model 2 because the final pipeline will classify disease based on heartbeat-centered segments.
