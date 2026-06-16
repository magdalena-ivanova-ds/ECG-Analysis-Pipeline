# Model 2 — Final Report: Beat-Level ECG Disease Classification

*Author: Capstone project team (Model 2 pipeline)*  
*Date: June 2026*

---

## 1. Introduction

Model 2 is the second stage of our ECG Analysis Pipeline. After Model 1 detects R-peaks and extracts heartbeat segments, Model 2 classifies each 70-sample beat into one of five PTB-XL disease superclasses:


| Label | Meaning                |
| ----- | ---------------------- |
| NORM  | Normal ECG             |
| MI    | Myocardial infarction  |
| HYP   | Hypertrophy            |
| CD    | Conduction disturbance |
| STTC  | ST/T wave change       |


My goal was to build a reproducible, interpretable classifier that works on pre-processed beat-level data and is evaluated fairly on an official held-out test fold. Because the classes are highly imbalanced, I used **macro F1** as the primary metric rather than accuracy alone.

---

## 2. Dataset Analysis

I used only the prepared files:

- `data/processed/model2/X_model2_beats.npy` — shape `(320,871, 70)`
- `data/processed/model2/y_model2_beats.npy`
- `data/processed/model2/model2_beat_splits.csv` — official PTB-XL stratified folds

**Class distribution (full dataset):**


| Class | Beats   | Share |
| ----- | ------- | ----- |
| NORM  | 179,381 | 55.9% |
| MI    | 50,228  | 15.7% |
| STTC  | 48,659  | 15.2% |
| CD    | 32,618  | 10.2% |
| HYP   | 9,985   | 3.1%  |


**Split sizes:** 255,905 train · 32,352 validation (fold 9) · 32,614 test (fold 10).

### Why macro F1 is difficult here

Macro F1 averages F1 across all five classes with equal weight. A model that predicts NORM most of the time can still achieve high accuracy (~~65%) while scoring poorly on macro F1 (~~0.37) because minority classes (especially HYP with only ~1,200 validation beats) contribute little to accuracy but heavily to macro F1. This makes the task genuinely hard: we must detect rare diseases without destroying performance on the majority class.

---

## 3. Initial Problems

When I started, I observed these issues:

1. **Low macro F1 (~0.37)** on the Random Forest baseline without imbalance handling.
2. **Strong bias toward NORM** — recall for NORM exceeded 95% while HYP recall was 0%.
3. **Poor minority detection** — MI, HYP, and CD had very low recall in early models.
4. **Fragmented project structure** — code was spread across `src/`, `reports/`, and a training notebook, which made the pipeline hard to follow and reproduce.
5. **Mixed responsibilities** — the notebook was training models and evaluating them in the same place, which is bad practice for a capstone submission.

---

## 4. Fixes Applied

### 4.1 Class imbalance handling

- **Mandatory class weights** in CNN loss:  
`weight_i = total / (num_classes × count_i)` computed from training labels only.
- **Random Forest**: `class_weight="balanced_subsample"` with `n_estimators=300`.
- **Mild NORM undersampling** on the training set only (`keep_ratio=0.5`) — reduces majority dominance without touching validation or test splits.
- I did **not** use SMOTE (see Section 6).

### 4.2 Model tuning (simple architectures only)

- Kept a **lightweight 1D CNN** (two Conv1D blocks + linear head). No transformers, LSTMs, or attention.
- Added **early stopping** on validation macro F1 (patience = 10).
- Used **Adam** optimizer, learning-rate reduction on plateau, and gradient clipping.
- Increased RF trees from 200 → 300 for the improved baseline.

### 4.3 Pipeline restructuring

I removed the `src/` folder entirely and consolidated Model 2 under `models/model2/` with descriptive subfolder names:

```
models/model2/
├── beatDataLoading/          # load beats + official splits
├── neuralNetworkArchitecture/  # BeatCNNClassifier
├── evaluationMetrics/        # metrics, plots, inference
├── training/                 # config + train_rf.py + train_cnn.py
├── savedTrainedWeights/    # .joblib / .pt / metrics JSON
└── reports/                  # this document
```

### 4.4 Notebook refactor

- **Deleted** `notebooks/model2_pipeline.ipynb` (training + EDA mixed).
- **Created** `notebooks/model2_evaluation.ipynb` — evaluation only:
  - loads saved models
  - computes metrics
  - displays confusion matrices, per-class F1, ROC curves, comparison table
  - does **not** train or tune

### 4.5 Logging improvements

Training scripts now print:

- class distribution before and after undersampling
- model hyperparameters and parameter counts
- per-epoch loss, accuracy, macro F1, weighted F1
- per-class F1 every epoch
- confusion matrix summaries at checkpoints
- wall-clock timing per stage

Example log format:

```
Epoch 3/40
Loss: 1.1605
Accuracy: 0.5175
Macro F1: 0.3965
Class F1:
  NORM: 0.6867
  MI: 0.2102
  ...
```

### 4.6 File cleanup

- Removed duplicate scripts from `src/model2/`
- Stopped writing plots to `reports/model2/` (evaluation is notebook-inline)
- Centralised metrics in `savedTrainedWeights/training_metrics_summary.json`

---

## 5. What Worked

### Macro F1 improvements (validation)


| Model                   | Macro F1  | Δ vs RF baseline |
| ----------------------- | --------- | ---------------- |
| RF Baseline             | 0.374     | —                |
| RF Improved             | 0.394     | +0.020           |
| CNN (no weights)        | 0.409     | +0.035           |
| **CNN + class weights** | **0.414** | **+0.040**       |


This is a **~11% relative improvement** in macro F1 over the original RF baseline.

### Per-class improvements (validation, best CNN vs RF baseline)


| Class | RF Baseline F1 | CNN + Weights F1 | Change                      |
| ----- | -------------- | ---------------- | --------------------------- |
| NORM  | 0.793          | 0.749            | −0.044 (expected trade-off) |
| MI    | 0.183          | 0.185            | +0.002                      |
| HYP   | 0.000          | 0.182            | **+0.182**                  |
| CD    | 0.431          | 0.466            | +0.035                      |
| STTC  | 0.464          | 0.487            | +0.023                      |


The largest gain is **HYP detection** — from zero F1 to 0.18 recall of 23%. This was the main benefit of class weighting and mild undersampling.

### Confusion matrix changes

The improved CNN predicts HYP and STTC more often instead of collapsing everything into NORM. The trade-off is slightly lower NORM recall (73% vs 96%), which is acceptable for a clinical screening context where missing a rare disease is costlier than extra false alarms on normal beats.

### Test set (fold 10, evaluated once)


| Metric       | CNN + Weights |
| ------------ | ------------- |
| Accuracy     | 0.577         |
| **Macro F1** | **0.411**     |
| Weighted F1  | 0.566         |


---

## 6. What Did NOT Work


| Approach                                                   | Result                                  | Why rejected                                                                                                                                        |
| ---------------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Weighted random sampler** (prior experiment)             | Macro F1 dropped to ~0.20; NORM F1 ≈ 0  | Oversampling minorities too aggressively destabilised training                                                                                      |
| **Deeper “tuned” CNN + focal loss** (prior `src/` version) | Marginal gain over simple CNN + weights | Added complexity without meaningful F1 improvement; removed to stay within constraints                                                              |
| **SMOTE**                                                  | Not used                                | Synthetic beats could introduce unrealistic morphologies; beat segments are only 70 samples and SMOTE on raw waveforms is hard to justify medically |
| **Heavy oversampling**                                     | Not used                                | Same concern as SMOTE; mild NORM undersampling was sufficient                                                                                       |
| **Optimising for accuracy**                                | Misleading                              | High accuracy (~65%) masked zero HYP recall                                                                                                         |


---

## 7. Final Architecture

### Pipeline flow

```
PTB-XL beats (preprocessed)
        │
        ▼
beatDataLoading/loader.py  ── official train/val/test split
        │
        ├─► training/train_rf.py  ──► savedTrainedWeights/rf_*.joblib
        │
        └─► training/train_cnn.py ──► savedTrainedWeights/cnn_beat_classifier.pt
                    │
                    ▼
        notebooks/model2_evaluation.ipynb  (metrics + plots only)
                    │
                    ▼
        evaluationMetrics/inference.py  (Model 1 beats → predictions)
```

### How to run (from project root)

```bash
python models/model2/training/train_rf.py
python models/model2/training/train_cnn.py
# Then open notebooks/model2_evaluation.ipynb
python models/model2/evaluationMetrics/inference.py
```

---

## 8. Results Comparison

### Validation set (fold 9)


| Model                   | Accuracy  | Macro F1  | Weighted F1 |
| ----------------------- | --------- | --------- | ----------- |
| RF Baseline             | 0.649     | 0.374     | 0.586       |
| RF Improved             | 0.642     | 0.394     | 0.599       |
| CNN (no weights)        | 0.648     | 0.409     | 0.609       |
| **CNN + class weights** | **0.582** | **0.414** | **0.576**   |


*Note: CNN + weights trades accuracy for better minority recall — the correct trade-off for macro F1.*

### Test set (fold 10)


| Model                   | Accuracy  | Macro F1  | Weighted F1 |
| ----------------------- | --------- | --------- | ----------- |
| RF Improved             | 0.632     | 0.395     | 0.584       |
| **CNN + class weights** | **0.577** | **0.411** | **0.566**   |


---

## 9. Conclusion

I restructured Model 2 into a clean, reproducible pipeline under `models/model2/`, separated training from evaluation, and applied principled imbalance handling (class weights + mild NORM undersampling). Macro F1 improved from **0.374 → 0.414** on validation (+0.040) and reached **0.411** on the held-out test set. The most meaningful per-class gain was detecting HYP, which the baseline completely ignored.

### Limitations

- Macro F1 remains moderate (~0.41) because beat-level 70-sample windows carry limited disease signal and HYP is extremely rare.Lower accuracy on the weighted CNN is
-  an expected consequence of prioritising minority recall.
- I did not tune on the test set; all model selection used validation macro F1 only.

### Future work (light suggestions)

- Record-level aggregation (majority vote over beats per ECG) might stabilise predictions.
- Threshold tuning per class on validation could further improve macro F1 without changing the model.
- Integrating Model 1 predicted peaks end-to-end and measuring pipeline-level macro F1 would complete the clinical story.

---

*All metrics in this report are stored in `models/model2/savedTrainedWeights/training_metrics_summary.json` and can be reproduced by re-running the training scripts with `RANDOM_SEED=42`.*