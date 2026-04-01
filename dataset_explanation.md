Add this section to your README. It explains exactly what those SCP codes are and how they become your final labels.

---

# Label Processing and SCP Code Mapping (Model 2)

## Raw Labels in PTB-XL

The PTB-XL dataset does not store labels as simple classes.
Instead, each ECG has a column called `scp_codes`.

This column contains a **dictionary of diagnostic codes with confidence values**.

Example:

```text
{'NORM': 100.0, 'SR': 0.0}
{'IMI': 35.0, 'ABQRS': 0.0}
{'AFLT': 100.0}
```

Meaning:

* each key = a medical diagnosis code (SCP code)
* each value = confidence or relevance score

So one ECG can have **multiple diagnoses at once**.

---

## Problem with Raw SCP Codes

Using these directly is not practical because:

* too many different labels
* highly imbalanced classes
* multi-label problem (complex to train)

So the labels must be simplified.

---

## Solution: Mapping to Superclasses

PTB-XL provides a file:

```text
scp_statements.csv
```

This file maps each SCP code to a **diagnostic superclass**.

You used this mapping to convert detailed labels into 5 main categories.

---

## Final Label Classes Used

After processing, every ECG (or beat) is assigned **one main class**:

| Final Label | Meaning                                           |
| ----------- | ------------------------------------------------- |
| NORM        | Normal ECG                                        |
| MI          | Myocardial infarction (heart attack)              |
| HYP         | Hypertrophy (thickened heart muscle)              |
| CD          | Conduction disturbance (electrical signal issues) |
| STTC        | ST/T wave abnormalities                           |

---

## Example Mappings

### Example 1

```text
{'NORM': 100.0}
```

→ Final label:

```text
NORM
```

---

### Example 2

```text
{'IMI': 35.0, 'ABQRS': 0.0}
```

* IMI = Inferior Myocardial Infarction
* ABQRS = abnormal QRS

→ Final label:

```text
MI
```

---

### Example 3

```text
{'AFLT': 100.0}
```

* AFLT = atrial flutter

→ Final label:

```text
CD
```

---

### Example 4

```text
{'NST_': 100.0, 'DIG': 100.0}
```

* NST_ = non-specific ST change
* DIG = digitalis effect

→ Final label:

```text
STTC
```

---

## How the Mapping Works in the Pipeline

The transformation process is:

```text
scp_codes (raw dictionaries)
        ↓
map each SCP code → superclass (using scp_statements.csv)
        ↓
select one final label per ECG
        ↓
store as numeric/class labels
```

---

## Important Simplification

Original dataset:

* multi-label (multiple conditions per ECG)

Your pipeline:

* converted to **single-label classification**

Each sample gets:

* one dominant class

This makes:

* training simpler
* evaluation clearer

---

## Where These Labels Are Stored

After preprocessing:

### Beat-level labels

```text
y_model2_beats.npy
```

### Record-level labels

```text
y_model2_records.npy
```

Each entry corresponds to:

* one ECG segment (beat-level) or
* one full ECG (record-level)

---

## Why This Is Correct

This approach:

* reduces complexity
* keeps clinically meaningful categories
* aligns with common ECG classification tasks

---

## One-line Explanation

The original PTB-XL SCP codes are detailed multi-label diagnoses, which are mapped into five main diagnostic groups (NORM, MI, HYP, CD, STTC) to create a simplified single-label classification problem.

---

## Key Takeaway

* CSV shows **raw medical codes**
* your model uses **processed class labels**
* mapping step is essential for making the problem learnable
