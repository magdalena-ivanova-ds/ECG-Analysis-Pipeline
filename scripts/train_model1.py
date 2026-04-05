from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from scipy.signal import find_peaks
import random


# =============== SET SEED FOR REPRODUCIBILITY ===============


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for reproducibility (slightly slower but safer)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# =============== IMPORT DATA: INPUT (X), LABELS (y) AND SPLIT ===============


data_dir = Path(__file__).resolve().parent.parent / "data/processed/model1"

# ----- input data model 1 -----
X = np.load(data_dir / "X_model1.npy")
# ----- labels model 1 -----
y = np.load(data_dir / "y_model1.npy")
# ----- splits for training, validating and testing -----
splits = pd.read_csv(data_dir / "model1_splits.csv")


# =============== SPLIT DATA ===============


train_indices = splits["train_idx"].dropna().astype(int)
val_indices = splits["val_idx"].dropna().astype(int)
test_indices = splits["test_idx"].dropna().astype(int)

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]


# =============== PREPARE DATA FOR A 1D NEURAL NETWORK ===============


# ----- specify signal channel expected for 1D NN-----

# Conv1D in PyTorch expects input like: (batch_size, channels, sequence_length)
# channel is 1, as our signal is just one waveform

X_train = X_train[:, np.newaxis, :]
X_val = X_val[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]

# print("X Training Shape after specifying the channel:", X_train.shape)
# print("X Validation Shape after specifying the channel:",X_val.shape)
# print("X Test Shape after specifying the channel:",X_test.shape)

y_train = y_train[:, np.newaxis, :]
y_val = y_val[:, np.newaxis, :]
y_test = y_test[:, np.newaxis, :]

# print("y Training Shape after specifying the channel:", y_train.shape)
# print("y Validation Shape after specifying the channel:",y_val.shape)
# print("y Test Shape after specifying the channel:",y_test.shape)

# ----- convert data to torch.tensors -----

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ----- create tensor datasets, pairing each input with its correct label -----

# PyTorch models work with datasets and dataloaders not with raw tensors

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


# =============== CREATE DATA LOADERS FOR BATCHING ===============


# 64 ECG windows at a time
# shuffle while training for each epoch: prevents model from memorizing order

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# =============== ***** CREATE A CNN MODEL CLASS ***** ===============


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #layers slide across the signal to learn patterns

        #layer 1: detects small signals
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
            #in_channels = 1   ...we have 1 ECG signal
            #out_channels = 16 ...16 filters (= patern detectors) used, each specialising in something (= more ways to understand the signal)
            #kernel_size = 5   ...filter looks at 5 values at the same time
            #padding = 2       ...adds fake values to edges so output length stays the same

        self.relu1 = nn.ReLU() # introduction to non-linearity: kills negative values to be 0, only keeping strong positive signals

        #layer 2: detects local patterns
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()

        #layer 3: detects heartbeat shapes
        self.conv3 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()

        #layer 4: detects if it is an R-peak or not
        self.conv4 = nn.Conv1d(16, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid() #converts output into values between 0 and 1

    def forward(self, x):
        """Path that inputs are taking through the network"""
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)

        x = self.sigmoid(x)
        return x


# =============== CREATE THE MODEL ===============


model = SimpleCNN()
#print(model)


# =============== LOSS FUNCTION ===============

# ----- use of Binary Cross Entropy Loss for Binary Classification -----

criterion = nn.BCELoss()
# compatible with sigmoid (values lie between 0 and 1)


# =============== OPTIMIZER ===============

# ----- to adjust the model weights to reduce the loss -----

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
# model.parameters() ...targets all weights in the network
# Adam               ...smart optimizer (more efficient gradient descent)
# lr                 ...learning rate (=> too big = unstable, too small = slow)


# =============== ***** TRAINING OUR MODEL ***** ===============

# ---- going through the entire dataset 5 times -----
num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # set model to training mode (layers behave differently than in validation)

    total_loss = 0

    for X_batch, y_batch in train_loader:
        # reset gradients (don't mix old + new learnings)
        optimizer.zero_grad()

        # forward pass (prediction = calls forward() to process the input data)
        outputs = model(X_batch)

        # compute loss (comparing prediction vs. truth)
        loss = criterion(outputs, y_batch)

        # backward pass (compute gradients utilising backpropagation)
        loss.backward()

        # update weights to improve
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# ----- save the model -----

torch.save(model.state_dict(), "../models/model1_simplecnn.pth")
print("Model saved to model1_simplecnn.pth")


# =============== ***** EVALUATING LOSS ON EVALUATION SET ***** ===============


model.eval()  # switches model to evaluation mode

val_loss = 0

with torch.no_grad():  # no gradients needed for validating

    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)             #predictions for validation set
        loss = criterion(outputs, y_batch)   #prediction vs truth
        val_loss += loss.item()

avg_val_loss = val_loss / len(val_loader)

print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


# =============== ***** EVALUATING OUR MODEL ***** ===============


def match_peaks(true_indices, pred_indices, tolerance=10):
    """How many predicted peaks match real peaks (within some error)?"""
    matched_true = set()
    matched_pred = set()

    for pred_i, pred_peak in enumerate(pred_indices):
        for true_i, true_peak in enumerate(true_indices):
            if true_i in matched_true:
                continue

            if abs(pred_peak - true_peak) <= tolerance:
                matched_true.add(true_i)
                matched_pred.add(pred_i)
                break

    tp = len(matched_pred)
    fp = len(pred_indices) - tp
    fn = len(true_indices) - len(matched_true)

    return tp, fp, fn


def evaluate_peak_detection(model, data_loader, height=0.25, distance=30, tolerance=10):
    """Full evaluation of peak detection"""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    model.eval()

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)

            for i in range(X_batch.shape[0]):
                true_signal = y_batch[i].squeeze().numpy()
                pred_signal = outputs[i].squeeze().numpy()

                true_indices = [j for j in range(len(true_signal)) if true_signal[j] == 1]
                pred_indices, _ = find_peaks(pred_signal, height=height, distance=distance)

                tp, fp, fn = match_peaks(true_indices, pred_indices, tolerance=tolerance)

                total_tp += tp
                total_fp += fp
                total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# ----- Evaluation on validation set -----
val_precision, val_recall, val_f1 = evaluate_peak_detection(
    model,
    val_loader,
    height=0.25,
    distance=30,
    tolerance=10
)

print("Validation Metrics on VALIDATION SET")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1 Score:  {val_f1:.4f}")

# ----- Evaluation on test set -----
test_precision, test_recall, test_f1 = evaluate_peak_detection(
    model,
    test_loader,
    height=0.25,
    distance=30,
    tolerance=10
)

print("\nFinal Metrics on TEST SET")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")