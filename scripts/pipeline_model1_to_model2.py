from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
import pandas as pd

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.sigmoid(x)
        return x


def load_trained_model(model_path: str | Path) -> SimpleCNN:
    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_peak_probabilities(signal: np.ndarray, model: SimpleCNN) -> np.ndarray:
    """
    Input:
        signal shape: (720,)
    Output:
        probabilities shape: (720,)
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal of shape (720,), got shape {signal.shape}")

    x = torch.tensor(signal[np.newaxis, np.newaxis, :], dtype=torch.float32)

    with torch.no_grad():
        probs = model(x).squeeze().numpy()

    return probs


def detect_r_peaks(probabilities: np.ndarray, height: float = 0.25, distance: int = 30) -> np.ndarray:
    """
    Converts probability signal into predicted R-peak indices.
    """
    peak_indices, _ = find_peaks(probabilities, height=height, distance=distance)
    return peak_indices


def extract_beats(signal: np.ndarray, peak_indices: np.ndarray, beat_size: int = 70) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed-size beat segments centered on predicted peaks.

    Returns:
        beats: shape (n_beats, 70)
        valid_peak_indices: only peaks for which a full beat segment could be extracted
    """
    if beat_size % 2 != 0:
        raise ValueError("beat_size should be even so extraction is symmetric around the peak.")

    half_window = beat_size // 2
    beats = []
    valid_peaks = []

    for peak in peak_indices:
        start = peak - half_window
        end = peak + half_window

        if start >= 0 and end <= len(signal):
            beat = signal[start:end]
            if len(beat) == beat_size:
                beats.append(beat)
                valid_peaks.append(peak)

    if len(beats) == 0:
        return np.empty((0, beat_size), dtype=np.float32), np.empty((0,), dtype=int)

    return np.array(beats, dtype=np.float32), np.array(valid_peaks, dtype=int)


def process_signal_for_model2(
    signal: np.ndarray,
    model: SimpleCNN,
    height: float = 0.25,
    distance: int = 30,
    beat_size: int = 70,
) -> dict:
    """
    Full Model 1 -> Model 2 handoff for one ECG signal/window.

    Returns a dictionary with:
        - probabilities
        - predicted_peak_indices
        - extracted_beats
        - valid_peak_indices
    """
    probabilities = predict_peak_probabilities(signal, model)
    predicted_peak_indices = detect_r_peaks(probabilities, height=height, distance=distance)
    extracted_beats, valid_peak_indices = extract_beats(signal, predicted_peak_indices, beat_size=beat_size)

    return {
        "probabilities": probabilities,
        "predicted_peak_indices": predicted_peak_indices,
        "extracted_beats": extracted_beats,
        "valid_peak_indices": valid_peak_indices,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models/model1_simplecnn.pth"
    data_path = project_root / "data/processed/model1/X_model1.npy"

    model = load_trained_model(model_path)

    X = np.load(data_path)

    # Process the Signal Outputs of Model 1

    total_extracted_beats = 0
    all_extracted_beats = []

    beat_metadata = []

    for idx in range(len(X)):
        signal = X[idx]
        result = process_signal_for_model2(signal, model)
        total_extracted_beats += result["extracted_beats"].shape[0]

        if result["extracted_beats"].shape[0] > 0:
            all_extracted_beats.append(result["extracted_beats"])

        for peak in result["valid_peak_indices"]:
            beat_metadata.append({
                "signal_idx": idx,
                "peak_idx": int(peak)
            })

    print("\n=== Summary ===")
    print(f"Total extracted beats from first 5 signals: {total_extracted_beats}")

    combined_beats = np.vstack(all_extracted_beats)
    print(f"Combined beats array shape: {combined_beats.shape}")

    output_path = project_root / "data/processed/model2/X_model2_beats_from_model1.npy"
    np.save(output_path, combined_beats)

    print(f"Saved combined beats to: {output_path}")

    metadata_df = pd.DataFrame(beat_metadata)
    metadata_path = project_root / "data/processed/model2/model2_beats_from_model1_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"Saved beat metadata to: {metadata_path}")

if __name__ == "__main__":
    main()