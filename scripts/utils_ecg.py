import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def make_dirs(paths):
    """Create folders if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def zscore_signal(signal):
    """Simple z-score normalization."""
    signal = np.asarray(signal, dtype=np.float32)
    std = np.std(signal)
    if std == 0:
        return signal
    return (signal - np.mean(signal)) / std


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=3):
    """
    Simple ECG bandpass filter.
    Keeps the code understandable and not over-engineered.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def create_binary_peak_label(window_length, peak_positions):
    """
    Create binary label sequence for one ECG window.
    1 means an R-peak at that sample.
    """
    y = np.zeros(window_length, dtype=np.int8)
    for pos in peak_positions:
        if 0 <= pos < window_length:
            y[pos] = 1
    return y


def simple_peak_detector(signal, fs):
    """
    Very simple automatic peak detector for PTB-XL preprocessing.
    This is NOT the final model.
    It is just to create heartbeat-centered windows for later use.
    """
    min_distance = int(0.25 * fs)  # at least 250 ms between peaks
    prominence = max(0.3, 0.5 * np.std(signal))

    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
    return peaks


def parse_scp_codes(text):
    """
    PTB-XL stores diagnosis codes as text dictionaries.
    Example: "{'NORM': 100.0, 'SR': 0.0}"
    """
    if pd.isna(text):
        return {}
    return ast.literal_eval(text)


def choose_single_superclass(code_dict, scp_df, target_superclasses):
    """
    Map detailed PTB-XL codes to one broad class.

    To keep the dataset simple for a student project:
    - we only keep records that map to exactly one target superclass
    - records with multiple target superclasses are skipped
    """
    matched = set()

    for code in code_dict.keys():
        if code not in scp_df.index:
            continue

        row = scp_df.loc[code]
        diagnostic_class = row.get("diagnostic_class")

        if pd.isna(diagnostic_class):
            continue

        if diagnostic_class in target_superclasses:
            matched.add(diagnostic_class)

    if len(matched) == 1:
        return list(matched)[0]

    return None


def extract_fixed_windows(signal, peaks, fs, window_sec, stride_sec):
    """
    Split long ECG into overlapping windows.
    Also create binary R-peak labels per window.
    """
    window_size = int(window_sec * fs)
    stride = int(stride_sec * fs)

    X = []
    Y = []
    meta = []

    for start in range(0, len(signal) - window_size + 1, stride):
        end = start + window_size
        window = signal[start:end]

        peak_positions = [p - start for p in peaks if start <= p < end]
        label = create_binary_peak_label(window_size, peak_positions)

        X.append(window)
        Y.append(label)
        meta.append((start, end, len(peak_positions)))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int8), meta


def extract_beat_windows(signal, peaks, fs, before_sec, after_sec):
    """
    Extract heartbeat-centered windows around each detected peak.
    """
    before = int(before_sec * fs)
    after = int(after_sec * fs)

    windows = []

    for peak in peaks:
        start = peak - before
        end = peak + after

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]
        windows.append(beat)

    return np.array(windows, dtype=np.float32)