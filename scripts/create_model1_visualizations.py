from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pipeline_model1_to_model2 import load_trained_model, process_signal_for_model2
from scipy.signal import find_peaks

# DISCLAIMER: THESE VISUALIZATIONS ARE FOR PRESENTATION PURPOSES AND EXPLANATIONS AND ARE PARTIALLY HARD-CODED!

def plot_beat_extraction(signal, peak_indices, beat_size=70, output_path=None):
    half_window = beat_size // 2

    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="ECG signal")

    for peak in peak_indices:
        start = peak - half_window
        end = peak + half_window

        if start >= 0 and end <= len(signal):
            plt.axvspan(start, end, alpha=0.2)
            plt.axvline(peak, linestyle="--", linewidth=1.5)
            plt.scatter(peak, signal[peak], s=50, label="R-peak" if peak == peak_indices[0] else None)

            plt.text(
                peak,
                signal[peak] + 0.3,
                f"peak {peak}",
                ha="center",
                fontsize=9
            )

    plt.title("Beat extraction around predicted R-peaks")
    plt.xlabel("Sample position")
    plt.ylabel("ECG amplitude")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Saved figure to: {output_path}")

    plt.show()

def plot_ecg_window_with_labels(signal, labels, output_path):
    peak_indices = np.where(labels == 1)[0]
    plt.figure(figsize=(8, 3))
    plt.plot(signal, linewidth=1.5, label="ECG input window")
    plt.scatter(
        peak_indices,
        signal[peak_indices],
        color="red",
        s=35,
        label="Ground-truth R-peaks"
    )

    plt.title("Example ECG Window Used for Model 1 Training")
    plt.xlabel("Sample position")
    plt.ylabel("Normalized ECG amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_prediction_vs_truth(signal, true_labels, model, output_path):

    # Get true R-peak positions from binary labels
    true_peaks = np.where(true_labels == 1)[0]

    # Get model predictions and post-processed predicted R-peaks
    result = process_signal_for_model2(signal, model)

    pred_peaks = result["predicted_peak_indices"]

    plt.figure(figsize=(8, 3))
    plt.plot(signal, linewidth=1.5, label="ECG signal")

    # Show tolerance windows around true peaks
    tolerance = 10
    for peak in true_peaks:
        plt.axvspan(
            peak - tolerance,
            peak + tolerance,
            alpha=0.15
        )

    # True peaks
    plt.scatter(
        true_peaks,
        signal[true_peaks],
        color="green",
        s=45,
        label="True R-peaks"
    )

    # Predicted peaks
    plt.scatter(
        pred_peaks,
        signal[pred_peaks],
        color="red",
        marker="x",
        s=60,
        label="Predicted R-peaks"
    )

    plt.title("Model 1 Prediction Example")
    plt.xlabel("Sample position")
    plt.ylabel("Normalized ECG amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to: {output_path}")

    plt.show()

def plot_training_loss_curve(output_path):
    epochs = [1, 2, 3, 4, 5]
    train_loss = [0.0247, 0.0064, 0.0056, 0.0053, 0.0051]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o", label="Training loss")

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to: {output_path}")
    plt.show()



def find_missed_peaks(true_peaks, pred_peaks, tolerance=10):
    """
    Find true R-peaks that were not matched by any predicted R-peak.

    A predicted peak counts as a match if it is within +/- tolerance samples
    of a ground-truth peak.
    """
    missed_peaks = []

    for true_peak in true_peaks:
        distances = np.abs(pred_peaks - true_peak)

        if len(distances) == 0 or np.min(distances) > tolerance:
            missed_peaks.append(true_peak)

    return np.array(missed_peaks, dtype=int)


def plot_model1_failure_example(X, y, model, output_path, height=0.25, distance=30, tolerance=10):
    """
    Search through ECG windows and plot the first example where Model 1 misses
    at least one true R-peak.

    This is used for the presentation slide 'Where does Model 1 fail?'.
    """
    for signal_idx in range(len(X)):
        signal = X[signal_idx]
        true_labels = y[signal_idx]

        true_peaks = np.where(true_labels == 1)[0]

        result = process_signal_for_model2(
            signal=signal,
            model=model,
            height=height,
            distance=distance,
            beat_size=70
        )
        pred_peaks = result["predicted_peak_indices"]

        missed_peaks = find_missed_peaks(
            true_peaks=true_peaks,
            pred_peaks=pred_peaks,
            tolerance=tolerance
        )

        if len(missed_peaks) > 0:
            print("Found failure example")
            print(f"Signal index: {signal_idx}")
            print(f"True peaks: {true_peaks}")
            print(f"Predicted peaks: {pred_peaks}")
            print(f"Missed peaks: {missed_peaks}")

            plt.figure(figsize=(9, 3.5))
            plt.plot(signal, linewidth=1.5, label="ECG signal")

            # Show tolerance windows around true peaks.
            for peak in true_peaks:
                plt.axvspan(peak - tolerance, peak + tolerance, alpha=0.12)

            # Ground-truth peaks.
            plt.scatter(
                true_peaks,
                signal[true_peaks],
                color="green",
                s=45,
                label="True R-peaks"
            )

            # Predicted peaks.
            plt.scatter(
                pred_peaks,
                signal[pred_peaks],
                color="red",
                marker="x",
                s=65,
                label="Predicted R-peaks"
            )

            # Missed true peaks.
            plt.scatter(
                missed_peaks,
                signal[missed_peaks],
                facecolors="none",
                edgecolors="orange",
                linewidths=2.5,
                s=130,
                label="Missed true peak"
            )

            plt.title("Model 1 Failure Example: Missed R-peak")
            plt.xlabel("Sample position")
            plt.ylabel("Normalized ECG amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            print(f"Saved figure to: {output_path}")
            plt.show()
            return

    print("No missed-peak example found with the current settings.")


# New function: plot_model1_interior_failure_example
def plot_model1_interior_failure_example(
        X,
        y,
        model,
        output_path,
        height=0.25,
        distance=30,
        tolerance=10,
        boundary_margin=35
):
    """
    Search through ECG windows and plot a failure example where Model 1 misses
    a true R-peak that is not close to the signal boundary.

    Boundary peaks are ignored here because they mainly show beat-extraction
    limitations, not necessarily true model detection failures.
    """
    for signal_idx in range(len(X)):
        signal = X[signal_idx]
        true_labels = y[signal_idx]

        true_peaks = np.where(true_labels == 1)[0]

        result = process_signal_for_model2(
            signal=signal,
            model=model,
            height=height,
            distance=distance,
            beat_size=70
        )
        pred_peaks = result["predicted_peak_indices"]

        missed_peaks = find_missed_peaks(
            true_peaks=true_peaks,
            pred_peaks=pred_peaks,
            tolerance=tolerance
        )

        # Remove boundary misses so we focus on a real interior detection issue.
        interior_missed_peaks = missed_peaks[
            (missed_peaks >= boundary_margin) &
            (missed_peaks <= len(signal) - boundary_margin)
        ]

        if len(interior_missed_peaks) > 0:
            print("Found interior failure example")
            print(f"Signal index: {signal_idx}")
            print(f"True peaks: {true_peaks}")
            print(f"Predicted peaks: {pred_peaks}")
            print(f"Missed peaks before boundary filtering: {missed_peaks}")
            print(f"Interior missed peaks: {interior_missed_peaks}")

            plt.figure(figsize=(9, 3.5))
            plt.plot(signal, linewidth=1.5, label="ECG signal")

            # Show tolerance windows around true peaks.
            for peak in true_peaks:
                plt.axvspan(peak - tolerance, peak + tolerance, alpha=0.12)

            # Ground-truth peaks.
            plt.scatter(
                true_peaks,
                signal[true_peaks],
                color="green",
                s=45,
                label="True R-peaks"
            )

            # Predicted peaks.
            plt.scatter(
                pred_peaks,
                signal[pred_peaks],
                color="red",
                marker="x",
                s=65,
                label="Predicted R-peaks"
            )

            # Interior missed true peaks.
            plt.scatter(
                interior_missed_peaks,
                signal[interior_missed_peaks],
                facecolors="none",
                edgecolors="orange",
                linewidths=2.5,
                s=130,
                label="Missed true peak"
            )

            plt.title("Model 1 Failure Example: Interior Missed R-peak")
            plt.xlabel("Sample position")
            plt.ylabel("Normalized ECG amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            print(f"Saved figure to: {output_path}")
            plt.show()
            return

    print("No interior missed-peak example found with the current settings.")


def plot_multiple_interior_failure_examples(
        X,
        y,
        model,
        output_dir,
        height=0.25,
        distance=30,
        tolerance=10,
        boundary_margin=35,
        max_examples=6
):
    """
    Save multiple interior missed-peak examples.

    This helps inspect whether Model 1 misses similar-looking peaks or whether
    the failure cases differ across ECG windows.
    """
    output_dir.mkdir(exist_ok=True)

    saved_examples = 0

    for signal_idx in range(len(X)):
        signal = X[signal_idx]
        true_labels = y[signal_idx]

        true_peaks = np.where(true_labels == 1)[0]

        result = process_signal_for_model2(
            signal=signal,
            model=model,
            height=height,
            distance=distance,
            beat_size=70
        )
        pred_peaks = result["predicted_peak_indices"]

        missed_peaks = find_missed_peaks(
            true_peaks=true_peaks,
            pred_peaks=pred_peaks,
            tolerance=tolerance
        )

        interior_missed_peaks = missed_peaks[
            (missed_peaks >= boundary_margin) &
            (missed_peaks <= len(signal) - boundary_margin)
        ]

        if len(interior_missed_peaks) == 0:
            continue

        saved_examples += 1

        print(f"\nSaved interior failure example {saved_examples}")
        print(f"Signal index: {signal_idx}")
        print(f"True peaks: {true_peaks}")
        print(f"Predicted peaks: {pred_peaks}")
        print(f"Interior missed peaks: {interior_missed_peaks}")

        output_path = output_dir / f"model1_interior_failure_{saved_examples:02d}_signal_{signal_idx}.png"

        plt.figure(figsize=(9, 3.5))
        plt.plot(signal, linewidth=1.5, label="ECG signal")

        # Show tolerance windows around true peaks.
        for peak in true_peaks:
            plt.axvspan(peak - tolerance, peak + tolerance, alpha=0.12)

        # Ground-truth peaks.
        plt.scatter(
            true_peaks,
            signal[true_peaks],
            color="green",
            s=45,
            label="True R-peaks"
        )

        # Predicted peaks.
        plt.scatter(
            pred_peaks,
            signal[pred_peaks],
            color="red",
            marker="x",
            s=65,
            label="Predicted R-peaks"
        )

        # Interior missed true peaks.
        plt.scatter(
            interior_missed_peaks,
            signal[interior_missed_peaks],
            facecolors="none",
            edgecolors="orange",
            linewidths=2.5,
            s=130,
            label="Missed true peak"
        )

        plt.title(f"Interior Missed R-peak Example {saved_examples}")
        plt.xlabel("Sample position")
        plt.ylabel("Normalized ECG amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved figure to: {output_path}")

        if saved_examples >= max_examples:
            break

    print(f"\nTotal interior failure examples saved: {saved_examples}")

    if saved_examples == 0:
        print("No interior missed-peak examples found with the current settings.")


def main():
    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    X = np.load(project_root / "data/processed/model1/X_model1.npy")
    y = np.load(project_root / "data/processed/model1/y_model1.npy")

    signal = X[0]
    labels = y[0]

    model_path = project_root / "models/model1_simplecnn.pth"
    model = load_trained_model(model_path)

    # ---------- enable for plot 1 ----------
    # peak_indices = np.array([77, 370, 662])
    # output_path = figures_dir / "model1_beat_extraction_window.png"

    # plot_beat_extraction(
        # signal=signal,
        # peak_indices=peak_indices,
        # beat_size=70,
        # output_path=output_path)

    # ---------- enable for plot 2 ----------
    # output_path = figures_dir / "model1_example_input_window.png"
    # plot_ecg_window_with_labels(signal, labels, output_path)

    # ---------- enable for plot 3 ----------
    # prediction_output_path = figures_dir / "model1_prediction_vs_truth.png"
    # plot_prediction_vs_truth(
        # signal=X[0],
        # true_labels=y[0],
        # model=model,
        # output_path=prediction_output_path)

    # ---------- enable for plot 4 ----------
    # loss_curve_output_path = figures_dir / "model1_training_loss_curve.png"

    # plot_training_loss_curve(
        # output_path=loss_curve_output_path
    # )

    # ---------- enable for plot 5 ----------
    # failure_output_path = figures_dir / "model1_failure_example.png"

    # plot_model1_failure_example(
        # X=X,
        # y=y,
        # model=model,
        # output_path=failure_output_path,
        # height=0.25,
        # distance=30,
        # tolerance=10
    # )

    # ---------- enable for plot 6 ----------
    # interior_failure_output_path = figures_dir / "model1_interior_failure_example.png"

    # plot_model1_interior_failure_example(
        # X=X,
        # y=y,
        # model=model,
        # output_path=interior_failure_output_path,
        # height=0.25,
        # distance=30,
        # tolerance=10,
        # boundary_margin=35
    # )

    # ---------- enable for plot 7 ----------
    multiple_failure_dir = figures_dir / "model1_multiple_interior_failures"

    plot_multiple_interior_failure_examples(
        X=X,
        y=y,
        model=model,
        output_dir=multiple_failure_dir,
        height=0.25,
        distance=30,
        tolerance=10,
        boundary_margin=35,
        max_examples=6
    )

if __name__ == "__main__":
    main()