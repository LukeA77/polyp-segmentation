"""Pixel-level confidence calibration: ECE + reliability bins.

Each pixel's sigmoid output is treated as its predicted probability of being polyp;
bins are formed directly on that probability (not max-confidence), matching the
common binary-calibration convention (e.g. scikit-learn's calibration curves).
"""

from __future__ import annotations

import numpy as np


def reliability_bins(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean confidence, empirical accuracy, pixel count) for a reliability
    diagram, over flattened probability/target arrays.
    """
    probs = np.asarray(probs).ravel()
    targets = np.asarray(targets).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(probs, bin_edges[1:-1]), 0, n_bins - 1)

    confidences = np.zeros(n_bins)
    accuracies = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        in_bin = bin_indices == b
        counts[b] = in_bin.sum()
        if counts[b] > 0:
            confidences[b] = probs[in_bin].mean()
            accuracies[b] = targets[in_bin].mean()

    return confidences, accuracies, counts


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Pixel-level ECE: the count-weighted mean |confidence - accuracy| across bins."""
    confidences, accuracies, counts = reliability_bins(probs, targets, n_bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    weights = counts / total
    return float(np.sum(weights * np.abs(confidences - accuracies)))
