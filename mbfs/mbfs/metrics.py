"""Evaluation metrics for feature selection results."""

from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np

__all__ = ["evaluate_selection"]


def evaluate_selection(
    prob_scores: Union[np.ndarray, Iterable[float]],
    true_idx: Union[np.ndarray, List[int]],
) -> dict:
    """
    Evaluate feature selection by thresholding theta at 0.5.

    Parameters
    ----------
    prob_scores : array-like of shape (p,)
        Posterior mean of each feature, i.e. alpha / (alpha + beta).
    true_idx : sequence of int
        Indices of the truly relevant features.

    Returns
    -------
    dict
        Keys: ``FDP``, ``Precision``, ``Recall``, ``F1``, ``Accuracy``,
        ``Hamming``, ``selected_features``.
    """
    prob_scores = np.asarray(prob_scores)
    p = prob_scores.shape[0]

    selected_indices = np.where(prob_scores > 0.5)[0]

    sel = np.zeros(p, dtype=int)
    sel[selected_indices] = 1

    truth = np.zeros(p, dtype=int)
    truth[list(true_idx)] = 1

    TP = int(np.sum((sel == 1) & (truth == 1)))
    FP = int(np.sum((sel == 1) & (truth == 0)))
    FN = int(np.sum((sel == 0) & (truth == 1)))
    TN = int(np.sum((sel == 0) & (truth == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (
        (TP + TN) / (TP + TN + FP + FN)
        if (TP + TN + FP + FN) > 0
        else 0.0
    )
    fdp = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    hamming = int(np.sum(sel != truth))

    return {
        "FDP": fdp,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "Hamming": hamming,
        "selected_features": selected_indices.tolist(),
    }