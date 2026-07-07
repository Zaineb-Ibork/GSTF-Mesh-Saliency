"""
Evaluation metrics matching the Schelling dataset evaluation protocol.

PLCC  — Pearson Linear Correlation Coefficient between predicted saliency S
         and ground-truth Gaussian-smoothed distribution G.
AUC   — Area Under the ROC Curve, thresholding S to predict Schelling points.
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


def plcc(S: np.ndarray, G: np.ndarray) -> float:
    """Pearson correlation between prediction S and ground-truth G."""
    if S.std() < 1e-10 or G.std() < 1e-10:
        return 0.0
    r, _ = pearsonr(S, G)
    return float(r)


def auc_schelling(S: np.ndarray, schelling_idx: np.ndarray,
                   n_vertices: int) -> float:
    """
    AUC: binary label = 1 for Schelling points, 0 otherwise.
    S: per-vertex predicted saliency.
    schelling_idx: 0-based vertex indices of the ground-truth salient points.
    """
    labels = np.zeros(n_vertices, dtype=np.int32)
    labels[schelling_idx] = 1
    if labels.sum() == 0 or labels.sum() == n_vertices:
        return 0.5
    return float(roc_auc_score(labels, S))


def evaluate_dataset(predictions: list, ground_truths: list,
                      schelling_pts: list) -> dict:
    """
    predictions  : list of per-vertex np.ndarray (one per mesh)
    ground_truths: list of per-vertex np.ndarray
    schelling_pts: list of np.ndarray of Schelling point indices
    Returns dict with mean PLCC and mean AUC.
    """
    plccs, aucs = [], []
    for S, G, pts in zip(predictions, ground_truths, schelling_pts):
        plccs.append(plcc(S, G))
        aucs.append(auc_schelling(S, pts, len(S)))
    # PLCC_list / AUC_list are the per-mesh values (same order as the inputs);
    # kept so downstream paired significance tests don't need a re-run.
    return {"PLCC": float(np.mean(plccs)), "AUC": float(np.mean(aucs)),
            "PLCC_std": float(np.std(plccs)), "AUC_std": float(np.std(aucs)),
            "PLCC_list": [float(x) for x in plccs],
            "AUC_list": [float(x) for x in aucs]}
