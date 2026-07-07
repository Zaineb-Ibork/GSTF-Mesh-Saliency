"""
Contribution 2 — Contextual Rank Features (Section 3.2 of the proposal).

All three existing features (R, GS, SS) are differential — they measure
contrast relative to a local neighbourhood.  A nipple on a smooth body
may score low in absolute terms but still be among the most geometrically
distinct points of that specific mesh.  The MLP never sees this global
context.  Contextual rank provides it.

For feature f and scale k:
  CR^k_f(vi) = (1/m) * |{vj in Mk : f^k(vj) <= f^k(vi)}|

This is the empirical CDF of f evaluated at vi — a value in [0, 1] encoding
"this vertex is in the p-th percentile of feature f for this mesh."
It is mesh-normalised: a smooth region that still outranks 80% of other
vertices receives CR = 0.80 regardless of absolute magnitude.

Applied to each of the 3 existing features at each of the 3 scales:
  CR_R^k, CR_GS^k, CR_SS^k   for k in {1, 2, 3}   → 9 new features
Computed in O(m log m) — a simple sort.  No new parameters.
"""
import numpy as np


def empirical_cdf(values: np.ndarray) -> np.ndarray:
    """
    Compute the empirical CDF of a 1-D array.
    Returns an array of the same shape with values in (0, 1].

    CR_f(vi) = (1/m) * |{vj : f(vj) <= f(vi)}|
    Ties are broken by average rank (consistent with the definition).
    """
    m = len(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = np.arange(1, m + 1)   # rank 1 = smallest

    # Handle ties: assign the average rank to all tied values
    sorted_vals = values[order]
    i = 0
    while i < m:
        j = i + 1
        while j < m and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
        ranks[order[i:j]] = avg
        i = j

    return ranks / m


def contextual_rank_features(phi_raw: np.ndarray) -> np.ndarray:
    """
    Given raw (un-normalised) single-scale features phi_raw of shape (m, 3)
    = [R, GS, SS], compute the 9 contextual rank features:
      CR_R, CR_GS, CR_SS   (3 features, same order as input)

    Returns array of shape (m, 3) with values in (0, 1].
    Computed separately per feature column — each is ranked within the mesh.
    """
    m, n_feat = phi_raw.shape
    cr = np.empty_like(phi_raw, dtype=np.float64)
    for j in range(n_feat):
        cr[:, j] = empirical_cdf(phi_raw[:, j])
    return cr
