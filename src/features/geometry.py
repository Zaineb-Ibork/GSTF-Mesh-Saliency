"""
Geometric quantities on triangle meshes:
  - cotangent Laplacian (sparse, symmetric)
  - mean curvature κ per vertex
  - angle-weighted vertex normals
  - γ-step smoothed curvature via normalised adjacency power
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


# ---------------------------------------------------------------------------
# Cotangent weights and Laplacian
# ---------------------------------------------------------------------------

def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised cotangent of the angle between vectors a and b."""
    dot   = np.einsum("ij,ij->i", a, b)
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    return dot / (cross + 1e-12)


def cotangent_laplacian(vertices: np.ndarray, triangles: np.ndarray):
    """
    Return the (symmetric) cotangent Laplacian as a scipy sparse CSR matrix.
    L[i,j] = 0.5 * (cot α_ij + cot β_ij)  for i≠j adjacent
    L[i,i] = -sum_{j} L[i,j]
    """
    m = len(vertices)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    # Cotangent at each corner (angle opposite to the edge)
    # cot at v0: angle between edges v0v1 and v0v2
    cot0 = _cot(v1 - v0, v2 - v0)
    # cot at v1: angle between edges v1v0 and v1v2
    cot1 = _cot(v0 - v1, v2 - v1)
    # cot at v2: angle between edges v2v0 and v2v1
    cot2 = _cot(v0 - v2, v1 - v2)

    # Edge (i1, i2) has cot0 at corner v0 opposite to it
    rows = np.concatenate([i1, i2, i0, i2, i0, i1])
    cols = np.concatenate([i2, i1, i2, i0, i1, i0])
    vals = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]) * 0.5

    W = csr_matrix((vals, (rows, cols)), shape=(m, m))
    # Symmetrize in case of numerical issues
    W = (W + W.T) * 0.5

    # Diagonal: -sum of row weights
    diag_vals = np.array(W.sum(axis=1)).ravel()
    D = csr_matrix((diag_vals, (np.arange(m), np.arange(m))), shape=(m, m))
    L = D - W
    return L, W


def row_normalise(W):
    """Return row-stochastic version of a non-negative sparse matrix."""
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = csr_matrix((1.0 / row_sums, (np.arange(len(row_sums)),
                                        np.arange(len(row_sums)))),
                     shape=W.shape)
    return inv @ W


# ---------------------------------------------------------------------------
# Mean curvature
# ---------------------------------------------------------------------------

def mean_curvature(vertices: np.ndarray, triangles: np.ndarray,
                   L=None, W=None) -> np.ndarray:
    """
    Unsigned mean curvature κ(vi) = ||H_i|| / (2 * A_i)
    where H_i = L @ x  (cotangent Laplacian applied to vertex positions)
    and   A_i is the mixed (Voronoi) area.
    Returns array of shape (m,).
    """
    m = len(vertices)
    if L is None or W is None:
        L, W = cotangent_laplacian(vertices, triangles)

    # Mean-curvature normal (unnormalised)
    Hn = L.dot(vertices)  # (m, 3)

    # Vertex areas: one-third of each incident triangle's area
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    areas = np.zeros(m)
    for k in range(3):
        np.add.at(areas, triangles[:, k], tri_areas / 3.0)
    areas = np.maximum(areas, 1e-12)

    kappa = np.linalg.norm(Hn, axis=1) / (2.0 * areas)
    return kappa


# ---------------------------------------------------------------------------
# Vertex normals (angle-weighted)
# ---------------------------------------------------------------------------

def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Angle-weighted vertex normals (eq. 9 of Bærentzen & Aanæs [13]).
    Returns unit normals of shape (m, 3).
    """
    m = len(vertices)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    e01 = v1 - v0
    e02 = v2 - v0
    face_n = np.cross(e01, e02)           # (t, 3) unnormalised

    normals = np.zeros((m, 3))
    for k, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
        ea = vertices[triangles[:, a]] - vertices[triangles[:, k]]
        eb = vertices[triangles[:, b]] - vertices[triangles[:, k]]
        cos_ang = np.einsum("ij,ij->i", ea, eb) / (
            np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-12)
        angle = np.arccos(np.clip(cos_ang, -1.0, 1.0))   # (t,)
        np.add.at(normals, triangles[:, k], face_n * angle[:, None])

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-12)
    return normals


# ---------------------------------------------------------------------------
# γ-step smoothed curvature for roughness
# ---------------------------------------------------------------------------

def smoothed_curvature(kappa: np.ndarray, W_norm, gamma: int) -> np.ndarray:
    """
    Compute (W_norm^γ @ κ) efficiently by repeated sparse matrix-vector
    multiplication. W_norm must be row-stochastic.
    Returns smoothed curvature of shape (m,).
    """
    s = kappa.copy()
    for _ in range(gamma):
        s = W_norm.dot(s)
    return s
