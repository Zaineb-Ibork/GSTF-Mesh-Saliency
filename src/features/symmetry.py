"""
Contribution 1 — Symmetry Consistency Features (Section 3.1 of the proposal).

Steps:
  1. Estimate bilateral symmetry plane via PCA (smallest-eigenvalue axis)
  2. Find symmetric counterpart v_sym(vi) for every vertex via reflected
     nearest-neighbour (k-d tree, O(m log m))
  3. Compute confidence score conf_sym(M) that suppresses the feature on
     asymmetric meshes (e.g. a hand in a grasping pose)
  4. SymF^k(vi) = conf_sym(Mk) * ||patch_vec[vi] - patch_vec[v_sym(vi)]||_2

Post-processing pass (applied after MLP, in evaluation):
  S_tilde(vi) = alpha * S(vi) + (1 - alpha) * conf_sym * S(v_sym(vi))
"""
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Symmetry plane (Step 1)
# ---------------------------------------------------------------------------

def symmetry_plane(vertices: np.ndarray):
    """
    Estimate the bilateral symmetry plane normal via PCA.
    The axis of least variance of the vertex cloud approximates the
    bilateral axis for symmetric objects.

    Returns (mu, n_hat):
      mu    — centroid of vertex positions (3,)
      n_hat — unit normal of the symmetry plane (3,), smallest eigenvalue
    """
    mu = vertices.mean(axis=0)
    centered = vertices - mu
    C = (centered.T @ centered) / len(vertices)
    eigenvalues, eigenvectors = np.linalg.eigh(C)   # sorted ascending
    n_hat = eigenvectors[:, 0]                        # smallest eigenvalue
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-12)
    return mu, n_hat


def reflect_points(points: np.ndarray, mu: np.ndarray,
                   n_hat: np.ndarray) -> np.ndarray:
    """
    Reflect points through the plane defined by (mu, n_hat).
    reflect(p) = p - 2 * [(p - mu) . n_hat] * n_hat
    """
    d    = points - mu                                # (m, 3)
    proj = (d @ n_hat)[:, None] * n_hat               # (m, 3)
    return points - 2 * proj


# ---------------------------------------------------------------------------
# Symmetric vertex correspondence (Step 2)
# ---------------------------------------------------------------------------

def symmetric_correspondence(vertices: np.ndarray,
                              mu: np.ndarray,
                              n_hat: np.ndarray) -> np.ndarray:
    """
    For each vertex vi find its symmetric counterpart:
      v_sym(vi) = argmin_{vj in V} ||reflect(p(vi)) - p(vj)||_2
    Returns integer index array of shape (m,).
    """
    reflected = reflect_points(vertices, mu, n_hat)
    tree = cKDTree(vertices)
    _, v_sym = tree.query(reflected, k=1)
    return v_sym.astype(np.int32)


# ---------------------------------------------------------------------------
# Symmetry confidence (Step 3)
# ---------------------------------------------------------------------------

def symmetry_confidence(vertices: np.ndarray,
                         v_sym:    np.ndarray,
                         mu:       np.ndarray,
                         n_hat:    np.ndarray) -> float:
    """
    conf_sym(M) = exp(-sigma2_refl / sigma2_pos)

    sigma2_refl = (1/m) sum_i ||p(v_sym(vi)) - reflect(p(vi))||^2
                  (mean squared reflection residual)
    sigma2_pos  = total vertex position variance (scale reference)
    """
    reflected    = reflect_points(vertices, mu, n_hat)
    residuals    = np.linalg.norm(vertices[v_sym] - reflected, axis=1) ** 2
    sigma2_refl  = float(residuals.mean())
    sigma2_pos   = float(np.var(vertices))        # scalar variance over all coords
    return float(np.exp(-sigma2_refl / (sigma2_pos + 1e-12)))


# ---------------------------------------------------------------------------
# SymF feature (Step 4)
# ---------------------------------------------------------------------------

def compute_symf(patch_vecs: np.ndarray,
                 v_sym:      np.ndarray,
                 conf_sym:   float) -> np.ndarray:
    """
    SymF(vi) = conf_sym * ||patch_vec[vi] - patch_vec[v_sym(vi)]||_2

    Uses the precomputed fixed-size patch vectors (57-D) so the distance
    is computed in a single vectorised operation.
    Returns array of shape (m,) normalised to [0, 1] per-mesh.
    """
    diff = patch_vecs - patch_vecs[v_sym]         # (m, D)
    dist = np.linalg.norm(diff, axis=1)            # (m,)
    symf = conf_sym * dist

    mn, mx = symf.min(), symf.max()
    return (symf - mn) / (mx - mn + 1e-12)


# ---------------------------------------------------------------------------
# Full per-scale SymF pipeline
# ---------------------------------------------------------------------------

def extract_symf_single_scale(vertices:   np.ndarray,
                               patch_vecs: np.ndarray) -> tuple:
    """
    Run the full symmetry pipeline on one decimation level.
    Returns (symf [m,], conf_sym scalar, v_sym [m,]).
    """
    mu, n_hat   = symmetry_plane(vertices)
    v_sym       = symmetric_correspondence(vertices, mu, n_hat)
    conf        = symmetry_confidence(vertices, v_sym, mu, n_hat)
    symf        = compute_symf(patch_vecs, v_sym, conf)
    return symf, conf, v_sym


# ---------------------------------------------------------------------------
# Post-processing pass (applied to MLP output, not to features)
# ---------------------------------------------------------------------------

def symmetry_propagation(S:       np.ndarray,
                          v_sym:   np.ndarray,
                          conf:    float,
                          alpha:   float = 0.7) -> np.ndarray:
    """
    S_tilde(vi) = alpha * S(vi) + (1 - alpha) * conf * S(v_sym(vi))
    alpha in [0,1]; expected optimum ~0.7 (tuned on validation set).
    """
    return alpha * S + (1.0 - alpha) * conf * S[v_sym]
