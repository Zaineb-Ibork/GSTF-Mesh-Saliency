"""
Extract the three spiral-patch saliency features (Roughness, Geometric
Saliency, Spectral Saliency) at a single decimation scale, then assemble
the 9-dimensional multi-scale feature vector Phi(vi).

Equations follow Lezoray & Nouri, MMSP 2025.

Speed note: the inner loop over (vi, vj) pairs is kept in Python but all
distance arithmetic is delegated to numpy by representing each spiral patch
as a fixed-length normal vector (rings resampled to fixed sizes).  This gives
~30x speedup over the naive per-pair Python implementation.
"""
import numpy as np

from .geometry import (cotangent_laplacian, mean_curvature, vertex_normals,
                        row_normalise, smoothed_curvature)
from .spiral  import (build_adjacency, get_gamma_hop, build_all_patches)


# Fixed ring sizes used for the patch-vector representation (rings 0,1,2).
# Ring 0 always has 1 vertex; rings 1 and 2 are resampled to these sizes.
_RING_SIZES = (1, 6, 12)   # k=2 spiral patch: dim = (1+6+12)*3 = 57


# ---------------------------------------------------------------------------
# Vectorised patch representation
# ---------------------------------------------------------------------------

def _resample_ring(ring_normals: np.ndarray, target: int) -> np.ndarray:
    """Linearly resample a ring of shape (r,3) to (target,3)."""
    r = len(ring_normals)
    if r == target:
        return ring_normals
    x_old = np.linspace(0.0, 1.0, r)
    x_new = np.linspace(0.0, 1.0, target)
    return np.column_stack([
        np.interp(x_new, x_old, ring_normals[:, c]) for c in range(3)
    ])


def build_patch_vectors(patches: list, normals: np.ndarray,
                         ring_sizes: tuple = _RING_SIZES) -> np.ndarray:
    """
    Represent every spiral patch as a fixed-length float32 vector by
    resampling each ring to ring_sizes[l] vertices and flattening.
    Shape: (m, D) where D = sum(ring_sizes) * 3.
    """
    D = sum(ring_sizes) * 3
    m = len(patches)
    vecs = np.zeros((m, D), dtype=np.float32)

    for vi, patch in enumerate(patches):
        offset = 0
        for ring, ts in zip(patch, ring_sizes):
            if ring:
                rn = normals[ring]            # (r, 3)
                rn = _resample_ring(rn, ts)   # (ts, 3)
                vecs[vi, offset:offset + ts * 3] = rn.ravel()
            offset += ts * 3
    return vecs


# ---------------------------------------------------------------------------
# Single-scale feature extraction
# ---------------------------------------------------------------------------

def extract_roughness(kappa: np.ndarray, W_norm, gamma: int) -> np.ndarray:
    """
    R(vi) = |kappa(vi) - (W_norm^gamma @ kappa)(vi)|   eq. (10).
    kappa: mean curvature (m,)
    W_norm: row-stochastic cotangent weight matrix
    """
    smoothed = smoothed_curvature(kappa, W_norm, gamma)
    return np.abs(kappa - smoothed)


def extract_gs_ss(patch_vecs: np.ndarray, adj: list, gamma: int):
    """
    For each vertex vi:
      grad_f(vi) = [||pv[vi] - pv[vj]||  for vj in gamma-hop(vi)]   eq. (11)
      GS(vi) = mean |grad_f|                                         eq. (12)
      SS(vi) = sum  grad_f^2     (= ||J||_F, rank-1 structure tensor) eq. (13)

    Using precomputed patch vectors enables fully numpy inner loops.
    Returns (GS, SS) each of shape (m,).
    """
    m = len(patch_vecs)
    GS = np.zeros(m, dtype=np.float64)
    SS = np.zeros(m, dtype=np.float64)

    # Precompute gamma-hop for all vertices once
    gamma_hops = [get_gamma_hop(adj, vi, gamma) for vi in range(m)]

    for vi in range(m):
        nhop = gamma_hops[vi]
        if not nhop:
            continue
        diff  = patch_vecs[nhop] - patch_vecs[vi]   # (|nhop|, D)
        dists = np.linalg.norm(diff, axis=1)          # (|nhop|,)
        GS[vi] = dists.mean()
        SS[vi] = (dists ** 2).sum()

    return GS, SS


def _safe_norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-12)


def extract_features_single_scale(vertices:  np.ndarray,
                                   triangles: np.ndarray,
                                   gamma:     int,
                                   k:         int) -> np.ndarray:
    """
    Compute [R, GS, SS] for every vertex on a single mesh (one scale).
    Returns normalised feature matrix of shape (m, 3).
    """
    L, W   = cotangent_laplacian(vertices, triangles)
    W_norm = row_normalise(W)
    kappa  = mean_curvature(vertices, triangles, L, W)
    norms  = vertex_normals(vertices, triangles)

    R = extract_roughness(kappa, W_norm, gamma)

    patches, adj = build_all_patches(vertices, triangles, norms, k)
    patch_vecs   = build_patch_vectors(patches, norms)
    GS, SS = extract_gs_ss(patch_vecs, adj, gamma)

    return np.stack([_safe_norm(R), _safe_norm(GS), _safe_norm(SS)], axis=1)


def extract_features_single_scale_enhanced(vertices:  np.ndarray,
                                            triangles: np.ndarray,
                                            gamma:     int,
                                            k:         int):
    """
    Compute baseline [R, GS, SS] PLUS all Week-2/3 contributions.
    Returns a dict with keys:
      'baseline'  — (m, 3)  normalised [R, GS, SS]
      'symf'      — (m,)    normalised SymF                  (Contribution 1)
      'cr'        — (m, 3)  contextual rank in (0,1]         (Contribution 2)
      'pers'      — (m,)    normalised persistence in [0,1]  (Contribution 3)
      'conf_sym'  — float   symmetry confidence for this scale
      'v_sym'     — (m,)    int32 symmetric vertex indices
    """
    from .symmetry    import extract_symf_single_scale
    from .context     import contextual_rank_features
    from .persistence import extract_persistence_single_scale

    L, W   = cotangent_laplacian(vertices, triangles)
    W_norm = row_normalise(W)
    kappa  = mean_curvature(vertices, triangles, L, W)
    norms  = vertex_normals(vertices, triangles)

    R_raw = extract_roughness(kappa, W_norm, gamma)

    patches, adj = build_all_patches(vertices, triangles, norms, k)
    patch_vecs   = build_patch_vectors(patches, norms)
    GS_raw, SS_raw = extract_gs_ss(patch_vecs, adj, gamma)

    # Contextual rank (applied to raw values before normalisation)
    raw_3 = np.stack([R_raw, GS_raw, SS_raw], axis=1)
    cr    = contextual_rank_features(raw_3)

    # Baseline normalised features
    baseline = np.stack([_safe_norm(R_raw), _safe_norm(GS_raw),
                          _safe_norm(SS_raw)], axis=1)

    # Symmetry features
    symf, conf_sym, v_sym = extract_symf_single_scale(vertices, patch_vecs)

    # Topological persistence (Contribution 3) — reuses kappa
    pers = extract_persistence_single_scale(triangles, kappa, smooth_radius=1)

    return {
        "baseline": baseline,
        "symf":     symf,
        "cr":       cr,
        "pers":     pers,
        "conf_sym": conf_sym,
        "v_sym":    v_sym,
    }


# ---------------------------------------------------------------------------
# Multi-scale feature vector Φ(vi)  [9-dimensional]
# ---------------------------------------------------------------------------

def extract_multiscale_features(vertices_orig: np.ndarray,
                                  triangles_orig: np.ndarray,
                                  scale_meshes:   list,
                                  gamma: int,
                                  k:     int) -> np.ndarray:
    """
    scale_meshes: list of (vertices_k, triangles_k) for each decimation level.
      scale_meshes[0] must be the original mesh (ratio 1.0).
      scale_meshes[1] is ~50%, scale_meshes[2] is ~25%.

    Returns Φ of shape (m_orig, 9) = [R^1,GS^1,SS^1, R^2,GS^2,SS^2, R^3,GS^3,SS^3].
    """
    from ..data.decimate import map_features_to_original

    all_feats = []
    for scale_idx, (vk, tk) in enumerate(scale_meshes):
        feats_k = extract_features_single_scale(vk, tk, gamma, k)  # (m_k, 3)

        if scale_idx == 0:
            all_feats.append(feats_k)
        else:
            # Map decimated features back to original resolution
            feats_mapped = map_features_to_original(vertices_orig, vk, feats_k)
            all_feats.append(feats_mapped)

    Phi = np.concatenate(all_feats, axis=1)   # (m_orig, 9)
    return Phi


# ---------------------------------------------------------------------------
# Enhanced multi-scale vector  [24-dimensional]  (Weeks 2 & 3)
# ---------------------------------------------------------------------------

def extract_multiscale_features_enhanced(vertices_orig: np.ndarray,
                                          triangles_orig: np.ndarray,
                                          scale_meshes:   list,
                                          gamma: int,
                                          k:     int) -> np.ndarray:
    """
    Returns Phi_enh of shape (m_orig, 24):
      cols  0- 8: baseline [R,GS,SS] x 3 scales
      cols  9-11: SymF x 3 scales                       (Contribution 1)
      cols 12-20: [CR_R,CR_GS,CR_SS] x 3 scales         (Contribution 2)
      cols 21-23: Persistence x 3 scales                (Contribution 3)

    Also saves per-scale symmetry metadata (conf_sym, v_sym) needed for
    the post-processing pass at inference time, returned as a separate list.
    """
    from ..data.decimate import map_features_to_original

    baseline_parts = []
    symf_parts     = []
    cr_parts       = []
    pers_parts     = []
    sym_meta       = []

    for scale_idx, (vk, tk) in enumerate(scale_meshes):
        out = extract_features_single_scale_enhanced(vk, tk, gamma, k)

        b = out["baseline"]
        s = out["symf"]
        c = out["cr"]
        p = out["pers"]

        if scale_idx == 0:
            baseline_parts.append(b)
            symf_parts.append(s)
            cr_parts.append(c)
            pers_parts.append(p)
            sym_meta.append((out["conf_sym"], out["v_sym"]))
        else:
            baseline_parts.append(
                map_features_to_original(vertices_orig, vk, b))
            symf_parts.append(
                map_features_to_original(vertices_orig, vk, s))
            cr_parts.append(
                map_features_to_original(vertices_orig, vk, c))
            pers_parts.append(
                map_features_to_original(vertices_orig, vk, p))
            from scipy.spatial import cKDTree as _KDT
            tree = _KDT(vertices_orig)
            _, mapped_sym = tree.query(vk[out["v_sym"]], k=1)
            sym_meta.append((out["conf_sym"], mapped_sym.astype(np.int32)))

    # Reshape 1-D parts to (m, 1) before concatenation
    symf_2d = [f[:, None] for f in symf_parts]
    pers_2d = [f[:, None] for f in pers_parts]
    Phi_enh = np.concatenate(
        baseline_parts + symf_2d + cr_parts + pers_2d, axis=1
    )   # (m_orig, 24)

    return Phi_enh, sym_meta
