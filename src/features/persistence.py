"""
Contribution 3 — Topological Persistence Features (Section 3.3 of the proposal).

Sublevel-set persistent homology on the mean-curvature function kappa:V->R.
This is the deepest solution to Failure Mode B: vertices like the navel,
the glass bottom, or an ear centre are local minima of kappa whose magnitude
may be small but whose topological persistence is high relative to the
mesh's noise floor.  None of the differential features (R, GS, SS) capture
this property.

Key definition
--------------
For each vertex v that births a connected component in the sublevel set
filtration kappa_t = {v in V : kappa(v) <= t}, define
    Pers(v) = kappa(v_death) - kappa(v_birth)
where v_death is the saddle (edge) that merges v's component with an older
one.  Vertices that never give birth (i.e. not local minima) get Pers(v) = 0.
The essential class (global minimum) is assigned the full kappa-range so it
is not lost.

Multi-scale vector:
    Pers_full(vi) = [Pers^1(vi), Pers^2(vi), Pers^3(vi)]   (3-D)

Total complexity: O(m * alpha(m)) — essentially linear (alpha = inverse
Ackermann), since the underlying union-find dominates.
"""
import numpy as np
import gudhi


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def mesh_edges(triangles: np.ndarray) -> np.ndarray:
    """
    Return unique undirected edges as an (n_edges, 2) int array, with
    entries sorted so edge[i,0] < edge[i,1].
    """
    e = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]],
    ])
    e = np.sort(e, axis=1)                    # ensure (small, large) order
    e = np.unique(e, axis=0)                  # deduplicate
    return e.astype(np.int32)


# ---------------------------------------------------------------------------
# Persistence computation
# ---------------------------------------------------------------------------

def _persistence_oneside(triangles: np.ndarray,
                          field:     np.ndarray) -> np.ndarray:
    """
    Sublevel-set 0-dim persistent homology of the given scalar field.
    Returns per-vertex persistence array of shape (m,).
    Vertices that are not local minima of the field get 0.
    """
    m         = len(field)
    edges     = mesh_edges(triangles)
    f_arr     = field.astype(np.float64)
    f_rng     = max(float(f_arr.max() - f_arr.min()), 1e-12)

    st = gudhi.SimplexTree()
    for v in range(m):
        st.insert([int(v)], filtration=float(f_arr[v]))
    for (u, v) in edges:
        f = max(float(f_arr[u]), float(f_arr[v]))
        st.insert([int(u), int(v)], filtration=f)

    st.compute_persistence()
    pairs = st.persistence_pairs()

    pers = np.zeros(m, dtype=np.float64)
    for birth_simp, death_simp in pairs:
        if len(birth_simp) != 1:
            continue
        v_birth = int(birth_simp[0])
        if len(death_simp) == 0:
            persistence = f_rng                          # essential class
        else:
            f_death = max(float(f_arr[death_simp[0]]),
                          float(f_arr[death_simp[1]]))
            persistence = f_death - float(f_arr[v_birth])
        if persistence > pers[v_birth]:
            pers[v_birth] = persistence
    return pers


def compute_persistence(triangles: np.ndarray,
                         kappa:    np.ndarray) -> np.ndarray:
    """
    Combined persistence: max(superlevel, sublevel) on kappa.

    - Superlevel set persistence (filtration of -kappa) captures *local maxima*
      of kappa — high-curvature features like ear tips, nose, finger tips.
      These are the majority of Schelling points.
    - Sublevel set persistence (filtration of  kappa) captures *local minima*
      of kappa — flat regions like the navel, glass bottom, ear centres.
      These are the Mode-B failure cases the proposal targets.

    Taking the element-wise max gives every topologically significant vertex
    (max OR min of kappa) a non-zero, normalised-on-kappa-range signal.
    Returns shape (m,), non-negative.
    """
    pers_sub  = _persistence_oneside(triangles,  kappa)   # captures minima
    pers_sup  = _persistence_oneside(triangles, -kappa)   # captures maxima
    return np.maximum(pers_sub, pers_sup)


# ---------------------------------------------------------------------------
# Optional: smooth persistence over a small neighbourhood
# ---------------------------------------------------------------------------

def smooth_persistence(pers: np.ndarray,
                        triangles: np.ndarray,
                        radius_hops: int = 1) -> np.ndarray:
    """
    Spread the (sparse) persistence signal to each critical vertex's
    radius_hops-ring neighbourhood by taking the max over a small window.
    This stabilises the feature when subsequent mapping/decimation jitters
    which exact vertex is identified as the local minimum.
    """
    if radius_hops <= 0:
        return pers

    from .spiral import build_adjacency

    m   = len(pers)
    adj = build_adjacency(triangles, m)
    out = pers.copy()

    for vi in range(m):
        if pers[vi] == 0:
            continue
        # propagate to radius_hops-ring of vi via BFS
        visited = {vi}
        frontier = [vi]
        for _ in range(radius_hops):
            nxt = []
            for u in frontier:
                for nb in adj[u]:
                    if nb not in visited:
                        visited.add(nb)
                        nxt.append(nb)
            frontier = nxt
        for w in visited:
            if pers[vi] > out[w]:
                out[w] = pers[vi]

    return out


# ---------------------------------------------------------------------------
# Full per-scale pipeline (drop-in for extract.py)
# ---------------------------------------------------------------------------

def extract_persistence_single_scale(triangles: np.ndarray,
                                       kappa:    np.ndarray,
                                       smooth_radius: int = 1) -> np.ndarray:
    """
    Compute persistence at one decimation scale, smoothed and normalised
    to [0, 1] per mesh.
    """
    pers = compute_persistence(triangles, kappa)
    if smooth_radius > 0:
        pers = smooth_persistence(pers, triangles, radius_hops=smooth_radius)

    mn, mx = pers.min(), pers.max()
    return (pers - mn) / (mx - mn + 1e-12)
