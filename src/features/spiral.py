"""
Spiral patch construction and comparison for 3D meshes.

References:
  Lézoray & Nouri, MMSP 2025  — eqs (1)–(9)
  Lim et al., ECCV 2019       — original spiral operator [11]

Key definitions
---------------
k-disk(vi)  = all vertices reachable from vi in 0 to k walks
k-ring(vi)  = k-disk(vi) minus (k-1)-disk(vi)  (vertices exactly k hops away)
k-hop(vi)   = k-disk(vi) minus {vi}            (everything but vi itself)
Sp(vi, k)   = (vi, ordered-1-ring, ..., ordered-k-ring)

The first vertex in the 1-ring, R^1_1(vi), is the 1-hop neighbour with
minimum Euclidean distance to vi (a proxy for geodesic distance, eq. 6).
Subsequent vertices are sorted clockwise by angle in vi's tangent plane.
"""
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def build_adjacency(triangles: np.ndarray, n_vertices: int) -> list[set]:
    """Return adjacency as a list of sets: adj[i] = {neighbours of i}."""
    adj = [set() for _ in range(n_vertices)]
    for t in triangles:
        i, j, k = int(t[0]), int(t[1]), int(t[2])
        adj[i].update([j, k])
        adj[j].update([i, k])
        adj[k].update([i, j])
    return adj


# ---------------------------------------------------------------------------
# k-ring / k-hop neighbourhood
# ---------------------------------------------------------------------------

def get_k_rings(adj: list, vi: int, k: int) -> list[list[int]]:
    """
    Return a list of k lists where rings[l] = l-ring vertices (l=1…k).
    Uses BFS.
    """
    visited = {vi}
    current_ring = [vi]
    rings = []
    for _ in range(k):
        next_ring = []
        for u in current_ring:
            for nb in adj[u]:
                if nb not in visited:
                    visited.add(nb)
                    next_ring.append(nb)
        rings.append(next_ring)
        current_ring = next_ring
    return rings


def get_gamma_hop(adj: list, vi: int, gamma: int) -> list[int]:
    """Return all vertices in γ-hop(vi) (within γ hops, excluding vi)."""
    visited = {vi}
    frontier = [vi]
    for _ in range(gamma):
        nxt = []
        for u in frontier:
            for nb in adj[u]:
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
        frontier = nxt
    visited.remove(vi)
    return list(visited)


# ---------------------------------------------------------------------------
# Clockwise ordering in the tangent plane
# ---------------------------------------------------------------------------

def _tangent_basis(normal: np.ndarray):
    """Return two orthonormal tangent vectors (t1, t2) for a given unit normal."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    # Pick an arbitrary vector not parallel to n
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-12
    return t1, t2


def order_ring_clockwise(vi: int, ring: list[int],
                          vertices: np.ndarray,
                          normals:  np.ndarray,
                          start_vertex: int = None) -> list[int]:
    """
    Sort ring vertices clockwise in the tangent plane at vi.
    If start_vertex is given, rotate the sorted list so it begins there.
    Vectorised: all angle computations are done in numpy.
    """
    if not ring:
        return ring

    t1, t2 = _tangent_basis(normals[vi])
    ring_arr = np.array(ring, dtype=np.int32)

    d = vertices[ring_arr] - vertices[vi]   # (r, 3)
    x = d @ t1                               # (r,)
    y = d @ t2                               # (r,)
    angles = np.arctan2(-y, x)              # CW: negate y

    order   = np.argsort(angles)
    ordered = ring_arr[order].tolist()

    if start_vertex is not None:
        try:
            idx = ordered.index(start_vertex)
            ordered = ordered[idx:] + ordered[:idx]
        except ValueError:
            pass

    return ordered


# ---------------------------------------------------------------------------
# Spiral patch
# ---------------------------------------------------------------------------

def build_spiral_patch(vi: int, k: int,
                        adj: list,
                        vertices: np.ndarray,
                        normals:  np.ndarray) -> list[list[int]]:
    """
    Build Sp(vi, k) = [ring_0, ring_1, …, ring_k] where ring_0 = [vi].
    ring_1 is ordered clockwise starting from the closest 1-ring neighbour.
    ring_l starts from the ring-l vertex adjacent to the start of ring_{l-1}.
    Returns a list of k+1 lists (each list = one ring, ordered).
    """
    rings_raw = get_k_rings(adj, vi, k)

    # ring_0
    result = [[vi]]

    if not rings_raw:
        return result

    # ring_1: start from closest 1-hop neighbour (eq. 6)
    r1 = rings_raw[0]
    if not r1:
        result.append([])
        for l in range(1, k):
            result.append(rings_raw[l] if l < len(rings_raw) else [])
        return result

    dists = np.linalg.norm(vertices[r1] - vertices[vi], axis=1)
    start1 = r1[int(np.argmin(dists))]
    r1_ordered = order_ring_clockwise(vi, r1, vertices, normals, start_vertex=start1)
    result.append(r1_ordered)

    # ring_l (l >= 2): start from the ring-l vertex adjacent to start of ring_{l-1}
    for l in range(1, k):
        if l >= len(rings_raw) or not rings_raw[l]:
            result.append([])
            continue
        rl = rings_raw[l]
        prev_start = result[-1][0] if result[-1] else None
        start_l = None
        if prev_start is not None:
            for nb in adj[prev_start]:
                if nb in rl:
                    start_l = nb
                    break
        r_ordered = order_ring_clockwise(vi, rl, vertices, normals, start_vertex=start_l)
        result.append(r_ordered)

    return result


# ---------------------------------------------------------------------------
# Patch distance  (eqs 7–9)
# ---------------------------------------------------------------------------

def _align_indices(n_large: int, n_small: int, idx_large: int) -> int:
    """Map index in the larger ring to corresponding index in the smaller ring."""
    return int(round(idx_large * n_small / n_large)) % n_small


def ring_distance(ring_i: list[int], ring_j: list[int],
                  normals: np.ndarray) -> float:
    """
    d(R^l(vi), R^l(vj)) — eq. (8).
    Compare ring_i and ring_j element-wise using normal-vector L2 distance.
    The larger ring is the reference; elements in the smaller are index-mapped.
    """
    if not ring_i or not ring_j:
        return 0.0

    ni, nj = len(ring_i), len(ring_j)
    if ni >= nj:
        large, small = ring_i, ring_j
    else:
        large, small = ring_j, ring_i
        ni, nj = nj, ni

    total = 0.0
    for n, vl in enumerate(large):
        n_prime = _align_indices(ni, nj, n)
        vs = small[n_prime]
        diff = normals[vl] - normals[vs]
        total += float(np.dot(diff, diff) ** 0.5)   # L2 distance (eq. 9)
    return total


def patch_distance(patch_i: list[list[int]],
                   patch_j: list[list[int]],
                   normals: np.ndarray) -> float:
    """
    d(Sp(vi, k), Sp(vj, k)) = sum_{l=0}^{k} d(R^l(vi), R^l(vj))  — eq. (7).
    """
    total = 0.0
    for ring_i, ring_j in zip(patch_i, patch_j):
        total += ring_distance(ring_i, ring_j, normals)
    return total


# ---------------------------------------------------------------------------
# Precompute all spiral patches for a mesh
# ---------------------------------------------------------------------------

def build_all_patches(vertices: np.ndarray, triangles: np.ndarray,
                      normals: np.ndarray, k: int) -> list:
    """
    Return a list of m spiral patches, one per vertex.
    Each patch is a list of k+1 ring lists.
    """
    m = len(vertices)
    adj = build_adjacency(triangles, m)
    patches = []
    for vi in range(m):
        patches.append(build_spiral_patch(vi, k, adj, vertices, normals))
    return patches, adj
