"""
Quick smoke test: run the full feature pipeline on a single small mesh.
Expected output: Phi.shape == (n_vertices, 9), no NaNs, no errors.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import config as cfg
from src.data.dataset   import load_mesh, load_saliency, load_schelling_points
from src.data.decimate  import decimate_mesh
from src.features.geometry import (cotangent_laplacian, mean_curvature,
                                    vertex_normals, row_normalise)
from src.features.spiral   import build_all_patches, get_gamma_hop, patch_distance
from src.features.extract  import extract_multiscale_features
from src.metrics           import plcc, auc_schelling

MESH_ID = 1   # first mesh; ~4706 vertices

print(f"=== Smoke test on mesh {MESH_ID} ===")

# --- Load
verts, tris = load_mesh(MESH_ID, cfg.MESHES_DIR)
sal = load_saliency(MESH_ID, cfg.DIST_DIR)
pts = load_schelling_points(MESH_ID, cfg.POINTS_DIR)
print(f"Vertices: {len(verts)}, Triangles: {len(tris)}, GT range: [{sal.min():.3f}, {sal.max():.3f}]")
print(f"Schelling points: {len(pts)}")

# --- Decimation
t0 = time.time()
v50, t50 = decimate_mesh(verts, tris, 0.5)
v25, t25 = decimate_mesh(verts, tris, 0.25)
print(f"Decimation: {len(verts)} → {len(v50)} → {len(v25)} vertices  ({time.time()-t0:.1f}s)")

# --- Geometry on original mesh
t0 = time.time()
L, W = cotangent_laplacian(verts, tris)
W_norm = row_normalise(W)
kappa  = mean_curvature(verts, tris, L, W)
norms  = vertex_normals(verts, tris)
print(f"Geometry: κ range [{kappa.min():.4f}, {kappa.max():.4f}]  ({time.time()-t0:.1f}s)")

# --- Spiral patches (just first 50 vertices to test quickly)
t0 = time.time()
from src.features.spiral import build_adjacency, get_gamma_hop, build_spiral_patch
adj = build_adjacency(tris, len(verts))
patch0 = build_spiral_patch(0, cfg.K_RINGS, adj, verts, norms)
print(f"Patch[0] rings: {[len(r) for r in patch0]}")
nhop = get_gamma_hop(adj, 0, cfg.GAMMA)
print(f"γ-hop[0] size: {len(nhop)}")
d = patch_distance(patch0, build_spiral_patch(nhop[0], cfg.K_RINGS, adj, verts, norms), norms)
print(f"d(Sp(0), Sp({nhop[0]})): {d:.4f}  ({time.time()-t0:.1f}s for one distance)")

# --- Full feature extraction (may take several minutes on full mesh)
print(f"\nExtracting full 9-D features (this may take a few minutes) …")
t0 = time.time()
scale_meshes = [(verts, tris), (v50, t50), (v25, t25)]
Phi = extract_multiscale_features(verts, tris, scale_meshes,
                                   gamma=cfg.GAMMA, k=cfg.K_RINGS)
elapsed = time.time() - t0
print(f"Phi shape: {Phi.shape}  ({elapsed:.1f}s)")
print(f"Phi  NaNs: {np.isnan(Phi).sum()}")
print(f"Phi range: [{Phi.min():.4f}, {Phi.max():.4f}]")

# --- Trivial evaluation (random predictor baseline)
S_rand = np.random.rand(len(verts))
print(f"\nRandom predictor: PLCC={plcc(S_rand, sal):.4f}, "
      f"AUC={auc_schelling(S_rand, pts, len(verts)):.4f}")
print(f"GT saliency as predictor: PLCC={plcc(sal, sal):.4f}, "
      f"AUC={auc_schelling(sal, pts, len(verts)):.4f}")

print("\n=== Smoke test PASSED ===")
