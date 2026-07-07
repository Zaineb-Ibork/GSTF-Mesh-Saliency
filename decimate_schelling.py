"""
DS-Net protocol: quadric-decimate all 400 Schelling meshes to ~7,500 vertices
(matching Chen et al. 2023 / Lézoray & Nouri 2025) and regenerate the
ground-truth saliency on the decimated meshes.

For each mesh we produce:
  decimated_data/Meshes_ply/{mid}.ply      -- decimated mesh
  decimated_data/Points/{mid}.pid          -- Schelling point indices on dec mesh
  decimated_data/Distributions/{mid}.val   -- Gaussian-smoothed saliency

Schelling points are mapped from the original mesh to the decimated mesh by
nearest-neighbour in 3D space, then a Gaussian map with σ = 0.01·R is
regenerated on the decimated vertex set (matching the conference paper).
"""
import os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset import (load_mesh, load_schelling_points, list_mesh_ids)

OUT_ROOT  = os.path.join(_HERE, "decimated_data")
OUT_MESH  = os.path.join(OUT_ROOT, "Meshes_ply")
OUT_POINT = os.path.join(OUT_ROOT, "Points")
OUT_DIST  = os.path.join(OUT_ROOT, "Distributions")
for d in [OUT_MESH, OUT_POINT, OUT_DIST]:
    os.makedirs(d, exist_ok=True)

TARGET_VERTICES = 7500   # DS-Net protocol


def decimate_one(verts, tris, target_v):
    """Quadric-decimate to approximately target_v vertices via Open3D."""
    if len(verts) <= target_v:
        return verts.astype(np.float64), tris.astype(np.int32)
    # Watertight: V ≈ T/2 + 2; aim for T ≈ 2·target_v
    target_t = int(2 * target_v)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
    dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target_t)
    dec.remove_unreferenced_vertices()
    dec.remove_degenerate_triangles()
    new_v = np.asarray(dec.vertices, dtype=np.float64)
    new_t = np.asarray(dec.triangles, dtype=np.int32)
    return new_v, new_t


def map_points_to_decimated(orig_verts, orig_pid, new_verts):
    """Map Schelling points (3D positions from the original mesh) to the nearest
    vertex on the decimated mesh."""
    pts_3d = orig_verts[orig_pid]
    tree   = cKDTree(new_verts)
    _, idx = tree.query(pts_3d, k=1)
    # de-duplicate (rare: two Schelling points snap to the same dec vertex)
    return np.unique(idx).astype(np.int32)


def regenerate_saliency(verts, schelling_pid, sigma_scale=cfg.GT_SIGMA_SCALE):
    """Per-vertex Gaussian-smoothed saliency on the decimated mesh, matching the
    σ = sigma_scale·R convention of the paper (R = bounding-sphere radius)."""
    R = float(np.linalg.norm(verts - verts.mean(axis=0), axis=1).max())
    sigma = sigma_scale * R
    if len(schelling_pid) == 0:
        return np.zeros(len(verts), dtype=np.float64)
    pts  = verts[schelling_pid]
    diff = verts[:, None, :] - pts[None, :, :]
    d2   = (diff * diff).sum(axis=2)
    sal  = np.exp(-d2 / (2.0 * sigma * sigma)).sum(axis=1)
    return sal


def process_one(mesh_id):
    out_ply = os.path.join(OUT_MESH,  f"{mesh_id}.ply")
    out_pid = os.path.join(OUT_POINT, f"{mesh_id}.pid")
    out_val = os.path.join(OUT_DIST,  f"{mesh_id}.val")
    if all(os.path.exists(p) for p in (out_ply, out_pid, out_val)):
        return mesh_id, "cached"

    verts, tris = load_mesh(mesh_id, cfg.MESHES_DIR)
    orig_pid    = load_schelling_points(mesh_id, cfg.POINTS_DIR)
    new_v, new_t = decimate_one(verts, tris, TARGET_VERTICES)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(new_v)
    mesh.triangles = o3d.utility.Vector3iVector(new_t)
    o3d.io.write_triangle_mesh(out_ply, mesh, write_ascii=True)

    new_pid = map_points_to_decimated(verts, orig_pid, new_v)
    np.savetxt(out_pid, new_pid, fmt="%d")

    new_sal = regenerate_saliency(new_v, new_pid)
    np.savetxt(out_val, new_sal, fmt="%.10g")

    return mesh_id, f"{len(verts)} -> {len(new_v)} verts"


def main():
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Decimating {len(mesh_ids)} meshes to ~{TARGET_VERTICES} vertices ...",
          flush=True)
    t0 = time.time()
    summary = []
    for i, mid in enumerate(mesh_ids):
        try:
            _, info = process_one(mid)
            summary.append((mid, info))
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(mesh_ids)}] mesh {mid}: {info}  "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
        except Exception as e:
            print(f"  mesh {mid} FAILED: {e}", flush=True)
            summary.append((mid, f"FAILED: {e}"))
    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s "
          f"({len(summary)}/{len(mesh_ids)} processed)", flush=True)
    print(f"Outputs:\n  {OUT_MESH}\n  {OUT_POINT}\n  {OUT_DIST}")


if __name__ == "__main__":
    main()
