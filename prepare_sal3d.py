"""
Prepare SAL3D (Martin et al. 2024, Visual Computer) for our cross-database test.

The SAL3D release ships three things per mesh:
  Meshes/{name}.obj                : full-resolution OBJ
  Gaze/{name}.txt                  : 20,000 point-cloud samples,
                                     cols 0-2 xyz, 3-5 normals, 6 raw gaze, 7 binary
  Smooth Gaze/{name}_neighbors.txt : kNN neighbor lists used by SAL3D's own loader

For our pipeline we want, per mesh:
  sal3d_data/Meshes_ply/{id}.ply       (decimated mesh, ~7,500 vertices)
  sal3d_data/Distributions/{id}.val    (per-vertex continuous saliency in [0,1])
  sal3d_data/Points/{id}.pid           (indices of binary-salient vertices, for AUC)

Mesh IDs are assigned 1001..(1000+N) in alphabetical order of mesh name. A
name <-> id map is written to sal3d_data/name_to_id.json.

Saliency projection: nearest-neighbour from the 20k Gaze samples to each
decimated mesh vertex (k=5, inverse-distance weighted) for continuous; the
binary column is mapped to "this vertex is salient if any of its k-NN gaze
samples are positive".
"""
import os, sys, time, json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC   = os.path.join(_HERE, "SAL3D", "SAL3D_Dataset")
OUT   = os.path.join(_HERE, "sal3d_data")

OUT_MESH  = os.path.join(OUT, "Meshes_ply")
OUT_POINT = os.path.join(OUT, "Points")
OUT_DIST  = os.path.join(OUT, "Distributions")
for d in [OUT_MESH, OUT_POINT, OUT_DIST]:
    os.makedirs(d, exist_ok=True)

TARGET_VERTICES = 7500   # parity with decimated_data/ pipeline (DS-Net protocol)
KNN_K           = 5      # k for kNN projection from Gaze samples to mesh vertices


def load_obj(path):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices,  dtype=np.float64)
    t = np.asarray(m.triangles, dtype=np.int32)
    return v, t


def decimate(verts, tris, target_v):
    if len(verts) <= target_v:
        return verts.astype(np.float64), tris.astype(np.int32)
    target_t = int(2 * target_v)
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(verts.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
    d = m.simplify_quadric_decimation(target_number_of_triangles=target_t)
    d.remove_unreferenced_vertices()
    d.remove_degenerate_triangles()
    return (np.asarray(d.vertices,  dtype=np.float64),
            np.asarray(d.triangles, dtype=np.int32))


def project_continuous(mesh_verts, gaze_pts, gaze_vals, k=KNN_K):
    """kNN inverse-distance projection from Gaze samples to mesh vertices."""
    tree = cKDTree(gaze_pts)
    dists, idx = tree.query(mesh_verts, k=k)
    w = 1.0 / (dists + 1e-8)
    w /= w.sum(axis=1, keepdims=True)
    return (gaze_vals[idx] * w).sum(axis=1)


def project_binary(mesh_verts, gaze_pts, gaze_bin, k=KNN_K):
    """A mesh vertex is binary-salient if any of its k-NN Gaze samples are."""
    tree = cKDTree(gaze_pts)
    _, idx = tree.query(mesh_verts, k=k)
    return (gaze_bin[idx] > 0.5).any(axis=1)


def process(name, mid):
    obj  = os.path.join(SRC, "Meshes",  name + ".obj")
    gaze = os.path.join(SRC, "Gaze",    name + ".txt")
    if not (os.path.exists(obj) and os.path.exists(gaze)):
        return None, "missing OBJ or Gaze"

    out_ply = os.path.join(OUT_MESH,  f"{mid}.ply")
    out_val = os.path.join(OUT_DIST,  f"{mid}.val")
    out_pid = os.path.join(OUT_POINT, f"{mid}.pid")
    if all(os.path.exists(p) for p in (out_ply, out_val, out_pid)):
        return mid, "cached"

    verts, tris = load_obj(obj)
    g           = np.loadtxt(gaze)
    g_xyz       = g[:, :3]
    g_raw       = g[:, 6]
    g_bin       = g[:, 7]

    dec_v, dec_t = decimate(verts, tris, TARGET_VERTICES)

    # Mesh
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(dec_v)
    m.triangles = o3d.utility.Vector3iVector(dec_t)
    o3d.io.write_triangle_mesh(out_ply, m, write_ascii=True)

    # Continuous saliency: project + min-max normalise per mesh
    sal_cont = project_continuous(dec_v, g_xyz, g_raw, k=KNN_K)
    lo, hi = float(sal_cont.min()), float(sal_cont.max())
    if hi - lo > 1e-12:
        sal_cont = (sal_cont - lo) / (hi - lo)
    np.savetxt(out_val, sal_cont, fmt="%.10g")

    # Binary salient vertex indices (for AUC)
    sal_bin = project_binary(dec_v, g_xyz, g_bin, k=KNN_K)
    np.savetxt(out_pid, np.where(sal_bin)[0].astype(np.int32), fmt="%d")

    return mid, f"{len(verts)}->{len(dec_v)} verts, " \
                f"{int(sal_bin.sum())} salient"


def main():
    obj_names  = {f[:-4] for f in os.listdir(os.path.join(SRC, "Meshes"))
                  if f.endswith(".obj")}
    gaze_names = {f[:-4] for f in os.listdir(os.path.join(SRC, "Gaze"))
                  if f.endswith(".txt")}
    names = sorted(obj_names & gaze_names)
    print(f"SAL3D meshes with both OBJ and Gaze: {len(names)}", flush=True)

    # Assign 1001.. as numeric mesh IDs so the existing precompute/training
    # pipeline (which expects {id}.ply) works unchanged.
    name_to_id = {n: 1000 + i + 1 for i, n in enumerate(names)}
    json.dump(name_to_id,
              open(os.path.join(OUT, "name_to_id.json"), "w"), indent=2)

    t0 = time.time()
    summary = []
    for i, name in enumerate(names):
        try:
            mid, info = process(name, name_to_id[name])
            summary.append((mid, name, info))
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{len(names)}] {name} (id {mid}): {info}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {name} FAILED: {e}", flush=True)
            summary.append((None, name, f"FAILED: {e}"))
    print(f"\nFinished in {time.time()-t0:.0f}s "
          f"({sum(1 for s in summary if s[0] is not None)}/{len(names)} ok)",
          flush=True)
    print(f"Outputs:")
    print(f"  meshes : {OUT_MESH}")
    print(f"  dists  : {OUT_DIST}")
    print(f"  points : {OUT_POINT}")
    print(f"  id map : {os.path.join(OUT, 'name_to_id.json')}")


if __name__ == "__main__":
    main()
