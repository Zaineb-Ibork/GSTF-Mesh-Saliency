"""
Full-resolution SAL3D preparation. Identical to prepare_sal3d.py except no
quadric decimation is applied to the source OBJ meshes.

Outputs go to sal3d_data_fullres/.
"""
import os, sys, time, json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC   = os.path.join(_HERE, "SAL3D", "SAL3D_Dataset")
OUT   = os.path.join(_HERE, "sal3d_data_fullres")

OUT_MESH  = os.path.join(OUT, "Meshes_ply")
OUT_POINT = os.path.join(OUT, "Points")
OUT_DIST  = os.path.join(OUT, "Distributions")
for d in [OUT_MESH, OUT_POINT, OUT_DIST]:
    os.makedirs(d, exist_ok=True)

KNN_K = 5


def load_obj(path):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices,  dtype=np.float64)
    t = np.asarray(m.triangles, dtype=np.int32)
    return v, t


def project_continuous(mesh_verts, gaze_pts, gaze_vals, k=KNN_K):
    tree = cKDTree(gaze_pts)
    dists, idx = tree.query(mesh_verts, k=k)
    w = 1.0 / (dists + 1e-8)
    w /= w.sum(axis=1, keepdims=True)
    return (gaze_vals[idx] * w).sum(axis=1)


def project_binary(mesh_verts, gaze_pts, gaze_bin, k=KNN_K):
    tree = cKDTree(gaze_pts)
    _, idx = tree.query(mesh_verts, k=k)
    return (gaze_bin[idx] > 0.5).any(axis=1)


def process(name, mid):
    obj  = os.path.join(SRC, "Meshes", name + ".obj")
    gaze = os.path.join(SRC, "Gaze",   name + ".txt")
    if not (os.path.exists(obj) and os.path.exists(gaze)):
        return None, "missing OBJ or Gaze"

    out_ply = os.path.join(OUT_MESH,  f"{mid}.ply")
    out_val = os.path.join(OUT_DIST,  f"{mid}.val")
    out_pid = os.path.join(OUT_POINT, f"{mid}.pid")
    if all(os.path.exists(p) for p in (out_ply, out_val, out_pid)):
        return mid, "cached"

    verts, tris = load_obj(obj)
    g = np.loadtxt(gaze)
    g_xyz, g_raw, g_bin = g[:, :3], g[:, 6], g[:, 7]

    # Write the mesh as PLY (no decimation), so downstream pipeline reads it like
    # any other dataset under cfg.MESHES_DIR
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(tris)
    o3d.io.write_triangle_mesh(out_ply, m, write_ascii=True)

    # Continuous saliency, normalised to [0, 1] per mesh
    sal = project_continuous(verts, g_xyz, g_raw, k=KNN_K)
    lo, hi = float(sal.min()), float(sal.max())
    if hi - lo > 1e-12:
        sal = (sal - lo) / (hi - lo)
    np.savetxt(out_val, sal, fmt="%.10g")

    # Binary salient vertex indices for AUC
    sb = project_binary(verts, g_xyz, g_bin, k=KNN_K)
    np.savetxt(out_pid, np.where(sb)[0].astype(np.int32), fmt="%d")

    return mid, f"{len(verts)} verts, {int(sb.sum())} salient"


def main():
    obj_names  = {f[:-4] for f in os.listdir(os.path.join(SRC, "Meshes"))
                  if f.endswith(".obj")}
    gaze_names = {f[:-4] for f in os.listdir(os.path.join(SRC, "Gaze"))
                  if f.endswith(".txt")}
    names = sorted(obj_names & gaze_names)

    # Reuse the same name->id mapping as the decimated prep so the two variants
    # share IDs (useful when joining downstream)
    map_file = os.path.join(_HERE, "sal3d_data", "name_to_id.json")
    if os.path.exists(map_file):
        name_to_id = {k: int(v) for k, v in json.load(open(map_file)).items()}
    else:
        name_to_id = {n: 1000 + i + 1 for i, n in enumerate(names)}
    json.dump(name_to_id,
              open(os.path.join(OUT, "name_to_id.json"), "w"), indent=2)

    print(f"SAL3D full-resolution: {len(names)} meshes", flush=True)
    t0 = time.time()
    ok = 0
    for i, name in enumerate(names):
        try:
            mid, info = process(name, name_to_id[name])
            if mid is not None:
                ok += 1
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{len(names)}] {name} (id {mid}): {info}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  {name} FAILED: {e}", flush=True)
    print(f"\nFinished in {time.time()-t0:.0f}s ({ok}/{len(names)} ok)")
    print(f"Outputs:")
    print(f"  meshes : {OUT_MESH}")
    print(f"  dists  : {OUT_DIST}")
    print(f"  points : {OUT_POINT}")


if __name__ == "__main__":
    main()
