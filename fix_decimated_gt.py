"""
Replace decimated_data/Distributions/*.val with a kNN interpolation of the
ORIGINAL Schelling .val files onto the decimated vertex sets. The previous
Gaussian-from-scratch regeneration in decimate_schelling.py used a different
sigma convention than the dataset's own .val files, so values overflowed [0,1]
and the MLP sigmoid output saturated.

Features and decimated meshes are NOT touched (cache_decimated/ stays valid).
"""
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset import load_mesh, load_saliency

DEC_MESH = os.path.join(_HERE, "decimated_data", "Meshes_ply")
OUT_DIST = os.path.join(_HERE, "decimated_data", "Distributions")
KNN_K    = 5


def interp(orig_verts, orig_sal, dec_verts, k=KNN_K):
    tree = cKDTree(orig_verts)
    dists, idx = tree.query(dec_verts, k=k)
    w = 1.0 / (dists + 1e-8)
    w /= w.sum(axis=1, keepdims=True)
    return (orig_sal[idx] * w).sum(axis=1)


def main():
    ids = sorted(int(f[:-4]) for f in os.listdir(DEC_MESH) if f.endswith(".ply"))
    print(f"Re-generating GT for {len(ids)} decimated meshes "
          f"(kNN interpolation, k={KNN_K})", flush=True)
    t0 = time.time()
    for i, mid in enumerate(ids):
        orig_v, _ = load_mesh(mid, cfg.MESHES_DIR)
        orig_sal  = load_saliency(mid, cfg.DIST_DIR)
        dec_v, _  = load_mesh(mid, DEC_MESH)
        new_sal   = interp(orig_v, orig_sal, dec_v)
        np.savetxt(os.path.join(OUT_DIST, f"{mid}.val"),
                   new_sal, fmt="%.10g")
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ids)}] mesh {mid}: "
                  f"orig max {orig_sal.max():.3f} -> dec max {new_sal.max():.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
