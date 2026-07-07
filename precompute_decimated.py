"""Recompute the 24-D enhanced features on the decimated Schelling meshes.
Cache is written to ./cache_decimated/{mid}_phi24.npy + {mid}_sym_meta.pkl.

The script monkey-patches cfg.MESHES_DIR before forking workers, so the
existing precompute_enhanced.compute_one is reused without modification.
"""
import os, sys, time
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
cfg.MESHES_DIR = os.path.join(_HERE, "decimated_data", "Meshes_ply")
cfg.DIST_DIR   = os.path.join(_HERE, "decimated_data", "Distributions")
cfg.POINTS_DIR = os.path.join(_HERE, "decimated_data", "Points")

import precompute_enhanced as pe
from src.data.dataset import list_mesh_ids

CACHE_DIR = os.path.join(_HERE, "cache_decimated")
os.makedirs(CACHE_DIR, exist_ok=True)


def main():
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Decimated meshes: {len(mesh_ids)} | cache: {CACHE_DIR}", flush=True)
    t0 = time.time()
    with mp.Pool(processes=4) as pool:
        args = [(mid, CACHE_DIR) for mid in mesh_ids]
        for i, r in enumerate(pool.imap_unordered(pe._worker, args)):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(mesh_ids)}] {r}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\nFinished {len(mesh_ids)} meshes in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
