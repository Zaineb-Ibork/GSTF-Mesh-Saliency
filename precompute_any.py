"""Run precompute_enhanced.compute_one on an arbitrary mesh directory.

Reuses the existing 24-D feature pipeline; only swaps cfg.MESHES_DIR (before
fork) and the output cache directory.

Usage:
  python precompute_any.py --mesh-dir <dir> --cache-dir <dir> [--workers 4]
"""
import argparse, os, sys, time
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--mesh-dir",  required=True)
ap.add_argument("--cache-dir", required=True)
ap.add_argument("--workers",   type=int, default=4)
args = ap.parse_args()

import config as cfg
cfg.MESHES_DIR = os.path.abspath(args.mesh_dir)

import precompute_enhanced as pe
from src.data.dataset import list_mesh_ids

CACHE = os.path.abspath(args.cache_dir)
os.makedirs(CACHE, exist_ok=True)


def main():
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Meshes ({len(mesh_ids)}): {cfg.MESHES_DIR}", flush=True)
    print(f"Cache: {CACHE}  Workers: {args.workers}\n", flush=True)
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        work = [(mid, CACHE) for mid in mesh_ids]
        for i, r in enumerate(pool.imap_unordered(pe._worker, work)):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{len(mesh_ids)}] {r}  ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"\nFinished {len(mesh_ids)} meshes in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
