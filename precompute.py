"""
Standalone feature pre-computation script.
Computes and caches Phi (9-D feature vectors) for all Schelling meshes.

Usage:
  python precompute.py [--workers 4] [--cache-dir ./cache]

With 8 cores, expect ~35 minutes for all 400 meshes.
"""
import argparse
import os
import sys
import time
import numpy as np

# Ensure imports work from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset  import load_mesh, list_mesh_ids
from src.data.decimate import decimate_mesh
from src.features.extract import extract_multiscale_features


def _worker(args):
    """Top-level function (required for ProcessPoolExecutor pickling)."""
    mesh_id, cache_dir = args
    return compute_one(mesh_id, cache_dir)


def compute_one(mesh_id: int, cache_dir: str) -> str:
    cp = os.path.join(cache_dir, f"{mesh_id}_phi.npy")
    if os.path.exists(cp):
        return f"mesh {mesh_id}: cached"

    t0 = time.time()
    try:
        verts, tris = load_mesh(mesh_id, cfg.MESHES_DIR)
        scale_meshes = [(verts, tris)]
        for ratio in cfg.DECIM_RATIOS[1:]:
            dv, dt = decimate_mesh(verts, tris, ratio)
            scale_meshes.append((dv, dt))
        Phi = extract_multiscale_features(verts, tris, scale_meshes,
                                           gamma=cfg.GAMMA, k=cfg.K_RINGS)
        np.save(cp, Phi)
        return f"mesh {mesh_id}: {time.time()-t0:.1f}s  shape={Phi.shape}"
    except Exception as e:
        return f"mesh {mesh_id}: ERROR — {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",   type=int, default=1)
    parser.add_argument("--cache-dir", default="./cache")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)

    todo = [mid for mid in mesh_ids
            if not os.path.exists(os.path.join(args.cache_dir, f"{mid}_phi.npy"))]
    already = len(mesh_ids) - len(todo)
    print(f"{already}/{len(mesh_ids)} meshes already cached. {len(todo)} to compute.")

    if not todo:
        print("Nothing to do.")
        return

    t_start = time.time()

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        work = [(mid, args.cache_dir) for mid in todo]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, w): w[0] for w in work}
            for f in as_completed(futures):
                done += 1
                elapsed = time.time() - t_start
                remaining = (elapsed / done) * (len(todo) - done)
                print(f"  [{done}/{len(todo)}] {f.result()}"
                      f"  (ETA {remaining/60:.1f}min)", flush=True)
    else:
        for i, mid in enumerate(todo, 1):
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (len(todo) - i) if i > 1 else 0
            msg = compute_one(mid, args.cache_dir)
            print(f"  [{i}/{len(todo)}] {msg}  (ETA {eta/60:.1f}min)", flush=True)

    total = time.time() - t_start
    print(f"\nDone. Total time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
