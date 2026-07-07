"""
Pre-compute and cache the 24-D enhanced feature vectors (Weeks 2 & 3):
  [R,GS,SS]*3 + SymF*3 + CR*9 + Pers*3 = 24

Output per mesh: {cache_dir}/{mesh_id}_phi24.npy   shape (m, 24)
Also saves symmetry metadata: {cache_dir}/{mesh_id}_sym_meta.pkl

Usage:
  python precompute_enhanced.py [--workers 4] [--cache-dir ./cache]
"""
import argparse
import os
import sys
import time
import pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset  import load_mesh, list_mesh_ids
from src.data.decimate import decimate_mesh
from src.features.extract import extract_multiscale_features_enhanced


def _worker(args):
    mesh_id, cache_dir = args
    return compute_one(mesh_id, cache_dir)


def compute_one(mesh_id: int, cache_dir: str) -> str:
    cp_feat = os.path.join(cache_dir, f"{mesh_id}_phi24.npy")
    cp_sym  = os.path.join(cache_dir, f"{mesh_id}_sym_meta.pkl")
    if os.path.exists(cp_feat) and os.path.exists(cp_sym):
        return f"mesh {mesh_id}: cached"

    t0 = time.time()
    try:
        verts, tris = load_mesh(mesh_id, cfg.MESHES_DIR)
        scale_meshes = [(verts, tris)]
        for ratio in cfg.DECIM_RATIOS[1:]:
            dv, dt = decimate_mesh(verts, tris, ratio)
            scale_meshes.append((dv, dt))

        Phi_enh, sym_meta = extract_multiscale_features_enhanced(
            verts, tris, scale_meshes, gamma=cfg.GAMMA, k=cfg.K_RINGS
        )

        np.save(cp_feat, Phi_enh)
        with open(cp_sym, "wb") as f:
            pickle.dump(sym_meta, f)

        return f"mesh {mesh_id}: {time.time()-t0:.1f}s  shape={Phi_enh.shape}"
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
            if not os.path.exists(os.path.join(args.cache_dir,
                                                f"{mid}_phi24.npy"))]
    print(f"{len(mesh_ids)-len(todo)}/{len(mesh_ids)} already cached. "
          f"{len(todo)} to compute.")

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
                eta = (elapsed / done) * (len(todo) - done)
                print(f"  [{done}/{len(todo)}] {f.result()}"
                      f"  (ETA {eta/60:.1f}min)", flush=True)
    else:
        for i, mid in enumerate(todo, 1):
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (len(todo) - i) if i > 1 else 0
            msg = compute_one(mid, args.cache_dir)
            print(f"  [{i}/{len(todo)}] {msg}  (ETA {eta/60:.1f}min)",
                  flush=True)

    print(f"\nDone. Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
