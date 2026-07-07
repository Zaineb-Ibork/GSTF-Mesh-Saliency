"""
Run all four CV protocols on the DECIMATED Schelling dataset (DS-Net protocol,
~7,500 vertices per mesh).

Reuses pooled_cv / heldout_cv from run_schelling_cv.py and per_category_cv /
loco from run_schelling_cv_phase2.py, only swapping the dataset paths and the
feature cache directory.

Output: schelling_decimated_cv_results.json (resumable, written after each cell).
"""
import os, sys, time, json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
# Point cfg at the decimated dataset before importing anything that reads it.
cfg.MESHES_DIR = os.path.join(_HERE, "decimated_data", "Meshes_ply")
cfg.DIST_DIR   = os.path.join(_HERE, "decimated_data", "Distributions")
cfg.POINTS_DIR = os.path.join(_HERE, "decimated_data", "Points")

from src.data.dataset import (load_saliency, load_schelling_points, list_mesh_ids)
from train_week2 import load_sym_meta, select_columns, EXPERIMENTS
import run_schelling_cv as r1
import run_schelling_cv_phase2 as r2

CACHE        = os.path.join(_HERE, "cache_decimated")
RESULTS_PATH = os.path.join(_HERE, "schelling_decimated_cv_results.json")

# Make the imported modules use the decimated cache.
r1.CACHE = CACHE
r2.CACHE = CACHE


def save_atomic(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    os.replace(tmp, RESULTS_PATH)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Device: {device} | {len(mesh_ids)} decimated meshes", flush=True)
    print(f"Cache:  {CACHE}\n", flush=True)

    # Preload phi24 (baseline takes its first 9 columns via select_columns),
    # ground-truth saliency, Schelling points, and symmetry metadata.
    print("Preloading features and meta ...", flush=True)
    X24, gts, points, mesh_sym = {}, {}, {}, {}
    for mid in mesh_ids:
        X24[mid]    = np.load(os.path.join(CACHE, f"{mid}_phi24.npy"))
        gts[mid]    = load_saliency(mid, cfg.DIST_DIR).astype(np.float32)
        points[mid] = load_schelling_points(mid, cfg.POINTS_DIR)
        mesh_sym[mid] = load_sym_meta(mid, CACHE)
    X9 = X24   # baseline experiment reads first 9 columns via select_columns
    print(f"Preloaded {len(mesh_ids)} meshes.\n", flush=True)

    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}

    K = 10  # K=5 vs K=10 was equivalent on the full dataset

    # ------------------------------------------------------------------
    # 1) Pooled-vertex CV
    # ------------------------------------------------------------------
    results.setdefault("pooled_K10", {})
    for exp in EXPERIMENTS:
        if exp in results["pooled_K10"]:
            s = results["pooled_K10"][exp]
            print(f"=== SKIP pooled K=10 {exp:<18} "
                  f"(PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}) ===", flush=True)
            continue
        print(f"\n=== pooled K=10 :: {exp} ===", flush=True)
        t = time.time()
        X_per = {m: select_columns(X24[m], exp).astype(np.float32) for m in mesh_ids}
        folds = r1.pooled_cv(K, mesh_ids, exp, device,
                             X_per, gts, points, mesh_sym)
        s = r1.summarize(folds); s["time_s"] = time.time() - t
        results["pooled_K10"][exp] = s
        save_atomic(results)
        print(f"  -> {exp:<18} PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
              f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}  ({s['time_s']:.0f}s)",
              flush=True)

    # ------------------------------------------------------------------
    # 2) Held-out-mesh CV
    # ------------------------------------------------------------------
    results.setdefault("heldout_K10", {})
    for exp in EXPERIMENTS:
        if exp in results["heldout_K10"]:
            s = results["heldout_K10"][exp]
            print(f"=== SKIP heldout K=10 {exp:<18} "
                  f"(PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}) ===", flush=True)
            continue
        print(f"\n=== heldout K=10 :: {exp} ===", flush=True)
        t = time.time()
        X_per = {m: select_columns(X24[m], exp).astype(np.float32) for m in mesh_ids}
        folds = r1.heldout_cv(K, mesh_ids, exp, device,
                              X_per, gts, points, mesh_sym)
        s = r1.summarize(folds); s["time_s"] = time.time() - t
        results["heldout_K10"][exp] = s
        save_atomic(results)
        print(f"  -> {exp:<18} PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
              f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}  ({s['time_s']:.0f}s)",
              flush=True)

    # ------------------------------------------------------------------
    # 3) Per-category K=10
    # ------------------------------------------------------------------
    results.setdefault("percat_K10", {})
    for exp in EXPERIMENTS:
        if exp in results["percat_K10"]:
            s = results["percat_K10"][exp]
            print(f"=== SKIP percat K=10 {exp:<18} "
                  f"(PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}) ===", flush=True)
            continue
        print(f"\n=== percat K=10 :: {exp} ===", flush=True)
        t = time.time()
        s = r2.per_category_cv(K, exp, device, X9, X24, gts, points, mesh_sym)
        s["time_s"] = time.time() - t
        results["percat_K10"][exp] = s
        save_atomic(results)
        print(f"  -> {exp:<18} PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
              f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}  ({s['time_s']:.0f}s)",
              flush=True)

    # ------------------------------------------------------------------
    # 4) LOCO
    # ------------------------------------------------------------------
    results.setdefault("loco", {})
    for exp in EXPERIMENTS:
        if exp in results["loco"]:
            s = results["loco"][exp]
            print(f"=== SKIP loco {exp:<18} "
                  f"(PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}) ===", flush=True)
            continue
        print(f"\n=== loco :: {exp} ===", flush=True)
        t = time.time()
        s = r2.loco(exp, device, X9, X24, gts, points, mesh_sym)
        s["time_s"] = time.time() - t
        results["loco"][exp] = s
        save_atomic(results)
        print(f"  -> {exp:<18} PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
              f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}  ({s['time_s']:.0f}s)",
              flush=True)

    print("\n" + "=" * 80)
    print("DECIMATED-SCHELLING CV: FINAL SUMMARY")
    for stage in ["pooled_K10", "heldout_K10", "percat_K10", "loco"]:
        print(f"\n[{stage}]")
        for exp in EXPERIMENTS:
            if exp not in results[stage]:
                continue
            s = results[stage][exp]
            print(f"  {exp:<18} dim={s['in_dim']:>2}  "
                  f"PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
                  f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
