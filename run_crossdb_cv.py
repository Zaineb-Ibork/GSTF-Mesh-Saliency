"""
Cross-database mesh saliency test, K = 10 folds throughout (consistent with the
Schelling CV in the journal version).

Three groups of cells go into schelling_to_sal3d_results.json:

  CROSS-DB (train on source, predict on target, fold-aggregated on the target)
    A   Schelling full       -> SAL3D full
    B   Schelling decimated  -> SAL3D decimated
    rA  SAL3D full           -> Schelling full
    rB  SAL3D decimated      -> Schelling decimated

  IN-DOMAIN SAL3D (K=10 fold CV over SAL3D meshes, sanity check that the
  features can predict the SAL3D ground truth at all)
    inA  SAL3D full,    held-out-mesh CV  K=10
    inB  SAL3D decim,   held-out-mesh CV  K=10

For the cross-DB cells: a single MLP is trained on the source (all meshes),
predictions made on the target (all meshes), and the K=10 fold split is taken
on the target side only -- so the source is the static training distribution
and the fold std measures across-mesh variance of the cross-DB prediction.

For the in-domain cells: standard K=10 mesh-level CV (320/80-style train/test
rotation).

Output is written cell-by-cell and the script is resumable.
"""
import os, sys, time, json, pickle
import numpy as np
import torch
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset      import list_mesh_ids, load_saliency, load_schelling_points
from src.metrics           import evaluate_dataset
from src.features.symmetry import symmetry_propagation
from train_week2           import select_columns, EXPERIMENTS
from run_schelling_cv      import train_one_fold, heldout_cv, summarize

RESULTS_PATH = os.path.join(_HERE, "schelling_to_sal3d_results.json")
K = 10
SEED = cfg.SEED

DATASETS = {
    "schelling_full": {
        "meshes":  cfg.MESHES_DIR,
        "dists":   cfg.DIST_DIR,
        "points":  cfg.POINTS_DIR,
        "cache":   os.path.join(_HERE, "cache"),
    },
    "schelling_decim": {
        "meshes":  os.path.join(_HERE, "decimated_data", "Meshes_ply"),
        "dists":   os.path.join(_HERE, "decimated_data", "Distributions"),
        "points":  os.path.join(_HERE, "decimated_data", "Points"),
        "cache":   os.path.join(_HERE, "cache_decimated"),
    },
    "sal3d_full": {
        "meshes":  os.path.join(_HERE, "sal3d_data_fullres", "Meshes_ply"),
        "dists":   os.path.join(_HERE, "sal3d_data_fullres", "Distributions"),
        "points":  os.path.join(_HERE, "sal3d_data_fullres", "Points"),
        "cache":   os.path.join(_HERE, "cache_sal3d_fullres"),
    },
    "sal3d_decim": {
        "meshes":  os.path.join(_HERE, "sal3d_data", "Meshes_ply"),
        "dists":   os.path.join(_HERE, "sal3d_data", "Distributions"),
        "points":  os.path.join(_HERE, "sal3d_data", "Points"),
        "cache":   os.path.join(_HERE, "cache_sal3d_decimated"),
    },
}


def preload(name):
    d = DATASETS[name]
    mesh_ids = list_mesh_ids(d["meshes"])
    X24, gts, points, mesh_sym = {}, {}, {}, {}
    for mid in mesh_ids:
        X24[mid]    = np.load(os.path.join(d["cache"], f"{mid}_phi24.npy"))
        gts[mid]    = load_saliency(mid, d["dists"]).astype(np.float32)
        points[mid] = load_schelling_points(mid, d["points"])
        with open(os.path.join(d["cache"], f"{mid}_sym_meta.pkl"), "rb") as f:
            mesh_sym[mid] = pickle.load(f)
    return mesh_ids, X24, gts, points, mesh_sym


def _per_mesh_predict(model, mesh_ids, X24, gts, points, mesh_sym, exp, device):
    """Run the trained MLP over each mesh; return parallel lists of predictions,
    ground-truth maps, and discrete-salient indices (for AUC)."""
    model.eval()
    S, G, P = [], [], []
    with torch.no_grad():
        for mid in mesh_ids:
            X = select_columns(X24[mid], exp).astype(np.float32)
            s = model(torch.from_numpy(X).to(device)).cpu().numpy()
            if exp == "full+postproc":
                conf, v_sym = mesh_sym[mid][0]
                s = symmetry_propagation(s, v_sym, conf, alpha=0.7)
            S.append(s); G.append(gts[mid]); P.append(points[mid])
    return S, G, P


def crossdb_cv(exp, src_ids, src_X, src_g,
               tgt_ids, tgt_X, tgt_g, tgt_p, tgt_s, device, K=K, seed=SEED):
    """Train ONCE on the source, then aggregate per-mesh PLCC/AUC over K=10
    fold splits of the target meshes. The training set never changes; the
    fold std reflects across-target-mesh variance of the cross-DB number."""
    X_tr = np.concatenate([select_columns(src_X[m], exp).astype(np.float32)
                           for m in src_ids]).astype(np.float32)
    Y_tr = np.concatenate([src_g[m] for m in src_ids]).astype(np.float32)
    in_dim = X_tr.shape[1]
    model, ep = train_one_fold(X_tr, Y_tr, in_dim, device, seed=seed)

    # Predict on every target mesh (one inference)
    S, G, P = _per_mesh_predict(model, tgt_ids, tgt_X, tgt_g, tgt_p, tgt_s,
                                exp, device)

    # K-fold aggregation on the target side: each fold = ~|tgt|/K meshes
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    tgt_arr = np.arange(len(tgt_ids))
    fold_records = []
    for fi, (_, te_pos) in enumerate(kf.split(tgt_arr)):
        fS = [S[i] for i in te_pos]
        fG = [G[i] for i in te_pos]
        fP = [P[i] for i in te_pos]
        r = evaluate_dataset(fS, fG, fP)
        r["fold"] = fi; r["n_test_meshes"] = len(te_pos)
        r["in_dim"] = in_dim
        fold_records.append(r)
    summary = summarize(fold_records)
    summary["epochs"] = ep
    return summary


def in_domain_cv(exp, mesh_ids, X24, gts, points, mesh_sym, device, K=K):
    """Standard K-fold mesh-level CV over a single dataset (train K-1 folds,
    eval on the held-out fold). Reuses heldout_cv from run_schelling_cv.py."""
    X_per = {m: select_columns(X24[m], exp).astype(np.float32) for m in mesh_ids}
    folds = heldout_cv(K, mesh_ids, exp, device, X_per, gts, points, mesh_sym)
    return summarize(folds)


def save_atomic(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    os.replace(tmp, RESULTS_PATH)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  K = {K} folds throughout", flush=True)

    pre = {}
    for ds in DATASETS:
        if not (os.path.isdir(DATASETS[ds]["cache"])
                and os.listdir(DATASETS[ds]["cache"])):
            print(f"  ! cache missing for {ds}", flush=True)
            continue
        print(f"Preloading {ds} ...", flush=True)
        pre[ds] = preload(ds)
        print(f"  -> {len(pre[ds][0])} meshes", flush=True)
    print()

    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}

    # ----- cross-DB cells (K=10 fold aggregation on target side) ------------
    cross_pairs = [
        ("A_schelling_full_to_sal3d_full",       "schelling_full",  "sal3d_full"),
        ("B_schelling_decim_to_sal3d_decim",     "schelling_decim", "sal3d_decim"),
        ("rA_sal3d_full_to_schelling_full",      "sal3d_full",      "schelling_full"),
        ("rB_sal3d_decim_to_schelling_decim",    "sal3d_decim",     "schelling_decim"),
    ]
    for tag, src, tgt in cross_pairs:
        if src not in pre or tgt not in pre:
            print(f"=== SKIP {tag} (missing dataset) ===", flush=True)
            continue
        results.setdefault(tag, {})
        src_ids, src_X, src_g, _,       _       = pre[src]
        tgt_ids, tgt_X, tgt_g, tgt_p, tgt_sym   = pre[tgt]
        print(f"\n=== {tag}  ({len(src_ids)} src -> {len(tgt_ids)} tgt, K={K} folds on target) ===",
              flush=True)
        for exp in EXPERIMENTS:
            if exp in results[tag]:
                s = results[tag][exp]
                print(f"  SKIP {exp:<18} PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}",
                      flush=True)
                continue
            t = time.time()
            s = crossdb_cv(exp, src_ids, src_X, src_g,
                           tgt_ids, tgt_X, tgt_g, tgt_p, tgt_sym, device)
            s["time_s"] = time.time() - t
            results[tag][exp] = s
            save_atomic(results)
            print(f"  {exp:<18} dim={s['in_dim']:>2}  "
                  f"PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
                  f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}  "
                  f"({s['time_s']:.0f}s)", flush=True)

    # ----- in-domain SAL3D K=10 CV (sanity check) ---------------------------
    in_domain_pairs = [
        ("inA_sal3d_full_kfold",  "sal3d_full"),
        ("inB_sal3d_decim_kfold", "sal3d_decim"),
    ]
    for tag, ds in in_domain_pairs:
        if ds not in pre:
            print(f"=== SKIP {tag} (missing dataset) ===", flush=True)
            continue
        results.setdefault(tag, {})
        mesh_ids, X24, gts, points, mesh_sym = pre[ds]
        print(f"\n=== {tag}  ({len(mesh_ids)} meshes, K={K} held-out CV) ===",
              flush=True)
        for exp in EXPERIMENTS:
            if exp in results[tag]:
                s = results[tag][exp]
                print(f"  SKIP {exp:<18} PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}",
                      flush=True)
                continue
            t = time.time()
            s = in_domain_cv(exp, mesh_ids, X24, gts, points, mesh_sym, device)
            s["time_s"] = time.time() - t
            results[tag][exp] = s
            save_atomic(results)
            print(f"  {exp:<18} dim={s['in_dim']:>2}  "
                  f"PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
                  f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}  "
                  f"({s['time_s']:.0f}s)", flush=True)

    print("\n" + "=" * 80)
    print("CROSS-DATABASE: FINAL SUMMARY")
    for tag in (t for t, _, _ in cross_pairs) :
        if tag not in results: continue
        print(f"\n[{tag}]")
        for exp in EXPERIMENTS:
            if exp not in results[tag]: continue
            s = results[tag][exp]
            print(f"  {exp:<18}  PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
                  f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}")
    print()
    for tag in (t for t, _ in in_domain_pairs):
        if tag not in results: continue
        print(f"\n[{tag}]")
        for exp in EXPERIMENTS:
            if exp not in results[tag]: continue
            s = results[tag][exp]
            print(f"  {exp:<18}  PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
                  f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
