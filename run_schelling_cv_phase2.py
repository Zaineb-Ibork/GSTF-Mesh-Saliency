"""
Phase 2: per-category K-fold CV (K = 5, 10) + leave-one-category-out (LOCO).
Saves intermediate JSON to ``schelling_cv_phase2_results.json`` after each cell,
so the script supports resume. Reuses the exact training regime of Phase 1.

Per-category K-fold: for each of the 20 categories, K-fold over its 20 meshes;
report mean ± std over categories of the per-category CV PLCC (analogous to the
single-pass per-cat table, with the within-category seed effect removed).

LOCO: K = 20 with category-aligned folds; train on the 19 other categories'
380 meshes, test on the held-out category's 20 meshes.
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset import (load_saliency, load_schelling_points, list_mesh_ids)
from src.model.mlp import SaliencyMLP
from src.metrics import evaluate_dataset, plcc, auc_schelling
from src.features.symmetry import symmetry_propagation
from train_week2 import load_phi, load_sym_meta, select_columns, EXPERIMENTS

CACHE = os.path.join(_HERE, "cache")
RESULTS_PATH = os.path.join(_HERE, "schelling_cv_phase2_results.json")
EPOCHS, LR, BS, PATIENCE, MIN_DELTA = cfg.EPOCHS, cfg.LR, cfg.BATCH_SIZE, 50, 1e-5


def category_ids(c):
    return list(range(20 * c + 1, 20 * c + 21))


def train_one(X_train, Y_train, in_dim, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR); loss_fn = nn.MSELoss()
    Xg = torch.from_numpy(X_train).to(device); Yg = torch.from_numpy(Y_train).to(device)
    n = len(Xg)
    best, bad, best_state, ep_used = float("inf"), 0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(n, device=device); el = 0.0
        for i in range(0, n, BS):
            idx = perm[i:i + BS]; xb, yb = Xg[idx], Yg[idx]
            p = model(xb); l = loss_fn(p, yb)
            opt.zero_grad(); l.backward(); opt.step()
            el = el + l.detach() * len(idx)
        avg = (el / n).item()
        if avg < best - MIN_DELTA:
            best = avg; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        ep_used = ep
        if bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ep_used


def predict_meshes(model, X_per_mesh, mesh_ids, device, exp, mesh_sym):
    model.eval()
    out = {}
    with torch.no_grad():
        for mid in mesh_ids:
            s = model(torch.from_numpy(X_per_mesh[mid]).to(device)).cpu().numpy()
            if exp == "full+postproc":
                conf, v_sym = mesh_sym[mid][0]
                s = symmetry_propagation(s, v_sym, conf, alpha=0.7)
            out[mid] = s
    return out


# ---------------------------------------------------------------------------
# Per-category K-fold
# ---------------------------------------------------------------------------
def per_category_cv(K, exp, device, X9, X24, gts, points, mesh_sym):
    cat_results = []
    in_dim = None
    for c in range(cfg.N_CATEGORIES):
        cat_meshes = category_ids(c)
        # Per-mesh feature views
        X_per_mesh = {m: select_columns(X9[m] if exp == "baseline" else X24[m], exp)
                        .astype(np.float32)
                      for m in cat_meshes}
        kf = KFold(n_splits=K, shuffle=True, random_state=cfg.SEED + c)
        fold_pl, fold_au, fold_eps = [], [], []
        for fi, (tr_pos, te_pos) in enumerate(kf.split(cat_meshes)):
            train_ids = [cat_meshes[i] for i in tr_pos]
            test_ids  = [cat_meshes[i] for i in te_pos]
            X_tr = np.concatenate([X_per_mesh[m] for m in train_ids]).astype(np.float32)
            Y_tr = np.concatenate([gts[m]        for m in train_ids]).astype(np.float32)
            in_dim = X_tr.shape[1]
            model, ep = train_one(X_tr, Y_tr, in_dim, device,
                                  seed=cfg.SEED + c * K + fi)
            preds = predict_meshes(model, X_per_mesh, test_ids, device, exp, mesh_sym)
            pls = [plcc(preds[m], gts[m]) for m in test_ids]
            aus = [auc_schelling(preds[m], points[m], len(preds[m])) for m in test_ids]
            fold_pl.append(float(np.mean(pls)))
            fold_au.append(float(np.mean(aus)))
            fold_eps.append(ep)
        cat_results.append({
            "cat": c,
            "PLCC_mean": float(np.mean(fold_pl)),
            "PLCC_std":  float(np.std(fold_pl, ddof=1)),
            "AUC_mean":  float(np.mean(fold_au)),
            "AUC_std":   float(np.std(fold_au, ddof=1)),
            "fold_PLCC": fold_pl, "fold_AUC": fold_au, "fold_epochs": fold_eps,
        })
        print(f"  cat {c+1:>2}/{cfg.N_CATEGORIES}: "
              f"PLCC={cat_results[-1]['PLCC_mean']:.4f}±{cat_results[-1]['PLCC_std']:.4f}  "
              f"AUC={cat_results[-1]['AUC_mean']:.4f}±{cat_results[-1]['AUC_std']:.4f}",
              flush=True)
    plccs = [r["PLCC_mean"] for r in cat_results]
    aucs  = [r["AUC_mean"]  for r in cat_results]
    return {
        "in_dim":   in_dim,
        "PLCC_mean": float(np.mean(plccs)),
        "PLCC_std":  float(np.std(plccs, ddof=1)),
        "AUC_mean":  float(np.mean(aucs)),
        "AUC_std":   float(np.std(aucs, ddof=1)),
        "per_category": cat_results,
    }


# ---------------------------------------------------------------------------
# Leave-one-category-out
# ---------------------------------------------------------------------------
def loco(exp, device, X9, X24, gts, points, mesh_sym):
    folds = []
    in_dim = None
    for c in range(cfg.N_CATEGORIES):
        train_ids = [m for cc in range(cfg.N_CATEGORIES) if cc != c
                       for m in category_ids(cc)]
        test_ids  = category_ids(c)
        X_per_mesh = {m: select_columns(X9[m] if exp == "baseline" else X24[m], exp)
                        .astype(np.float32)
                      for m in train_ids + test_ids}
        X_tr = np.concatenate([X_per_mesh[m] for m in train_ids]).astype(np.float32)
        Y_tr = np.concatenate([gts[m]        for m in train_ids]).astype(np.float32)
        in_dim = X_tr.shape[1]
        t0 = time.time()
        model, ep = train_one(X_tr, Y_tr, in_dim, device, seed=cfg.SEED + c)
        preds = predict_meshes(model, X_per_mesh, test_ids, device, exp, mesh_sym)
        S = [preds[m] for m in test_ids]
        G = [gts[m]   for m in test_ids]
        P = [points[m] for m in test_ids]
        r = evaluate_dataset(S, G, P)
        r["held_out_cat"] = c; r["epochs"] = ep
        r["time_s"] = time.time() - t0; r["in_dim"] = in_dim
        folds.append(r)
        print(f"  cat {c+1:>2}/{cfg.N_CATEGORIES} (held-out): "
              f"PLCC={r['PLCC']:.4f}  AUC={r['AUC']:.4f}  "
              f"({r['time_s']:.0f}s, {ep} ep)", flush=True)
    plccs = [r["PLCC"] for r in folds]
    aucs  = [r["AUC"]  for r in folds]
    return {
        "in_dim":   in_dim,
        "PLCC_mean": float(np.mean(plccs)),
        "PLCC_std":  float(np.std(plccs, ddof=1)),
        "AUC_mean":  float(np.mean(aucs)),
        "AUC_std":   float(np.std(aucs, ddof=1)),
        "folds":     folds,
    }


# ---------------------------------------------------------------------------
def save_atomic(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    os.replace(tmp, RESULTS_PATH)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Device: {device} | {len(mesh_ids)} meshes\n", flush=True)

    print("Preloading features and meta ...", flush=True)
    X9, X24, gts, points, mesh_sym = {}, {}, {}, {}, {}
    for mid in mesh_ids:
        X9[mid]  = np.load(os.path.join(CACHE, f"{mid}_phi.npy"))
        X24[mid] = np.load(os.path.join(CACHE, f"{mid}_phi24.npy"))
        gts[mid]    = load_saliency(mid, cfg.DIST_DIR).astype(np.float32)
        points[mid] = load_schelling_points(mid, cfg.POINTS_DIR)
        mesh_sym[mid] = load_sym_meta(mid, CACHE)
    print(f"Preloaded {len(mesh_ids)} meshes.\n", flush=True)

    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}

    # 1) Per-category K-fold (K = 5 and K = 10)
    for K in [5, 10]:
        key = f"percat_K{K}"
        results.setdefault(key, {})
        for exp in EXPERIMENTS:
            if exp in results[key]:
                s = results[key][exp]
                print(f"=== SKIP {key} :: {exp} (PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}) ===",
                      flush=True)
                continue
            print(f"\n=== {key} :: {exp} ===", flush=True)
            t = time.time()
            r = per_category_cv(K, exp, device, X9, X24, gts, points, mesh_sym)
            r["time_s"] = time.time() - t
            results[key][exp] = r
            save_atomic(results)
            print(f"  -> {exp:<18} dim={r['in_dim']:>2}  "
                  f"PLCC={r['PLCC_mean']:.4f}±{r['PLCC_std']:.4f}  "
                  f"AUC={r['AUC_mean']:.4f}±{r['AUC_std']:.4f}  "
                  f"({r['time_s']:.0f}s)", flush=True)

    # 2) LOCO
    results.setdefault("loco", {})
    for exp in EXPERIMENTS:
        if exp in results["loco"]:
            s = results["loco"][exp]
            print(f"=== SKIP loco :: {exp} (PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}) ===",
                  flush=True)
            continue
        print(f"\n=== loco :: {exp} ===", flush=True)
        t = time.time()
        r = loco(exp, device, X9, X24, gts, points, mesh_sym)
        r["time_s"] = time.time() - t
        results["loco"][exp] = r
        save_atomic(results)
        print(f"  -> {exp:<18} dim={r['in_dim']:>2}  "
              f"PLCC={r['PLCC_mean']:.4f}±{r['PLCC_std']:.4f}  "
              f"AUC={r['AUC_mean']:.4f}±{r['AUC_std']:.4f}  "
              f"({r['time_s']:.0f}s)", flush=True)

    print("\n" + "=" * 80)
    print("PHASE 2 FINAL SUMMARY")
    for key in ["percat_K5", "percat_K10", "loco"]:
        print(f"\n[{key}]")
        for exp in EXPERIMENTS:
            if exp not in results[key]:
                continue
            s = results[key][exp]
            print(f"  {exp:<18} dim={s['in_dim']:>2}  "
                  f"PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
                  f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
