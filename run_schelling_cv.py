"""
Schelling K-fold cross-validation: K in {5, 10} on the pooled-vertex and
held-out-mesh protocols. Runs every variant in EXPERIMENTS (9 total) under each
(protocol, K) combination.

Intermediate results are written after every (protocol, K, experiment) finishes
to ``schelling_cv_results.json``. If the script is re-launched, already-finished
cells are skipped, so it doubles as a resume mechanism.

  protocol \ K           5-fold        10-fold
  pooled-vertex     mean PLCC/AUC   mean PLCC/AUC
  held-out mesh     mean PLCC/AUC   mean PLCC/AUC

Per-mesh PLCC/AUC are computed on full meshes for pooled (matching the original
protocol of the conference paper) and on the unseen test meshes only for
held-out (320-360 train / 40-80 test depending on K).

Reuses the exact training regime of train_week2.py / eval_heldout_mesh.py
(same MLP, MSE+Adam, lr 1e-3, batch 4096, 500 epochs, patience 50).
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
from src.metrics import evaluate_dataset
from src.features.symmetry import symmetry_propagation
from train_week2 import load_phi, load_sym_meta, select_columns, EXPERIMENTS

CACHE = os.path.join(_HERE, "cache")
RESULTS_PATH = os.path.join(_HERE, "schelling_cv_results.json")

EPOCHS    = cfg.EPOCHS
LR        = cfg.LR
BS        = cfg.BATCH_SIZE
PATIENCE  = 50
MIN_DELTA = 1e-5


# ---------------------------------------------------------------------------
# Training and inference primitives (shared by both protocols)
# ---------------------------------------------------------------------------
def train_one_fold(X_train, Y_train, in_dim, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    Xg = torch.from_numpy(X_train).to(device)
    Yg = torch.from_numpy(Y_train).to(device)
    n = len(Xg)
    best, bad, best_state, ep_used = float("inf"), 0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        el = 0.0
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            xb, yb = Xg[idx], Yg[idx]
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


def eval_meshes(preds_by_mid, mesh_ids, gts, points):
    S = [preds_by_mid[mid] for mid in mesh_ids]
    G = [gts[mid] for mid in mesh_ids]
    P = [points[mid] for mid in mesh_ids]
    return evaluate_dataset(S, G, P)


# ---------------------------------------------------------------------------
# Protocol-specific CV loops
# ---------------------------------------------------------------------------
def pooled_cv(K, mesh_ids, exp, device, X_per_mesh, gts, points, mesh_sym):
    """K-fold across all pooled vertices; eval per fold over full meshes (paper style)."""
    all_X = np.concatenate([X_per_mesh[m] for m in mesh_ids]).astype(np.float32)
    all_Y = np.concatenate([gts[m] for m in mesh_ids]).astype(np.float32)
    in_dim = all_X.shape[1]
    kf = KFold(n_splits=K, shuffle=True, random_state=cfg.SEED)
    folds = []
    for fi, (tr_idx, _te_idx) in enumerate(kf.split(np.arange(len(all_X)))):
        t0 = time.time()
        model, ep = train_one_fold(all_X[tr_idx], all_Y[tr_idx], in_dim, device,
                                   seed=cfg.SEED + fi)
        preds = predict_meshes(model, X_per_mesh, mesh_ids, device, exp, mesh_sym)
        r = eval_meshes(preds, mesh_ids, gts, points)
        r["fold"] = fi; r["epochs"] = ep; r["time_s"] = time.time() - t0
        r["in_dim"] = in_dim
        folds.append(r)
        print(f"      fold {fi+1}/{K}: PLCC={r['PLCC']:.4f}  AUC={r['AUC']:.4f}  "
              f"({r['time_s']:.0f}s, {ep} ep)", flush=True)
    return folds


def heldout_cv(K, mesh_ids, exp, device, X_per_mesh, gts, points, mesh_sym):
    """K-fold over mesh IDs; train on K-1 folds of meshes, eval on the held-out fold."""
    kf = KFold(n_splits=K, shuffle=True, random_state=cfg.SEED)
    mesh_arr = np.asarray(mesh_ids)
    folds = []
    for fi, (tr_pos, te_pos) in enumerate(kf.split(mesh_arr)):
        train_ids = [int(x) for x in mesh_arr[tr_pos]]
        test_ids  = [int(x) for x in mesh_arr[te_pos]]
        X_tr = np.concatenate([X_per_mesh[m] for m in train_ids]).astype(np.float32)
        Y_tr = np.concatenate([gts[m]        for m in train_ids]).astype(np.float32)
        in_dim = X_tr.shape[1]
        t0 = time.time()
        model, ep = train_one_fold(X_tr, Y_tr, in_dim, device, seed=cfg.SEED + fi)
        preds = predict_meshes(model, X_per_mesh, test_ids, device, exp, mesh_sym)
        r = eval_meshes(preds, test_ids, gts, points)
        r["fold"] = fi; r["epochs"] = ep; r["time_s"] = time.time() - t0
        r["in_dim"] = in_dim
        r["n_train_meshes"] = len(train_ids); r["n_test_meshes"] = len(test_ids)
        folds.append(r)
        print(f"      fold {fi+1}/{K}: PLCC={r['PLCC']:.4f}  AUC={r['AUC']:.4f}  "
              f"({len(train_ids)}t/{len(test_ids)}e, {r['time_s']:.0f}s, {ep} ep)", flush=True)
    return folds


def summarize(folds):
    plccs = [r["PLCC"] for r in folds]
    aucs  = [r["AUC"]  for r in folds]
    return {
        "PLCC_mean": float(np.mean(plccs)),
        "PLCC_std":  float(np.std(plccs, ddof=1)),
        "AUC_mean":  float(np.mean(aucs)),
        "AUC_std":   float(np.std(aucs, ddof=1)),
        "in_dim":    folds[0]["in_dim"],
        "folds":     folds,
    }


# ---------------------------------------------------------------------------
# Driver
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

    Ks         = [5, 10]
    protocols  = ["pooled", "heldout"]
    experiments = EXPERIMENTS

    # Preload features, GTs, points, and sym meta once
    print("Preloading features and meta ...", flush=True)
    X9, X24, gts, points, mesh_sym = {}, {}, {}, {}, {}
    for mid in mesh_ids:
        X9[mid]  = np.load(os.path.join(CACHE, f"{mid}_phi.npy"))
        X24[mid] = np.load(os.path.join(CACHE, f"{mid}_phi24.npy"))
        gts[mid]    = load_saliency(mid, cfg.DIST_DIR).astype(np.float32)
        points[mid] = load_schelling_points(mid, cfg.POINTS_DIR)
        mesh_sym[mid] = load_sym_meta(mid, CACHE)
    print(f"Preloaded {len(mesh_ids)} meshes.\n", flush=True)

    # Resume support
    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}

    for protocol in protocols:
        results.setdefault(protocol, {})
        for K in Ks:
            Kk = str(K)
            results[protocol].setdefault(Kk, {})
            for exp in experiments:
                if exp in results[protocol][Kk]:
                    s = results[protocol][Kk][exp]
                    print(f"=== SKIP {protocol} K={K} {exp:<18} "
                          f"(have PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}) ===",
                          flush=True)
                    continue
                print(f"\n=== {protocol} K={K} :: {exp} ===", flush=True)
                t_start = time.time()

                # Build per-mesh feature views for this experiment
                if exp == "baseline":
                    X_per_mesh = {m: select_columns(X9[m], exp).astype(np.float32)
                                  for m in mesh_ids}
                else:
                    X_per_mesh = {m: select_columns(X24[m], exp).astype(np.float32)
                                  for m in mesh_ids}

                if protocol == "pooled":
                    folds = pooled_cv(K, mesh_ids, exp, device,
                                      X_per_mesh, gts, points, mesh_sym)
                else:
                    folds = heldout_cv(K, mesh_ids, exp, device,
                                       X_per_mesh, gts, points, mesh_sym)

                summary = summarize(folds)
                summary["time_s"] = time.time() - t_start
                results[protocol][Kk][exp] = summary
                save_atomic(results)
                print(f"  -> {exp:<18} dim={summary['in_dim']:>2}  "
                      f"PLCC={summary['PLCC_mean']:.4f}±{summary['PLCC_std']:.4f}  "
                      f"AUC={summary['AUC_mean']:.4f}±{summary['AUC_std']:.4f}  "
                      f"({summary['time_s']:.0f}s)", flush=True)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    for protocol in protocols:
        for K in Ks:
            print(f"\n[{protocol}  K={K}]")
            for exp in experiments:
                if exp not in results[protocol][str(K)]:
                    continue
                s = results[protocol][str(K)][exp]
                print(f"  {exp:<18} dim={s['in_dim']:>2}  "
                      f"PLCC={s['PLCC_mean']:.4f}±{s['PLCC_std']:.4f}  "
                      f"AUC={s['AUC_mean']:.4f}±{s['AUC_std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
