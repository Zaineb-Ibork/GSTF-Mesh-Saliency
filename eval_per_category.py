"""
Per-category evaluation (paper Table II style) on the FULL-mesh protocol.

Same protocol as the v3 sweep: vertex-level 80/20 train split, train on the 80%,
then full-mesh inference over ALL meshes. Difference: this reports BOTH
  - per-mesh-averaged PLCC/AUC   (mean over the 400 meshes — what train_week2 does)
  - per-category-averaged PLCC/AUC (PLCC per category, then mean over 20 categories
                                    — directly comparable to the paper's 0.6062)

Categories: Schelling IDs 1-400 in contiguous blocks of 20 -> cat = (id-1)//20.

Usage:
  python eval_per_category.py [--seed 42] [--epochs 500] [--patience 50]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import config as cfg
from src.data.dataset import (load_saliency, load_schelling_points,
                              list_mesh_ids, mesh_id_to_category)
from src.model.mlp import SaliencyMLP
from src.metrics import plcc, auc_schelling
from src.features.symmetry import symmetry_propagation
from train_week2 import load_phi, load_sym_meta, select_columns, EXPERIMENTS


def aggregate(per_mesh, mesh_ids):
    """per_mesh: list aligned with mesh_ids. Returns (per_mesh_mean, per_cat_mean)."""
    per_mesh = np.asarray(per_mesh)
    cats = np.array([mesh_id_to_category(m) for m in mesh_ids])
    cat_means = [per_mesh[cats == c].mean() for c in range(cfg.N_CATEGORIES)
                 if np.any(cats == c)]
    return float(per_mesh.mean()), float(np.mean(cat_means))


def run_experiment(experiment, mesh_ids, cache_dir, epochs, lr, bs,
                   seed, device, patience, min_delta):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    need_enh = experiment != "baseline"

    all_X, all_Y = [], []
    mesh_slices, mesh_gt, mesh_pts, mesh_sym = [], {}, {}, {}
    total = 0
    for mid in mesh_ids:
        Phi = load_phi(mid, cache_dir, need_enh)
        X = select_columns(Phi, experiment).astype(np.float32)
        sal = load_saliency(mid, cfg.DIST_DIR)
        all_X.append(X); all_Y.append(sal)
        mesh_slices.append((total, total + len(X), mid))
        mesh_gt[mid] = sal
        mesh_pts[mid] = load_schelling_points(mid, cfg.POINTS_DIR)
        if need_enh:
            mesh_sym[mid] = load_sym_meta(mid, cache_dir)
        total += len(X)

    X_all = np.concatenate(all_X).astype(np.float32)
    Y_all = np.concatenate(all_Y).astype(np.float32)

    idx = rng.permutation(len(X_all))
    n_train = int(cfg.TRAIN_SPLIT * len(X_all))
    train_idx = idx[:n_train]
    in_dim = X_all.shape[1]

    model = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()

    Xg = torch.from_numpy(X_all[train_idx]).to(device)
    Yg = torch.from_numpy(Y_all[train_idx]).to(device)
    n = len(Xg)
    t0 = time.time(); best = float("inf"); bad = 0; best_state = None
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); el = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]; xb, yb = Xg[j], Yg[j]
            p = model(xb); l = loss_fn(p, yb)
            opt.zero_grad(); l.backward(); opt.step()
            el = el + l.detach() * len(j)
        avg = (el / n).item()
        if avg < best - min_delta:
            best = avg; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Full-mesh inference + per-mesh metrics
    model.eval()
    ids_order, plcc_m, auc_m = [], [], []
    with torch.no_grad():
        for start, end, mid in mesh_slices:
            sm = model(torch.from_numpy(X_all[start:end]).to(device)).cpu().numpy()
            if experiment == "full+postproc":
                conf, v_sym = mesh_sym[mid][0]
                sm = symmetry_propagation(sm, v_sym, conf, alpha=0.7)
            ids_order.append(mid)
            plcc_m.append(plcc(sm, mesh_gt[mid]))
            auc_m.append(auc_schelling(sm, mesh_pts[mid], len(sm)))

    pm_plcc, pc_plcc = aggregate(plcc_m, ids_order)
    pm_auc,  pc_auc  = aggregate(auc_m,  ids_order)
    return {"in_dim": in_dim, "time_s": time.time() - t0, "epochs": ep,
            "per_mesh_PLCC": pm_plcc, "per_cat_PLCC": pc_plcc,
            "per_mesh_AUC": pm_auc,   "per_cat_AUC": pc_auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Device: {device}  |  {len(mesh_ids)} meshes, full-mesh protocol "
          f"(vertex-level {int(cfg.TRAIN_SPLIT*100)}/{int((1-cfg.TRAIN_SPLIT)*100)} train split)")
    print(f"Regime: epochs={args.epochs}  patience={args.patience}\n")

    rows = []
    for exp in EXPERIMENTS:
        r = run_experiment(exp, mesh_ids, args.cache_dir, args.epochs, args.lr,
                           args.batch_size, args.seed, device,
                           args.patience, args.min_delta)
        rows.append((exp, r))
        print(f"  {exp:<16} dim={r['in_dim']:>2}  "
              f"PLCC[mesh={r['per_mesh_PLCC']:.4f} cat={r['per_cat_PLCC']:.4f}]  "
              f"AUC[mesh={r['per_mesh_AUC']:.4f} cat={r['per_cat_AUC']:.4f}]  "
              f"({r['time_s']:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print(f"{'Experiment':<16} {'Dim':>3}  {'PLCC/mesh':>9} {'PLCC/cat':>9}  "
          f"{'AUC/mesh':>9} {'AUC/cat':>9}")
    print("-" * 78)
    for exp, r in rows:
        print(f"{exp:<16} {r['in_dim']:>3}  {r['per_mesh_PLCC']:>9.4f} {r['per_cat_PLCC']:>9.4f}  "
              f"{r['per_mesh_AUC']:>9.4f} {r['per_cat_AUC']:>9.4f}")
    print("=" * 78)
    print("Paper: per-mesh PLCC=0.5567/AUC=0.8196 (Table I) ; per-category PLCC=0.6062 (Table II)")


if __name__ == "__main__":
    main()
