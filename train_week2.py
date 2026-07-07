"""
Week 2 + 3 training: baseline + SymF + CR + Persistence = 24-D.

Ablation experiments run:
  - Baseline only           (9-D)
  - + SymF only             (12-D)
  - + CR only               (18-D)
  - + Pers only             (12-D)
  - + SymF + CR             (21-D)
  - + SymF + CR + Pers      (24-D)
  - full + post-proc        (24-D + symmetry propagation on output)

Usage:
  python train_week2.py [--experiment all] [--seed 42] [--epochs 200]
"""
import argparse
import os
import sys
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset  import (load_mesh, load_saliency, load_schelling_points,
                                 list_mesh_ids)
from src.model.mlp     import SaliencyMLP
from src.metrics       import evaluate_dataset
from src.features.symmetry import symmetry_propagation


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_phi(mesh_id: int, cache_dir: str, enhanced: bool) -> np.ndarray:
    if enhanced:
        return np.load(os.path.join(cache_dir, f"{mesh_id}_phi24.npy"))
    return np.load(os.path.join(cache_dir, f"{mesh_id}_phi.npy"))


def load_sym_meta(mesh_id: int, cache_dir: str):
    path = os.path.join(cache_dir, f"{mesh_id}_sym_meta.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)   # list of (conf_sym, v_sym) per scale


def select_columns(Phi: np.ndarray, experiment: str) -> np.ndarray:
    """
    Phi has 24 columns:
      0- 8: baseline [R,GS,SS] x 3 scales
      9-11: SymF x 3 scales
      12-20: CR x 3 scales (each scale: CR_R, CR_GS, CR_SS)
      21-23: Persistence x 3 scales
    """
    if experiment == "baseline":
        return Phi[:, :9]
    elif experiment == "symf":
        return np.concatenate([Phi[:, :9], Phi[:, 9:12]], axis=1)        # 12-D
    elif experiment == "cr":
        return np.concatenate([Phi[:, :9], Phi[:, 12:21]], axis=1)        # 18-D
    elif experiment == "pers":
        return np.concatenate([Phi[:, :9], Phi[:, 21:24]], axis=1)        # 12-D
    elif experiment == "symf+cr":
        return np.concatenate([Phi[:, :9], Phi[:, 9:21]], axis=1)         # 21-D
    elif experiment == "cr+pers":
        return np.concatenate([Phi[:, :9], Phi[:, 12:24]], axis=1)        # 21-D
    elif experiment == "symf+pers":
        return np.concatenate([Phi[:, :9], Phi[:, 9:12], Phi[:, 21:24]],
                              axis=1)                                      # 15-D
    elif experiment in ("symf+cr+pers", "full+postproc"):
        return Phi   # all 24-D
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def run_experiment(experiment: str, mesh_ids: list, cache_dir: str,
                   epochs: int, lr: float, batch_size: int,
                   seed: int, device: torch.device,
                   patience: int = 50,
                   min_delta: float = 1e-5) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    need_enhanced = experiment != "baseline"

    # Load all data
    all_X, all_Y = [], []
    mesh_slices, mesh_gt, mesh_pts, mesh_sym = [], {}, {}, {}
    total = 0

    for mesh_id in mesh_ids:
        Phi_full = load_phi(mesh_id, cache_dir, enhanced=need_enhanced)
        X        = select_columns(Phi_full, experiment).astype(np.float32)
        sal      = load_saliency(mesh_id, cfg.DIST_DIR)
        pts      = load_schelling_points(mesh_id, cfg.POINTS_DIR)

        m = len(X)
        all_X.append(X)
        all_Y.append(sal)
        mesh_slices.append((total, total + m, mesh_id))
        mesh_gt[mesh_id]  = sal
        mesh_pts[mesh_id] = pts
        if need_enhanced:
            mesh_sym[mesh_id] = load_sym_meta(mesh_id, cache_dir)
        total += m

    X_all = np.concatenate(all_X, axis=0).astype(np.float32)
    Y_all = np.concatenate(all_Y, axis=0).astype(np.float32)

    # 80/20 vertex-level split
    idx     = rng.permutation(len(X_all))
    n_train = int(cfg.TRAIN_SPLIT * len(X_all))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    in_dim = X_all.shape[1]
    model  = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Move the full training set to the GPU once (no per-batch host->device copies)
    X_tr_t = torch.from_numpy(X_all[train_idx]).to(device)
    Y_tr_t = torch.from_numpy(Y_all[train_idx]).to(device)
    n_tr   = len(X_tr_t)

    t0 = time.time()
    best_loss   = float("inf")
    epochs_bad  = 0
    best_state  = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        epoch_loss = 0.0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr_t[idx], Y_tr_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss = epoch_loss + loss.detach() * len(idx)   # stay on GPU

        avg_loss = (epoch_loss / n_tr).item()   # single host sync per epoch
        # Patience check: save best only on a > min_delta improvement,
        # which also resets the early-stop counter.
        if avg_loss < best_loss - min_delta:
            best_loss  = avg_loss
            epochs_bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_bad += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch}/{epochs}  loss={avg_loss:.6f}  "
                  f"best={best_loss:.6f}  bad={epochs_bad}/{patience}",
                  flush=True)

        if epochs_bad >= patience:
            print(f"    early-stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Full-mesh inference
    model.eval()
    all_S, all_G, all_P = [], [], []

    with torch.no_grad():
        for start, end, mesh_id in mesh_slices:
            xm = torch.from_numpy(X_all[start:end]).to(device)
            sm = model(xm).cpu().numpy()

            # Optional symmetry propagation post-processing
            if experiment == "full+postproc":
                conf, v_sym = mesh_sym[mesh_id][0]
                sm = symmetry_propagation(sm, v_sym, conf, alpha=0.7)

            all_S.append(sm)
            all_G.append(mesh_gt[mesh_id])
            all_P.append(mesh_pts[mesh_id])

    results = evaluate_dataset(all_S, all_G, all_P)
    results["time_s"] = time.time() - t0
    results["in_dim"] = in_dim
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = ["baseline", "symf", "cr", "pers",
               "symf+cr", "cr+pers", "symf+pers",
               "symf+cr+pers", "full+postproc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=EXPERIMENTS + ["all"])
    parser.add_argument("--cache-dir",  default="./cache")
    parser.add_argument("--epochs",     type=int, default=cfg.EPOCHS)
    parser.add_argument("--seed",       type=int, default=cfg.SEED)
    parser.add_argument("--lr",         type=float, default=cfg.LR)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--patience",   type=int,   default=50)
    parser.add_argument("--min-delta",  type=float, default=1e-5)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)

    exps = EXPERIMENTS if args.experiment == "all" else [args.experiment]

    # Check required cache files exist
    sample = mesh_ids[0]
    if not os.path.exists(os.path.join(args.cache_dir, f"{sample}_phi.npy")):
        print("ERROR: baseline cache missing. Run precompute.py first.")
        sys.exit(1)
    if any(e != "baseline" for e in exps):
        if not os.path.exists(os.path.join(args.cache_dir,
                                            f"{sample}_phi24.npy")):
            print("ERROR: enhanced cache missing. Run precompute_enhanced.py first.")
            sys.exit(1)

    print(f"Device: {device}  |  Meshes: {len(mesh_ids)}")
    print(f"Experiments: {exps}\n")

    results_table = []
    for exp in exps:
        print(f"--- {exp} ---")
        r = run_experiment(exp, mesh_ids, args.cache_dir,
                           args.epochs, args.lr, args.batch_size,
                           args.seed, device,
                           patience=args.patience, min_delta=args.min_delta)
        results_table.append((exp, r))
        print(f"  PLCC={r['PLCC']:.4f} ±{r['PLCC_std']:.4f}  "
              f"AUC={r['AUC']:.4f} ±{r['AUC_std']:.4f}  "
              f"({r['time_s']:.0f}s)\n")

    # Summary table
    print("=" * 70)
    print(f"{'Experiment':<22} {'Dim':>4}  {'PLCC':>8}  {'AUC':>8}")
    print("-" * 70)
    for exp, r in results_table:
        print(f"{exp:<22} {r['in_dim']:>4}  "
              f"{r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
              f"{r['AUC']:.4f}±{r['AUC_std']:.4f}")
    print("=" * 70)
    print("Baseline target (paper): PLCC=0.5567  AUC=0.8196")


if __name__ == "__main__":
    main()
