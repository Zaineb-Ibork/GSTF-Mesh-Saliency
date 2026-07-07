"""
Clean evaluation: held-out MESH split.

The training scripts compute PLCC/AUC via full-mesh inference over ALL vertices,
including the 80% used to fit the model -> the numbers include training-set fit.

This script gives a leakage-free generalisation number: split at the MESH level
(80% train / 20% test meshes), train only on train-mesh vertices, and evaluate
PLCC/AUC on the held-out test meshes (entirely unseen). AUC is well defined here
because each test mesh is scored in full (Schelling positive points are present).

Same training regime as the v3 sweep (config.EPOCHS / patience / batching).

Usage:
  python eval_heldout_mesh.py [--seed 42] [--epochs 500] [--patience 50]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import config as cfg
from src.data.dataset import (load_saliency, load_schelling_points, list_mesh_ids)
from src.model.mlp import SaliencyMLP
from src.metrics import evaluate_dataset
from src.features.symmetry import symmetry_propagation
from train_week2 import load_phi, load_sym_meta, select_columns, EXPERIMENTS


def mesh_split(cache_dir, seed, test_frac=0.20):
    """Random 80/20 split over meshes that have a phi24 cache."""
    ids = [m for m in list_mesh_ids(cfg.MESHES_DIR)
           if os.path.exists(os.path.join(cache_dir, f"{m}_phi24.npy"))]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    n_test = int(round(test_frac * len(perm)))
    test = sorted(int(x) for x in perm[:n_test])
    train = sorted(int(x) for x in perm[n_test:])
    return train, test


def run_experiment(experiment, train_ids, test_ids, cache_dir,
                   epochs, lr, bs, seed, device, patience, min_delta):
    torch.manual_seed(seed); np.random.seed(seed)
    need_enh = experiment != "baseline"

    Xtr, Ytr = [], []
    for mid in train_ids:
        Phi = load_phi(mid, cache_dir, need_enh)
        Xtr.append(select_columns(Phi, experiment).astype(np.float32))
        Ytr.append(load_saliency(mid, cfg.DIST_DIR).astype(np.float32))
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    in_dim = Xtr.shape[1]

    model = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()

    Xg = torch.from_numpy(Xtr).to(device); Yg = torch.from_numpy(Ytr).to(device)
    n = len(Xg)
    t0 = time.time(); best = float("inf"); bad = 0; best_state = None
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=device); el = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb, yb = Xg[idx], Yg[idx]
            p = model(xb); l = loss_fn(p, yb)
            opt.zero_grad(); l.backward(); opt.step()
            el = el + l.detach() * len(idx)
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

    # Evaluate on held-out (unseen) test meshes
    model.eval(); allS, allG, allP = [], [], []
    with torch.no_grad():
        for mid in test_ids:
            Phi = load_phi(mid, cache_dir, need_enh)
            X = select_columns(Phi, experiment).astype(np.float32)
            sm = model(torch.from_numpy(X).to(device)).cpu().numpy()
            if experiment == "full+postproc":
                conf, v_sym = load_sym_meta(mid, cache_dir)[0]
                sm = symmetry_propagation(sm, v_sym, conf, alpha=0.7)
            allS.append(sm)
            allG.append(load_saliency(mid, cfg.DIST_DIR))
            allP.append(load_schelling_points(mid, cfg.POINTS_DIR))

    r = evaluate_dataset(allS, allG, allP)
    r["in_dim"] = in_dim; r["time_s"] = time.time() - t0; r["epochs"] = ep
    return r


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
    train_ids, test_ids = mesh_split(args.cache_dir, args.seed)
    print(f"Device: {device}  |  held-out MESH split: "
          f"{len(train_ids)} train / {len(test_ids)} test meshes  (seed {args.seed})")
    print(f"Regime: epochs={args.epochs}  patience={args.patience}\n")

    rows = []
    for exp in EXPERIMENTS:
        r = run_experiment(exp, train_ids, test_ids, args.cache_dir,
                           args.epochs, args.lr, args.batch_size, args.seed,
                           device, args.patience, args.min_delta)
        rows.append((exp, r))
        print(f"  {exp:<16} dim={r['in_dim']:>2}  PLCC={r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
              f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}  "
              f"({r['time_s']:.0f}s, {r['epochs']} ep)", flush=True)

    print("\n" + "=" * 70)
    print(f"{'Experiment':<16} {'Dim':>4}  {'PLCC (test meshes)':>20}  {'AUC':>16}")
    print("-" * 70)
    for exp, r in rows:
        print(f"{exp:<16} {r['in_dim']:>4}  {r['PLCC']:.4f}±{r['PLCC_std']:.4f}      "
              f"{r['AUC']:.4f}±{r['AUC_std']:.4f}")
    print("=" * 70)
    print("Held-out-mesh = no training vertices in the eval set (clean generalisation).")


if __name__ == "__main__":
    main()
