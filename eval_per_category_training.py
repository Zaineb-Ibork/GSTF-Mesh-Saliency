"""
Per-category TRAINING evaluation (the MF-SAE / paper Table II protocol).

For each of the 20 Schelling categories we train a SEPARATE model:
  - category c -> mesh IDs [20c+1 .. 20c+20]
  - random (seeded) mesh-level split, default 80/20 -> 16 train, 4 test meshes
  - train a fresh model on the train meshes' vertices
  - full-mesh inference on the held-out test meshes -> per-mesh PLCC/AUC
  - category score = mean over its test meshes
Final number = mean (± std) over the 20 categories.

This is both per-category (specialised models) and leakage-free (test meshes are
never in training), so it is directly comparable to the paper's per-category 0.6062.

Usage:
  python eval_per_category_training.py [--seed 42] [--epochs 500] [--patience 50]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import config as cfg
from src.data.dataset import (load_saliency, load_schelling_points, list_mesh_ids)
from src.model.mlp import SaliencyMLP
from src.metrics import plcc, auc_schelling
from src.features.symmetry import symmetry_propagation
from train_week2 import select_columns, load_sym_meta, EXPERIMENTS


def category_ids(c):
    """Category c (0-based) -> its 20 mesh IDs."""
    return list(range(20 * c + 1, 20 * c + 21))


def train_eval_category(train_ids, test_ids, feat_fn, experiment, cache_dir,
                        epochs, lr, bs, device, patience, min_delta, seed):
    torch.manual_seed(seed)
    Xtr = np.concatenate([feat_fn(m) for m in train_ids]).astype(np.float32)
    Ytr = np.concatenate([load_saliency(m, cfg.DIST_DIR) for m in train_ids]).astype(np.float32)
    in_dim = Xtr.shape[1]

    model = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.MSELoss()
    Xg = torch.from_numpy(Xtr).to(device); Yg = torch.from_numpy(Ytr).to(device)
    n = len(Xg)

    best = float("inf"); bad = 0; best_state = None
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

    model.eval(); pl, au = [], []
    with torch.no_grad():
        for m in test_ids:
            X = feat_fn(m).astype(np.float32)
            sm = model(torch.from_numpy(X).to(device)).cpu().numpy()
            if experiment == "full+postproc":
                conf, v_sym = load_sym_meta(m, cache_dir)[0]
                sm = symmetry_propagation(sm, v_sym, conf, alpha=0.7)
            pl.append(plcc(sm, load_saliency(m, cfg.DIST_DIR)))
            au.append(auc_schelling(sm, load_schelling_points(m, cfg.POINTS_DIR), len(sm)))
    return float(np.mean(pl)), float(np.mean(au)), in_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--train-frac", type=float, default=cfg.TRAIN_SPLIT,
                    help="Fraction of each category's meshes used for training")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)

    # Preload features once (9-D and 24-D) to avoid re-reading per category/experiment.
    print("Preloading features ...", flush=True)
    phi, phi24 = {}, {}
    for m in mesh_ids:
        phi[m] = np.load(os.path.join(args.cache_dir, f"{m}_phi.npy"))
        phi24[m] = np.load(os.path.join(args.cache_dir, f"{m}_phi24.npy"))

    def make_feat_fn(experiment):
        def f(m):
            P = phi[m] if experiment == "baseline" else phi24[m]
            return select_columns(P, experiment)
        return f

    # Fixed mesh split per category (seeded, same split reused across experiments)
    rng = np.random.default_rng(args.seed)
    splits = {}
    n_per_cat = cfg.MESHES_PER_CAT
    n_train = int(round(args.train_frac * n_per_cat))
    n_train = max(1, min(n_per_cat - 1, n_train))   # keep >=1 train and >=1 test
    for c in range(cfg.N_CATEGORIES):
        ids = rng.permutation(category_ids(c))
        splits[c] = (sorted(int(x) for x in ids[:n_train]),
                     sorted(int(x) for x in ids[n_train:]))

    print(f"Device: {device}  |  per-category training: {cfg.N_CATEGORIES} models, "
          f"{n_train} train / {n_per_cat - n_train} test meshes each "
          f"({int(args.train_frac*100)}/{100-int(args.train_frac*100)} split)")
    print(f"Regime: epochs={args.epochs}  patience={args.patience}\n")

    rows = []
    for exp in EXPERIMENTS:
        feat_fn = make_feat_fn(exp)
        cat_pl, cat_au = [], []
        in_dim = None
        t0 = time.time()
        for c in range(cfg.N_CATEGORIES):
            tr, te = splits[c]
            pl, au, in_dim = train_eval_category(
                tr, te, feat_fn, exp, args.cache_dir, args.epochs, args.lr,
                args.batch_size, device, args.patience, args.min_delta,
                seed=args.seed + c)
            cat_pl.append(pl); cat_au.append(au)
        r = {"in_dim": in_dim,
             "PLCC": float(np.mean(cat_pl)), "PLCC_std": float(np.std(cat_pl)),
             "AUC": float(np.mean(cat_au)),  "AUC_std": float(np.std(cat_au)),
             "cat_pl": cat_pl, "time_s": time.time() - t0}
        rows.append((exp, r))
        print(f"  {exp:<16} dim={in_dim:>2}  "
              f"PLCC={r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
              f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}  "
              f"(20 models, {r['time_s']:.0f}s)", flush=True)

    print("\n" + "=" * 72)
    print(f"{'Experiment':<16} {'Dim':>3}  {'PLCC (per-cat)':>16}  {'AUC (per-cat)':>16}")
    print("-" * 72)
    for exp, r in rows:
        print(f"{exp:<16} {r['in_dim']:>3}  {r['PLCC']:.4f}±{r['PLCC_std']:.4f}    "
              f"{r['AUC']:.4f}±{r['AUC_std']:.4f}")
    print("=" * 72)
    print(f"Mean over {cfg.N_CATEGORIES} category-specialised models "
          f"({n_train} train / {n_per_cat - n_train} test meshes each).")
    print("Paper per-category (Table II): PLCC=0.6062.")


if __name__ == "__main__":
    main()
