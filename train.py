"""
Baseline reproduction of Lézoray & Nouri, MMSP 2025.
Week 1 goal: PLCC ≈ 0.5567 ± 0.005, AUC ≈ 0.8196.

Usage:
  python train.py [--cache-dir ./cache] [--epochs 200] [--seed 42]
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from src.data.dataset  import (load_mesh, load_saliency, load_schelling_points,
                                 mesh_id_to_category, list_mesh_ids)
from src.data.decimate  import decimate_mesh
from src.features.extract import extract_multiscale_features
from src.model.mlp   import SaliencyMLP
from src.metrics     import evaluate_dataset, plcc, auc_schelling


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------

def cache_path(cache_dir, mesh_id):
    return os.path.join(cache_dir, f"{mesh_id}_phi.npy")


def compute_and_cache(mesh_id: int, cache_dir: str) -> str:
    """Compute features for one mesh and save to cache. Returns status string."""
    cp = cache_path(cache_dir, mesh_id)
    if os.path.exists(cp):
        return f"mesh {mesh_id}: cached"

    t0 = time.time()
    verts, tris = load_mesh(mesh_id, cfg.MESHES_DIR)
    scale_meshes = [(verts, tris)]
    for ratio in cfg.DECIM_RATIOS[1:]:
        dv, dt = decimate_mesh(verts, tris, ratio)
        scale_meshes.append((dv, dt))

    Phi = extract_multiscale_features(verts, tris, scale_meshes,
                                       gamma=cfg.GAMMA, k=cfg.K_RINGS)
    np.save(cp, Phi)
    return f"mesh {mesh_id}: {time.time()-t0:.1f}s  shape={Phi.shape}"


def load_or_compute_features(mesh_id: int, cache_dir: str) -> np.ndarray:
    cp = cache_path(cache_dir, mesh_id)
    if not os.path.exists(cp):
        msg = compute_and_cache(mesh_id, cache_dir)
        print(f"  [feat] {msg}")
    return np.load(cp)


def precompute_all_features(mesh_ids: list, cache_dir: str, workers: int = 1):
    """Precompute and cache features for all meshes (parallel if workers > 1)."""
    todo = [mid for mid in mesh_ids
            if not os.path.exists(cache_path(cache_dir, mid))]
    if not todo:
        print(f"All {len(mesh_ids)} meshes already cached.")
        return

    print(f"Computing features for {len(todo)} meshes "
          f"({'parallel x' + str(workers) if workers > 1 else 'sequential'}) ...")
    done = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(compute_and_cache, mid, cache_dir): mid
                       for mid in todo}
            for f in as_completed(futures):
                done += 1
                print(f"  [{done}/{len(todo)}] {f.result()}", flush=True)
    else:
        for mid in todo:
            done += 1
            msg = compute_and_cache(mid, cache_dir)
            print(f"  [{done}/{len(todo)}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--epochs",    type=int, default=cfg.EPOCHS)
    parser.add_argument("--seed",      type=int, default=cfg.SEED)
    parser.add_argument("--batch-size",type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",        type=float, default=cfg.LR)
    parser.add_argument("--patience",  type=int, default=50,
                        help="Early-stopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=1e-5,
                        help="Minimum loss improvement to reset patience counter")
    parser.add_argument("--workers",   type=int, default=1,
                        help="Parallel workers for feature pre-computation")
    parser.add_argument("--no-cache",  action="store_true")
    parser.add_argument("--precompute-only", action="store_true",
                        help="Only compute features, skip training")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)   # 1–400
    print(f"Dataset: {len(mesh_ids)} meshes")

    # ---- Optional: clear cache ------------------------------------------
    if args.no_cache:
        for mid in mesh_ids:
            cp = cache_path(args.cache_dir, mid)
            if os.path.exists(cp):
                os.remove(cp)

    # ---- Pre-compute features (parallel) --------------------------------
    precompute_all_features(mesh_ids, args.cache_dir, workers=args.workers)

    if args.precompute_only:
        print("Feature pre-computation done. Exiting (--precompute-only).")
        return

    # ---- 80/20 random vertex-level split --------------------------------
    # Following the paper: split vertices across ALL meshes jointly, not per-mesh
    rng = np.random.default_rng(args.seed)

    all_features, all_saliency = [], []
    mesh_slices   = []   # (start, end, mesh_id) for per-mesh evaluation
    mesh_gt       = {}
    mesh_pts      = {}

    total = 0
    for mesh_id in mesh_ids:
        Phi = load_or_compute_features(mesh_id, args.cache_dir)
        sal = load_saliency(mesh_id, cfg.DIST_DIR)
        pts = load_schelling_points(mesh_id, cfg.POINTS_DIR)

        m = len(Phi)
        all_features.append(Phi)
        all_saliency.append(sal)
        mesh_slices.append((total, total + m, mesh_id))
        mesh_gt[mesh_id] = sal
        mesh_pts[mesh_id] = pts
        total += m

    X = np.concatenate(all_features, axis=0).astype(np.float32)  # (N, 9)
    Y = np.concatenate(all_saliency, axis=0).astype(np.float32)  # (N,)
    print(f"Total vertices: {len(X)}")

    # Vertex-level random split
    idx = rng.permutation(len(X))
    n_train = int(cfg.TRAIN_SPLIT * len(X))
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_te, Y_te = X[test_idx],  Y[test_idx]

    # ---- Model / optimiser -----------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    elif os.environ.get("ALLOW_CPU") == "1":
        device = torch.device("cpu")
        print("Device: cpu (CUDA unavailable, ALLOW_CPU=1 set)")
    else:
        raise RuntimeError(
            "CUDA is not available. Refusing to train on CPU silently.\n"
            "If the GPU driver is in a bad state, try: "
            "sudo modprobe nvidia_uvm  (or reload nvidia kernel modules).\n"
            "To override and train on CPU anyway, set ALLOW_CPU=1.")

    model = SaliencyMLP(in_dim=cfg.FEAT_DIM, hidden=cfg.MLP_HIDDEN).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Move the full training set to the GPU once — avoids the per-batch
    # host->device copies and per-batch .item() syncs that dominated runtime.
    X_tr_t = torch.from_numpy(X_tr).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device)
    n_tr   = len(X_tr_t)
    bs     = args.batch_size

    # ---- Training loop with early stopping -------------------------------
    print(f"\nTraining up to {args.epochs} epochs (patience={args.patience}) ...")
    best_loss   = float("inf")
    epochs_bad  = 0
    best_state  = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        epoch_loss = 0.0
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            xb, yb = X_tr_t[idx], Y_tr_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss = epoch_loss + loss.detach() * len(idx)   # stay on GPU

        avg_loss = (epoch_loss / n_tr).item()   # single host sync per epoch

        # Patience check: best weights are saved only on a meaningful
        # improvement (> min_delta), which also resets the early-stop counter.
        if avg_loss < best_loss - args.min_delta:
            best_loss  = avg_loss
            epochs_bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_bad += 1

        if epoch % 5 == 0 or epoch == 1 or epochs_bad >= args.patience:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={avg_loss:.6f}  best={best_loss:.6f}  "
                  f"bad={epochs_bad}/{args.patience}", flush=True)

        if epochs_bad >= args.patience:
            print(f"  Early stop at epoch {epoch} (no improvement in {args.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Evaluation: per-mesh PLCC & AUC --------------------------------
    print("\nEvaluating on all meshes (test vertices only per mesh) …")
    model.eval()

    # Build a test-vertex mask per mesh
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_idx] = True

    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(device)
        S_te = model(X_te_t).cpu().numpy()

    # Reconstruct per-mesh predictions (use full-mesh inference for PLCC/AUC)
    print("Running full-mesh inference for metric computation …")
    all_S, all_G, all_P = [], [], []
    with torch.no_grad():
        for start, end, mesh_id in mesh_slices:
            xm = torch.from_numpy(X[start:end]).to(device)
            sm = model(xm).cpu().numpy()
            all_S.append(sm)
            all_G.append(mesh_gt[mesh_id])
            all_P.append(mesh_pts[mesh_id])

    results = evaluate_dataset(all_S, all_G, all_P)
    print(f"\n{'='*50}")
    print(f"  PLCC = {results['PLCC']:.4f}  ±{results['PLCC_std']:.4f}")
    print(f"  AUC  = {results['AUC']:.4f}  ±{results['AUC_std']:.4f}")
    print(f"{'='*50}")
    print(f"\nBaseline target: PLCC≈0.5567, AUC≈0.8196")

    # Save model
    save_path = os.path.join(args.cache_dir, "baseline_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
