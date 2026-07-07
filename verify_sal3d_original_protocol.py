"""
Verification: apply the exact original Schelling protocol (train_week2.py)
to SAL3D, byte-for-byte:
  - concatenate all vertices from all meshes
  - random permutation, 80% training / 20% test, seed = cfg.SEED
  - MLP 256->100->1, ReLU/sigmoid, MSE, Adam lr=1e-3, batch=4096
  - early stop patience 50, min_delta 1e-5, max 500 epochs
  - eval = full-mesh inference, per-mesh PLCC + AUC, averaged over meshes

Runs on three datasets so we can directly compare:
  1) schelling_full   - reference (should match the conference paper ~0.68 PLCC)
  2) sal3d_full       - the new dataset, full resolution
  3) sal3d_decim      - the new dataset, decimated to ~7,500 verts

Output: sal3d_verification_results.json
"""
import os, sys, time, json, pickle
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset      import list_mesh_ids, load_saliency, load_schelling_points
from src.model.mlp         import SaliencyMLP
from src.metrics           import evaluate_dataset
from src.features.symmetry import symmetry_propagation
from train_week2           import select_columns, EXPERIMENTS

RESULTS_PATH = os.path.join(_HERE, "sal3d_verification_results.json")
PATIENCE     = 50
MIN_DELTA    = 1e-5

DATASETS = {
    "schelling_full": {
        "meshes":  cfg.MESHES_DIR,
        "dists":   cfg.DIST_DIR,
        "points":  cfg.POINTS_DIR,
        "cache":   os.path.join(_HERE, "cache"),
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


def preload(ds):
    d = DATASETS[ds]
    mesh_ids = list_mesh_ids(d["meshes"])
    X24, gts, points, mesh_sym = {}, {}, {}, {}
    for mid in mesh_ids:
        X24[mid]    = np.load(os.path.join(d["cache"], f"{mid}_phi24.npy"))
        gts[mid]    = load_saliency(mid, d["dists"]).astype(np.float32)
        points[mid] = load_schelling_points(mid, d["points"])
        with open(os.path.join(d["cache"], f"{mid}_sym_meta.pkl"), "rb") as f:
            mesh_sym[mid] = pickle.load(f)
    return mesh_ids, X24, gts, points, mesh_sym


def run_pooled_8020(exp, mesh_ids, X24, gts, points, mesh_sym, device,
                    seed=cfg.SEED, epochs=cfg.EPOCHS):
    """Byte-for-byte replica of train_week2.run_experiment (the original)."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X_per = {m: select_columns(X24[m], exp).astype(np.float32) for m in mesh_ids}
    all_X = np.concatenate([X_per[m] for m in mesh_ids]).astype(np.float32)
    all_Y = np.concatenate([gts[m]   for m in mesh_ids]).astype(np.float32)

    # 80/20 vertex-level split (exact code from train_week2.py)
    idx     = rng.permutation(len(all_X))
    n_train = int(cfg.TRAIN_SPLIT * len(all_X))
    train_idx = idx[:n_train]

    in_dim = all_X.shape[1]
    model  = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.MSELoss()

    Xg = torch.from_numpy(all_X[train_idx]).to(device)
    Yg = torch.from_numpy(all_Y[train_idx]).to(device)
    n_tr = len(Xg)

    best_loss, bad, best_state, ep_used = float("inf"), 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        el = 0.0
        for i in range(0, n_tr, cfg.BATCH_SIZE):
            j = perm[i:i + cfg.BATCH_SIZE]
            xb, yb = Xg[j], Yg[j]
            p = model(xb); l = loss_fn(p, yb)
            opt.zero_grad(); l.backward(); opt.step()
            el = el + l.detach() * len(j)
        avg = (el / n_tr).item()
        if avg < best_loss - MIN_DELTA:
            best_loss = avg; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        ep_used = ep
        if bad >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Full-mesh inference (paper's pooled evaluation)
    model.eval()
    S, G, P = [], [], []
    with torch.no_grad():
        for mid in mesh_ids:
            xm = torch.from_numpy(X_per[mid]).to(device)
            sm = model(xm).cpu().numpy()
            if exp == "full+postproc":
                conf, v_sym = mesh_sym[mid][0]
                sm = symmetry_propagation(sm, v_sym, conf, alpha=0.7)
            S.append(sm); G.append(gts[mid]); P.append(points[mid])
    r = evaluate_dataset(S, G, P)
    r["in_dim"] = in_dim
    r["epochs"] = ep_used
    return r


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  protocol: 80/20 vertex pool, seed {cfg.SEED}\n",
          flush=True)

    results = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}

    for ds in DATASETS:
        d = DATASETS[ds]
        if not (os.path.isdir(d["cache"]) and os.listdir(d["cache"])):
            print(f"SKIP {ds}: cache empty"); continue
        print(f"=== {ds} ===  preloading ...", flush=True)
        mesh_ids, X24, gts, points, mesh_sym = preload(ds)
        total_v = sum(len(X24[m]) for m in mesh_ids)
        print(f"  {len(mesh_ids)} meshes, {total_v:,} total vertices, "
              f"{int(total_v*cfg.TRAIN_SPLIT):,} train / "
              f"{total_v - int(total_v*cfg.TRAIN_SPLIT):,} test\n", flush=True)
        results.setdefault(ds, {})
        for exp in EXPERIMENTS:
            if exp in results[ds]:
                s = results[ds][exp]
                print(f"  SKIP {exp:<18}  PLCC={s['PLCC']:.4f}±{s['PLCC_std']:.4f}",
                      flush=True)
                continue
            t = time.time()
            r = run_pooled_8020(exp, mesh_ids, X24, gts, points, mesh_sym, device)
            r["time_s"] = time.time() - t
            results[ds][exp] = r
            tmp = RESULTS_PATH + ".tmp"
            json.dump(results, open(tmp, "w"), indent=2,
                      default=lambda x: float(x))
            os.replace(tmp, RESULTS_PATH)
            print(f"  {exp:<18} dim={r['in_dim']:>2}  "
                  f"PLCC={r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
                  f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}  "
                  f"({r['time_s']:.0f}s, {r['epochs']} ep)", flush=True)
        print()

    print("=" * 80)
    print("VERIFICATION SUMMARY (original 80/20 pooled-vertex single-pass)")
    for ds in DATASETS:
        if ds not in results: continue
        print(f"\n[{ds}]")
        for exp in EXPERIMENTS:
            if exp not in results[ds]: continue
            r = results[ds][exp]
            print(f"  {exp:<18} dim={r['in_dim']:>2}  "
                  f"PLCC={r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
                  f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
