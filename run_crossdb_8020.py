"""
Cross-database test, redone under the EXACT original 80/20 pooled-vertex
single-pass protocol (matching train_week2.py and verify_sal3d_original_protocol.py).

For every (source, target) pair we:
  1. concatenate all source-mesh vertices, take a random 80% subset (seed 42),
  2. train the MLP with the conference paper's hyperparameters,
  3. perform full-mesh inference on EVERY target mesh,
  4. report per-mesh PLCC and AUC, aggregated as the mean and std
     across target meshes (= what evaluate_dataset returns).

Pairs:
  A   Schelling full       -> SAL3D full
  B   Schelling decimated  -> SAL3D decimated
  rA  SAL3D full           -> Schelling full
  rB  SAL3D decimated      -> Schelling decimated

In-domain cells are already in sal3d_verification_results.json under the same
protocol, so we don't re-run them.

Output: crossdb_8020_results.json (resumable).
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

RESULTS_PATH = os.path.join(_HERE, "crossdb_8020_results.json")
PATIENCE     = 50
MIN_DELTA    = 1e-5

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


def train_8020(exp, src_ids, src_X, src_g, device, seed=cfg.SEED, epochs=cfg.EPOCHS):
    """Exact 80/20 pooled-vertex training from train_week2.run_experiment."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    X_per = {m: select_columns(src_X[m], exp).astype(np.float32) for m in src_ids}
    all_X = np.concatenate([X_per[m] for m in src_ids]).astype(np.float32)
    all_Y = np.concatenate([src_g[m] for m in src_ids]).astype(np.float32)
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
    best, bad, best_state, ep_used = float("inf"), 0, None, 0
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
    return model, ep_used, in_dim


def predict_target(model, tgt_ids, tgt_X, tgt_g, tgt_p, tgt_s, exp, device):
    model.eval()
    S, G, P = [], [], []
    with torch.no_grad():
        for mid in tgt_ids:
            X = select_columns(tgt_X[mid], exp).astype(np.float32)
            s = model(torch.from_numpy(X).to(device)).cpu().numpy()
            if exp == "full+postproc":
                conf, v_sym = tgt_s[mid][0]
                s = symmetry_propagation(s, v_sym, conf, alpha=0.7)
            S.append(s); G.append(tgt_g[mid]); P.append(tgt_p[mid])
    return evaluate_dataset(S, G, P)


def save_atomic(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    os.replace(tmp, RESULTS_PATH)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  protocol: 80/20 vertex pool, seed {cfg.SEED}\n",
          flush=True)

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

    pairs = [
        ("A_schelling_full_to_sal3d_full",     "schelling_full",  "sal3d_full"),
        ("B_schelling_decim_to_sal3d_decim",   "schelling_decim", "sal3d_decim"),
        ("rA_sal3d_full_to_schelling_full",    "sal3d_full",      "schelling_full"),
        ("rB_sal3d_decim_to_schelling_decim",  "sal3d_decim",     "schelling_decim"),
    ]
    for tag, src, tgt in pairs:
        if src not in pre or tgt not in pre:
            print(f"=== SKIP {tag} (missing dataset) ===", flush=True)
            continue
        results.setdefault(tag, {})
        src_ids, src_X, src_g, _,       _       = pre[src]
        tgt_ids, tgt_X, tgt_g, tgt_p, tgt_sym   = pre[tgt]
        print(f"\n=== {tag}  ({len(src_ids)} src -> {len(tgt_ids)} tgt) ===",
              flush=True)
        for exp in EXPERIMENTS:
            if exp in results[tag]:
                s = results[tag][exp]
                print(f"  SKIP {exp:<18} PLCC={s['PLCC']:.4f}±{s['PLCC_std']:.4f}",
                      flush=True)
                continue
            t = time.time()
            model, ep, in_dim = train_8020(exp, src_ids, src_X, src_g, device)
            r = predict_target(model, tgt_ids, tgt_X, tgt_g, tgt_p, tgt_sym,
                               exp, device)
            r["in_dim"] = in_dim
            r["epochs"] = ep
            r["time_s"] = time.time() - t
            r["mesh_ids"] = [int(m) for m in tgt_ids]  # pairing key for sig tests
            results[tag][exp] = r
            save_atomic(results)
            print(f"  {exp:<18} dim={r['in_dim']:>2}  "
                  f"PLCC={r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
                  f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}  "
                  f"({r['time_s']:.0f}s, {r['epochs']} ep)", flush=True)

    print("\n" + "=" * 80)
    print("CROSSDB 80/20 SUMMARY")
    for tag, _, _ in pairs:
        if tag not in results: continue
        print(f"\n[{tag}]")
        for exp in EXPERIMENTS:
            if exp not in results[tag]: continue
            r = results[tag][exp]
            print(f"  {exp:<18}  PLCC={r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
                  f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
