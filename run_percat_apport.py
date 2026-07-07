"""
Fill in the per-category Table 3 rows for the two new ablation variants
(cr+pers, symf+pers) under both splits used by the paper: 10/10 and 80/20.

Reuses eval_per_category_training.train_eval_category so settings (epochs,
patience, seed, MLP, batching, decimation) exactly match the existing rows.
"""
import os, sys, time, json
import numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset import list_mesh_ids
from train_week2 import select_columns
from eval_per_category_training import train_eval_category, category_ids

CACHE = os.path.join(_HERE, "cache")
NEW_EXPS = ["cr+pers", "symf+pers"]
SPLITS = [0.5, 0.8]   # 10/10 and 80/20

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)

    # Preload features once (24-D enhanced is enough for the two new combos)
    print("Preloading 24-D features ...", flush=True)
    phi24 = {m: np.load(os.path.join(CACHE, f"{m}_phi24.npy")) for m in mesh_ids}

    def make_feat_fn(experiment):
        def f(m):
            return select_columns(phi24[m], experiment)
        return f

    results = {f"{int(s*100)}/{100-int(s*100)}": {} for s in SPLITS}

    for frac in SPLITS:
        split_name = f"{int(frac*100)}/{100-int(frac*100)}"
        # build per-category mesh split (seeded; matches eval_per_category_training)
        rng = np.random.default_rng(cfg.SEED)
        splits = {}
        n_per_cat = cfg.MESHES_PER_CAT
        n_train = max(1, min(n_per_cat - 1, int(round(frac * n_per_cat))))
        for c in range(cfg.N_CATEGORIES):
            ids = rng.permutation(category_ids(c))
            splits[c] = (sorted(int(x) for x in ids[:n_train]),
                         sorted(int(x) for x in ids[n_train:]))

        print(f"\n=== split {split_name}  ({n_train} train / {n_per_cat - n_train} test per category) ===", flush=True)

        for exp in NEW_EXPS:
            feat_fn = make_feat_fn(exp)
            cat_pl, cat_au = [], []
            in_dim = None
            t0 = time.time()
            for c in range(cfg.N_CATEGORIES):
                tr, te = splits[c]
                pl, au, in_dim = train_eval_category(
                    tr, te, feat_fn, exp, CACHE,
                    epochs=cfg.EPOCHS, lr=cfg.LR, bs=cfg.BATCH_SIZE,
                    device=device, patience=50, min_delta=1e-5,
                    seed=cfg.SEED + c)
                cat_pl.append(pl); cat_au.append(au)
            r = {"in_dim": in_dim,
                 "PLCC": float(np.mean(cat_pl)), "PLCC_std": float(np.std(cat_pl)),
                 "AUC": float(np.mean(cat_au)),  "AUC_std": float(np.std(cat_au)),
                 # per-category values (one per category) for paired sig tests
                 "PLCC_list": [float(x) for x in cat_pl],
                 "AUC_list": [float(x) for x in cat_au],
                 "time_s": time.time() - t0}
            results[split_name][exp] = r
            print(f"  {exp:<16} dim={in_dim:>2}  "
                  f"PLCC={r['PLCC']:.4f}±{r['PLCC_std']:.4f}  "
                  f"AUC={r['AUC']:.4f}±{r['AUC_std']:.4f}  "
                  f"({r['time_s']:.0f}s)", flush=True)

    # save raw JSON
    with open(os.path.join(_HERE, "percat_apport_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary in the paper's column format ------------------------------
    print("\n" + "=" * 70)
    print(f"{'Experiment':<22}{'Dim':>4}  {'PLCC (10/10)':>14}  {'PLCC (80/20)':>14}  {'AUC (80/20)':>13}")
    print("-" * 70)
    for exp in NEW_EXPS:
        r10 = results["50/50"][exp]
        r80 = results["80/20"][exp]
        print(f"{exp:<22}{r10['in_dim']:>4}  "
              f"{r10['PLCC']:.4f}±{r10['PLCC_std']:.4f}  "
              f"{r80['PLCC']:.4f}±{r80['PLCC_std']:.4f}  "
              f"{r80['AUC']:.4f}±{r80['AUC_std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
