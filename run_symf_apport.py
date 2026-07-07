"""
Journal extension run: fill the missing ablation rows so the SymF contribution
is visible in every context.

Runs cr+pers (21-D) and symf+pers (15-D) under both protocols:
  - pooled-vertex 80/20  (train_week2.run_experiment)
  - held-out mesh 320/80 (eval_heldout_mesh.run_experiment)

Then prints a focused table that pairs each "without SymF" row with the
corresponding "with SymF" row so the delta is unambiguous.
"""
import os, sys, time, json
import numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg
from src.data.dataset import list_mesh_ids
import train_week2
import eval_heldout_mesh

CACHE = os.path.join(_HERE, "cache")
NEW_EXPS = ["cr+pers", "symf+pers"]

# Known results (from previous v3 sweep, for the contextual comparison).
# Source: train_week2_v3.log / eval_heldout_mesh.log.
KNOWN_POOLED = {
    "baseline":     dict(in_dim=9,  PLCC=0.540, AUC=0.818),
    "symf":         dict(in_dim=12, PLCC=0.573, AUC=0.831),
    "cr":           dict(in_dim=18, PLCC=0.644, AUC=0.860),
    "pers":         dict(in_dim=12, PLCC=0.601, AUC=0.832),
    "symf+cr":      dict(in_dim=21, PLCC=0.668, AUC=0.870),
    "symf+cr+pers": dict(in_dim=24, PLCC=0.680, AUC=0.873),
}
KNOWN_HELD = {
    "baseline":     dict(in_dim=9,  PLCC=0.426, AUC=0.768),
    "symf":         dict(in_dim=12, PLCC=0.436, AUC=0.770),
    "cr":           dict(in_dim=18, PLCC=0.503, AUC=0.801),
    "pers":         dict(in_dim=12, PLCC=0.457, AUC=0.777),
    "symf+cr":      dict(in_dim=21, PLCC=0.503, AUC=0.811),
    "symf+cr+pers": dict(in_dim=24, PLCC=0.521, AUC=0.815),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    train_ids, test_ids = eval_heldout_mesh.mesh_split(CACHE, cfg.SEED)
    print(f"Device: {device} | meshes: {len(mesh_ids)}  "
          f"held-out: {len(train_ids)} train / {len(test_ids)} test\n")

    pooled, held = {}, {}
    for exp in NEW_EXPS:
        print(f"=== {exp} :: POOLED ===", flush=True)
        t0 = time.time()
        pooled[exp] = train_week2.run_experiment(
            exp, mesh_ids, CACHE, epochs=cfg.EPOCHS, lr=cfg.LR,
            batch_size=cfg.BATCH_SIZE, seed=cfg.SEED, device=device,
            patience=50, min_delta=1e-5)
        pooled[exp]["mesh_ids"] = [int(m) for m in mesh_ids]
        print(f"  -> dim={pooled[exp]['in_dim']}  "
              f"PLCC={pooled[exp]['PLCC']:.4f}  AUC={pooled[exp]['AUC']:.4f}  "
              f"({time.time()-t0:.0f}s)\n", flush=True)

        print(f"=== {exp} :: HELD-OUT MESH ===", flush=True)
        t0 = time.time()
        held[exp] = eval_heldout_mesh.run_experiment(
            exp, train_ids, test_ids, CACHE,
            epochs=cfg.EPOCHS, lr=cfg.LR, bs=cfg.BATCH_SIZE,
            seed=cfg.SEED, device=device, patience=50, min_delta=1e-5)
        held[exp]["mesh_ids"] = [int(m) for m in test_ids]
        print(f"  -> dim={held[exp]['in_dim']}  "
              f"PLCC={held[exp]['PLCC']:.4f}  AUC={held[exp]['AUC']:.4f}  "
              f"({time.time()-t0:.0f}s)\n", flush=True)

    # Stash for the writeup
    out = {"pooled": {k: {kk: float(vv) if isinstance(vv, (np.floating, float, int))
                          else vv for kk, vv in v.items() if kk != "time_s"}
                      for k, v in pooled.items()},
           "held": {k: {kk: float(vv) if isinstance(vv, (np.floating, float, int))
                        else vv for kk, vv in v.items() if kk != "time_s"}
                    for k, v in held.items()}}
    with open(os.path.join(_HERE, "symf_apport_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote symf_apport_results.json\n")

    # --- Apport of SymF table ------------------------------------------------
    def fmt(d):
        return f"{d['PLCC']:.4f} / {d['AUC']:.4f}"

    pairs = [
        ("alone",         "baseline",   "symf"),
        ("with CR",       "cr",         "symf+cr"),
        ("with Pers",     "pers",       "symf+pers"),
        ("with CR+Pers",  "cr+pers",    "symf+cr+pers"),
    ]

    def look(table, key):
        if key in table:                  # newly computed
            return {"PLCC": table[key]["PLCC"], "AUC": table[key]["AUC"]}
        return None

    def get(known, table, key):
        v = look(table, key)
        return v if v is not None else known.get(key)

    print("=" * 88)
    print("Apport of SymF (Δ when SymF is added on top of an existing variant)")
    print("=" * 88)
    print(f"{'Context':<14}{'Without SymF':<22}{'With SymF':<22}"
          f"{'ΔPLCC':>8} {'ΔAUC':>8}  Protocol")
    print("-" * 88)
    for proto, known, table in [("POOLED", KNOWN_POOLED, pooled),
                                  ("HELD-OUT", KNOWN_HELD, held)]:
        for label, without, with_ in pairs:
            a = get(known, table, without); b = get(known, table, with_)
            if a is None or b is None:
                continue
            print(f"{label:<14}{without+'  '+fmt(a):<22}"
                  f"{with_+'  '+fmt(b):<22}"
                  f"{b['PLCC']-a['PLCC']:+8.4f} {b['AUC']-a['AUC']:+8.3f}  {proto}")
        print("-" * 88)


if __name__ == "__main__":
    main()
