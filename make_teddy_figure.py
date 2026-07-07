"""
Teddy-bear figure for the journal: the locked-view bear with its features AND
the predictions of the three best journal models.

Layout 3 rows x 4 columns:
  Row 1: plain mesh   | ground truth | R | GS
  Row 2: SS           | SymF         | CR | Pers
  Row 3: Pred. full 24-D | Pred. +CR+Pers 21-D | Pred. Full + post-proc | (blank)

Models are trained once on the standard Schelling 80/20 pooled-vertex single-
pass protocol (matching train_week2.run_experiment), then the prediction for
mesh 161 is cached so the figure script is fast on re-runs.
"""
import os, sys, time, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import config as cfg
from src.data.dataset      import load_mesh, load_saliency, list_mesh_ids
from src.model.mlp         import SaliencyMLP
from src.features.symmetry import symmetry_propagation
from train_week2           import select_columns

MESH_ID = 161                                      # teddy bear
ELEV, AZIM, ROLL = 80, -90, 0                      # locked view from talk
PRED_CACHE = os.path.join(_HERE, "cache",
                          f"_teddy_predictions_{MESH_ID}.npz")
OUT_PNG = os.path.join(_HERE, "journal_paper", "fig_teddy.png")

VARIANTS = [
    ("symf+cr+pers",  "Pred. Full 24-D"),
    ("cr+pers",       r"Pred. +CR+Pers 21-D $\star$"),
    ("full+postproc", "Pred. Full + post-proc"),
]


# ---------------------------------------------------------------------------
# Training (only when the prediction cache is missing)
# ---------------------------------------------------------------------------
def train_predict_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Device: {device}  |  {len(mesh_ids)} meshes", flush=True)

    # Preload features + GT + sym meta
    X24, gts, mesh_sym = {}, {}, {}
    for mid in mesh_ids:
        X24[mid] = np.load(os.path.join(_HERE, "cache", f"{mid}_phi24.npy"))
        gts[mid] = load_saliency(mid, cfg.DIST_DIR).astype(np.float32)
        with open(os.path.join(_HERE, "cache", f"{mid}_sym_meta.pkl"), "rb") as f:
            mesh_sym[mid] = pickle.load(f)

    out = {}
    rng = np.random.default_rng(cfg.SEED)

    for exp, _label in VARIANTS:
        print(f"\n=== training {exp} ===", flush=True)
        torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
        X_per = {m: select_columns(X24[m], exp).astype(np.float32) for m in mesh_ids}
        all_X = np.concatenate([X_per[m] for m in mesh_ids]).astype(np.float32)
        all_Y = np.concatenate([gts[m]   for m in mesh_ids]).astype(np.float32)
        idx = np.random.default_rng(cfg.SEED).permutation(len(all_X))
        n_train = int(cfg.TRAIN_SPLIT * len(all_X))
        train_idx = idx[:n_train]

        in_dim = all_X.shape[1]
        model = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.LR)
        loss_fn = nn.MSELoss()
        Xg = torch.from_numpy(all_X[train_idx]).to(device)
        Yg = torch.from_numpy(all_Y[train_idx]).to(device)
        n_tr = len(Xg)

        best, bad, best_state = float("inf"), 0, None
        t0 = time.time()
        for ep in range(1, cfg.EPOCHS + 1):
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
            if avg < best - 1e-5:
                best = avg; bad = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= 50: break
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"  trained in {time.time()-t0:.0f}s ({ep} ep)", flush=True)

        # Predict on teddy
        model.eval()
        with torch.no_grad():
            X = torch.from_numpy(X_per[MESH_ID]).to(device)
            sm = model(X).cpu().numpy()
            if exp == "full+postproc":
                conf, v_sym = mesh_sym[MESH_ID][0]
                sm = symmetry_propagation(sm, v_sym, conf, alpha=0.7)
        out[exp] = sm
        print(f"  predicted on mesh {MESH_ID}: shape {sm.shape}", flush=True)

    np.savez(PRED_CACHE, **out)
    print(f"\nSaved predictions to {PRED_CACHE}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def shade(tri, rgba):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(n, axis=1, keepdims=True); nl[nl == 0] = 1.0
    n /= nl
    light = np.array([-0.3, -0.4, 0.85]); light /= np.linalg.norm(light)
    inten = 0.55 + 0.45 * np.clip(n @ light, 0.0, 1.0)
    rgba[:, :3] *= inten[:, None]
    return rgba


def panel(ax, verts, tris, scalar, title, plain=False, highlight=False):
    tri = verts[tris]
    if plain:
        rgba = np.tile(np.array([0.45, 0.6, 0.85, 1.0]), (len(tri), 1))
    else:
        z = scalar.astype(np.float64)
        lo, hi = np.percentile(z, [2, 98])
        z = np.clip((z - lo) / (hi - lo + 1e-12), 0.0, 1.0) ** 0.7
        rgba = cm.jet(z[tris].mean(axis=1))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=shade(tri, rgba),
                                          edgecolors="none", linewidths=0,
                                          antialiased=True))
    c = verts.mean(0); r = np.abs(verts - c).max()
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    try:    ax.set_box_aspect((1, 1, 1), zoom=1.55)
    except TypeError: ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    try:    ax.view_init(elev=ELEV, azim=AZIM, roll=ROLL)
    except TypeError: ax.view_init(elev=ELEV, azim=AZIM)
    weight = "bold" if highlight else "normal"
    colr   = "#1f3a5f" if highlight else "#333333"
    ax.set_title(title, fontsize=10.5, color=colr, fontweight=weight, pad=2)


def render(verts, tris, gt, preds):
    Phi  = np.load(os.path.join(_HERE, "cache", f"{MESH_ID}_phi24.npy"))
    R, GS, SS = Phi[:, 0], Phi[:, 3], Phi[:, 6]
    SymF = Phi[:, 9]
    CR   = np.stack([Phi[:, 12], Phi[:, 15], Phi[:, 18]], axis=1).mean(axis=1)
    Pers = Phi[:, 21]

    cells = [
        ("Plain mesh",                  None,            False),
        ("Ground truth",                gt,              False),
        ("$R$  roughness",              R,               False),
        ("$GS$  geometric",             GS,              False),
        ("$SS$  spectral",              SS,              False),
        ("SymF  symmetry",              SymF,            False),
        ("CR  contextual rank",         CR,              True),
        ("Pers  topological persist.",  Pers,            True),
        ("Pred. Full 24-D",                       preds["symf+cr+pers"], True),
        (r"Pred. $+$CR$+$Pers 21-D $\star$",      preds["cr+pers"],      True),
        ("Pred. Full $+$ post-proc",              preds["full+postproc"], True),
        (None, None, None),  # blank cell
    ]

    fig = plt.figure(figsize=(12.0, 9.4))
    for i, (title, sc, hl) in enumerate(cells):
        ax = fig.add_subplot(3, 4, i + 1, projection="3d")
        if title is None:
            ax.set_axis_off()
            continue
        panel(ax, verts, tris, sc, title, plain=(sc is None), highlight=hl)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005,
                        wspace=0.02, hspace=0.06)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", transparent=False)
    print("wrote", OUT_PNG)


def main():
    verts, tris = load_mesh(MESH_ID, cfg.MESHES_DIR)
    gt          = load_saliency(MESH_ID, cfg.DIST_DIR)
    if os.path.exists(PRED_CACHE):
        z = np.load(PRED_CACHE)
        preds = {k: z[k] for k in z.files}
        print(f"loaded cached predictions: {list(preds)}", flush=True)
    else:
        preds = train_predict_all()
    render(verts, tris, gt, preds)


if __name__ == "__main__":
    main()
