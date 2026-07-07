"""
Fourth journal figure: side-by-side qualitative comparison of our two top
configurations against the Spiral Regressor baseline (9-D, the
L\\'ezoray \\& Nouri 2025 descriptor we extend) on three representative meshes
that span the symmetry spectrum.

Layout 3 rows x 4 columns:
  Row 1: Teddy   (cat 8,  strongly bilateral)
  Row 2: Dolphin (cat 11, smooth + symmetric)
  Row 3: Chair   (cat 5,  weakly symmetric)

Columns: Ground truth | Spiral Regressor (9-D) | Ours, full 24-D | Ours, full+post-proc

The full+post-proc column reuses the full 24-D model and applies the bilateral
post-processing step at inference time, so the only meaningful difference
between columns 3 and 4 is the post-proc itself - which is exactly what we
want to highlight.
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

# Meshes chosen for symmetry-spectrum coverage and visual variety.
MESHES = [
    (161, "Teddy"),
    (221, "Dolphin"),
    (101, "Chair"),
]

# Each row's locked viewpoint (elev, azim, roll).
VIEWS = {
    161: (80, -90, 0),     # teddy as in talk
    221: (15, -70, 0),     # dolphin as in feature figs
    101: (15, -70, 0),     # chair, same convention
}

PRED_CACHE = os.path.join(_HERE, "cache", "_sota_predictions.npz")
OUT_PNG    = os.path.join(_HERE, "journal_paper", "fig_sota_qualitative.png")

VARIANTS = [("baseline", "9-D"), ("symf+cr+pers", "24-D")]


# ---------------------------------------------------------------------------
def train_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Device: {device}  |  {len(mesh_ids)} meshes", flush=True)

    X24, gts, mesh_sym = {}, {}, {}
    for mid in mesh_ids:
        X24[mid] = np.load(os.path.join(_HERE, "cache", f"{mid}_phi24.npy"))
        gts[mid] = load_saliency(mid, cfg.DIST_DIR).astype(np.float32)
        with open(os.path.join(_HERE, "cache", f"{mid}_sym_meta.pkl"), "rb") as f:
            mesh_sym[mid] = pickle.load(f)

    out = {}
    for exp, label in VARIANTS:
        print(f"\n=== training {exp} ({label}) ===", flush=True)
        torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
        rng = np.random.default_rng(cfg.SEED)
        X_per = {m: select_columns(X24[m], exp).astype(np.float32) for m in mesh_ids}
        all_X = np.concatenate([X_per[m] for m in mesh_ids]).astype(np.float32)
        all_Y = np.concatenate([gts[m]   for m in mesh_ids]).astype(np.float32)
        idx = rng.permutation(len(all_X))
        n_train = int(cfg.TRAIN_SPLIT * len(all_X))
        train_idx = idx[:n_train]

        in_dim = all_X.shape[1]
        model = SaliencyMLP(in_dim=in_dim, hidden=cfg.MLP_HIDDEN).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.LR)
        loss_fn = nn.MSELoss()
        Xg = torch.from_numpy(all_X[train_idx]).to(device)
        Yg = torch.from_numpy(all_Y[train_idx]).to(device)
        n_tr = len(Xg)
        best, bad, best_state, ep_used = float("inf"), 0, None, 0
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
            ep_used = ep
            if bad >= 50: break
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"  trained in {time.time()-t0:.0f}s ({ep_used} ep)", flush=True)

        model.eval()
        with torch.no_grad():
            for mid, _ in MESHES:
                X = torch.from_numpy(X_per[mid]).to(device)
                sm = model(X).cpu().numpy()
                out[f"{exp}_{mid}"] = sm
                if exp == "symf+cr+pers":
                    conf, v_sym = mesh_sym[mid][0]
                    out[f"postproc_{mid}"] = symmetry_propagation(sm, v_sym, conf,
                                                                    alpha=0.7)
    np.savez(PRED_CACHE, **out)
    print(f"\nSaved predictions to {PRED_CACHE}", flush=True)
    return out


# ---------------------------------------------------------------------------
def shade(tri, rgba):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(n, axis=1, keepdims=True); nl[nl == 0] = 1.0
    n /= nl
    light = np.array([-0.3, -0.4, 0.85]); light /= np.linalg.norm(light)
    inten = 0.55 + 0.45 * np.clip(n @ light, 0.0, 1.0)
    rgba[:, :3] *= inten[:, None]
    return rgba


def panel(ax, verts, tris, scalar, view, title=None, ylabel=None):
    elev, azim, roll = view
    tri = verts[tris]
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
    try:    ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError: ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11, color="#1f3a5f",
                     fontweight="bold", pad=2)
    if ylabel:
        ax.text2D(-0.02, 0.5, ylabel, transform=ax.transAxes, rotation=90,
                  fontsize=12, color="#1f3a5f", fontweight="bold",
                  ha="right", va="center")


def render(preds, sarma_preds):
    col_titles = ["Ground truth",
                  "Spiral Regressor (9-D)",
                  "SARMA (GNN)",
                  "Ours, full 24-D",
                  r"Ours, full $+$ post-proc"]
    fig = plt.figure(figsize=(15.0, 9.0))
    for ri, (mid, name) in enumerate(MESHES):
        verts, tris = load_mesh(mid, cfg.MESHES_DIR)
        gt = load_saliency(mid, cfg.DIST_DIR)
        view = VIEWS[mid]
        maps = [gt,
                preds[f"baseline_{mid}"],
                sarma_preds[f"sarma_{mid}"],
                preds[f"symf+cr+pers_{mid}"],
                preds[f"postproc_{mid}"]]
        for ci, sc in enumerate(maps):
            ax = fig.add_subplot(3, 5, ri*5 + ci + 1, projection="3d")
            title = col_titles[ci] if ri == 0 else None
            ylabel = name if ci == 0 else None
            panel(ax, verts, tris, sc, view, title=title, ylabel=ylabel)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.965, bottom=0.005,
                        wspace=0.02, hspace=0.04)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", transparent=False)
    print("wrote", OUT_PNG)


def main():
    if os.path.exists(PRED_CACHE):
        z = np.load(PRED_CACHE)
        preds = {k: z[k] for k in z.files}
        print(f"loaded cached predictions ({len(preds)} entries)", flush=True)
    else:
        preds = train_predict()
    sarma_path = os.path.join(_HERE, "cache", "_sarma_predictions.npz")
    assert os.path.exists(sarma_path), \
        f"SARMA predictions missing: run infer_sarma.py first ({sarma_path})"
    z = np.load(sarma_path)
    sarma_preds = {k: z[k] for k in z.files}
    print(f"loaded SARMA predictions ({len(sarma_preds)} entries)", flush=True)
    render(preds, sarma_preds)


if __name__ == "__main__":
    main()
