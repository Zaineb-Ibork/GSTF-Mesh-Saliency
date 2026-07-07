"""
Generate the two paper figures into the LaTeX directory:
  fig_qualitative.png   - GT vs baseline vs full-model saliency on example meshes
  fig_percategory.pdf   - per-category PLCC, baseline vs full model (pooled protocol)

Trains a pooled baseline (9-D) and full (24-D) model under the original
vertex-level 80/20 protocol (same regime as Table 1), runs full-mesh inference,
then renders heatmaps and the per-category breakdown.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from src.data.dataset import (load_mesh, load_saliency, load_schelling_points,
                               list_mesh_ids, mesh_id_to_category)
from src.model.mlp import SaliencyMLP
from src.metrics import plcc
from train_week2 import load_phi, select_columns

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S+SSPR 2026")
CACHE = "./cache"
# (mesh id, elev, azim) -- diverse, recognisable Schelling categories
EXAMPLES = [(1, 12, -75), (221, 15, -70), (61, 20, -60),
            (101, 20, -120), (181, 12, -75)]
CAT_NAMES = ["Human", "Dolphin", "Airplane", "Chair", "Hand"]
SEED = cfg.SEED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pooled(experiment, mesh_ids, epochs=cfg.EPOCHS, patience=50, bs=cfg.BATCH_SIZE):
    """Train one pooled model; return per-mesh prediction dict and per-mesh PLCC dict."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    need_enh = experiment != "baseline"

    all_X, all_Y, slices, gt = [], [], [], {}
    total = 0
    for mid in mesh_ids:
        X = select_columns(load_phi(mid, CACHE, need_enh), experiment).astype(np.float32)
        sal = load_saliency(mid, cfg.DIST_DIR)
        all_X.append(X); all_Y.append(sal); gt[mid] = sal
        slices.append((total, total + len(X), mid)); total += len(X)
    X_all = np.concatenate(all_X).astype(np.float32)
    Y_all = np.concatenate(all_Y).astype(np.float32)

    idx = rng.permutation(len(X_all)); ntr = int(cfg.TRAIN_SPLIT * len(X_all))
    model = SaliencyMLP(in_dim=X_all.shape[1], hidden=cfg.MLP_HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR); loss_fn = nn.MSELoss()
    Xg = torch.from_numpy(X_all[idx[:ntr]]).to(DEVICE)
    Yg = torch.from_numpy(Y_all[idx[:ntr]]).to(DEVICE); n = len(Xg)

    best = float("inf"); bad = 0; best_state = None
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n, device=DEVICE); el = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]; xb, yb = Xg[j], Yg[j]
            p = model(xb); l = loss_fn(p, yb)
            opt.zero_grad(); l.backward(); opt.step(); el = el + l.detach() * len(j)
        avg = (el / n).item()
        if avg < best - 1e-5:
            best = avg; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state: model.load_state_dict(best_state)

    preds, plccs = {}, {}
    model.eval()
    with torch.no_grad():
        for s, e, mid in slices:
            sm = model(torch.from_numpy(X_all[s:e]).to(DEVICE)).cpu().numpy()
            preds[mid] = sm; plccs[mid] = plcc(sm, gt[mid])
    return preds, plccs


def render(ax, verts, tris, scalar, title, elev=12, azim=-75):
    s = scalar.astype(np.float64)
    # Robust display normalisation (2nd-98th percentile clip + mild gamma) so sparse
    # outliers do not crush the map to one colour. Display-only; data unchanged.
    lo, hi = np.percentile(s, [2, 98])
    s = np.clip((s - lo) / (hi - lo + 1e-12), 0.0, 1.0) ** 0.7
    tri = verts[tris]
    rgba = cm.jet(s[tris].mean(axis=1))
    # Lambertian shading from face normals for a proper 3-D look
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(nrm, axis=1, keepdims=True); nl[nl == 0] = 1.0
    nrm = nrm / nl
    light = np.array([-0.3, -0.4, 0.85]); light = light / np.linalg.norm(light)
    inten = 0.55 + 0.45 * np.clip(nrm @ light, 0.0, 1.0)
    rgba[:, :3] *= inten[:, None]
    pc = Poly3DCollection(tri, facecolors=rgba, edgecolors="none",
                          linewidths=0, antialiased=True)
    ax.add_collection3d(pc)
    c = verts.mean(0); r = np.abs(verts - c).max()
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    try: ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError:
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
    ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=10, pad=-4)


def main():
    print(f"Device: {DEVICE}", flush=True)
    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)

    import pickle
    pred_cache = os.path.join("cache", "_fig_predictions.pkl")
    if os.path.exists(pred_cache) and "--retrain" not in sys.argv:
        with open(pred_cache, "rb") as fh:
            pb, plb, pf, plf = pickle.load(fh)
        print("Loaded cached predictions (pass --retrain to retrain).", flush=True)
    else:
        print("Training baseline (9-D) ...", flush=True)
        pb, plb = train_pooled("baseline", mesh_ids)
        print("Training full (24-D) ...", flush=True)
        pf, plf = train_pooled("symf+cr+pers", mesh_ids)
        with open(pred_cache, "wb") as fh:
            pickle.dump((pb, plb, pf, plf), fh)
        print("Saved predictions to cache for fast re-rendering.", flush=True)
    print(f"  mean PLCC  baseline={np.mean(list(plb.values())):.4f}  "
          f"full={np.mean(list(plf.values())):.4f}", flush=True)

    # ---- Figure 1: qualitative -- ground truth vs our prediction, across categories ----
    # Both columns use the SAME relative-saliency scale (each map percentile-normalised
    # to [0,1] with the same parameters), so colours are directly comparable; a shared
    # colour bar is drawn as the reference.
    print("Rendering qualitative figure ...", flush=True)
    from matplotlib.colors import Normalize
    n = len(EXAMPLES)
    fig = plt.figure(figsize=(1.9 * n, 4.4))           # wide and short
    for col, (mid, elev, azim) in enumerate(EXAMPLES):
        v, t = load_mesh(mid, cfg.MESHES_DIR)
        gt = load_saliency(mid, cfg.DIST_DIR)
        axg = fig.add_subplot(2, n, col + 1, projection="3d")
        render(axg, v, t, gt, "", elev=elev, azim=azim)
        axg.set_title(CAT_NAMES[col], fontsize=10)
        axp = fig.add_subplot(2, n, n + col + 1, projection="3d")
        render(axp, v, t, pf[mid], "", elev=elev, azim=azim)
        axp.set_title(f"PLCC {plf[mid]:.2f}", fontsize=9)
    fig.text(0.012, 0.71, "Ground truth", rotation=90, va="center", fontsize=9)
    fig.text(0.012, 0.30, "Our prediction", rotation=90, va="center", fontsize=9)
    fig.subplots_adjust(wspace=0.0, hspace=0.18, left=0.045, right=0.995,
                        top=0.90, bottom=0.12)
    cax = fig.add_axes([0.37, 0.045, 0.26, 0.022])
    cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap="jet"),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.set_label("relative saliency (shared scale)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(OUT, "fig_qualitative.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: per-category PLCC breakdown ----
    print("Rendering per-category breakdown ...", flush=True)
    ncat = cfg.N_CATEGORIES
    base_cat = np.array([np.mean([plb[m] for m in mesh_ids
                                  if mesh_id_to_category(m) == c]) for c in range(ncat)])
    full_cat = np.array([np.mean([plf[m] for m in mesh_ids
                                  if mesh_id_to_category(m) == c]) for c in range(ncat)])
    x = np.arange(ncat); w = 0.4
    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.bar(x - w/2, base_cat, w, label="Baseline (9-D)", color="#bbbbbb")
    ax2.bar(x + w/2, full_cat, w, label="Full (24-D)", color="#1f77b4")
    ax2.set_xlabel("Schelling category index"); ax2.set_ylabel("PLCC")
    ax2.set_xticks(x); ax2.set_xticklabels([str(c) for c in range(ncat)], fontsize=7)
    ax2.axhline(np.mean(base_cat), ls="--", lw=0.8, color="#888888")
    ax2.axhline(np.mean(full_cat), ls="--", lw=0.8, color="#1f77b4")
    ax2.legend(loc="lower right", fontsize=8); ax2.set_ylim(0, 1)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "fig_percategory.pdf"), bbox_inches="tight")
    plt.close(fig2)
    print("Done. Figures written to", OUT, flush=True)


if __name__ == "__main__":
    main()
