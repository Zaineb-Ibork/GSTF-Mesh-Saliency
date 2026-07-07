"""
Combined feature-relevance figure (analogue of Fig. 2 in Lezoray & Nouri): two
meshes (a human and a dolphin) rendered under every feature channel, so the reader
sees what each channel captures and that the behaviour generalises across
categories.

Rows: meshes (human, dolphin). Columns: ground truth, roughness, geometric saliency,
spectral saliency, symmetry (SymF), contextual rank, persistence (scale 1).
Display only: each map is percentile-normalised (2nd-98th) with a mild gamma.
Output: "S+SSPR 2026/fig_features.png"
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from src.data.dataset import load_mesh, load_saliency

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S+SSPR 2026")

# (mesh id, elev, azim, row label)
MESHES = [(1, 12, -75, "Human"), (221, 15, -70, "Dolphin")]
COLS = ["Ground truth", "Roughness", "Geometric", "Spectral",
        "Symmetry", "Contextual rank", "Persistence"]


def channels(P, gt):
    """Scale-1 maps in column order."""
    return [gt, P[:, 0], P[:, 1], P[:, 2], P[:, 9],
            P[:, 12:15].mean(axis=1), P[:, 21]]


def render(ax, verts, tris, scalar, elev, azim):
    s = scalar.astype(np.float64)
    lo, hi = np.percentile(s, [2, 98])
    s = np.clip((s - lo) / (hi - lo + 1e-12), 0.0, 1.0) ** 0.7
    tri = verts[tris]
    rgba = cm.jet(s[tris].mean(axis=1))
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(nrm, axis=1, keepdims=True); nl[nl == 0] = 1.0
    nrm = nrm / nl
    light = np.array([-0.3, -0.4, 0.85]); light = light / np.linalg.norm(light)
    inten = 0.55 + 0.45 * np.clip(nrm @ light, 0.0, 1.0)
    rgba[:, :3] *= inten[:, None]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, edgecolors="none",
                                         linewidths=0, antialiased=True))
    c = verts.mean(0); r = np.abs(verts - c).max()
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    try: ax.set_box_aspect((1, 1, 1), zoom=1.55)
    except TypeError:
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
    ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)


def main():
    ncol = len(COLS); nrow = len(MESHES)
    fig = plt.figure(figsize=(1.55 * ncol, 1.7 * nrow + 0.2))
    for r, (mid, elev, azim, label) in enumerate(MESHES):
        v, t = load_mesh(mid, cfg.MESHES_DIR)
        gt = load_saliency(mid, cfg.DIST_DIR)
        P = np.load(os.path.join("cache", f"{mid}_phi24.npy"))
        for c, sc in enumerate(channels(P, gt)):
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            render(ax, v, t, sc, elev, azim)
            if r == 0:
                ax.set_title(COLS[c], fontsize=8)
        fig.text(0.008, 0.71 - r * 0.44, label, rotation=90, va="center", fontsize=9)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.025, right=0.997,
                        top=0.92, bottom=0.01)
    out = os.path.join(OUT, "fig_features.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
