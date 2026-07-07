"""
Figures for the global/detailed presentation (presentation_global/img/):
  - dataset_categories.png : one representative mesh per Schelling category (4x5),
    captioned with category labels.
  - category_variance.png  : all 20 meshes of a single category, to show the
    within-category shape variance that explains the per-category PLCC spread.

Reuses the shaded-render style of make_slide_images.py.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from src.data.dataset import load_mesh

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "presentation_global", "img")
os.makedirs(OUT, exist_ok=True)

# Category labels, cat 0..19 -> representative mesh id = 20*cat+1.
# Identified visually from the representative renders.
CATEGORY_NAMES = [
    "Human", "Cup", "Glasses", "Airplane", "Ant",
    "Chair", "Octopus", "Table", "Teddy", "Hand",
    "Plier", "Fish", "Bird", "Spring", "Armadillo",
    "Bust", "Mech", "Bearing", "Vase", "FourLeg",
]


def _shade(tri, rgba):
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(nrm, axis=1, keepdims=True); nl[nl == 0] = 1.0
    nrm = nrm / nl
    light = np.array([-0.3, -0.4, 0.85]); light = light / np.linalg.norm(light)
    inten = 0.55 + 0.45 * np.clip(nrm @ light, 0.0, 1.0)
    rgba[:, :3] *= inten[:, None]
    return rgba


def _draw(ax, mid, color=(0.5, 0.62, 0.85), elev=12, azim=-75):
    vv, tt = load_mesh(mid, cfg.MESHES_DIR)
    tri = vv[tt]
    rgba = np.tile(np.array([*color, 1.0]), (len(tri), 1))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=_shade(tri, rgba),
                                         edgecolors="none", linewidths=0))
    c = vv.mean(0); r = np.abs(vv - c).max()
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    try: ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError: ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)


def dataset_categories():
    fig = plt.figure(figsize=(11, 5))
    for c in range(cfg.N_CATEGORIES):
        ax = fig.add_subplot(4, 5, c + 1, projection="3d")
        _draw(ax, 20 * c + 1)
        ax.set_title(f"{c+1}. {CATEGORY_NAMES[c]}", fontsize=9, pad=-2)
    fig.subplots_adjust(wspace=0.02, hspace=0.12, left=0.01, right=0.99,
                        top=0.96, bottom=0.01)
    p = os.path.join(OUT, "dataset_categories.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def category_variance(cat, title):
    ids = [20 * cat + 1 + i for i in range(cfg.MESHES_PER_CAT)]
    fig = plt.figure(figsize=(11, 5))
    for i, mid in enumerate(ids):
        ax = fig.add_subplot(4, 5, i + 1, projection="3d")
        _draw(ax, mid, color=(0.85, 0.6, 0.45))
        ax.set_title(f"#{mid}", fontsize=8, pad=-2)
    # No in-figure title: the slide supplies the caption (avoids duplication).
    fig.subplots_adjust(wspace=0.02, hspace=0.1, left=0.01, right=0.99,
                        top=0.98, bottom=0.01)
    p = os.path.join(OUT, "category_variance.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "cats"):
        dataset_categories()
    if which in ("all", "var"):
        category_variance(2, f"Category '{CATEGORY_NAMES[2]}' (20 meshes): within-category shape variance")
