"""
Render images for the conference slides into presentation/img/:
  - one standalone heatmap per feature on the human mesh (gt, roughness, geometric,
    spectral, symf, cr, pers)
  - a plain shaded view of the human and dolphin
  - a 4x5 dataset montage (one mesh per Schelling category)
Display normalisation matches the paper (2nd-98th percentile + mild gamma).
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presentation", "img")
os.makedirs(OUT, exist_ok=True)


def _shade(tri, rgba):
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(nrm, axis=1, keepdims=True); nl[nl == 0] = 1.0
    nrm = nrm / nl
    light = np.array([-0.3, -0.4, 0.85]); light = light / np.linalg.norm(light)
    inten = 0.55 + 0.45 * np.clip(nrm @ light, 0.0, 1.0)
    rgba[:, :3] *= inten[:, None]
    return rgba


def render_single(verts, tris, scalar, elev, azim, fname, plain=False, roll=0):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    tri = verts[tris]
    if plain:
        rgba = np.tile(np.array([0.45, 0.6, 0.85, 1.0]), (len(tri), 1))
    else:
        s = scalar.astype(np.float64)
        lo, hi = np.percentile(s, [2, 98])
        s = np.clip((s - lo) / (hi - lo + 1e-12), 0.0, 1.0) ** 0.7
        rgba = cm.jet(s[tris].mean(axis=1))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=_shade(tri, rgba),
                                         edgecolors="none", linewidths=0, antialiased=True))
    c = verts.mean(0); r = np.abs(verts - c).max()
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    try: ax.set_box_aspect((1, 1, 1), zoom=1.6)
    except TypeError: ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(fname, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", os.path.basename(fname))


def main():
    # Ground truth on mesh 161 (a teddy bear) for the motivation slide;
    # the FEATURE maps on the dolphin.
    vh, th = load_mesh(161, cfg.MESHES_DIR)
    render_single(vh, th, load_saliency(161, cfg.DIST_DIR), 80, -90,
                  os.path.join(OUT, "feat_gt.png"))
    render_single(vh, th, None, 80, -90, os.path.join(OUT, "human_plain.png"), plain=True)
    vd, td = load_mesh(221, cfg.MESHES_DIR)
    Pd = np.load(os.path.join("cache", "221_phi24.npy"))
    dfeats = {"roughness": Pd[:, 0], "geometric": Pd[:, 1], "spectral": Pd[:, 2],
              "symf": Pd[:, 9], "cr": Pd[:, 12:15].mean(axis=1), "pers": Pd[:, 21]}
    for name, sc in dfeats.items():
        render_single(vd, td, sc, 15, -70, os.path.join(OUT, f"feat_{name}.png"))

    # dataset montage
    ids = [20 * c + 1 for c in range(cfg.N_CATEGORIES)]
    fig = plt.figure(figsize=(10, 4))
    for i, mid in enumerate(ids):
        vv, tt = load_mesh(mid, cfg.MESHES_DIR)
        ax = fig.add_subplot(4, 5, i + 1, projection="3d")
        tri = vv[tt]
        rgba = np.tile(np.array([0.5, 0.62, 0.85, 1.0]), (len(tri), 1))
        ax.add_collection3d(Poly3DCollection(tri, facecolors=_shade(tri, rgba),
                                             edgecolors="none", linewidths=0))
        c = vv.mean(0); r = np.abs(vv - c).max()
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        try: ax.set_box_aspect((1, 1, 1), zoom=1.5)
        except TypeError: ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off(); ax.view_init(elev=12, azim=-75)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(OUT, "dataset_montage.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote dataset_montage.png")


if __name__ == "__main__":
    main()
