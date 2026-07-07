"""Draw a clean MLP architecture diagram -> presentation/img/mlp.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presentation", "img", "mlp.png")
NAVY = "#1f3a5f"; ACC = "#2a6fb0"; GREY = "#888888"

layers = [("$\\Phi(v)$", "9–24", 5), ("hidden", "256", 7),
          ("hidden", "100", 6), ("saliency", "1", 1)]
xs = [0, 1, 1, 1]
X = np.cumsum([0, 2.2, 2.2, 2.2])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
node_pos = []
for li, (name, size, ndraw) in enumerate(layers):
    ys = np.linspace(-1.4, 1.4, ndraw) if ndraw > 1 else [0.0]
    col = ACC if 0 < li < 3 else NAVY
    pos = []
    for y in ys:
        c = Circle((X[li], y), 0.13, color=col, zorder=3)
        ax.add_patch(c); pos.append((X[li], y))
    node_pos.append(pos)
    # ellipsis to suggest more units
    if ndraw > 1:
        ax.text(X[li], 1.72, "$\\vdots$", ha="center", va="bottom", fontsize=14, color=GREY)
    ax.text(X[li], -1.95, name, ha="center", va="top", fontsize=13,
            color=NAVY, fontweight="bold")
    ax.text(X[li], -2.28, "" if name == "hidden" else f"dim {size}",
            ha="center", va="top", fontsize=10, color=GREY)
    ax.text(X[li], 1.95, size if name == "hidden" else "", ha="center", fontsize=12,
            color=ACC, fontweight="bold")

# edges
for a, b in zip(node_pos[:-1], node_pos[1:]):
    for (x1, y1) in a:
        for (x2, y2) in b:
            ax.plot([x1, x2], [y1, y2], color=GREY, lw=0.35, alpha=0.45, zorder=1)

ax.text(np.mean(X[1:3]), 2.45, "ReLU activations", ha="center", fontsize=11,
        color=ACC, style="italic")
ax.text(X[3], 0.45, "sigmoid", ha="center", fontsize=10, color=ACC, style="italic")
ax.set_xlim(-0.6, X[-1] + 0.9); ax.set_ylim(-2.75, 2.85)
ax.axis("off"); ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", transparent=True)
print("wrote", OUT)
