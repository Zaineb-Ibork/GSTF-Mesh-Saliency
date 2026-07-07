"""
Slide figure: improvement of each variant over the baseline, with bootstrap 95%
CI error bars and Holm-corrected Wilcoxon significance stars. Reads
significance_results.json (produced by compute_significance.py).

Two panels (PLCC, AUC) for the per-mesh protocol (n=80, the most rigorous paired
comparison). Output: presentation_global/img/significance_bars.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "presentation_global", "img")
os.makedirs(OUT, exist_ok=True)

LABELS = {
    "symf": "+SymF", "cr": "+CR", "pers": "+Pers", "symf+cr": "+SymF+CR",
    "symf+pers": "+SymF+Pers", "cr+pers": "+CR+Pers",
    "symf+cr+pers": "+SymF+CR+Pers", "full+postproc": "Full+post-proc",
}
ORDER = list(LABELS.keys())


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def panel(ax, rows, title):
    by = {r["variant"]: r for r in rows}
    ys = np.arange(len(ORDER))[::-1]
    for y, key in zip(ys, ORDER):
        r = by[key]
        d = r["mean_diff"]
        lo, hi = r["ci95"]
        sig = r["wilcoxon_p_holm"] < 0.05
        color = ("#2a8f4e" if d > 0 else "#c0392b") if sig else "#9aa0a6"
        ax.barh(y, d, color=color, height=0.62, zorder=2)
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="none",
                    ecolor="#333", elinewidth=1.1, capsize=3, zorder=3)
        s = stars(r["wilcoxon_p_holm"])
        off = 0.002 if d >= 0 else -0.002
        ax.text(hi + 0.002 if d >= 0 else lo - 0.002, y, s,
                va="center", ha="left" if d >= 0 else "right",
                fontsize=9, color=color, fontweight="bold")
    ax.axvline(0, color="#000", lw=0.8, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([LABELS[k] for k in ORDER], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$\Delta$ vs baseline", fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.5)


def main():
    sig = json.load(open(os.path.join(HERE, "significance_results.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panel(axes[0], sig["sarma_permesh_PLCC"], "PLCC improvement (per-mesh, n=80)")
    panel(axes[1], sig["sarma_permesh_AUC"], "AUC improvement (per-mesh, n=80)")
    fig.suptitle("Each variant vs. baseline: paired improvement with 95% CI  "
                 "(green = significant gain, grey = n.s.; Holm-corrected Wilcoxon)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "significance_bars.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
