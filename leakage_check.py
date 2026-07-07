"""
CR leakage sanity check.

Question: the contextual-rank (CR) block gave a surprisingly large PLCC gain
(+0.10). Is that genuine signal or label leakage from the GT?

CR is, by construction, the per-mesh empirical CDF (rank) of each RAW feature
(R/GS/SS) — see src/features/context.py. It never sees the ground-truth
saliency. This script provides the supporting evidence:

  1. CR is a strictly monotonic transform of the raw feature
     -> Spearman(CR_col, baseline_col) should be ~1.0 (no new information source).
  2. CR correlates with GT *more linearly* than the min-max-normalised baseline
     -> mean |Pearson(CR_col, GT)| > mean |Pearson(baseline_col, GT)| would
        explain the PLCC gain (PLCC is a *linear* correlation; rank-encoding
        linearises the feature->saliency relationship). That is a legitimate
        conditioning effect, not leakage.

phi24 columns:  0-8 baseline [R,GS,SS]x3 ; 12-20 CR [CR_R,CR_GS,CR_SS]x3
CR col (12 + i) corresponds to baseline col (i)  for i in 0..8.
"""
import os, sys, glob
import numpy as np
from scipy.stats import pearsonr, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import config as cfg
from src.data.dataset import load_saliency

CACHE = "./cache"
NAMES = ["R1","GS1","SS1","R2","GS2","SS2","R3","GS3","SS3"]

def mesh_id_from_path(p):
    return int(os.path.basename(p).split("_")[0])

def main():
    files = sorted(glob.glob(os.path.join(CACHE, "*_phi24.npy")))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    files = files[:80]   # sample for speed

    n = len(NAMES)
    pear_base = [[] for _ in range(n)]
    pear_cr   = [[] for _ in range(n)]
    spear_mono = [[] for _ in range(n)]
    used = 0
    for f in files:
        mid = mesh_id_from_path(f)
        try:
            G = load_saliency(mid, cfg.DIST_DIR)
        except Exception:
            continue
        P = np.load(f)
        if len(P) != len(G):
            continue
        used += 1
        if G.std() < 1e-12:
            continue
        for i in range(n):
            base = P[:, i]        # min-max normalised baseline feature
            cr   = P[:, 12 + i]   # matching CR feature
            if base.std() > 1e-12:
                pear_base[i].append(abs(pearsonr(base, G)[0]))
                spear_mono[i].append(spearmanr(cr, base)[0])
            if cr.std() > 1e-12:
                pear_cr[i].append(abs(pearsonr(cr, G)[0]))

    print(f"Meshes used: {used}\n")
    print(f"{'feat':>5} | {'|PLCC|(base,GT)':>16} | {'|PLCC|(CR,GT)':>14} | "
          f"{'gain':>7} | {'Spearman(CR,base)':>18}")
    print("-" * 78)
    gains = []
    for i in range(n):
        b = np.mean(pear_base[i]); c = np.mean(pear_cr[i])
        sp = np.mean(spear_mono[i]); gains.append(c - b)
        print(f"{NAMES[i]:>5} | {b:>16.4f} | {c:>14.4f} | {c-b:>+7.4f} | {sp:>18.4f}")
    print("-" * 78)
    print(f"mean |PLCC| baseline->GT : {np.mean([np.mean(x) for x in pear_base]):.4f}")
    print(f"mean |PLCC| CR      ->GT : {np.mean([np.mean(x) for x in pear_cr]):.4f}")
    print(f"mean linear-corr gain    : {np.mean(gains):+.4f}")
    print(f"mean Spearman(CR,base)   : {np.mean([np.mean(x) for x in spear_mono]):.4f}  "
          f"(=1.0 => CR is a pure monotonic re-encoding of the raw feature)")

if __name__ == "__main__":
    main()
