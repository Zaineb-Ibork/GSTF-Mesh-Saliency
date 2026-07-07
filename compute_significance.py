#!/usr/bin/env python3
"""
Statistical significance of each variant vs the baseline, computed from already
saved results (no re-training needed). Reports Wilcoxon signed-rank and paired
t-test side by side, with effect size and a bootstrap 95% CI on the mean diff.

Two paired sources are used:
  * sarma_ours/results/sarma_ours_results.json  -> per-mesh PLCC/AUC (n=80),
    same mesh order across all variants  -> proper paired test.
  * schelling_cv_phase2_results.json (percat_K10) -> per-fold PLCC/AUC
    (20 categories x 10 folds = 200 values), aligned across variants.

For each comparison the paired unit is identical across the two variants, so the
pairing is valid. p-values are Holm-corrected across the 8 variant-vs-baseline
comparisons within each (source, metric) family.
"""
import json
import numpy as np
from scipy import stats

np.random.seed(0)


def cohen_dz(diff):
    """Effect size for paired samples: mean(diff)/std(diff)."""
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def cliffs_delta(a, b):
    """Non-parametric effect size; >0 means a tends to exceed b."""
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


def bootstrap_ci(diff, n=10000, alpha=0.05):
    idx = np.random.randint(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def paired_tests(base, var):
    """base, var: 1-D arrays of paired metric values (var - base = improvement)."""
    base = np.asarray(base, float)
    var = np.asarray(var, float)
    diff = var - base
    # Wilcoxon needs at least one non-zero difference
    try:
        w_stat, w_p = stats.wilcoxon(var, base)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0
    t_stat, t_p = stats.ttest_rel(var, base)
    lo, hi = bootstrap_ci(diff)
    return {
        "n": len(diff),
        "mean_base": float(base.mean()),
        "mean_var": float(var.mean()),
        "mean_diff": float(diff.mean()),
        "ci95": (lo, hi),
        "wilcoxon_p": float(w_p),
        "ttest_p": float(t_p),
        "cohen_dz": cohen_dz(diff),
        "cliffs_delta": cliffs_delta(var, base),
    }


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def run_family(name, units):
    """units: dict variant -> 1-D array. 'baseline' is the reference."""
    variants = [v for v in units if v != "baseline"]
    base = units["baseline"]
    rows = []
    for v in variants:
        r = paired_tests(base, units[v])
        r["variant"] = v
        rows.append(r)
    w_adj = holm([r["wilcoxon_p"] for r in rows])
    t_adj = holm([r["ttest_p"] for r in rows])
    for r, wa, ta in zip(rows, w_adj, t_adj):
        r["wilcoxon_p_holm"] = float(wa)
        r["ttest_p_holm"] = float(ta)

    print(f"\n{'='*100}\n{name}   (baseline mean = {base.mean():.4f}, n = {len(base)} paired units)\n{'='*100}")
    hdr = (f"{'variant':<16}{'mean':>8}{'Δ':>9}{'95% CI':>20}"
           f"{'Wilcoxon':>13}{'(Holm)':>9}{'t-test':>11}{'(Holm)':>9}{'dz':>7}{'δ':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
        print(f"{r['variant']:<16}{r['mean_var']:>8.4f}{r['mean_diff']:>+9.4f}{ci:>20}"
              f"{r['wilcoxon_p']:>10.2e}{stars(r['wilcoxon_p_holm']):>4}"
              f"{r['wilcoxon_p_holm']:>9.2e}"
              f"{r['ttest_p']:>11.2e}{r['ttest_p_holm']:>9.2e}"
              f"{r['cohen_dz']:>7.2f}{r['cliffs_delta']:>7.2f}")
    return rows


def _load(path):
    import os
    return json.load(open(path)) if os.path.exists(path) else None


def _family_from_block(block, metric):
    """block: dict variant -> result dict that may carry PLCC_list/AUC_list.
    Returns variant->array dict if every variant has aligned lists, else None."""
    key = f"{metric}_list"
    if "baseline" not in block:
        return None
    units = {}
    n = None
    for v, r in block.items():
        if not isinstance(r, dict) or key not in r:
            return None
        arr = np.asarray(r[key], float)
        if n is None:
            n = len(arr)
        elif len(arr) != n:
            return None          # not aligned -> can't pair
        units[v] = arr
    return units if n and n > 1 else None


def run_optional_sources(out):
    """Add the three ablation files once they've been re-run with per-unit
    logging. Each is silently skipped while it still holds only mean+/-std."""
    # symf_apport: two protocols, per-mesh lists from evaluate_dataset
    d = _load("symf_apport_results.json")
    if d:
        for proto in ("pooled", "held"):
            for metric in ("PLCC", "AUC"):
                units = _family_from_block(d.get(proto, {}), metric)
                if units:
                    out[f"symf_apport_{proto}_{metric}"] = run_family(
                        f"SymF apport ({proto}) per-mesh  —  {metric}", units)

    # percat_apport: per-category lists, under each split
    d = _load("percat_apport_results.json")
    if d:
        for split, block in d.items():
            for metric in ("PLCC", "AUC"):
                units = _family_from_block(block, metric)
                if units:
                    out[f"percat_apport_{split.replace('/','-')}_{metric}"] = \
                        run_family(f"Per-category apport ({split}) —  {metric}", units)

    # crossdb_8020: per-mesh lists, under each transfer pair
    d = _load("crossdb_8020_results.json")
    if d:
        for tag, block in d.items():
            for metric in ("PLCC", "AUC"):
                units = _family_from_block(block, metric)
                if units:
                    out[f"crossdb_{tag}_{metric}"] = run_family(
                        f"Cross-DB {tag} per-mesh  —  {metric}", units)


def main():
    out = {}

    # ---- Source 1: SOTA comparison set, per-mesh (n=80) -------------------
    d = json.load(open("sarma_ours/results/sarma_ours_results.json"))
    order = [m["mesh"] for m in d["baseline"]["per_mesh"]]
    for metric in ("PLCC", "AUC"):
        units = {}
        for v in d:
            by_mesh = {m["mesh"]: m[metric] for m in d[v]["per_mesh"]}
            units[v] = np.array([by_mesh[mid] for mid in order], float)
        out[f"sarma_permesh_{metric}"] = run_family(
            f"Per-mesh (SAL3D->Schelling SOTA protocol)  —  {metric}", units)

    # ---- Source 2: Schelling per-category CV, per-fold (n=200) ------------
    d2 = json.load(open("schelling_cv_phase2_results.json"))["percat_K10"]
    for metric, key in (("PLCC", "fold_PLCC"), ("AUC", "fold_AUC")):
        units = {}
        for v in d2:
            vals = []
            for cat in d2[v]["per_category"]:
                vals.extend(cat[key])
            units[v] = np.array(vals, float)
        out[f"schelling_perfold_{metric}"] = run_family(
            f"Per-fold (Schelling per-category 10-fold CV)  —  {metric}", units)

    # ---- Optional sources: only present after re-running with per-unit
    #      logging (PLCC_list / AUC_list / per-category lists). Skipped silently
    #      if a file still holds only mean+/-std. -----------------------------
    run_optional_sources(out)

    json.dump(out, open("significance_results.json", "w"), indent=2)
    print("\nSaved -> significance_results.json")
    print("\nLegend: Δ = mean(variant) - mean(baseline);  dz = Cohen's dz (paired);")
    print("        δ = Cliff's delta;  stars on Holm-corrected Wilcoxon "
          "(*** p<.001, ** p<.01, * p<.05, ns).")


if __name__ == "__main__":
    main()
