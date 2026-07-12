# GSTF : Graph-Structural and Topological Features for 3D Mesh Saliency

Code for the paper **“Graph-Structural and Topological Features for 3D Mesh
Saliency”** (Zaineb Ibork and Akihiro Sugimoto, National Institute of Informatics,
Tokyo), *S+SSPR 2026*.

Mesh saliency predicts which vertices of a 3D surface attract human attention. Most
methods use purely local, differential descriptors and miss where a point sits within
the global shape. **GSTF** adds three *learning-free* feature families on top of a
multi-scale spiral-patch descriptor:

- **Symmetry consistency (SymF)** : distance between a vertex's spiral patch and that of
  its bilateral mirror, weighted by a global symmetry-confidence term.
- **Contextual rank (CR)** : the per-mesh empirical CDF of each base feature, giving the
  model a mesh-global sense of how a vertex ranks.
- **Topological persistence (Pers)** : the persistence of each vertex under a curvature
  filtration, capturing topologically significant extrema regardless of local magnitude.

On the Schelling benchmark, GSTF raises PLCC from **0.540 → 0.680** and AUC from
**0.818 → 0.873** over the spiral-patch baseline ; the highest PLCC among compared
methods — and the gain survives a strict, leakage-free held-out-mesh test.

📄 **Project page:** https://zaineb-ibork.github.io/GSTF-Mesh-Saliency/

## Repository layout

```
src/
  features/     spiral, geometry, symmetry (SymF), context (CR), persistence (Pers), extract
  data/         dataset loading, decimation
  model/        the MLP regressor
  metrics.py    PLCC / AUC
precompute*.py  cache per-vertex features (base + GSTF) at all scales
train.py        train the MLP on cached features
run_schelling_cv*.py   k-fold cross-validation (pooled / held-out / per-category)
eval_*.py       held-out-mesh and per-category evaluation
compute_significance.py   paired significance tests (Wilcoxon / t-test)
make_*figure*.py, make_*table*.py   paper figures and tables
config.py       paths and hyperparameters
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12. `gudhi` provides the persistent-homology backend;
`torch-geometric` is only needed for the optional graph-network baselines (commented
out in `requirements.txt`).

## Dataset

This repository contains **code only**. The experiments use the third-party
**Schelling** dataset (Chen et al., *Schelling points on 3D surface meshes*, SIGGRAPH
2012), obtainable from the original authors. The loader expects the standard layout:

```
SchellingData/
  Meshes_ply/     # 400 meshes, 20 categories
  Distributions/  # continuous saliency maps
  Points/         # discrete Schelling points
```

Point the code at your copy via an environment variable:

```bash
export SCHELLING_ROOT=/path/to/SchellingData
```

The cross-dataset scripts (`run_crossdb*.py`, `prepare_sal3d*.py`) additionally use
the **SAL3D** dataset, obtainable from its authors.

## Usage

```bash
# 1. Precompute and cache per-vertex features (base + GSTF) at all scales
python precompute_enhanced.py

# 2. Train the MLP under the original pooled-vertex protocol
python train.py

# 3. Cross-validation (pooled, held-out-mesh, per-category)
python run_schelling_cv.py
python run_schelling_cv_phase2.py

# 4. Held-out-mesh and per-category evaluation
python eval_heldout_mesh.py
python eval_per_category.py

# 5. Statistical significance (Wilcoxon / paired-t, Holm-corrected)
python compute_significance.py

# 6. Reproduce paper figures/tables
python make_figures.py
python make_sota_comparison_figure.py
```

All hyperparameters (γ, ring depth, scales, seed, MLP widths) are in `config.py` and
match the paper.

## Citation

```bibtex
@inproceedings{ibork2026gstf,
  title     = {Graph-Structural and Topological Features for 3D Mesh Saliency},
  author    = {Ibork, Zaineb and Sugimoto, Akihiro},
  booktitle = {Structural, Syntactic, and Statistical Pattern Recognition (S+SSPR)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE). The Schelling and SAL3D datasets are the
property of their respective authors and are **not** included or redistributed here.
