# GSTF: Graph-Structural and Topological Features for 3D Mesh Saliency

This is the code for our S+SSPR 2026 paper, **"Graph-Structural and Topological
Features for 3D Mesh Saliency"**, by Zaineb Ibork and Akihiro Sugimoto (National
Institute of Informatics, Tokyo).

Mesh saliency is about predicting which parts of a 3D surface people look at. Most
methods describe each vertex with local, differential features, so they can't tell
where a point sits in the shape as a whole. GSTF adds three feature families on top of
a multi-scale spiral-patch descriptor. None of them needs training:

- **Symmetry consistency (SymF):** how much a vertex's spiral patch differs from the
  patch of its mirror vertex, weighted by how symmetric the whole mesh is.
- **Contextual rank (CR):** where each base feature of a vertex ranks within its own
  mesh (the per-mesh empirical CDF).
- **Topological persistence (Pers):** how long each curvature extremum survives in a
  persistence filtration, so that strong extrema stand out even when the curvature
  itself is small.

On the Schelling benchmark, GSTF raises PLCC from **0.540 to 0.680** and AUC from
**0.818 to 0.873** over the spiral-patch baseline. That is the highest PLCC among the
methods we compared, and the gain holds up when whole meshes are held out of training.

**Project page:** https://zaineb-ibork.github.io/GSTF-Mesh-Saliency/

The code in this repository reproduces the **conference paper**.

## Journal extension (under revision, coming soon)

We have extended this work into a journal paper, which is now under revision. In it we:

- evaluate the features under six named protocols (unseen meshes, unseen categories,
  training per category, decimated meshes, and the published split of a learned graph
  network), with paired statistical tests and error measures (MSE, MAE) as well as
  correlation;
- replace each feature with a simpler signal to find out where its gain comes from;
- compare with a learned graph neural network on its own test meshes, and also train
  that network with our features as input;
- refine the symmetry feature and the symmetry-plane estimation, re-implement the
  spiral-patch baseline exactly as published, test transfer to eye-tracking data
  (SAL3D), and measure the cost of processing a new mesh.

We'll release the paper and its code here once it is published.

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

We tested it with Python 3.12. `gudhi` computes the persistent homology.
`torch-geometric` is only needed for the optional graph-network baselines, so it is
commented out in `requirements.txt`.

## Dataset

This repository contains **code only**. The experiments use the **Schelling** dataset
(Chen et al., *Schelling points on 3D surface meshes*, SIGGRAPH 2012), which you can get
from its authors. The loader expects this layout:

```
SchellingData/
  Meshes_ply/     # 400 meshes, 20 categories
  Distributions/  # continuous saliency maps
  Points/         # discrete Schelling points
```

Then tell the code where your copy is:

```bash
export SCHELLING_ROOT=/path/to/SchellingData
```

The cross-dataset scripts (`run_crossdb*.py`, `prepare_sal3d*.py`) also need the
**SAL3D** dataset, which is likewise available from its authors.

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

# 6. Reproduce the figures and tables of the paper
python make_figures.py
python make_sota_comparison_figure.py
```

All hyperparameters (γ, ring depth, scales, seed, MLP widths) live in `config.py`
and are the ones used in the paper.

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

The code is released under the [MIT License](LICENSE). The Schelling and SAL3D
datasets belong to their authors; they are **not** included here and we do not
redistribute them.
