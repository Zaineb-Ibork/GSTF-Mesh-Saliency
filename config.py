import os

# Path to the Schelling dataset root. Override with the SCHELLING_ROOT environment
# variable, e.g.  export SCHELLING_ROOT=/path/to/SchellingData
# (dataset is third-party and not redistributed here; see README for the link).
DATA_ROOT = os.environ.get(
    "SCHELLING_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SchellingData"),
)
MESHES_DIR  = os.path.join(DATA_ROOT, "Meshes_ply")
DIST_DIR    = os.path.join(DATA_ROOT, "Distributions")
POINTS_DIR  = os.path.join(DATA_ROOT, "Points")

# Mesh IDs: 1–400, 20 per category (cat c → IDs 20c+1 … 20c+20)
N_MESHES        = 400
N_CATEGORIES    = 20
MESHES_PER_CAT  = 20

# Feature hyperparameters (must match the paper exactly)
GAMMA   = 3   # hop neighbourhood for roughness / gradient
K_RINGS = 2   # spiral patch depth
N_SCALES = 3  # number of decimation levels

# Decimation targets (fraction of original vertices)
DECIM_RATIOS = [1.0, 0.5, 0.25]

# Training
TRAIN_SPLIT = 0.80
SEED        = 42
BATCH_SIZE  = 4096
LR          = 1e-3
EPOCHS      = 500
MLP_HIDDEN  = [256, 100]   # layer widths before the final 1-unit output

# Gaussian saliency smoothing (paper: σ = 0.01 * R, applied to ground truth)
GT_SIGMA_SCALE = 0.01  # σ = GT_SIGMA_SCALE * mesh_radius

# Feature dimensionality: 3 features × 3 scales = 9
FEAT_DIM = 9
