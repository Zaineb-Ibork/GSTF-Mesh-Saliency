"""
Reproduce a SARMA-style FeaStConv GNN (Lezoray, Ibork, Nouri, Charrier, VCIP
2025) and predict saliency on the three meshes used in Figure 4.

Architecture (paper notation):
  Input features per vertex: (x, y, z, kappa)  -- 4-D
  3 x FeaStConv (hidden 64) + ReLU + residual projection
  Final FeaStConv -> 1-D output, sigmoid
  Loss: MSE against the Schelling Gaussian-smoothed ground truth
  Optimiser: Adam, lr 1e-3
  Training: mesh-level batches, 80/20 mesh split, 300 epochs with early stop
  (the paper trains 1400 but converges much earlier in practice)

Predictions for meshes 161, 221, 101 are saved to cache for use by
make_sota_comparison_figure.py.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import config as cfg
from src.data.dataset       import load_mesh, load_saliency, list_mesh_ids
from src.features.geometry  import mean_curvature

from torch_geometric.nn   import FeaStConv
from torch_geometric.data import Data, Batch

CACHE_PRED = os.path.join(_HERE, "cache", "_sarma_predictions.npz")
CURV_CACHE = os.path.join(_HERE, "cache", "_curvature")
os.makedirs(CURV_CACHE, exist_ok=True)
TEST_MESHES = [161, 221, 101]
HIDDEN   = 64
EPOCHS   = 300
PATIENCE = 30
LR       = 1e-3
SEED     = cfg.SEED


# ---------------------------------------------------------------------------
class SARMA(nn.Module):
    def __init__(self, in_dim=4, hidden=HIDDEN, heads=4):
        super().__init__()
        self.conv1 = FeaStConv(in_dim, hidden, heads=heads)
        self.res1  = nn.Linear(in_dim, hidden)
        self.conv2 = FeaStConv(hidden, hidden, heads=heads)
        self.conv3 = FeaStConv(hidden, hidden, heads=heads)
        self.head  = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index) + self.res1(x))
        h2 = F.relu(self.conv2(h1, edge_index) + h1)
        h3 = self.conv3(h2, edge_index) + h2
        return torch.sigmoid(self.head(h3)).squeeze(-1)


def edges_from_tris(tris):
    """Undirected edge list (2 x E) from triangle faces."""
    e0 = tris[:, [0, 1]]; e1 = tris[:, [1, 2]]; e2 = tris[:, [2, 0]]
    E = np.vstack([e0, e1, e2])
    E = np.vstack([E, E[:, ::-1]])   # symmetric
    return torch.from_numpy(E.T.astype(np.int64))


def get_curvature(mid, verts, tris):
    path = os.path.join(CURV_CACHE, f"{mid}.npy")
    if os.path.exists(path):
        return np.load(path)
    k = mean_curvature(verts.astype(np.float64), tris.astype(np.int32))
    np.save(path, k.astype(np.float32))
    return k.astype(np.float32)


def build_data(mid):
    verts, tris = load_mesh(mid, cfg.MESHES_DIR)
    kappa = get_curvature(mid, verts, tris)
    # Normalisations as the paper does: centre vertices to origin and scale to
    # unit sphere; standardise curvature (zero mean, unit std) per mesh.
    v = verts - verts.mean(axis=0)
    v = v / max(np.linalg.norm(v, axis=1).max(), 1e-12)
    k = (kappa - kappa.mean()) / (kappa.std() + 1e-12)
    x = np.concatenate([v, k[:, None]], axis=1).astype(np.float32)
    y = load_saliency(mid, cfg.DIST_DIR).astype(np.float32)
    return Data(x=torch.from_numpy(x),
                edge_index=edges_from_tris(tris),
                y=torch.from_numpy(y),
                mesh_id=mid,
                num_nodes=len(v))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    mesh_ids = list_mesh_ids(cfg.MESHES_DIR)
    print(f"Preparing {len(mesh_ids)} mesh graphs (curvature cached) ...",
          flush=True)
    data_all = []
    t0 = time.time()
    for i, mid in enumerate(mesh_ids):
        data_all.append(build_data(mid))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(mesh_ids)}] {time.time()-t0:.0f}s",
                  flush=True)
    print(f"All graphs ready in {time.time()-t0:.0f}s\n", flush=True)

    # 80/20 mesh split (mesh-level since it's a GNN). Make sure the three test
    # visualisation meshes are in the training set so the columns in Fig. 4
    # are comparable: the other models also have these meshes in training.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(mesh_ids))
    n_train = int(cfg.TRAIN_SPLIT * len(mesh_ids))
    train_idx = set(perm[:n_train].tolist())
    train_idx.update(mesh_ids.index(t) for t in TEST_MESHES)
    train_set = [data_all[i] for i in train_idx]
    val_idx   = [i for i in range(len(mesh_ids)) if i not in train_idx]
    val_set   = [data_all[i] for i in val_idx]
    print(f"Train meshes: {len(train_set)}  /  Val meshes: {len(val_set)}\n",
          flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SARMA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best, bad, best_state = float("inf"), 0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        rng2 = np.random.default_rng(SEED + ep)
        order = rng2.permutation(len(train_set))
        epoch_loss = 0.0
        n_v = 0
        # Mesh-level minibatches (batch_size = 8 meshes per step for speed)
        BATCH = 8
        for k in range(0, len(order), BATCH):
            batch = Batch.from_data_list([train_set[i] for i in order[k:k+BATCH]])
            batch = batch.to(device)
            pred  = model(batch.x, batch.edge_index)
            loss  = loss_fn(pred, batch.y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss = epoch_loss + loss.item() * batch.num_nodes
            n_v += batch.num_nodes
        avg = epoch_loss / max(n_v, 1)

        # Validation loss
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for k in range(0, len(val_set), BATCH):
                batch = Batch.from_data_list(val_set[k:k+BATCH]).to(device)
                pred = model(batch.x, batch.edge_index)
                vl += loss_fn(pred, batch.y).item() * batch.num_nodes
                vn += batch.num_nodes
            val_avg = vl / max(vn, 1)
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{EPOCHS}  train={avg:.5f}  val={val_avg:.5f}  "
                  f"best_val={best:.5f}  bad={bad}/{PATIENCE}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if val_avg < best - 1e-5:
            best = val_avg; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early-stop at ep {ep}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference on the three Fig-4 meshes
    model.eval()
    preds = {}
    with torch.no_grad():
        for mid in TEST_MESHES:
            d = build_data(mid).to(device)
            preds[f"sarma_{mid}"] = model(d.x, d.edge_index).cpu().numpy()
            print(f"  predicted on mesh {mid}: shape {preds[f'sarma_{mid}'].shape}",
                  flush=True)
    np.savez(CACHE_PRED, **preds)
    print(f"\nSaved SARMA predictions to {CACHE_PRED}")


if __name__ == "__main__":
    main()
