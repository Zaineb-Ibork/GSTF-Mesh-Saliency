"""Run the user's trained SARMA model (best_model_FeastConv.pt) on the three
Figure 4 meshes (161, 221, 101) and cache the predictions.
"""
import os, sys
import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import config as cfg

# Plug the user's SARMA implementation in
SARMA_DIR = "/home/nii/Documents/Saliency/SARMA_SchellingData/GNNSaliency"
sys.path.insert(0, SARMA_DIR)
from MeshGNN import MeshGNN

WEIGHTS    = os.path.join(SARMA_DIR, "best_model_FeastConv.pt")
PRED_CACHE = os.path.join(_HERE, "cache", "_sarma_predictions.npz")
MESH_IDS   = [161, 221, 101]


def normalize_coords(pos):
    pos = pos - pos.mean(dim=0)
    pos = pos / pos.norm(p=2, dim=1).max()
    return pos


def build_data(mid):
    """Reproduce SARMA's MeshDataset preprocessing for a single mesh."""
    ply = os.path.join(cfg.MESHES_DIR, f"{mid}.ply")
    mesh = trimesh.load(ply, process=False)
    pos = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float)
    pos = normalize_coords(pos)

    edges = []
    for f in np.asarray(mesh.faces):
        for i in range(3):
            edges.append([int(f[i]),         int(f[(i + 1) % 3])])
            edges.append([int(f[(i + 1) % 3]), int(f[i])])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # SARMA uses discrete mean curvature at radius=0.05 of the normalised mesh
    kappa = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, np.asarray(mesh.vertices), radius=0.05)
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    kappa = torch.tensor(kappa, dtype=torch.float).view(-1, 1)
    kappa = (kappa - kappa.mean()) / (kappa.std() + 1e-8)

    x = torch.cat([pos, kappa], dim=1)
    return Data(x=x, edge_index=edge_index)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading SARMA weights: {WEIGHTS}", flush=True)
    model = MeshGNN().to(device)
    state = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out = {}
    with torch.no_grad():
        for mid in MESH_IDS:
            d = build_data(mid).to(device)
            pred = model(d).squeeze(-1).cpu().numpy()
            pred = np.clip(pred, 0, None)            # matches inference.py
            out[f"sarma_{mid}"] = pred
            print(f"  mesh {mid}: {pred.shape}  min {pred.min():.4f}  "
                  f"max {pred.max():.4f}  mean {pred.mean():.4f}", flush=True)

    np.savez(PRED_CACHE, **out)
    print(f"\nSaved SARMA predictions to {PRED_CACHE}", flush=True)


if __name__ == "__main__":
    main()
