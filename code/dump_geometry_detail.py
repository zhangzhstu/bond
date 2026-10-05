"""
Dump the mapped support sets of each head for plotting, as a complement to the diameter reported
by diagnose_geometry.py.

For each head (dnm, branch_linear, l2norm), stage (init, trained) and task index, the .npz holds

  <head>|<stage>|<task>|coords   the 20 support molecules projected to 2D by PCA, in feature-space
                                 units (not normalised, so changes of scale stay visible)
  <head>|<stage>|<task>|dists    all 190 pairwise distances

Trained weights are the first seed's step_2000.pth, for BOND from --main_root and for the other
heads from --ckpt_root.

The kernel length-scale is initialised by the median heuristic, so a pure rescaling of the
feature map leaves the initial kernel matrix unchanged. Dividing the distances by their median
therefore separates a change of scale from a change in the shape of the distance distribution.

Usage:
    python dump_geometry_detail.py --dataset sider --tasks 6 --out geom_detail.npz
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch

from diagnose_geometry import make_model, support_batch, pairwise, upper, FP_DIM

HEADS = ["dnm", "branch_linear", "l2norm"]


@torch.no_grad()
def features(model, batch, device):
    batch = batch.to(device)
    g, _ = model.mol_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    f = batch.fingerprints.view(-1, FP_DIM)
    return model.dnm(torch.cat([g, f], 1)).cpu().numpy()


def pca2(z):
    """Project to the first two principal components, keeping the original units."""
    c = z - z.mean(0, keepdims=True)
    # economy SVD: n=20 rows, so this is cheap even at d=2048
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--main_root", default="results_kdd")
    ap.add_argument("--ckpt_root", default="results_abl")
    ap.add_argument("--out", default="geom_detail.npz")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    batches = [support_batch(args.dataset, t) for t in range(args.tasks)]
    store = {}

    for head in HEADS:
        # initialised
        m = make_model(head).to(dev)
        for ti, b in enumerate(batches):
            z = features(m, b, dev)
            d = upper(pairwise(torch.tensor(z))).numpy()
            store[f"{head}|init|{ti}|coords"] = pca2(z)
            store[f"{head}|init|{ti}|dists"] = d

        # trained: BOND's weights are in the main grid, the other arms in the ablation grid
        pat = (os.path.join(args.main_root, f"{args.dataset}_pre0_seed*", "*", "1", "step_2000.pth")
               if head == "dnm" else
               os.path.join(args.ckpt_root, f"{args.dataset}_pre0_{head}_fp1_seed*", "*", "1",
                            "step_2000.pth"))
        paths = sorted(glob.glob(pat))
        if not paths:
            print(f"  {head}: no trained checkpoint, skipping")
            continue
        m = make_model(head, ckpt=paths[0]).to(dev)      # first seed only
        for ti, b in enumerate(batches):
            z = features(m, b, dev)
            d = upper(pairwise(torch.tensor(z))).numpy()
            store[f"{head}|trained|{ti}|coords"] = pca2(z)
            store[f"{head}|trained|{ti}|dists"] = d
        print(f"  {head}: dumped init + trained ({os.path.basename(os.path.dirname(paths[0]))})")

    np.savez_compressed(args.out, dataset=args.dataset, **store)
    print(f"\nwrote {args.out}  ({len(store)} arrays)")


if __name__ == "__main__":
    main()
