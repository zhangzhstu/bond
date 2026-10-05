"""
Measure the feature geometry the GP sees: the fingerprint share of the input distance, how far
the head rebalances it, and how the diameter of the mapped support set changes in training.

All quantities are computed on the 20-molecule support sets (10 per class, random.seed(0)) of
--tasks consecutive tasks starting at --first_task. Initialised models use the pre-trained GIN
encoder and a freshly initialised head.

Input share. At the head's input (300-d graph embedding g, 2048-bit fingerprint f) the squared
pairwise distance splits into the two blocks,

    ||x_i - x_j||^2 = ||g_i - g_j||^2 + ||f_i - f_j||^2

and the fingerprint block's share, averaged over pairs, is reported.

Block influence after the head. The head mixes the blocks, so the split above does not apply to
its output. Instead the head is re-run with one block zeroed, and the Pearson correlation over
pairs between these output distances and the output distances for the full input is reported. A
head that ignored the graph block would be unchanged when that block is zeroed.

Diameter. The largest pairwise distance of the mapped support set, at initialisation and, with
--ckpt_root, after 2000 epochs of meta-training, for bounded and unbounded heads.

Usage:
    python diagnose_geometry.py --dataset sider --tasks 6
    python diagnose_geometry.py --dataset sider --ckpt_root results_abl_local
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
from torch_geometric.data import DataLoader

from chem_lib.models import BONDModel
from chem_lib.datasets import MoleculeDataset, sample_meta_datasets

GRAPH_DIM, FP_DIM = 300, 2048


def make_model(head="dnm", pretrained=1, M=30, fp=1, ckpt=None):
    a = argparse.Namespace(
        emb_dim=300, gpu_id=0, enc_layer=5, JK="last", dropout=0.5, enc_pooling="mean",
        enc_gnn="gin", enc_batch_norm=1, pretrained=pretrained,
        pretrained_weight_path="chem_lib/model_gin/supervised_contextpred.pth",
        dnm_M=M, use_fingerprints=fp, head_type=head, mlp_match_params=1)
    m = BONDModel(a)
    if ckpt and os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(sd, strict=False)
    return m.eval()


# Root of the preprocessed MoleculeNet panels; ./data as in the training scripts. Overridden by
# --data_root, e.g. for ToxCast, which is preprocessed under the original ADKF-IFT checkout
# (too large to copy here).
DATA_ROOT = "data"


def support_batch(ds, task, seed=0):
    import random
    random.seed(seed)
    d = MoleculeDataset(os.path.join(DATA_ROOT, ds, "new", str(task + 1)), dataset=ds)
    s, _ = sample_meta_datasets(d, ds, task, 10, 16)
    for b in DataLoader(s, batch_size=len(s), shuffle=False):
        return b


def pairwise(x):
    return torch.cdist(x, x)


def upper(v):
    n = v.shape[0]
    iu = torch.triu_indices(n, n, 1)
    return v[iu[0], iu[1]]


@torch.no_grad()
def analyse(model, batch, device):
    """Input fingerprint share, output diameter and block influence for one support set."""
    batch = batch.to(device)
    g, _ = model.mol_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    f = batch.fingerprints.view(-1, FP_DIM)
    x = torch.cat([g, f], 1)

    d_graph = upper(pairwise(g)) ** 2
    d_fp = upper(pairwise(f)) ** 2
    share_fp = (d_fp / (d_graph + d_fp + 1e-12)).mean().item()

    z_full = model.dnm(x)
    d_full = upper(pairwise(z_full))

    # block influence: zero one block and compare the output distances with the full ones
    x_no_g = x.clone(); x_no_g[:, :GRAPH_DIM] = 0
    x_no_f = x.clone(); x_no_f[:, GRAPH_DIM:] = 0
    d_no_g = upper(pairwise(model.dnm(x_no_g)))
    d_no_f = upper(pairwise(model.dnm(x_no_f)))

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

    return dict(
        share_fp_input=share_fp,
        diam=float(d_full.max()),
        keep_fp_only=corr(d_full, d_no_g),   # how well fingerprint alone reproduces the geometry
        keep_graph_only=corr(d_full, d_no_f),
    )


def main():
    global DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--main_root", default="results_kdd",
                    help="directory of the main-grid runs; BOND's checkpoints are read from here")
    ap.add_argument("--ckpt_root", default=None,
                    help="directory holding <tag>/*/1/step_2000.pth; if given, trained "
                         "checkpoints are analysed alongside the initialised models")
    ap.add_argument("--csv", default=None, help="append the measurements to this CSV")
    ap.add_argument("--data_root", default=DATA_ROOT,
                    help="root holding <dataset>/new/<task>; ToxCast is under the original "
                         "ADKF-IFT checkout")
    ap.add_argument("--first_task", type=int, default=0,
                    help="first task index; ToxCast's meta-test assays start at 454")
    ap.add_argument("--dnm_ckpt_glob", default=None,
                    help="glob for BOND's trained checkpoints, for panels whose runs are not "
                         "laid out like the main grid")
    args = ap.parse_args()
    DATA_ROOT = args.data_root

    import csv as _csv
    rows = []

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    batches = [support_batch(args.dataset, args.first_task + t) for t in range(args.tasks)]

    heads = [("dnm", "BOND (bounded, branched)"),
             ("branch_linear", "branched, unbounded"),
             ("l2norm", "L2-normalised"),
             ("mlp", "MLP, matched")]

    print(f"dataset = {args.dataset},  {args.tasks} meta-train tasks,  device = {dev}\n")
    print("INPUT TO THE HEAD -- share of squared pairwise distance carried by the fingerprint block")
    m = make_model("dnm").to(dev)
    shares = [analyse(m, b, dev)["share_fp_input"] for b in batches]
    print(f"    fingerprint share: {np.mean(shares):.4f} +/- {np.std(shares):.4f}"
          f"   (graph share {1 - np.mean(shares):.4f})\n")
    rows.append(dict(dataset=args.dataset, head="input", stage="concat", n=len(batches),
                     fp_share=np.mean(shares), fp_share_sd=np.std(shares),
                     corr_fp=float("nan"), corr_graph=float("nan"),
                     diam=float("nan"), diam_sd=float("nan")))

    print("AFTER THE HEAD -- correlation of the output pairwise distance with each block alone")
    print(f"  {'head':<26}{'init: fp-only':>15}{'graph-only':>13}{'diam':>10}")
    for key, label in heads:
        m = make_model(key).to(dev)
        r = [analyse(m, b, dev) for b in batches]
        print(f"  {label:<26}{np.mean([x['keep_fp_only'] for x in r]):>15.3f}"
              f"{np.mean([x['keep_graph_only'] for x in r]):>13.3f}"
              f"{np.mean([x['diam'] for x in r]):>10.2f}")
        rows.append(dict(dataset=args.dataset, head=key, stage="init", n=len(r),
                         fp_share=float("nan"), fp_share_sd=float("nan"),
                         corr_fp=np.mean([x["keep_fp_only"] for x in r]),
                         corr_graph=np.mean([x["keep_graph_only"] for x in r]),
                         diam=np.mean([x["diam"] for x in r]),
                         diam_sd=np.std([x["diam"] for x in r])))

    if args.ckpt_root:
        print(f"\nAFTER 2000 EPOCHS -- same quantities from the trained checkpoints in {args.ckpt_root}")
        print(f"  {'head':<26}{'fp-only':>10}{'graph-only':>12}{'diam':>10}{'n':>4}")
        for key, label in heads:
            # BOND's trained weights come from the main grid: the ablation grid's `dnm` runs
            # were trained without the fingerprint.
            if key == "dnm":
                pat = args.dnm_ckpt_glob or os.path.join(
                    args.main_root, f"{args.dataset}_pre0_seed*", "*", "1", "step_2000.pth")
            else:
                pat = os.path.join(args.ckpt_root, f"{args.dataset}_pre0_{key}_fp1_seed*",
                                   "*", "1", "step_2000.pth")
            paths = sorted(glob.glob(pat))
            if not paths:
                print(f"  {label:<26}{'--':>10}{'--':>12}{'--':>10}{0:>4}")
                continue
            acc = []
            for p in paths:
                m = make_model(key, ckpt=p).to(dev)
                acc += [analyse(m, b, dev) for b in batches]
            print(f"  {label:<26}{np.mean([x['keep_fp_only'] for x in acc]):>10.3f}"
                  f"{np.mean([x['keep_graph_only'] for x in acc]):>12.3f}"
                  f"{np.mean([x['diam'] for x in acc]):>10.2f}{len(paths):>4}")
            rows.append(dict(dataset=args.dataset, head=key, stage="trained", n=len(paths),
                             fp_share=float("nan"), fp_share_sd=float("nan"),
                             corr_fp=np.mean([x["keep_fp_only"] for x in acc]),
                             corr_graph=np.mean([x["keep_graph_only"] for x in acc]),
                             diam=np.mean([x["diam"] for x in acc]),
                             diam_sd=np.std([x["diam"] for x in acc])))


    if args.csv:
        fields = ["dataset", "head", "stage", "n", "fp_share", "fp_share_sd",
                  "corr_fp", "corr_graph", "diam", "diam_sd"]
        new = not os.path.isfile(args.csv)
        with open(args.csv, "a", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
