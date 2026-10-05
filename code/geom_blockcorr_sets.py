"""
Block influence after the head (diagnose_geometry.analyse) for two families of BOND checkpoints
on two molecule sets, to separate the effect of the checkpoints from that of the molecules.

Checkpoint families:

  submitted   main-grid checkpoints, trained under the legacy protocol
  clean       clean-protocol checkpoints

Molecule sets:

  support20   20-molecule support sets of meta-training tasks 0..T-1 (T = 6 for SIDER, 9 for
              Tox21), random.seed(0), as in diagnose_geometry. This is the setting of the paper's
              SIDER values: correlation 0.96 with the fingerprint-only path, 0.10 with the
              graph-only path.
  full        SIDER: all molecules of the meta-test assays (the six share one molecule set);
              Tox21: for each meta-test assay, a fixed random sample of up to 2000 molecules

Per (dataset, family, set) the CSV gives, as mean and sd over seeds, corr_fp (keep_fp_only) and
corr_graph (keep_graph_only) as Pearson correlations, and the same two as Spearman correlations
(rank_fp, rank_graph), since a Pearson correlation over a million or more pairs can be dominated
by a few outlying molecules. fp_share_input, the fingerprint share of the squared distance at the
head's input, is given as a mean only. Missing encoder or head keys raise an error, because the
state dict is loaded with strict=False.

Usage:
    SUBMITTED_RUNS=results_kdd CLEAN_RUNS=work/runs python geom_blockcorr_sets.py --out geom_blockcorr_sets.csv

SUBMITTED_RUNS holds the main-grid runs (<ds>_pre0_seed<k>), CLEAN_RUNS the clean-protocol runs
(clean_bond_<ds>_pre0_s<k>), each with its step_2000.pth.
"""
import argparse
import csv
import glob
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch

import diagnose_geometry as dg
from chem_lib.datasets import MoleculeDataset

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

SUBMITTED = os.environ.get("SUBMITTED_RUNS", "results_kdd")
CLEAN = os.environ.get("CLEAN_RUNS", "work/runs")
TEST_TASKS = {"sider": list(range(21, 27)), "tox21": [9, 10, 11]}
N_SUPPORT_TASKS = {"sider": 6, "tox21": 9}


def rank_corrs(model, batch):
    """Spearman versions of keep_fp_only / keep_graph_only, same distances as dg.analyse."""
    batch = batch.to("cpu")
    g, _ = model.mol_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    f = batch.fingerprints.view(-1, dg.FP_DIM)
    x = torch.cat([g, f], 1)
    x_no_g = x.clone(); x_no_g[:, :dg.GRAPH_DIM] = 0
    x_no_f = x.clone(); x_no_f[:, dg.GRAPH_DIM:] = 0
    d_full = dg.upper(dg.pairwise(model.dnm(x)))
    d_no_g = dg.upper(dg.pairwise(model.dnm(x_no_g)))
    d_no_f = dg.upper(dg.pairwise(model.dnm(x_no_f)))

    def rk(v):
        r = torch.empty_like(v)
        r[torch.argsort(v)] = torch.arange(len(v), dtype=v.dtype)
        return r

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

    rf = rk(d_full)
    return corr(rf, rk(d_no_g)), corr(rf, rk(d_no_f))


def ckpts(family, ds):
    if family == "submitted":
        pat = os.path.join(SUBMITTED, f"{ds}_pre0_seed*", "*", "*", "step_2000.pth")
    else:
        pat = os.path.join(CLEAN, f"clean_bond_{ds}_pre0_s*", "*", "*", "step_2000.pth")
    return sorted(glob.glob(pat))


def load(path):
    m = dg.make_model("dnm", pretrained=0, ckpt=None)
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    head_missing = [k for k in missing if k.startswith(("dnm", "mol_encoder"))]
    if head_missing:
        raise RuntimeError(f"{path}: {len(head_missing)} encoder/head keys missing, e.g. {head_missing[:3]}")
    return m.eval(), len(missing), len(unexpected)


def full_batches(ds):
    out = []
    if ds == "sider":
        d = MoleculeDataset(os.path.join(dg.DATA_ROOT, ds, "new", str(TEST_TASKS[ds][0] + 1)), dataset=ds)
        out.append(next(iter(DataLoader(d, batch_size=len(d), shuffle=False))))
    else:
        for t in TEST_TASKS[ds]:
            d = MoleculeDataset(os.path.join(dg.DATA_ROOT, ds, "new", str(t + 1)), dataset=ds)
            idx = list(range(len(d)))
            random.Random(1000 + t).shuffle(idx)
            sub = d[torch.tensor(sorted(idx[:2000]))]
            out.append(next(iter(DataLoader(sub, batch_size=len(sub), shuffle=False))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="geom_blockcorr_sets.csv")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for ds in ("sider", "tox21"):
        sets = {"support20": [dg.support_batch(ds, t) for t in range(N_SUPPORT_TASKS[ds])],
                "full": full_batches(ds)}
        print(ds, {k: [b.num_graphs for b in v] for k, v in sets.items()}, flush=True)
        for family in ("submitted", "clean"):
            paths = ckpts(family, ds)
            per = {k: [] for k in sets}
            for p in paths:
                m, nmiss, nunexp = load(p)
                for k, bs in sets.items():
                    with torch.no_grad():
                        r = [dg.analyse(m, b, "cpu") for b in bs]
                        rr = [rank_corrs(m, b) for b in bs]
                    per[k].append((np.mean([x["keep_fp_only"] for x in r]),
                                   np.mean([x["keep_graph_only"] for x in r]),
                                   np.mean([x["share_fp_input"] for x in r]),
                                   np.mean([x[0] for x in rr]), np.mean([x[1] for x in rr])))
                print(f"  {family:<9} {os.path.relpath(p, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p)))))}"
                      f"  missing={nmiss} unexpected={nunexp}  "
                      + "  ".join(f"{k}: fp={per[k][-1][0]:.3f} graph={per[k][-1][1]:.3f} rank fp={per[k][-1][3]:.3f} graph={per[k][-1][4]:.3f}" for k in sets), flush=True)
            for k in sets:
                a = np.array(per[k])
                rows.append(dict(dataset=ds, family=family, set=k, n_seeds=len(a),
                                 corr_fp=a[:, 0].mean(), corr_fp_sd=a[:, 0].std(),
                                 corr_graph=a[:, 1].mean(), corr_graph_sd=a[:, 1].std(),
                                 fp_share_input=a[:, 2].mean(),
                                 rank_fp=a[:, 3].mean(), rank_fp_sd=a[:, 3].std(),
                                 rank_graph=a[:, 4].mean(), rank_graph_sd=a[:, 4].std()))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['dataset']:<6}{r['family']:<10}{r['set']:<10} n={r['n_seeds']}  corr_fp {r['corr_fp']:.3f}±{r['corr_fp_sd']:.3f}"
              f"  corr_graph {r['corr_graph']:.3f}±{r['corr_graph_sd']:.3f}  fp_share_in {r['fp_share_input']:.3f}"
              f"  | rank_fp {r['rank_fp']:.3f}±{r['rank_fp_sd']:.3f}  rank_graph {r['rank_graph']:.3f}±{r['rank_graph_sd']:.3f}")


if __name__ == "__main__":
    main()
