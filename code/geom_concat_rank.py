"""
Block influence under plain concatenation: clean-trained ADKF-IFT on GIN + ECFP4 (identity head).

The GP sees x = [g, f], so its distance is sqrt(||g_i - g_j||^2 + ||f_i - f_j||^2). For each
checkpoint this reports how well the graph block alone and the fingerprint block alone reproduce
that distance, as Pearson and as Spearman correlations over all pairs, and the fingerprint share
of the squared distance summed over pairs. Values are averaged over the molecule sets, then over
seeds. The molecule sets are the "full" sets of geom_blockcorr_sets.py: SIDER's 1,427 meta-test
molecules, and for Tox21 a fixed sample of up to 2,000 molecules per meta-test assay.

Checkpoints are read from $CLEAN_RUNS/clean_adkfecfp_<ds>_pre0_s<k>/*/*/step_2000.pth.

Usage:
    CLEAN_RUNS=work/runs python geom_concat_rank.py
"""
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch

import diagnose_geometry as dg
from geom_blockcorr_sets import full_batches

RUNS = os.environ.get("CLEAN_RUNS", "work/runs")


def rk(v):
    r = torch.empty_like(v)
    r[torch.argsort(v)] = torch.arange(len(v), dtype=v.dtype)
    return r


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


@torch.no_grad()
def main():
    torch.set_num_threads(6)
    for ds in ("sider", "tox21"):
        batches = full_batches(ds)
        res = []
        for p in sorted(glob.glob(os.path.join(RUNS, f"clean_adkfecfp_{ds}_pre0_s*", "*", "*", "step_2000.pth"))):
            m = dg.make_model("identity", pretrained=0, ckpt=None)
            sd = torch.load(p, map_location="cpu")
            missing, _ = m.load_state_dict(sd, strict=False)
            assert not [k for k in missing if k.startswith("mol_encoder")], p
            m.eval()
            per = []
            for b in batches:
                g, _ = m.mol_encoder(b.x, b.edge_index, b.edge_attr, b.batch)
                f = b.fingerprints.view(-1, dg.FP_DIM)
                dg2, df2 = dg.upper(dg.pairwise(g)) ** 2, dg.upper(dg.pairwise(f)) ** 2
                d = torch.sqrt(dg2 + df2)
                dgr, dfr = torch.sqrt(dg2), torch.sqrt(df2)
                rd = rk(d)
                per.append((corr(d, dfr), corr(d, dgr), corr(rd, rk(dfr)), corr(rd, rk(dgr)),
                            (df2.sum() / (dg2.sum() + df2.sum())).item()))
            res.append(np.mean(per, 0))
            print(ds, os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))),
                  "pearson fp %.3f graph %.3f | spearman fp %.3f graph %.3f | fp share %.3f" % tuple(res[-1]), flush=True)
        a = np.array(res)
        print(f"{ds} MEAN n={len(a)} pearson fp {a[:,0].mean():.3f} graph {a[:,1].mean():.3f} | "
              f"spearman fp {a[:,2].mean():.3f}±{a[:,2].std():.3f} graph {a[:,3].mean():.3f}±{a[:,3].std():.3f} | fp share {a[:,4].mean():.3f}")


if __name__ == "__main__":
    main()
