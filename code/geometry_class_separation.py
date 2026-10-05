"""
Class separation of the GP input, measured in the full feature space (no 2D projection).

For every meta-test task, each labelled molecule is mapped by a trained checkpoint to the GP
input. Models are the three clean-protocol models, trained from scratch on the panel they are
scored on, plus the raw fingerprint as a model-free reference (run directories under
--runs_root):

  ecfp        raw 2048-bit ECFP4, no model
  adkf        GIN embedding (300-d), identity head            clean_adkf_<ds>_pre0_s<k>
  adkfecfp    [GIN | ECFP4] (2348-d), identity head           clean_adkfecfp_<ds>_pre0_s<k>
  bond        DNM head on [GIN | ECFP4] (2048-d)              clean_bond_<ds>_pre0_s<k>

Features are taken in eval mode (BatchNorm running statistics, no dropout), the map used for the
GP posterior at test time. Labels are the stored per-molecule labels (data.y), not the index
threshold of obtain_distr_list, which is off by up to 7 molecules on Tox21.

Per task, over all unordered pairs i<j (y in {0,1}, s = 2y-1 in {-1,+1}):

  ratio       mean d_ij over different-class pairs / mean d_ij over same-class pairs (both
              classes pooled). Scale-free.
  pair_auroc  AUROC of -d_ij for same-class (positive) vs different-class pairs, ties 1/2.
              Equal to the AUROC of the kernel value, a strictly decreasing function of d.
              Scale-free.
  kta         <K, ss^T>_F / (||K||_F ||ss^T||_F), full n x n matrices, diagonal included.
              K is Matern-5/2 with output scale 1 and the model's median-heuristic length-scale
              (BONDModel.compute_median_lengthscale_init),
                  ell = sqrt(0.5 * median{d_ij^2 : i<j, d_ij > 0}),
              i.e. the initialisation before the per-task marginal-likelihood fit.
  ckta        as kta, with K and ss^T centred by H = I - 11^T/n (Cortes et al. 2012). Uncentred
              KTA is high for any near-constant kernel when one class dominates, so it reflects
              the class imbalance as well as the geometry.

For the models whose input concatenates the blocks (bond, adkfecfp), block influence on the
GP-input distance, as defined in diagnose_geometry.analyse:

  corr_fp     Pearson correlation over pairs between the GP-input distance and the distance
              after the same head with the graph block zeroed (fingerprint only).
  corr_graph  the same with the fingerprint block zeroed (graph only).
  fp_share    mean over pairs of ||f_i-f_j||^2 / (||g_i-g_j||^2 + ||f_i-f_j||^2) at the head's
              input.
  corr_fp_raw, corr_graph_raw
              correlation of the GP-input distance with the raw block distances ||f_i-f_j||
              and ||g_i-g_j|| (equal to corr_fp / corr_graph for the identity head).

The ecfp row sets the fingerprint columns to 1 and the graph columns to NaN; adkf has NaN in all
block columns.

With the default --cap of 2000, SIDER's meta-test tasks (1427 labelled molecules) are used whole.
Tox21's (5.8k-6.8k) are subsampled to --cap molecules per task, stratified by label (proportional
class counts, rounded), with a seed that depends only on the task and --subsample_seed, so all
models and seeds are scored on the same molecules.

Per-task rows are appended to the per-task CSV (--out_tasks) as they finish, and rows already
present with the same --cap and --subsample_seed are reused, so a rerun computes only what is
missing. The summary (--out) averages over tasks per seed and reports the mean and population sd
over seeds.

Usage:
    python geometry_class_separation.py --runs_root ./work/runs \
        --out ./work/geom_class_separation.csv
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import diagnose_geometry as dg
from diagnose_geometry import make_model, pairwise, upper, analyse, FP_DIM
from chem_lib.datasets import MoleculeDataset, obatin_train_test_tasks
from chem_lib.models import BONDModel

try:
    from torch_geometric.loader import DataLoader
except ImportError:                                   # older PyG
    from torch_geometric.data import DataLoader

MODELS = {"ecfp": None, "adkf": "clean_adkf", "adkfecfp": "clean_adkfecfp", "bond": "clean_bond"}
SEP = ["ratio", "pair_auroc", "kta", "ckta"]
BLOCK = ["corr_fp", "corr_graph", "fp_share", "corr_fp_raw", "corr_graph_raw"]
TASK_FIELDS = (["dataset", "model", "seed", "task", "n", "n_pos", "cap", "subsample_seed",
                "lengthscale", "mean_d_same", "mean_d_diff"] + SEP + BLOCK)


# ---------------------------------------------------------------------------------- data ----
def task_molecules(ds, task, cap, sub_seed):
    """All labelled molecules of a meta-test task, or a label-stratified subsample of `cap`."""
    d = MoleculeDataset(os.path.join(dg.DATA_ROOT, ds, "new", str(task + 1)), dataset=ds)
    y = d.data.y.view(len(d)).numpy().astype(int)
    assert set(np.unique(y)) <= {0, 1}, f"{ds} task {task}: labels {np.unique(y)}"
    idx = np.arange(len(d))
    if len(d) > cap:
        rng = np.random.RandomState(sub_seed * 100003 + task)
        pos, neg = idx[y == 1], idx[y == 0]
        n_pos = int(round(cap * len(pos) / len(d)))
        idx = np.sort(np.concatenate([rng.choice(neg, cap - n_pos, replace=False),
                                      rng.choice(pos, n_pos, replace=False)]))
    return d, idx, y[idx]


def one_batch(d, idx, y):
    sub = d[torch.tensor(idx, dtype=torch.long)]
    b = next(iter(DataLoader(sub, batch_size=len(sub), shuffle=False)))
    assert np.array_equal(b.y.view(-1).numpy().astype(int), y), "label/order mismatch"
    return b


# -------------------------------------------------------------------------------- models ----
def load_checkpoint(run_dir):
    """Rebuild the model from the run's args.json and load step_2000.pth (non-GP keys strict)."""
    ck = sorted(glob.glob(os.path.join(run_dir, "*", "1", "step_2000.pth")))
    if not ck:
        return None, None
    a = json.load(open(os.path.join(os.path.dirname(ck[0]), "args.json")))
    # make_model hard-codes the encoder; refuse anything it would rebuild differently
    fixed = dict(emb_dim=300, enc_layer=5, JK="last", enc_pooling="mean", enc_gnn="gin",
                 enc_batch_norm=1, use_gin=1, mlp_match_params=1)
    bad = {k: a.get(k) for k, v in fixed.items() if a.get(k, v) != v}
    assert not bad, f"{run_dir}: args differ from make_model: {bad}"
    m = make_model(a["head_type"], pretrained=0, M=a["dnm_M"], fp=a["use_fingerprints"])
    sd = torch.load(ck[0], map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # the GP's own state is not used here (the kernel is rebuilt from the median heuristic)
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith("gp_")]
    assert not bad, f"{ck[0]}: state_dict mismatch outside the GP: {bad[:5]}"
    return m.eval(), a


@torch.no_grad()
def gp_input(model, b, chunk=250):
    """Graph block, fingerprint block, and the features the GP sees."""
    g, _ = model.mol_encoder(b.x, b.edge_index, b.edge_attr, b.batch)
    f = b.fingerprints.view(-1, FP_DIM)
    x = torch.cat([g, f], 1) if model.use_fingerprints else g
    z = torch.cat([model.dnm(x[i:i + chunk]) for i in range(0, len(x), chunk)])
    return g, f, z


# ------------------------------------------------------------------------------- metrics ----
def matern52(r):
    s5r = np.sqrt(5.0) * r
    return (1.0 + s5r + 5.0 / 3.0 * r ** 2) * np.exp(-s5r)


def separation(z, y):
    z = torch.as_tensor(z, dtype=torch.float64)
    D = pairwise(z)
    D.fill_diagonal_(0.0)
    d = upper(D).numpy()
    iu = torch.triu_indices(len(y), len(y), 1).numpy()
    same = y[iu[0]] == y[iu[1]]
    ell = BONDModel.compute_median_lengthscale_init(None, z).item()
    K = matern52(D.numpy() / ell)
    s = 2.0 * y - 1.0
    sc = s - s.mean()
    Kc = K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
    return dict(
        lengthscale=ell, mean_d_same=d[same].mean(), mean_d_diff=d[~same].mean(),
        ratio=d[~same].mean() / d[same].mean(),
        pair_auroc=roc_auc_score(same.astype(int), -d),
        kta=float(s @ K @ s / (np.linalg.norm(K) * len(s))),          # ||ss^T||_F = n
        ckta=float(sc @ Kc @ sc / (np.linalg.norm(Kc) * (sc @ sc))),  # ||Hss^TH||_F = |Hs|^2
    )


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def blocks(model, b, g, f, z):
    r = analyse(model, b, "cpu")                 # definitions from diagnose_geometry
    dz = upper(pairwise(z.double())).numpy()
    return dict(corr_fp=r["keep_fp_only"], corr_graph=r["keep_graph_only"],
                fp_share=r["share_fp_input"],
                corr_fp_raw=pearson(dz, upper(pairwise(f.double())).numpy()),
                corr_graph_raw=pearson(dz, upper(pairwise(g.double())).numpy()))


# ---------------------------------------------------------------------------------- main ----
def read_cache(path, cap, sub_seed):
    done = {}
    if os.path.isfile(path):
        for r in csv.DictReader(open(path)):
            if int(r["cap"]) == cap and int(r["subsample_seed"]) == sub_seed:
                done[(r["dataset"], r["model"], int(r["seed"]), int(r["task"]))] = r
    return done


def summarise(rows, datasets, models):
    out = []
    for ds in datasets:
        for mo in models:
            rs = [r for r in rows if r["dataset"] == ds and r["model"] == mo]
            if not rs:
                continue
            seeds = sorted({int(r["seed"]) for r in rs})
            ntask = sorted({sum(1 for r in rs if int(r["seed"]) == s) for s in seeds})
            row = dict(dataset=ds, model=mo, n_seeds=len(seeds) if seeds != [-1] else 0,
                       seeds=" ".join(str(s) for s in seeds if s >= 0),
                       n_tasks="/".join(map(str, ntask)),
                       n_mol=" ".join(sorted({r["n"] for r in rs}, key=int)))
            for k in SEP + BLOCK:
                per_seed = [np.mean([float(r[k]) for r in rs if int(r["seed"]) == s])
                            for s in seeds]
                row[k + "_mean"] = float(np.mean(per_seed))
                row[k + "_sd"] = float(np.std(per_seed))          # population sd over seeds
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="./work/runs")
    ap.add_argument("--datasets", nargs="+", default=["sider", "tox21"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cap", type=int, default=2000, help="max molecules per task")
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--data_root", default=dg.DATA_ROOT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="geom_class_separation.csv")
    ap.add_argument("--out_tasks", default=None,
                    help="per-task rows, also the cache; default <out>_tasks.csv")
    args = ap.parse_args()
    dg.DATA_ROOT = args.data_root
    torch.set_num_threads(args.threads)
    tasks_csv = args.out_tasks or args.out.replace(".csv", "_tasks.csv")
    done = read_cache(tasks_csv, args.cap, args.subsample_seed)
    new_file = not os.path.isfile(tasks_csv)
    fh = open(tasks_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=TASK_FIELDS)
    if new_file:
        w.writeheader()
    t_all = time.time()

    for ds in args.datasets:
        test_tasks = obatin_train_test_tasks(ds)[1]
        mols = {t: task_molecules(ds, t, args.cap, args.subsample_seed) for t in test_tasks}
        for t, (_, idx, y) in mols.items():
            print(f"{ds} task {t}: {len(idx)} molecules, {int(y.sum())} positive")
        for mo in args.models:
            for seed in ([-1] if MODELS[mo] is None else args.seeds):
                todo = [t for t in test_tasks if (ds, mo, seed, t) not in done]
                if not todo:
                    continue
                if MODELS[mo] is None:
                    model = None
                else:
                    run = os.path.join(args.runs_root, f"{MODELS[mo]}_{ds}_pre0_s{seed}")
                    model, _ = load_checkpoint(run)
                    if model is None:
                        print(f"  {mo} s{seed}: no step_2000.pth in {run}, skipped")
                        continue
                for t in todo:
                    t0 = time.time()
                    d, idx, y = mols[t]
                    if model is None:
                        f = d.data.fingerprints.view(len(d), FP_DIM)[torch.tensor(idx)]
                        r = separation(f, y)
                        r.update(corr_fp=1.0, corr_graph=float("nan"), fp_share=1.0,
                                 corr_fp_raw=1.0, corr_graph_raw=float("nan"))
                    else:
                        b = one_batch(d, idx, y)
                        g, f, z = gp_input(model, b)
                        r = separation(z, y)
                        if model.use_fingerprints:
                            r.update(blocks(model, b, g, f, z))
                        else:
                            r.update({k: float("nan") for k in BLOCK})
                    r.update(dataset=ds, model=mo, seed=seed, task=t, n=len(idx),
                             n_pos=int(y.sum()), cap=args.cap,
                             subsample_seed=args.subsample_seed)
                    w.writerow(r)
                    fh.flush()
                    done[(ds, mo, seed, t)] = {k: str(v) for k, v in r.items()}
                    print(f"  {mo:<9} s{seed:<2} task {t:<3} n={len(idx):<5}"
                          f" ratio={r['ratio']:.4f} pairAUC={r['pair_auroc']:.4f}"
                          f" KTA={r['kta']:.4f} cKTA={r['ckta']:.4f}"
                          f" corr_fp={r['corr_fp']:.3f} corr_graph={r['corr_graph']:.3f}"
                          f"  ({time.time() - t0:.0f}s)", flush=True)
                del model
    fh.close()

    rows = [r for r in done.values() if r["dataset"] in args.datasets
            and r["model"] in args.models
            and (int(r["seed"]) in args.seeds or int(r["seed"]) == -1)]
    summ = summarise(rows, args.datasets, args.models)
    fields = ["dataset", "model", "n_seeds", "seeds", "n_tasks", "n_mol"] + \
             [k + s for k in SEP + BLOCK for s in ("_mean", "_sd")]
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        wo = csv.DictWriter(fo, fieldnames=fields)
        wo.writeheader()
        wo.writerows(summ)
    print(f"\n{'dataset':<7}{'model':<10}{'seeds':>6}" + "".join(f"{k:>18}" for k in SEP + BLOCK[:3]))
    for r in summ:
        print(f"{r['dataset']:<7}{r['model']:<10}{r['n_seeds']:>6}" + "".join(
            f"{r[k + '_mean']:>10.4f}+-{r[k + '_sd']:<6.4f}" for k in SEP + BLOCK[:3]))
    print(f"\nwrote {args.out} and {tasks_csv}  ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
