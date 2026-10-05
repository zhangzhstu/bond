"""Score a learning-free ECFP4 Tanimoto 1-nearest-neighbour baseline on the re-scoring episodes.

Nothing is trained. In every meta-test task of every support draw, a query molecule q is scored

    s(q) = max_{i in S, y_i = 1} T(q, x_i) - max_{i in S, y_i = 0} T(q, x_i)

where S is the support set and T the Tanimoto similarity of the 2048-bit ECFP4 fingerprints
(Morgan radius 2) stored in the processed datasets, i.e. the `fingerprints` field that the model
concatenates under --use_fingerprints 1. Support labels are the stored `y`, as in the trainer.
AUROC and AUPRC come from the same torchmetrics calls as the trainer's test step (the score is
passed as (s + 1) / 2, a monotone map into [0, 1], so torchmetrics does not apply a sigmoid).

The episodes are those of reeval_checkpoint.py: the same per-task datasets, the same
sample_test_datasets call (n_shot_test 10, n_query 16, update_step_test 1, --eval_protocol), and
before draw r the python, numpy and torch generators are seeded with episode_seed + r (default
1000), so draw r here has the same support set and scored query list as draw r in every other
re-score. --seed does not affect the results: as in reeval_checkpoint.py it seeds the generators
at start-up, the episode seed replaces it before every draw, and nothing else here is random.

Output in <result_path>: test_metrics.csv, one row per task per draw with the trainer's columns
(epoch holds the draw index, as in the re-scores); episodes.json, the arguments and the support
indices and labels of every episode; and DONE, written last, so that collect_tables.py lists
the run with the other re-scores.

    CUDA_VISIBLE_DEVICES= python ecfp_1nn.py --dataset sider --seed 0 --repeats 10 \
        --eval_protocol support_only \
        --result_path ./work/reeval/ecfp1nn_sider_s0_support_only

CPU only. A result_path that already holds DONE is refused unless --overwrite is given.
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chem_lib.datasets import MoleculeDataset, obatin_train_test_tasks, sample_test_datasets  # noqa: E402
from torchmetrics.functional import auroc, average_precision  # noqa: E402

COLUMNS = ['epoch', 'task', 'protocol', 'auroc', 'auprc', 'n_scored',
           'n_pos_scored', 'n_adapt', 'n_adapt_scored', 'n_pos_adapt_scored']
NBITS = 2048


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_task(data_dir, dataset, task):
    """The per-task dataset the trainer preloads, plus its fingerprints and labels as arrays."""
    d = MoleculeDataset(os.path.join(data_dir, dataset, "new", str(task + 1)), dataset=dataset)
    store = d._data if hasattr(d, "_data") else d.data
    n = len(d)
    # one 2048-bit fingerprint and one label per molecule, stored back to back
    if not torch.equal(d.slices["fingerprints"], torch.arange(0, (n + 1) * NBITS, NBITS)):
        raise ValueError(f"{dataset} task {task}: fingerprints are not {NBITS} bits per molecule")
    if not torch.equal(d.slices["y"], torch.arange(0, n + 1)):
        raise ValueError(f"{dataset} task {task}: expected one label per molecule")
    fps = store.fingerprints.view(n, NBITS).numpy().astype(np.float32)
    if not np.isin(fps, (0.0, 1.0)).all():
        raise ValueError(f"{dataset} task {task}: fingerprints are not binary")
    y = store.y.view(-1).numpy().astype(np.int64)
    return d, fps, y


def tanimoto(xq, xs):
    """Tanimoto similarity between the rows of two 0/1 matrices; 0 where both are empty."""
    inter = (xq @ xs.T).astype(np.float64)        # bit counts, exact in float32
    union = xq.sum(1, dtype=np.float64)[:, None] + xs.sum(1, dtype=np.float64)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nn_score(fps, y, s_idx, q_idx):
    sim = tanimoto(fps[q_idx], fps[s_idx])
    s_y = y[s_idx]
    if s_y.min() == s_y.max():
        raise ValueError("support set holds a single class; the 1-NN score is undefined")
    return sim[:, s_y == 1].max(1) - sim[:, s_y == 0].max(1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-d", "--dataset", required=True, choices=["tox21", "sider", "muv", "toxcast"])
    p.add_argument("--seed", type=int, default=0, help="recorded only; no effect on the scores")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--episode_seed", type=int, default=1000)
    p.add_argument("--eval_protocol", default="support_only",
                   choices=["legacy", "disjoint", "support_only"])
    p.add_argument("--n-shot-test", "--n_shot_test", dest="n_shot_test", type=int, default=10)
    p.add_argument("--n-query", "--n_query", dest="n_query", type=int, default=16)
    p.add_argument("--update_step_test", type=int, default=1)
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", default=os.path.join(HERE, "data"))
    p.add_argument("--result_path", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if os.path.exists(os.path.join(args.result_path, "DONE")) and not args.overwrite:
        sys.exit(f"{args.result_path} already holds DONE; pass --overwrite to redo it")
    os.makedirs(args.result_path, exist_ok=True)

    # same order as reeval_checkpoint.py: seed from --seed at start-up, load the test tasks,
    # then re-seed with the episode seed before every draw
    seed_all(args.seed)
    _, test_tasks = obatin_train_test_tasks(args.dataset)
    test_tasks = sorted(set(test_tasks))
    data = {t: load_task(args.data_dir, args.dataset, t) for t in test_tasks}
    print(f"ecfp 1-NN  {args.dataset}  tasks {test_tasks}  protocol {args.eval_protocol}  "
          f"{args.repeats} draw(s) from episode seed {args.episode_seed}  (--seed {args.seed})")

    rows, episodes, sk_gap = [], [], 0.0
    # optional cross-check of the torchmetrics values against sklearn on the raw score
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        roc_auc_score = None
    for r in range(args.repeats):
        seed_all(args.episode_seed + r)
        ep = {"draw": r, "episode_seed": args.episode_seed + r, "tasks": {}}
        for task in test_tasks:
            d, fps, y = data[task]
            s_data, q_data, _, stats = sample_test_datasets(
                d, args.dataset, task, args.n_shot_test, args.n_query, args.update_step_test,
                protocol=args.eval_protocol, return_stats=True)
            s_idx = np.asarray(s_data.indices(), dtype=np.int64)
            q_idx = np.asarray(q_data.indices(), dtype=np.int64)
            if len(q_idx) != stats["n_scored"]:
                raise RuntimeError("scored list does not match sample_test_datasets' statistics")

            score = nn_score(fps, y, s_idx, q_idx)
            preds = torch.from_numpy((score + 1.0) / 2.0)
            labels = torch.from_numpy(y[q_idx])
            auc = auroc(preds, labels, task="binary").item()
            auprc = average_precision(preds, labels.long(), task="binary").item()
            if roc_auc_score is not None:
                sk_gap = max(sk_gap, abs(auc - roc_auc_score(y[q_idx], score)),
                             abs(auprc - average_precision_score(y[q_idx], score)))

            rows.append([r, task, args.eval_protocol, round(auc, 6), round(auprc, 6),
                         stats["n_scored"], stats["n_pos_scored"], stats["n_adapt"],
                         stats["n_adapt_scored"], stats["n_pos_adapt_scored"]])
            ep["tasks"][str(task)] = {"support": s_idx.tolist(),
                                      "support_labels": y[s_idx].tolist(),
                                      "n_scored": int(len(q_idx)),
                                      "n_pos_scored_by_label": int(y[q_idx].sum())}
            print(f"draw {r}  task {task}  AUROC {auc:.4f}  AUPRC {auprc:.4f}  "
                  f"n_scored {stats['n_scored']}  n_pos_scored {stats['n_pos_scored']}")
        episodes.append(ep)

    def write(name, fn):
        tmp = os.path.join(args.result_path, name + ".tmp")
        with open(tmp, "w", newline="") as fh:
            fn(fh)
        os.replace(tmp, os.path.join(args.result_path, name))

    def write_csv(fh):
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)

    meta = {k: v for k, v in vars(args).items() if k != "overwrite"}
    write("test_metrics.csv", write_csv)
    write("episodes.json", lambda fh: json.dump({"args": meta, "draws": episodes}, fh))
    open(os.path.join(args.result_path, "DONE"), "w").close()

    # same summary as collect_tables.py: mean over draws per task, then over tasks
    per_task = {t: [x for x in rows if x[1] == t] for t in test_tasks}
    roc = np.mean([np.mean([x[3] for x in v]) for v in per_task.values()])
    prc = np.mean([np.mean([x[4] for x in v]) for v in per_task.values()])
    for t, v in per_task.items():
        print(f"task {t}: AUROC {100 * np.mean([x[3] for x in v]):6.2f}  "
              f"AUPRC {100 * np.mean([x[4] for x in v]):6.2f}")
    print(f"mean over tasks and draws: AUROC {100 * roc:.2f}  AUPRC {100 * prc:.2f}")
    if roc_auc_score is not None:
        print(f"largest |torchmetrics - sklearn| over all rows: {sk_gap:.2e}")
    print("done:", args.result_path)


if __name__ == "__main__":
    main()
