"""Collect the re-scores and training runs under a work directory into one table.

Reads, without modifying anything:
  reeval/<run>_<protocol>/.../test_metrics.csv   re-scores (reeval_checkpoint.py, ecfp_1nn.py)
  runs/clean_<cfg>_<ds>_pre<p>_s<seed>/.../test_metrics.csv   clean-protocol training runs
  runs/legacy_*, runs/scramble_*   training runs under the legacy protocol

Re-scores without a DONE file are skipped; training runs without one are listed as running.
For a re-score, a task's value is its mean over the support draws and a run's value is the mean
over tasks. For a training run the table gives the last evaluation and, as best_auroc, the
highest AUROC over all evaluations, which is optimistic and shown for comparison only. Writes
<work>/results_tables.csv and prints a summary.

    python collect_tables.py ./work
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

W = sys.argv[1] if len(sys.argv) > 1 else "./work"


def read_metrics(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["epoch"] = int(r["epoch"])
        r["auroc"] = float(r["auroc"])
        r["auprc"] = float(r["auprc"])
    return rows


def per_task_mean(rows):
    """Average each task over all rows (support draws), then average the tasks."""
    by = defaultdict(lambda: [[], []])
    for r in rows:
        by[r["task"]][0].append(r["auroc"])
        by[r["task"]][1].append(r["auprc"])
    roc = st.mean(st.mean(v[0]) for v in by.values())
    prc = st.mean(st.mean(v[1]) for v in by.values())
    return 100 * roc, 100 * prc


def at_epoch(rows, e):
    sel = [r for r in rows if r["epoch"] == e]
    return per_task_mean(sel) if sel else (float("nan"), float("nan"))


out = []

# ---- checkpoint re-scores -----------------------------------------------------------------
for d in sorted(glob.glob(os.path.join(W, "reeval", "*"))):
    name = os.path.basename(d)
    m = glob.glob(os.path.join(d, "**", "test_metrics.csv"), recursive=True)
    if not m or not os.path.exists(os.path.join(d, "DONE")):
        continue
    proto = re.search(r"_(legacy|disjoint|support_only)$", name).group(1)
    run = name[: -(len(proto) + 1)]
    roc, prc = per_task_mean(read_metrics(m[0]))
    out.append({"kind": "reeval", "run": run, "protocol": proto,
                "auroc": round(roc, 2), "auprc": round(prc, 2), "epoch": ""})

# ---- training runs (clean, legacy, scramble) -----------------------------------------------
for d in sorted(glob.glob(os.path.join(W, "runs", "clean_*")) + glob.glob(os.path.join(W, "runs", "legacy_*"))
                + glob.glob(os.path.join(W, "runs", "scramble_*"))):
    name = os.path.basename(d)
    m = glob.glob(os.path.join(d, "**", "test_metrics.csv"), recursive=True)
    if not m:
        continue
    rows = read_metrics(m[0])
    last = max(r["epoch"] for r in rows)
    roc, prc = at_epoch(rows, last)
    epochs = sorted({r["epoch"] for r in rows})
    best = max(at_epoch(rows, e)[0] for e in epochs)
    family = name.split("_", 1)[0]                     # clean / legacy / scramble
    proto = "support_only" if family == "clean" else "legacy"
    out.append({"kind": f"{family}_final" if os.path.exists(os.path.join(d, "DONE")) else f"{family}_running",
                "run": name, "protocol": proto, "auroc": round(roc, 2),
                "auprc": round(prc, 2), "epoch": last, "best_auroc": round(best, 2)})

with open(os.path.join(W, "results_tables.csv"), "w", newline="") as fh:
    keys = ["kind", "run", "protocol", "epoch", "auroc", "auprc", "best_auroc"]
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(out)

# ---- summaries ------------------------------------------------------------------------------
print("== main-grid checkpoints, re-scored (mean over seeds; per run: mean over tasks and draws)")
grid = defaultdict(list)
for r in out:
    mm = re.match(r"^(tox21|sider|muv)_pre([01])_seed(\d)$", r["run"])
    if r["kind"] == "reeval" and mm:
        grid[(mm.group(1), mm.group(2), r["protocol"])].append((r["auroc"], r["auprc"]))
print(f"{'dataset':<8}{'pre':>4}  {'protocol':<13}{'n':>3}{'AUROC':>9}{'sd':>6}{'AUPRC':>9}")
for k in sorted(grid):
    v = grid[k]
    a = [x[0] for x in v]
    p = [x[1] for x in v]
    sd = st.pstdev(a) if len(a) > 1 else 0
    print(f"{k[0]:<8}{k[1]:>4}  {k[2]:<13}{len(v):>3}{st.mean(a):>9.2f}{sd:>6.2f}{st.mean(p):>9.2f}")

print("\n== other re-scores (ablation checkpoints, fingerprint-only GP), support_only")
oth = defaultdict(list)
for r in out:
    if r["kind"] != "reeval" or re.match(r"^(tox21|sider|muv)_pre[01]_seed\d$", r["run"]):
        continue
    key = re.sub(r"_s(eed)?\d+$", "", r["run"])
    oth[key].append((r["auroc"], r["auprc"]))
for k in sorted(oth):
    a = [x[0] for x in oth[k]]
    p = [x[1] for x in oth[k]]
    print(f"  {k:<40} n={len(a)}  AUROC {st.mean(a):6.2f} (sd {st.pstdev(a) if len(a) > 1 else 0:4.2f})"
          f"  AUPRC {st.mean(p):6.2f}")

print("\n== training runs, scored in training (final evaluation; best-over-rounds in brackets)")
cl = defaultdict(list)
for r in out:
    if r["kind"] != "reeval":
        key = re.sub(r"_s\d+$", "", r["run"])
        cl[key].append(r)
for k in sorted(cl):
    rows = cl[k]
    done = [r for r in rows if r["kind"].endswith("_final")]
    desc = ", ".join(f"s{r['run'][-1]}:{r['auroc']:.2f}[{r['best_auroc']:.2f}]"
                     f"{'' if r['kind'].endswith('_final') else '@' + str(r['epoch'])}" for r in rows)
    mean = f"{st.mean(r['auroc'] for r in done):6.2f}" if done else "     -"
    best = f"{st.mean(r['best_auroc'] for r in done):6.2f}" if done else "     -"
    print(f"  {k:<32} done {len(done)}/{len(rows)}  final {mean}  best {best}   {desc}")
