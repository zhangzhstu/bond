"""Per-assay exposure of scored molecules to the test-time adaptation labels.

On the original (legacy) evaluation path each meta-test episode draws 10 positives and 10
negatives as the support set, then update_step_test * n_query / 2 more of each class (8 here)
whose labels drive a test-time adaptation step. Every molecule outside the support set is scored,
including the adaptation molecules. For each meta-test assay of Tox21, SIDER and MUV the script
counts how many of the scored positives are adaptation positives. The counts depend only on the
class sizes and the sampler, so no model is needed.

The printed summary averages the per-assay fractions; pooled fractions follow from the
adapt_pos_scored and scored_pos columns of the CSV.

    python leak_exposure.py            # writes leak_exposure.csv and prints a summary
"""
import csv
import sys

from chem_lib.datasets.samples import obatin_train_test_tasks, obtain_distr_list

N_SHOT, N_QUERY, UPDATE_STEP_TEST = 10, 16, 1
ADAPT_PER_CLASS = UPDATE_STEP_TEST * N_QUERY // 2

rows = []
for ds in ["tox21", "sider", "muv"]:
    _, test_tasks = obatin_train_test_tasks(ds)
    distri = obtain_distr_list(ds)
    for t in test_tasks:
        n_neg, n_pos = distri[t][0], distri[t][1]
        scored_pos = n_pos - N_SHOT
        scored = n_neg + n_pos - 2 * N_SHOT
        exposed = min(ADAPT_PER_CLASS, scored_pos)
        rows.append({
            "dataset": ds, "task": t, "n_pos": n_pos, "n_neg": n_neg,
            "scored": scored, "scored_pos": scored_pos,
            "adapt_pos_scored": exposed,
            "frac_scored_pos_exposed": round(exposed / scored_pos, 4) if scored_pos else "",
            "frac_scored_exposed": round(2 * ADAPT_PER_CLASS / scored, 4),
        })

with open("leak_exposure.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

sys.stdout.reconfigure(encoding="utf-8")
print(f"{'dataset':<8}{'task':>5}{'pos':>7}{'scored pos':>12}{'exposed':>9}{'% of scored pos':>17}")
for r in rows:
    print(f"{r['dataset']:<8}{r['task']:>5}{r['n_pos']:>7}{r['scored_pos']:>12}"
          f"{r['adapt_pos_scored']:>9}{100 * r['frac_scored_pos_exposed']:>16.1f}%")
for ds in ["tox21", "sider", "muv"]:
    f = [r["frac_scored_pos_exposed"] for r in rows if r["dataset"] == ds]
    print(f"  {ds}: mean {100 * sum(f) / len(f):.1f}% of scored positives were adaptation positives "
          f"(range {100 * min(f):.1f}-{100 * max(f):.1f}%)")
