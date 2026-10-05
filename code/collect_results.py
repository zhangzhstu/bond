"""
Summarise training runs that write results.txt (the main grid, the head ablation, the MUV
controls).

For each run directory it reports the best AUC-Avg over evaluation rounds, the AUC-Avg of the
last round, the mean of the last three rounds, and the difference between best and last.
Runs without a DONE marker are listed but left out of the per-dataset means.

    python collect_results.py --root results_kdd
    python collect_results.py --root results_kdd --csv summary.csv
"""
import argparse
import json
import os
import numpy as np


def parse_results_txt(path):
    """Return (header, rows) where rows are float lists. Tolerates repeated headers."""
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        return None, []
    header = lines[0].split("\t")
    header = [h for h in header if h != ""]
    rows = []
    for line in lines[1:]:
        parts = [p for p in line.split("\t") if p != ""]
        if len(parts) != len(header):
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # a repeated header line
    return header, rows


def summarize_run(run_dir):
    res = os.path.join(run_dir, "results.txt")
    if not os.path.isfile(res):
        return None
    header, rows = parse_results_txt(res)
    if not rows:
        return None
    try:
        i_avg = header.index("AUC-Avg")
    except ValueError:
        return None

    avgs = np.array([r[i_avg] for r in rows]) * 100.0
    epochs = np.array([r[0] for r in rows])

    info = {}
    args_path = os.path.join(run_dir, "args.json")
    if os.path.isfile(args_path):
        try:
            a = json.load(open(args_path))
            info = {
                "dataset": a.get("dataset"),
                "seed": a.get("seed"),
                "pretrained": a.get("pretrained"),
                "epochs": a.get("epochs"),
                "update_step_test": a.get("update_step_test"),
                "n_shot_test": a.get("n_shot_test"),
                "n_query": a.get("n_query"),
            }
        except Exception:
            pass

    # scripts/run_grid.sh writes a DONE marker at the top of a finished run's --result_path;
    # unfinished runs are listed but not averaged.
    done = False
    probe = os.path.abspath(run_dir)
    for _ in range(4):
        if os.path.isfile(os.path.join(probe, "DONE")):
            done = True
            break
        probe = os.path.dirname(probe)

    return {
        **info,
        "run_dir": run_dir,
        "complete": done,
        "n_evals": len(avgs),
        "last_epoch": int(epochs[-1]),
        "best": float(avgs.max()),
        "best_epoch": int(epochs[int(avgs.argmax())]),
        "final": float(avgs[-1]),
        "last3": float(avgs[-3:].mean()),
        "inflation": float(avgs.max() - avgs[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_kdd", help="directory to walk for runs")
    ap.add_argument("--csv", default=None, help="optional path to write the per-run CSV")
    args = ap.parse_args()

    runs = []
    for dirpath, _dirnames, filenames in os.walk(args.root):
        if "results.txt" in filenames:
            s = summarize_run(dirpath)
            if s:
                runs.append(s)

    if not runs:
        print(f"No runs with results.txt found under {args.root!r}")
        return

    runs.sort(key=lambda r: (str(r.get("dataset")), r.get("pretrained"), r.get("seed")))

    print("=" * 108)
    print("PER-RUN")
    print("=" * 108)
    hdr = f"{'dataset':<9}{'pre':>4}{'seed':>5}{'done':>6}{'evals':>7}{'lastEp':>7}" \
          f"{'best':>8}{'@ep':>7}{'final':>8}{'last3':>8}{'inflation':>11}"
    print(hdr)
    print("-" * 108)
    for r in runs:
        print(f"{str(r.get('dataset')):<9}{str(r.get('pretrained')):>4}{str(r.get('seed')):>5}"
              f"{('yes' if r['complete'] else 'NO'):>6}"
              f"{r['n_evals']:>7}{r['last_epoch']:>7}"
              f"{r['best']:>8.2f}{r['best_epoch']:>7}{r['final']:>8.2f}{r['last3']:>8.2f}"
              f"{r['inflation']:>+11.2f}")

    n_partial = sum(1 for r in runs if not r["complete"])
    print()
    print("=" * 108)
    print("AGGREGATED  (mean +/- std across seeds; COMPLETE runs only)")
    if n_partial:
        print(f"  excluding {n_partial} run(s) still in progress")
    print("=" * 108)
    print(f"{'dataset':<9}{'pre':>4}{'n':>4}   {'best':>24}   {'final':>24}   {'gap':>7}")
    print("-" * 108)

    groups = {}
    for r in runs:
        if not r["complete"]:
            continue
        groups.setdefault((r.get("dataset"), r.get("pretrained")), []).append(r)

    for (ds, pre), g in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        b = np.array([x["best"] for x in g])
        f = np.array([x["final"] for x in g])
        bs = f"{b.mean():.2f} +/- {b.std(ddof=0):.2f}"
        fs = f"{f.mean():.2f} +/- {f.std(ddof=0):.2f}"
        print(f"{str(ds):<9}{str(pre):>4}{len(g):>4}   {bs:>24}   {fs:>24}   {b.mean()-f.mean():>+7.2f}")

    print()
    print("best: maximum over evaluation rounds; final: last round.")

    if args.csv:
        import csv as _csv
        keys = ["dataset", "pretrained", "seed", "complete", "epochs", "update_step_test",
                "n_shot_test", "n_query", "n_evals", "last_epoch",
                "best", "best_epoch", "final", "last3", "inflation", "run_dir"]
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in runs:
                w.writerow(r)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
