"""
Summarise a geometry_log.csv written with --log_geometry 1.

The logged Hessian H is that of the inner objective in gpytorch's raw (inverse-softplus)
coordinates of (lengthscale, outputscale, noise). The inner objective is
-(log marginal likelihood + log priors) / n, with log-normal priors of scale 0.25 on the
lengthscale and on the noise, and n = 20 support points. In log coordinates each prior adds
1 / (0.25^2 n) to its diagonal entry; removing it and multiplying by n gives the observed
information of the marginal likelihood for log(lengthscale), reported as I_obs.

For each window of epochs the script prints the medians, over all solves in the window, of the
fitted lengthscale, r_min, r_max, I_obs and the condition number of H. If any solves were skipped
as singular, it also prints the largest outputscale, the median noise and the lengthscale range
in their pre-solve rows (matched by epoch and task), and, for the other solves, the largest
finite condition number and how many exceed 1e9.

    python summarise_geometry_log.py path/to/geometry_log.csv [--window 50]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

N_SUPPORT = 20
PRIOR = 1.0 / (0.25 ** 2 * N_SUPPORT)


def to_float(v):
    try:
        return float(v)
    except ValueError:
        return float("nan")


def log_information(r):
    """Observed information for log(lengthscale), prior removed; nan if H is not finite.

    Only the Jacobian term of the change of coordinates is kept, which is exact where the
    gradient of the inner objective vanishes.
    """
    ell = to_float(r["lengthscale"])
    j = (1.0 - np.exp(-ell)) / ell          # d log(ell) / d raw
    h = to_float(r["h_ll"])
    if not (np.isfinite(h) and np.isfinite(j) and j > 0):
        return float("nan")
    return N_SUPPORT * (h / j ** 2 - PRIOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    pre = [r for r in rows if r["event"] == "pre_solve"]
    skips = [r for r in rows if r["event"] == "singular_skip"]

    by = defaultdict(list)
    for r in pre:
        by[(int(r["epoch"]) - 1) // args.window].append(r)
    print(f"{len(pre)} solves, {len(skips)} singular solves skipped")
    print(f"{'epochs':>11} {'ell':>10} {'r_min':>7} {'r_max':>7} {'I_obs':>8} {'cond':>10}")
    for b in sorted(by):
        v = by[b]
        col = lambda k: np.array([to_float(r[k]) for r in v])
        info = np.array([log_information(r) for r in v])
        print(f"{b * args.window + 1:5d}-{(b + 1) * args.window:<5d} {np.median(col('lengthscale')):10.4g} "
              f"{np.median(col('r_min')):7.3f} {np.median(col('r_max')):7.3f} "
              f"{np.nanmedian(info):8.2f} {np.median(col('cond')):10.3g}")

    if skips:
        last = {}
        for r in rows:
            if r["event"] == "pre_solve":
                last[(r["epoch"], r["task"])] = r
        before = [last[(s["epoch"], s["task"])] for s in skips if (s["epoch"], s["task"]) in last]
        out = np.array([to_float(r["outputscale"]) for r in before])
        noise = np.array([to_float(r["noise"]) for r in before])
        ell = np.array([to_float(r["lengthscale"]) for r in before])
        print(f"before the skipped solves: outputscale max {out.max():.3g}, noise median {np.median(noise):.4f}, "
              f"lengthscale {ell.min():.4g} to {ell.max():.4g}")
        ok = [r for r in pre if r not in before]
        cond = np.array([to_float(r["cond"]) for r in ok])
        cond = cond[np.isfinite(cond)]
        print(f"other solves: largest finite condition number {cond.max():.3g}, "
              f"{int((cond > 1e9).sum())} above 1e9")


if __name__ == "__main__":
    main()
