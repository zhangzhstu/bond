"""Re-score a trained checkpoint under a chosen meta-test protocol.

Builds the model and trainer from the command-line options, loads the saved weights and runs
the test step `--repeats` times with different episode seeds, so each task's score can be
averaged over several independently drawn support sets. The trainer appends every task of every
repeat to <result_path>/.../test_metrics.csv.

    python reeval_checkpoint.py --ckpt results_kdd/sider_pre0_seed0/.../step_2000.pth \
        --dataset sider --pretrained 0 --eval_protocol support_only --repeats 10 \
        --result_path reeval/sider_pre0_seed0_support_only

All other options go to the training parser; those that define the model must match the run
that produced the checkpoint. Their defaults (head_type dnm, use_fingerprints 1, dnm_M 30) are
the configuration of the main results. The checkpoint holds only model weights, so the
optimiser is created afresh; under the protocols that adapt at test time (legacy, disjoint)
the scores are therefore close to, but not identical with, the in-training evaluation.
"""
import argparse
import random
import sys

import numpy as np
import torch


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt", required=True)
    pre.add_argument("--repeats", type=int, default=10)
    pre.add_argument("--episode_seed", type=int, default=1000)
    own, rest = pre.parse_known_args()

    # everything else is handed to the training parser unchanged
    sys.argv = [sys.argv[0]] + rest
    from adkf_parser import get_args
    from chem_lib.models import BONDModel, BOND_Meta_Trainer

    args = get_args(".")
    model = BONDModel(args)
    state = torch.load(own.ckpt, map_location=args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("state_dict mismatch -- missing:", missing, "unexpected:", unexpected)
        sys.exit(1)
    model = model.to(args.device)

    trainer = BOND_Meta_Trainer(args, model)
    trainer.train_epoch = 0
    print(f"checkpoint {own.ckpt}\nprotocol {args.eval_protocol}, {own.repeats} repeat(s)")

    for r in range(own.repeats):
        seed = own.episode_seed + r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        trainer.train_epoch = r          # the repeat index, recorded in test_metrics.csv
        trainer.test_step(args)

    print("done:", trainer.trial_path)


if __name__ == "__main__":
    main()
