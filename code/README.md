# BOND code

Code for *What the Kernel Measures: Bounding Feature Geometry for Few-Shot Molecular Property
Prediction* (KDD 2027).

BOND replaces the feature map between the molecular encoder and the Gaussian process of a
deep-kernel few-shot learner with a bounded, branch-structured head. The GIN encoder, the Gaussian
process and the bi-level meta-learning loop are the ADKF-IFT implementation for MoleculeNet (Chen
et al., ICLR 2023), which is itself built on PAR (Wang et al., NeurIPS 2021).

## Requirements

Python 3.10 with

    torch==2.1.0 (CUDA 12.1)   torch_geometric==2.5.3   gpytorch==1.10   botorch==0.8.5
    rdkit   torchmetrics   pandas   joblib   scikit-learn

Use these `gpytorch` and `botorch` versions; the Gaussian-process fit is sensitive to them. The
shell scripts assume Linux (bash, and `flock` from util-linux for the job queues).

## Data

The four MoleculeNet few-shot splits (Tox21, SIDER, MUV, ToxCast) in the format used by
[PAR](https://github.com/tata1661/PAR-NeurIPS21), placed under `data/`:

    data/tox21/new/<task>/raw/tox21.json
    data/sider/new/<task>/raw/sider.json
    ...

The processed files are built on first use. `chem_lib/model_gin/` holds the pre-trained GIN
weights of Hu et al. (ICLR 2020) used by the pre-trained protocol.

## Running

    python main_bond.py --dataset sider --test-dataset sider --pretrained 0 --seed 0 --epochs 2000

| option | default | meaning |
|---|---|---|
| `--head_type` | `dnm` | feature map: `dnm` (BOND: bounded, branched), `branch_linear` (branched, unbounded), `tanh_linear` / `l2norm` (bounded, unbranched), `mlp` (unbounded, parameter-matched), `identity` (no head, as in ADKF-IFT) |
| `--dnm_M` | `30` | branches per output coordinate |
| `--use_fingerprints` | `1` | concatenate the 2048-bit ECFP4 fingerprint to the GIN embedding |
| `--use_gin` | `1` | `0` zeroes the GIN embedding, leaving the fingerprint as the only input (needs `--use_fingerprints 1`) |
| `--update_step_test` | `1` | test-time adaptation steps per task; `0` disables adaptation |
| `--eval_protocol` | `legacy` | how a meta-test episode is scored (see below) |
| `--reset_optimizer_after_test` | `0` | diagnostic for the training-side channel (see below) |
| `--log_geometry` | `0` | `1` writes the support-set geometry and inner Hessian to `geometry_log.csv` (see below) |

`scripts/run_grid.sh` runs the main grid (SIDER, Tox21 and MUV, from scratch and pre-trained,
five seeds); `scripts/run_ablation.sh` the head ablation; `scripts/run_muv_controls.sh` the MUV
controls; `scripts/run_clean_grid.sh` the grid under the clean protocol. `scripts/run_queue.sh`
runs a file of `TAG ARGS...` lines with several workers on one GPU, each in `$W/runs/TAG`. The
scripts change to the code root, so relative paths given to them are taken from there.

## Evaluation protocols

The meta-test path inherited from ADKF-IFT's MoleculeNet code lets meta-test query labels reach the
model through two channels.

- *Evaluation side.* Besides the support set, each episode draws `update_step_test × n_query / 2`
  further labelled molecules of each class, uses their labels for a test-time adaptation step of
  the feature extractor, and then scores every molecule outside the support set, including those.
  See `sample_test_datasets` in `chem_lib/datasets/samples.py`, the adaptation loop in `test_step`
  of `chem_lib/models/bond_trainer.py`, and the query labels read from `q_data.y` in
  `chem_lib/models/bond_model.py`; the corresponding upstream files are
  `MoleculeNet/chem_lib/datasets/samples.py`, `adkfift_trainer.py` and `adkf_model.py` in the
  ADKF-IFT repository.
- *Training side.* The periodic meta-test evaluation during training (every `--eval_steps`
  epochs) takes its adaptation steps with the training AdamW optimiser, and afterwards restores the
  weights but not the optimiser's moment estimates. Gradients computed from meta-test labels
  therefore enter later meta-training updates.

`--eval_protocol` selects:

| value | adaptation | scored molecules |
|---|---|---|
| `legacy` | on the extra labelled molecules | everything outside the support set |
| `disjoint` | on the extra labelled molecules | everything outside the support set and the adaptation set |
| `support_only` | none | everything outside the support set |

`disjoint` removes the adaptation molecules from scoring but still adapts, so it closes the
evaluation-side channel only.

**Clean protocol.** Train and score with `--update_step_test 0 --eval_protocol support_only`, and
report the final evaluation, not the best one. No adaptation molecules are drawn and `test_step`
never adapts, so the support labels are the only labels used at test time and evaluation never
steps the optimiser. (`support_only` alone already skips the adaptation loop; `--update_step_test
0` makes this independent of the protocol flag.)

    CFGS="bond adkf adkfecfp" ./scripts/run_clean_grid.sh

`--reset_optimizer_after_test` is a diagnostic for the training-side channel under the legacy
protocol: `1` re-creates the optimiser after every evaluation (closes the channel but discards
momentum), `2` saves the optimiser state before the evaluation and restores it afterwards (closes
the channel and keeps momentum). The clean protocol does not need it.

Every evaluation appends one row per task to `<result_path>/.../test_metrics.csv`: `epoch`,
`task`, `protocol`, `auroc`, `auprc`, and the counts of scored molecules, scored positives,
adaptation molecules, and adaptation molecules (and positives) that were also scored.

`leak_exposure.py` reports, per meta-test assay, how many scored positives had their labels used
for adaptation under the legacy protocol: 1.0% on SIDER and 1.4% on Tox21 pooled over assays, and
40–57% on MUV, whose assays have 24–30 positives each.

`reeval_checkpoint.py` re-scores a saved model under any protocol, averaging over several support
draws (episode seeds 1000, 1001, ...) with a freshly built optimiser:

    python reeval_checkpoint.py --ckpt <run>/step_2000.pth --dataset sider --test-dataset sider \
        --pretrained 0 --eval_protocol support_only --repeats 10 --result_path work/reeval/sider

## Hypergradient

`chem_lib/models/cauchy_hypergradient.py` computes the outer gradient as the direct term
∂L_val/∂φ plus the implicit-function-theorem correction −∇_φ∇_ψ J · (∇²_ψ J)⁻¹ · ∇_ψ L_val, as in
ADKF-IFT. The direct term is dropped only when `ignore_direct_grad=True`, which no training path
sets.

With `--log_geometry 1` the trainer appends one row per meta-training task to
`<trial_path>/geometry_log.csv`. The row is written by a callback that `cauchy_hypergradient` calls
after assembling the 3×3 Hessian H of the inner objective and before the solve, so a task whose
solve fails still has its row. Logging does not change training. `summarise_geometry_log.py`
summarises a log per window of epochs.

| column | meaning |
|---|---|
| `epoch`, `step`, `task` | training epoch, outer update step within it, meta-training task |
| `lengthscale`, `outputscale`, `noise` | GP hyper-parameters after the inner fit |
| `r_min`, `r_max`, `r_median` | scaled support-set distances r_ij = ‖z_i − z_j‖ / ℓ over all pairs |
| `h_ll`, `h_ss`, `h_nn`, `h_ls`, `h_ln`, `h_sn` | entries of the symmetrised H (l length-scale, s output scale, n noise) |
| `eig_min`, `eig_max` | smallest and largest eigenvalue of the symmetrised H |
| `sv_min`, `cond` | smallest singular value and condition number of H as solved |
| `event` | `pre_solve`; an empty `singular_skip` row follows when the solve failed and the task was skipped; `_error:<type>` is appended if the values could not be computed |

H is taken in the coordinates the inner problem is solved in, gpytorch's raw (inverse-softplus)
parameters, not their logs; all six entries are kept so that it can be transformed to other
coordinates.

## Geometry measurements

`diagnose_geometry.py` measures the fingerprint share of the squared pairwise distance at the
input to the head, how far the head rebalances it, and the diameter of the mapped support set at
initialisation and after training. `dump_geometry_detail.py` writes the support-set coordinates
and pairwise distances behind the geometry figures. `geometry_class_separation.py` measures class
separation in the full space the GP sees, without projection: the between/within-class distance
ratio, the pair-AUROC of the kernel, kernel-target alignment and its centred form, and how much
each input block drives the distance; the definitions are in its docstring.
`geom_blockcorr_sets.py` repeats the block-influence measurement of `diagnose_geometry.py`, as
Pearson and Spearman correlations, for legacy-trained and clean-trained BOND on two molecule sets:
the 20-molecule support sets of meta-training tasks and the meta-test assays.
`geom_concat_rank.py` does the same for clean-trained ADKF-IFT on GIN + ECFP4 (identity head).

## Additional experiments

`scripts/reeval_*.sh`, `scripts/ecfpgp_eval.sh` and `scripts/after_scramble.sh` take the GPU as
their first argument. They, `run_clean_grid.sh` and `run_queue.sh` write to a work directory,
`work/` under the code root unless `W` is set, with `runs/`, `reeval/` and `logs/` inside it;
`run_grid.sh` and `run_ablation.sh` write to `results_kdd/` and `results_abl/`. Re-scores use 10
support draws per assay for `support_only` and 5 for the two adaptation protocols.

| result | produced by |
|---|---|
| clean protocol: BOND, ADKF-IFT, ADKF-IFT on GIN + ECFP4 and the other configurations of `run_clean_grid.sh` | `scripts/run_clean_grid.sh`, then `scripts/reeval_clean.sh 0` (each final checkpoint re-scored `support_only` by `reeval_checkpoint.py`) |
| clean protocol, ADKF-IFT at query size 32 | queue below (`--n-query 32`), then `scripts/reeval_clean.sh 0` |
| GP on ECFP4 | `scripts/ecfpgp_eval.sh 0` |
| ECFP4 Tanimoto 1-NN | `ecfp_1nn.py` (below; CPU, nothing trained, same episodes as the re-scores) |
| legacy-trained main grid under `legacy` / `disjoint` / `support_only` | `scripts/run_grid.sh`, then `scripts/reeval_all.sh 0 results_kdd` |
| legacy-trained head ablation and GIN-only arm, `support_only` | `scripts/run_ablation.sh`, then `scripts/reeval_ablation.sh 0 results_abl` |
| legacy-trained ADKF-IFT, `support_only` | queue below, then `scripts/reeval_legacy_adkf.sh 0` |
| label-scramble control | `scripts/after_scramble.sh 0` scores `work/runs/scramble_bond_<ds>_pre0_s0`, BOND trained under the legacy protocol with `--scramble_adapt_labels 1` |
| full-space class separation | `python geometry_class_separation.py --out work/geom_class_separation.csv` (clean runs from scratch, seeds 0–2) |
| block influence after the head, legacy-trained and clean-trained BOND | `SUBMITTED_RUNS=results_kdd python geom_blockcorr_sets.py --out work/geom_blockcorr_sets.csv` (main grid and clean `bond` runs from scratch) |
| block influence under plain concatenation | `python geom_concat_rank.py` (clean `adkfecfp` runs from scratch) |
| length-scale, scaled distances, inner-Hessian spectrum | `--log_geometry 1` runs (below), then `python summarise_geometry_log.py <run>/.../geometry_log.csv` |
| all tables | `python collect_tables.py`, which writes `work/results_tables.csv` and prints the summaries |

    python ecfp_1nn.py --dataset sider --repeats 10 --eval_protocol support_only \
        --result_path work/reeval/ecfp1nn_sider_s0_support_only

    mkdir -p work
    for ds in sider tox21; do for s in 0 1; do
      echo "legacy_adkf_${ds}_pre0_s$s --dataset $ds --test-dataset $ds --pretrained 0 --seed $s" \
           "--epochs 2000 --eval_steps 10 --save-steps 2000 --head_type identity --use_fingerprints 0 --use_gin 1"
    done; done > work/queue_legacy.txt
    ./scripts/run_queue.sh work/queue_legacy.txt 0 1

    for ds in sider tox21; do for cfg in "adkfq32 0" "adkfecfpq32 1"; do set -- $cfg
      echo "clean_$1_${ds}_pre0_s0 --dataset $ds --test-dataset $ds --pretrained 0 --seed 0" \
           "--update_step_test 0 --eval_protocol support_only --epochs 2000 --eval_steps 50" \
           "--save-steps 2000 --head_type identity --use_fingerprints $2 --use_gin 1 --n-query 32"
    done; done > work/queue_q32.txt
    cp work/queue_q32.txt work/queue_q32.txt.all
    ./scripts/run_queue.sh work/queue_q32.txt 0 1

    for head in dnm mlp; do
      python main_bond.py --dataset sider --test-dataset sider --pretrained 0 --seed 0 --epochs 400 \
          --eval_steps 10 --save-steps 400 --head_type $head --eval_protocol legacy --log_geometry 1 \
          --result_path work/runs/geom_${head}_sider_pre0_s0
    done

The `tanh_linear` log runs the full 2000 epochs and stops at a Cholesky failure near epoch 1857:

    python main_bond.py --dataset sider --test-dataset sider --pretrained 0 --seed 0 --epochs 2000 \
        --eval_steps 10 --save-steps 2000 --head_type tanh_linear --eval_protocol legacy --log_geometry 1 \
        --result_path work/runs/geom_tanh_sider_pre0_s0

`collect_tables.py` averages a re-score over draws per assay, then over assays. For a training
run it reports the final evaluation, with the best evaluation over training listed alongside for
comparison.

## Licence

MIT, as for ADKF-IFT; see `LICENSE`. ADKF-IFT's licence file is reproduced unchanged in
`LICENSE-ADKF-IFT`; its Creative Commons part covers the `datasets/` directory of FS-Mol, none of
which is included here. PAR, on which ADKF-IFT is built, is credited above.
