#!/usr/bin/env bash
# Train under the clean meta-test protocol (--update_step_test 0, support_only): no test-time
# adaptation, so scoring uses the support labels only and no meta-test gradients enter the
# optimiser state used for training. Report the final evaluation, not the best one.
#
# Writes one job per configuration, dataset, encoder setting and seed to $W/queue_clean.txt and
# runs it with scripts/run_queue.sh; each run goes to $W/runs/clean_<cfg>_<ds>_pre<p>_s<seed>.
# Every job is also kept in $W/queue_clean.txt.all, where reeval_clean.sh looks up its arguments.
#
#   CFGS="bond adkf adkfecfp" DATASETS="sider tox21" PRETRAINED="0 1" SEEDS="0 1 2" \
#       GPU=0 NWORKERS=2 ./scripts/run_clean_grid.sh
#
# Configurations:
#   bond      BOND head over GIN + ECFP4
#   adkf      no head, GIN only            (ADKF-IFT feature map)
#   adkfecfp  no head, GIN + ECFP4         (ADKF-IFT with the same input as BOND)
#   ecfpgp    no head, ECFP4 only          (fingerprint Gaussian process; see ecfpgp_eval.sh)
#   bondgin   BOND head, GIN only
#   bondecfp  BOND head, ECFP4 only
#   bondM60   BOND head with M=60 branches
#   branch    branched, unbounded head over GIN + ECFP4
set -u
cd "$(dirname "$0")/.."

W=${W:-$PWD/work}
GPU=${GPU:-0}
NWORKERS=${NWORKERS:-1}
CFGS=${CFGS:-"bond adkf adkfecfp"}
DATASETS=${DATASETS:-"sider tox21"}
PRETRAINED=${PRETRAINED:-"0 1"}
SEEDS=${SEEDS:-"0 1 2"}
CLEAN="--update_step_test 0 --eval_protocol support_only --epochs 2000 --eval_steps 50 --save-steps 2000"

cfg_args() {
  case $1 in
    bond)     echo "--head_type dnm --use_fingerprints 1 --use_gin 1" ;;
    adkf)     echo "--head_type identity --use_fingerprints 0 --use_gin 1" ;;
    adkfecfp) echo "--head_type identity --use_fingerprints 1 --use_gin 1" ;;
    ecfpgp)   echo "--head_type identity --use_fingerprints 1 --use_gin 0" ;;
    bondgin)  echo "--head_type dnm --use_fingerprints 0 --use_gin 1" ;;
    bondecfp) echo "--head_type dnm --use_fingerprints 1 --use_gin 0" ;;
    bondM60)  echo "--head_type dnm --use_fingerprints 1 --use_gin 1 --dnm_M 60" ;;
    branch)   echo "--head_type branch_linear --use_fingerprints 1 --use_gin 1" ;;
    *) echo "unknown configuration $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$W"
Q="$W/queue_clean.txt"
: > "$Q"
for seed in $SEEDS; do
  for cfg in $CFGS; do
    args=$(cfg_args "$cfg") || exit 1
    for ds in $DATASETS; do
      for pre in $PRETRAINED; do
        echo "clean_${cfg}_${ds}_pre${pre}_s${seed} --dataset $ds --test-dataset $ds" \
             "--pretrained $pre --seed $seed $CLEAN $args" >> "$Q"
      done
    done
  done
done
touch "$Q.all"
sort -u -o "$Q.all" "$Q.all" "$Q"
echo "queued $(wc -l < "$Q") clean runs -> $W/runs"

export W
exec bash scripts/run_queue.sh "$Q" "$GPU" "$NWORKERS"
