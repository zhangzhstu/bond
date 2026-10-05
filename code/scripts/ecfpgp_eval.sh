#!/usr/bin/env bash
# Fingerprint-only Gaussian process: identity head, GIN embedding zeroed. The features have no
# trainable parameters and the kernel hyper-parameters are refitted on every support set, so the
# checkpoint saved after one epoch is the final model. Scored under the support-only protocol
# with ten support draws per seed.
#
#   ./scripts/ecfpgp_eval.sh GPU
set -u
cd "$(dirname "$0")/.."
GPU=$1
W=${W:-$PWD/work}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
umask 002
mkdir -p "$W/runs" "$W/reeval" "$W/logs"
ARGS="--head_type identity --use_fingerprints 1 --use_gin 0 --pretrained 0"
for ds in sider tox21; do
  for seed in 0 1 2; do
    tag=ecfpgp_${ds}_s${seed}
    rp=$W/runs/$tag
    [ -f "$rp/DONE" ] && continue
    CUDA_VISIBLE_DEVICES=$GPU $PY main_bond.py --dataset $ds --test-dataset $ds --seed $seed \
      --epochs 1 --update_step_test 0 --eval_protocol support_only --save-steps 1 $ARGS \
      --result_path "$rp" > "$W/logs/$tag.log" 2>&1 || { echo "FAIL train $tag"; continue; }
    ck=$(find "$rp" -name step_1.pth | head -1)
    CUDA_VISIBLE_DEVICES=$GPU $PY reeval_checkpoint.py --ckpt "$ck" --repeats 10 \
      --dataset $ds --test-dataset $ds --seed $seed --eval_protocol support_only $ARGS \
      --result_path "$W/reeval/${tag}_support_only" > "$W/logs/reeval_${tag}.log" 2>&1 \
      && touch "$rp/DONE" "$W/reeval/${tag}_support_only/DONE" || echo "FAIL reeval $tag"
    echo "done $tag $(date '+%m-%d %H:%M')"
  done
done
