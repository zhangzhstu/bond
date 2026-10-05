#!/usr/bin/env bash
# Re-score every main-grid checkpoint under the three meta-test protocols.
#
#   ./scripts/reeval_all.sh GPU CKPT_ROOT
#
# support_only gets 10 support draws per checkpoint; legacy and disjoint, which run the
# test-time adaptation step and are slower, get 5. Output goes to $W/reeval/<run>_<protocol>.
set -u
cd "$(dirname "$0")/.."
GPU=$1; CKPT_ROOT=$2
W=${W:-$PWD/work}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
umask 002
mkdir -p "$W/runs" "$W/reeval" "$W/logs"

for proto in support_only disjoint legacy; do
  reps=5; [ "$proto" = support_only ] && reps=10
  for run in $(ls "$CKPT_ROOT" | sort); do
    ds=${run%%_*}
    pre=$(echo "$run" | sed -E 's/.*_pre([01])_.*/\1/')
    seed=$(echo "$run" | sed -E 's/.*_seed([0-9]+).*/\1/')
    ckpt=$(find "$CKPT_ROOT/$run" -name 'step_2000.pth' | head -1)
    out="$W/reeval/${run}_${proto}"
    [ -f "$out/DONE" ] && continue
    [ -n "$ckpt" ] || { echo "no checkpoint for $run"; continue; }
    echo "START $run $proto $(date '+%m-%d %H:%M')"
    CUDA_VISIBLE_DEVICES=$GPU $PY reeval_checkpoint.py --ckpt "$ckpt" --repeats $reps \
        --dataset "$ds" --test-dataset "$ds" --pretrained "$pre" --seed "$seed" \
        --eval_protocol "$proto" --result_path "$out" > "$W/logs/reeval_${run}_${proto}.log" 2>&1 \
      && touch "$out/DONE" || echo "FAIL $run $proto"
  done
done
echo "REEVAL DONE $(date)"
