#!/usr/bin/env bash
# MUV controls (appendix, MUV table): three arms that isolate the contribution of the
# test-time adaptation step to the score.
#
#   A  original evaluation path (test-time adaptation on)
#   E  as A with the adaptation labels permuted: removes the label information, keeps the
#      molecules, batches, optimiser steps and clipping
#   F  test-time adaptation disabled (clean baseline)
#
# Usage:  ./scripts/run_muv_controls.sh
#         SEEDS="0 1 2" ./scripts/run_muv_controls.sh
set -u
cd "$(dirname "$0")/.."

PY=${PY:-python}
SEEDS=${SEEDS:-"1 2"}
EPOCHS=${EPOCHS:-500}
DATASET=${DATASET:-muv}
OUT=${OUT:-results_controls}
LOGS=${LOGS:-logs_controls}

mkdir -p "$OUT" "$LOGS"

run() {   # run <tag> <extra-args...>
  local tag=$1; shift
  local rp="$OUT/$tag"
  if [ -f "$rp/DONE" ]; then echo "skip $tag"; return 0; fi
  mkdir -p "$rp"
  echo "START $tag  $(date +%H:%M:%S)"
  "$PY" main_bond.py \
    --dataset "$DATASET" --test-dataset "$DATASET" --pretrained 1 \
    --epochs "$EPOCHS" --eval_steps 10 \
    --result_path "$rp" "$@" > "$LOGS/$tag.log" 2>&1
  if [ $? -eq 0 ]; then touch "$rp/DONE"; echo "OK    $tag  $(date +%H:%M:%S)"
  else echo "FAIL  $tag  (see $LOGS/$tag.log)"; fi
}

for s in $SEEDS; do
  run "A_seed${s}" --seed "$s" --update_step_test 1
  run "E_seed${s}" --seed "$s" --update_step_test 1 --scramble_adapt_labels 1
  run "F_seed${s}" --seed "$s" --update_step_test 0
done

echo "CONTROLS FINISHED $(date)"
echo "Summarise with:  python collect_results.py --root $OUT"
