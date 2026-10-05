#!/usr/bin/env bash
# Wait for the label-scramble runs (BOND trained under the legacy protocol with
# --scramble_adapt_labels 1) to finish, score their final checkpoints under the support-only
# protocol with ten support draws, like the other re-scored checkpoints, and rebuild the summary.
#
#   ./scripts/after_scramble.sh GPU
set -u
cd "$(dirname "$0")/.."
GPU=$1
W=${W:-$PWD/work}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
umask 002
mkdir -p "$W/runs" "$W/reeval" "$W/logs"
RUNS="scramble_bond_sider_pre0_s0 scramble_bond_tox21_pre0_s0"
for t in $RUNS; do
  until [ -f "$W/runs/$t/DONE" ]; do
    # stop waiting if the run died: no live process for it and no DONE marker
    pgrep -u "$(id -u)" -f "result_path $W/runs/$t" > /dev/null || { [ -f "$W/runs/$t/DONE" ] || { echo "$t ended without DONE"; break; }; }
    sleep 300
  done
done
for t in $RUNS; do
  [ -f "$W/runs/$t/DONE" ] || continue
  ds=$(echo "$t" | cut -d_ -f3)
  ck=$(find "$W/runs/$t" -name step_2000.pth | head -1)
  out="$W/reeval/${t}_support_only"
  [ -f "$out/DONE" ] && continue
  echo "score $t $(date '+%m-%d %H:%M')"
  CUDA_VISIBLE_DEVICES=$GPU $PY reeval_checkpoint.py --ckpt "$ck" --repeats 10 \
      --dataset "$ds" --test-dataset "$ds" --pretrained 0 --seed 0 \
      --head_type dnm --use_fingerprints 1 --eval_protocol support_only \
      --result_path "$out" > "$W/logs/reeval_${t}.log" 2>&1 && touch "$out/DONE"
done
$PY collect_tables.py "$W" > "$W/results_summary.txt" 2>&1
echo "summary written $(date '+%m-%d %H:%M')"
