#!/usr/bin/env bash
# Re-score every finished clean-protocol run from its final checkpoint (support_only, ten
# support draws), so the score is averaged over support sets rather than taken from one draw.
# Each run's model options are read back from $W/queue_*.all (written by run_clean_grid.sh).
#
#   ./scripts/reeval_clean.sh GPU
set -u
cd "$(dirname "$0")/.."
GPU=$1
W=${W:-$PWD/work}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
umask 002
mkdir -p "$W/runs" "$W/reeval" "$W/logs"
for d in $(ls -d "$W"/runs/clean_* 2>/dev/null | sort); do
  tag=$(basename "$d")
  [ -f "$d/DONE" ] || continue
  out="$W/reeval/${tag}_support_only"
  [ -f "$out/DONE" ] && continue
  args=$(grep -h "^$tag " "$W"/queue_*.all 2>/dev/null | head -1 | cut -d' ' -f2-)
  [ -n "$args" ] || { echo "no args for $tag"; continue; }
  # keep model-defining options, drop the training schedule
  margs=$(echo "$args" | sed -E 's/--(epochs|eval_steps|save-steps|update_step_test|eval_protocol) [^ ]+//g')
  ckpt=$(find "$d" -name step_2000.pth | head -1)
  [ -n "$ckpt" ] || { echo "no checkpoint for $tag"; continue; }
  echo "START $tag $(date '+%m-%d %H:%M')"
  CUDA_VISIBLE_DEVICES=$GPU $PY reeval_checkpoint.py --ckpt "$ckpt" --repeats 10 $margs \
      --eval_protocol support_only --result_path "$out" > "$W/logs/reeval_${tag}.log" 2>&1 \
    && touch "$out/DONE" || echo "FAIL $tag"
done
echo "CLEAN REEVAL PASS DONE $(date)"
