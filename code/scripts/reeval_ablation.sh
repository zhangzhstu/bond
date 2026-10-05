#!/usr/bin/env bash
# Re-score the head-ablation checkpoints under the support-only protocol, ten support draws each.
# The run directory name carries the configuration: <ds>_pre<p>_<head>_fp<0|1>_seed<s>.
#
#   ./scripts/reeval_ablation.sh GPU ROOT [ROOT ...]
set -u
cd "$(dirname "$0")/.."
GPU=$1; shift
W=${W:-$PWD/work}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
umask 002
mkdir -p "$W/runs" "$W/reeval" "$W/logs"
for root in "$@"; do
  for run in $(ls "$root" | sort); do
    [[ $run =~ ^([a-z0-9]+)_pre([01])_([a-z0-9_]+)_fp([01])_seed([0-9]+)$ ]] || continue
    ds=${BASH_REMATCH[1]}; pre=${BASH_REMATCH[2]}; head=${BASH_REMATCH[3]}
    fp=${BASH_REMATCH[4]}; seed=${BASH_REMATCH[5]}
    out="$W/reeval/abl_${run}_support_only"
    [ -f "$out/DONE" ] && continue
    ckpt=$(find "$root/$run" -name step_2000.pth | head -1)
    [ -n "$ckpt" ] || { echo "no checkpoint: $run"; continue; }
    echo "START $run $(date '+%m-%d %H:%M')"
    CUDA_VISIBLE_DEVICES=$GPU $PY reeval_checkpoint.py --ckpt "$ckpt" --repeats 10 \
        --dataset "$ds" --test-dataset "$ds" --pretrained "$pre" --seed "$seed" \
        --head_type "$head" --use_fingerprints "$fp" --eval_protocol support_only \
        --result_path "$out" > "$W/logs/reeval_abl_${run}.log" 2>&1 \
      && touch "$out/DONE" || echo "FAIL $run"
  done
done
echo "ABLATION REEVAL DONE $(date)"
