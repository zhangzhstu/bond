#!/usr/bin/env bash
# Main MoleculeNet grid: SIDER, Tox21 and MUV by default (set DATASETS to include toxcast),
# both encoder settings (from scratch / pre-trained), five seeds. The seed sets the data split
# ordering, the episode sampling and the initialisation. Uses the evaluation path inherited from
# ADKF-IFT; see run_clean_grid.sh for the protocol without test-time adaptation.
#
# Each run gets its own --result_path because init_trial_path() (chem_lib/utils.py:22-29)
# races between os.path.exists() and os.makedirs() when runs start concurrently.
#
# Usage:
#   ./scripts/run_grid.sh                  # run the full queue
#   NWORKERS=4 EPOCHS=2000 ./scripts/run_grid.sh
#   DATASETS="sider tox21" SEEDS="0 1 2" ./scripts/run_grid.sh

set -u
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-results_kdd}
LOGS=${LOGS:-logs_kdd}
EPOCHS=${EPOCHS:-2000}
EVAL_STEPS=${EVAL_STEPS:-10}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-"sider tox21 muv"}   # cheapest first, so results arrive early
PRETRAINED=${PRETRAINED:-"0 1"}
NWORKERS=${NWORKERS:-4}
GPUS=${GPUS:-"0 1"}
# Each job already uses ~3.3 CPU cores; cap BLAS threads so 4 concurrent jobs
# do not oversubscribe a 12-core host.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p "$OUT" "$LOGS"
QUEUE="$LOGS/queue.txt"
LOCK="$LOGS/queue.lock"
: > "$QUEUE"

# Build the job list, cheapest dataset first.
for ds in $DATASETS; do
  for pre in $PRETRAINED; do
    for seed in $SEEDS; do
      echo "$ds $pre $seed" >> "$QUEUE"
    done
  done
done
TOTAL=$(wc -l < "$QUEUE")
echo "queued $TOTAL runs -> $OUT  (epochs=$EPOCHS, workers=$NWORKERS, seeds='$SEEDS')"

# Pop one line from the queue atomically.
pop_job() {
  flock "$LOCK" bash -c '
    q="$1"
    [ -s "$q" ] || exit 1
    head -n 1 "$q"
    tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
  ' _ "$QUEUE"
}

worker() {
  local wid=$1 gpu=$2
  while true; do
    local job
    job=$(pop_job) || { echo "[w$wid] queue empty, exiting"; return 0; }
    [ -n "$job" ] || return 0
    set -- $job
    local ds=$1 pre=$2 seed=$3
    local tag="${ds}_pre${pre}_seed${seed}"
    local rp="$OUT/$tag"

    if [ -f "$rp/DONE" ]; then
      echo "[w$wid] skip $tag (already DONE)"
      continue
    fi
    mkdir -p "$rp"
    echo "[w$wid gpu$gpu] START $tag  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu $PY main_bond.py \
      --dataset "$ds" --test-dataset "$ds" \
      --pretrained "$pre" --seed "$seed" \
      --epochs "$EPOCHS" --eval_steps "$EVAL_STEPS" \
      --result_path "$rp" > "$LOGS/$tag.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
      touch "$rp/DONE"
      echo "[w$wid gpu$gpu] OK    $tag  $(date +%H:%M:%S)"
    else
      echo "[w$wid gpu$gpu] FAIL  $tag rc=$rc  $(date +%H:%M:%S)  (see $LOGS/$tag.log)"
    fi
  done
}

# Spread workers across the available GPUs.
i=0
for w in $(seq 1 "$NWORKERS"); do
  gpu_arr=($GPUS)
  gpu=${gpu_arr[$(( i % ${#gpu_arr[@]} ))]}
  worker "$w" "$gpu" &
  i=$((i+1))
  sleep 3   # stagger so concurrent init_trial_path calls cannot collide
done
wait
echo "ALL RUNS FINISHED  $(date)"
