#!/usr/bin/env bash
# Head ablation: boundedness x branch aggregation.
#
#                     | branched (M=30)   | unbranched
#   ------------------+-------------------+---------------------------
#   bounded           | dnm  (BOND)       | tanh_linear, l2norm
#   unbounded         | branch_linear     | mlp (parameter-matched)
#
# tanh_linear and l2norm serve as normalisation-layer baselines. The default HEADS leave out
# dnm with the fingerprint, which is the main-grid configuration (run_grid.sh); NOFP=1 adds a
# GIN-only dnm arm (--use_fingerprints 0).
#
# Each run gets its own --result_path: init_trial_path (chem_lib/utils.py:22-29) races between
# os.path.exists and os.makedirs when runs start together.
#
# Usage:
#   HEADS="dnm mlp" DATASETS="sider" SEEDS="0 1 2" NWORKERS=4 GPUS="0 1" ./scripts/run_ablation.sh
set -u
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-results_abl}
LOGS=${LOGS:-logs_abl}
EPOCHS=${EPOCHS:-2000}
SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"sider tox21"}
PRETRAINED=${PRETRAINED:-"0"}
HEADS=${HEADS:-"branch_linear tanh_linear l2norm mlp"}
NOFP=${NOFP:-1}          # also run a dnm arm with the fingerprint removed
NWORKERS=${NWORKERS:-4}
GPUS=${GPUS:-"0 1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

mkdir -p "$OUT" "$LOGS"
QUEUE="$LOGS/queue.txt"; LOCK="$LOGS/queue.lock"; : > "$QUEUE"

for ds in $DATASETS; do
  for pre in $PRETRAINED; do
    for seed in $SEEDS; do
      for head in $HEADS; do
        echo "$ds $pre $seed $head 1" >> "$QUEUE"
      done
      [ "$NOFP" = "1" ] && echo "$ds $pre $seed dnm 0" >> "$QUEUE"
    done
  done
done
echo "queued $(wc -l < "$QUEUE") ablation runs -> $OUT"

pop_job() {
  flock "$LOCK" bash -c '
    q="$1"; [ -s "$q" ] || exit 1
    head -n 1 "$q"; tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"
  ' _ "$QUEUE"
}

worker() {
  local wid=$1 gpu=$2 job ds pre seed head fp tag rp
  while true; do
    job=$(pop_job) || { echo "[w$wid] queue empty"; return 0; }
    [ -n "$job" ] || return 0
    set -- $job; ds=$1; pre=$2; seed=$3; head=$4; fp=$5
    tag="${ds}_pre${pre}_${head}_fp${fp}_seed${seed}"
    rp="$OUT/$tag"
    [ -f "$rp/DONE" ] && { echo "[w$wid] skip $tag"; continue; }
    mkdir -p "$rp"
    echo "[w$wid gpu$gpu] START $tag $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu $PY main_bond.py \
      --dataset "$ds" --test-dataset "$ds" --pretrained "$pre" --seed "$seed" \
      --epochs "$EPOCHS" --eval_steps 10 \
      --head_type "$head" --use_fingerprints "$fp" \
      --result_path "$rp" > "$LOGS/$tag.log" 2>&1 \
      && { touch "$rp/DONE"; echo "[w$wid gpu$gpu] OK   $tag $(date +%H:%M:%S)"; } \
      || echo "[w$wid gpu$gpu] FAIL $tag $(date +%H:%M:%S)"
  done
}

i=0
for w in $(seq 1 "$NWORKERS"); do
  arr=($GPUS); gpu=${arr[$(( i % ${#arr[@]} ))]}
  worker "$w" "$gpu" & i=$((i+1)); sleep 3
done
wait
echo "ABLATION GRID FINISHED $(date)"
