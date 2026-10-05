#!/usr/bin/env bash
# Run a list of jobs with N workers on one GPU.
#
#   ./scripts/run_queue.sh QUEUE_FILE GPU NWORKERS
#
# Each line of QUEUE_FILE is "TAG ARGS...". ARGS go to main_bond.py unchanged; the result path
# and log are derived from TAG, so re-running the same queue skips anything already DONE. Jobs
# are popped atomically under a lock in /tmp, so one queue file must be consumed from one host.
set -u
cd "$(dirname "$0")/.."
QUEUE=$1; GPU=$2; NW=${3:-1}
W=${W:-$PWD/work}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
umask 002
mkdir -p "$W/runs" "$W/logs"
LOCK=/tmp/bond_queue_$(basename "$QUEUE").lock

pop() {
  flock "$LOCK" bash -c '
    q="$1"; [ -s "$q" ] || exit 1
    head -n 1 "$q"; tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"' _ "$QUEUE"
}

worker() {
  local wid=$1 fails=0
  while job=$(pop); do
    [ -n "$job" ] || continue
    local tag=${job%% *} args=${job#* }
    local rp="$W/runs/$tag"
    if [ -f "$rp/DONE" ]; then echo "[w$wid] skip $tag"; continue; fi
    mkdir -p "$rp"
    echo "[w$wid gpu$GPU] START $tag $(date '+%m-%d %H:%M')"
    ( CUDA_VISIBLE_DEVICES=$GPU $PY main_bond.py $args \
        --result_path "$rp" > "$W/logs/$tag.log" 2>&1 )
    if [ $? -eq 0 ]; then touch "$rp/DONE"; fails=0; echo "[w$wid] OK    $tag $(date '+%m-%d %H:%M')"
    else
      fails=$((fails+1)); echo "$job" >> "$QUEUE.failed"
      echo "[w$wid] FAIL  $tag $(date '+%m-%d %H:%M')  (kept in $(basename "$QUEUE").failed)"
      # consecutive failures usually mean a broken environment rather than bad jobs: stop
      # instead of draining the whole queue into .failed
      if [ $fails -ge 3 ]; then echo "[w$wid] 3 failures in a row, stopping"; return 1; fi
    fi
  done
  echo "[w$wid] queue empty"
}

for i in $(seq 1 "$NW"); do worker "$i" & sleep 5; done
wait
echo "ALL DONE $(date)"
