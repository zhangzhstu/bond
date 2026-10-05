#!/usr/bin/env bash
# Re-score the legacy-trained ADKF-IFT runs ($W/runs/legacy_adkf_<ds>_pre0_s<seed>) under the
# support-only protocol with ten support draws, like the BOND checkpoints and the clean runs.
#
#   ./scripts/reeval_legacy_adkf.sh GPU
set -u
cd "$(dirname "$0")/.."
GPU=$1
W=${W:-$PWD/work}
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} CUDA_DEVICE_ORDER=PCI_BUS_ID
umask 002
mkdir -p "$W/runs" "$W/reeval" "$W/logs"
for ds in sider tox21; do
  for s in 0 1; do
    t=legacy_adkf_${ds}_pre0_s${s}
    out="$W/reeval/${t}_support_only"
    [ -f "$out/DONE" ] && continue
    ck=$(find "$W/runs/$t" -name step_2000.pth | head -1)
    [ -n "$ck" ] || { echo "no checkpoint: $t"; continue; }
    CUDA_VISIBLE_DEVICES=$GPU $PY reeval_checkpoint.py --ckpt "$ck" --repeats 10 \
        --dataset $ds --test-dataset $ds --pretrained 0 --seed $s \
        --head_type identity --use_fingerprints 0 --use_gin 1 --eval_protocol support_only \
        --result_path "$out" > "$W/logs/reeval_${t}.log" 2>&1 && touch "$out/DONE" || echo "FAIL $t"
    echo "done $t $(date '+%m-%d %H:%M')"
  done
done
