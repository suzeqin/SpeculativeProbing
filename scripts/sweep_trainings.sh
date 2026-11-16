#!/usr/bin/env bash
# Sweep until all 45 experiments have final_results.json AND best.pt.
# Requeues anything missing either file; exits when complete.
cd /data1/suzeqin/SpeculativeProbing
PY=/home/suzeqin/miniconda3/envs/sp/bin/python
GPUS=${GPUS:-4,5,6,7}
for round in 1 2 3 4 5 6; do
  # count complete jobs (final_results.json + best.pt)
  complete=0
  for d in results_new_benchmark/*_4b/* results_new_benchmark/*_9b/* results_new_benchmark/*_mcpm/*; do
    [ -f "$d/final_results.json" ] && [ -f "$d/best.pt" ] && complete=$((complete+1))
  done
  echo "[sweep $round] complete: $complete/45"
  if [ "$complete" -ge 45 ]; then
    echo "[sweep] ALL DONE"
    break
  fi
  # remove incomplete dirs so the runner reruns them
  for d in results_new_benchmark/*_4b/* results_new_benchmark/*_9b/* results_new_benchmark/*_mcpm/*; do
    if [ -d "$d" ] && { [ ! -f "$d/final_results.json" ] || [ ! -f "$d/best.pt" ]; }; then
      # skip dirs of currently-running jobs
      if ! pgrep -f "$(basename $(dirname $d))" >/dev/null 2>&1; then
        echo "[sweep] removing incomplete $d"
        rm -rf "$d"
      fi
    fi
  done
  $PY new_benchmark_repro.py --out results_new_benchmark --gpus $GPUS >> logs_train_sweep.log 2>&1
  # also rerun anything on the 4090s if A6000s are saturated? (not needed; keep A6000-only)
done
echo "[sweep] final state:"
find results_new_benchmark -name best.pt | wc -l
