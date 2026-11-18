#!/bin/bash
# Disk-peak stagger: resume the paused mcpm precompute jobs one at a time,
# only after the previous wave of cache writers has fully closed.
cd /data1/suzeqin/SpeculativeProbing
REP_PID=2480384   # mcpm cot_repetitive (SIGSTOPed at ~62%)
BR_PID=2480386    # mcpm cot_branched   (SIGSTOPed at ~65%)

# 1. wait for both 9b jobs to finish (their writers close and free raws)
while ! grep -q "PRECOMPUTE_ONE_DONE 9b cot_repetitive" cot_local/pre_9b_rep.log 2>/dev/null; do sleep 120; done
while ! grep -q "PRECOMPUTE_ONE_DONE 9b cot_branched" cot_local/pre_9b_br.log 2>/dev/null; do sleep 120; done
echo "stagger: 9b both done at $(date '+%H:%M:%S'); resuming mcpm repetitive"
kill -CONT $REP_PID

# 2. wait for mcpm repetitive to finish before resuming branched
while ! grep -q "PRECOMPUTE_ONE_DONE mcpm cot_repetitive" cot_local/pre_mcpm_rep.log 2>/dev/null; do sleep 120; done
echo "stagger: mcpm rep done at $(date '+%H:%M:%S'); resuming mcpm branched"
kill -CONT $BR_PID
echo "stagger: complete"
