#!/bin/bash
# usage: run_all_variants.sh <worker 0|1>
cd "$(dirname "$0")"
i=0
for s in 1 2 3; do for v in base lateral5 shared_theta percell_theta percell_theta_small hetero_decay percell_double bal0.25 bal0.5; do
  if [ $((i % 2)) -eq $1 ] && [ ! -f variant_${v}_seed${s}.json ]; then python step10_variants.py $v $s 150 > log_${v}_$s.txt 2>&1; fi
  i=$((i+1))
done; done
echo worker $1 finished > worker_$1.done
