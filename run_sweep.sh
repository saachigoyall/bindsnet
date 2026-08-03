#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# Spiking CA parameter sweep — runs all configurations in parallel
# Usage: bash run_sweep.sh
# Results saved to results/ folder as JSON files
#
# DO NOT RUN THIS until the baseline check in README.md ("Reset fix —
# read before running anything below") has been done and confirms the
# reset fix changes the learn-then-forget curve. Running this sweep
# before that check produces results on the old (broken) reset logic,
# which become obsolete the moment the fix is applied — see Hazan's
# note. Requires spiking_ca_cluster.py with the reset fix applied.
# ══════════════════════════════════════════════════════════════════════

mkdir -p results

# Priority 1 — Eligibility trace timing
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 50  --tc_minus 50  --average_update 5  --output_file results/tc50_ma5.json &
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 100 --tc_minus 100 --average_update 10 --output_file results/tc100_ma10.json &
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 200 --tc_minus 200 --average_update 20 --output_file results/tc200_ma20.json &
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 500 --tc_minus 500 --average_update 50 --output_file results/tc500_ma50.json &

# Priority 2 — Exploration schedule
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 100  --output_file results/eps100.json &
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 300  --output_file results/eps300.json &
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 500  --output_file results/eps500.json &
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 1000 --output_file results/eps1000.json &

# Priority 3 — Target network update frequency
python examples/spiking_ca/spiking_ca_cluster.py --target_update 10  --output_file results/tgt10.json &
python examples/spiking_ca/spiking_ca_cluster.py --target_update 20  --output_file results/tgt20.json &
python examples/spiking_ca/spiking_ca_cluster.py --target_update 50  --output_file results/tgt50.json &
python examples/spiking_ca/spiking_ca_cluster.py --target_update 100 --output_file results/tgt100.json &

# Priority 4 — Longer training
python examples/spiking_ca/spiking_ca_cluster.py --n_episodes 1000 --output_file results/ep1000.json &
python examples/spiking_ca/spiking_ca_cluster.py --n_episodes 2000 --output_file results/ep2000.json &

# wait for all jobs to finish
wait
echo "All runs complete. Results saved to results/"
