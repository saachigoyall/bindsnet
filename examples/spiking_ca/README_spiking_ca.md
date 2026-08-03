# Spiking Neural CA for CartPole

Implementation of a spiking cellular automaton that learns to control CartPole
using reward-modulated STDP. Combines the neural CA architecture from
Variengien et al. (2021) with spiking neurons and local learning rules via BindsNET.

## Setup

```bash
pip install git+https://github.com/BindsNET/bindsnet.git
pip install gymnasium
```

## Reset fix — read before running anything below

`spiking_ca_cluster.py` previously never called `reset_state_variables()`
between episodes, and the installed `MSTDPET.reset_state_variables` itself
was incomplete — it only zeroed `eligibility` and `eligibility_trace`,
leaving `p_plus`, `p_minus`, and the moving-average buffer to silently
persist across episodes. Both are now fixed in this script (a patch
applied after import, plus the missing reset call at the top of each
training episode). A source-level fix for the second issue is up as a
PR: https://github.com/saachigoyall/bindsnet (pending review).

**Before running the full sweep below, run one baseline configuration
and confirm the learn-then-forget curve actually changes:**

```bash
python examples/spiking_ca/spiking_ca_cluster.py \
    --tc_plus 100 --tc_minus 100 --average_update 10 \
    --output_file results_baseline_fixed.json
```

Compare `results_baseline_fixed.json` against a pre-fix baseline run.
Only proceed to the full sweep (`run_sweep.sh`, 14 configs) once this
comparison shows the fix is doing something — otherwise the sweep numbers
below are measuring the old bug, not the parameters.

## Run a single configuration

```bash
python examples/spiking_ca/spiking_ca_cluster.py \
    --n_episodes 500 \
    --tc_plus 100.0 \
    --tc_minus 100.0 \
    --average_update 10 \
    --output_file results_default.json
```

## Parameter sweep (run all in parallel on cluster)

```bash
# sweep tc_plus / tc_minus
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 50  --tc_minus 50  --output_file results_tc50.json
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 100 --tc_minus 100 --output_file results_tc100.json
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 200 --tc_minus 200 --output_file results_tc200.json

# sweep average_update (moving average window)
python examples/spiking_ca/spiking_ca_cluster.py --average_update 5  --output_file results_ma5.json
python examples/spiking_ca/spiking_ca_cluster.py --average_update 10 --output_file results_ma10.json
python examples/spiking_ca/spiking_ca_cluster.py --average_update 20 --output_file results_ma20.json

# sweep epsilon decay
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 100 --output_file results_eps100.json
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 300 --output_file results_eps300.json
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 500 --output_file results_eps500.json

# sweep target network update frequency
python examples/spiking_ca/spiking_ca_cluster.py --target_update 10 --output_file results_tgt10.json
python examples/spiking_ca/spiking_ca_cluster.py --target_update 20 --output_file results_tgt20.json
python examples/spiking_ca/spiking_ca_cluster.py --target_update 40 --output_file results_tgt40.json
```

## All parameters

| Parameter | Default | Description |
|---|---|---|
| `--n_episodes` | 500 | Number of training episodes |
| `--max_steps` | 200 | Max steps per episode |
| `--tc_plus` | 100.0 | Pre-synaptic trace time constant |
| `--tc_minus` | 100.0 | Post-synaptic trace time constant |
| `--nu_ih` | 0.01 | Learning rate input→hidden |
| `--nu_ho` | 0.01 | Learning rate hidden→output |
| `--average_update` | 10 | Moving average window for eligibility trace |
| `--target_update` | 20 | Target network update frequency (episodes) |
| `--epsilon_decay` | 300 | Episodes over which epsilon decays from 1.0 to 0.1 |
| `--n_inference_steps` | 3 | CA steps per action for inference |
| `--grid_size` | 16 | CA grid size |
| `--hidden_n` | 32 | Number of hidden neurons |
| `--sparse_ih` | 0.4 | Fraction of input→hidden connections per neuron |
| `--sparse_ho` | 0.4 | Fraction of hidden→output connections per neuron |
| `--lateral_w` | 5.0 | Lateral inhibition weight magnitude |
| `--output_file` | results.json | File to save results |

## Recommended parameter sweep

Based on our experiments, the learn-then-forget pattern persists across configurations. The following combinations are most likely to reveal what is causing it and how to fix it. Each run takes roughly 30-60 minutes on CPU.

**Priority 1 — Eligibility trace timing (most likely cause)**
```bash
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 50  --tc_minus 50  --average_update 5  --output_file results_tc50_ma5.json
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 100 --tc_minus 100 --average_update 10 --output_file results_tc100_ma10.json
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 200 --tc_minus 200 --average_update 20 --output_file results_tc200_ma20.json
python examples/spiking_ca/spiking_ca_cluster.py --tc_plus 500 --tc_minus 500 --average_update 50 --output_file results_tc500_ma50.json
```

**Priority 2 — Exploration schedule**
```bash
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 100  --output_file results_eps100.json
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 300  --output_file results_eps300.json
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 500  --output_file results_eps500.json
python examples/spiking_ca/spiking_ca_cluster.py --epsilon_decay 1000 --output_file results_eps1000.json
```

**Priority 3 — Target network stability**
```bash
python examples/spiking_ca/spiking_ca_cluster.py --target_update 10  --output_file results_tgt10.json
python examples/spiking_ca/spiking_ca_cluster.py --target_update 20  --output_file results_tgt20.json
python examples/spiking_ca/spiking_ca_cluster.py --target_update 50  --output_file results_tgt50.json
python examples/spiking_ca/spiking_ca_cluster.py --target_update 100 --output_file results_tgt100.json
```

**Priority 4 — Longer training**
```bash
python examples/spiking_ca/spiking_ca_cluster.py --n_episodes 1000 --output_file results_1000ep.json
python examples/spiking_ca/spiking_ca_cluster.py --n_episodes 2000 --output_file results_2000ep.json
```

## Output format

Each run saves a JSON file with the configuration and all episode scores:

```json
{
  "config": { "n_episodes": 500, "tc_plus": 100.0, ... },
  "scores": [23, 17, 29, ...],
  "final_mean_20": 18.5,
  "best_episode": 88,
  "mean_all": 15.2
}
```

## Architecture

- 16x16 CA grid with 6 channels per cell
- Input: 8 special cells read CartPole observations (cart position, velocity, pole angle, pole angular velocity)
- Output: 2 special cells whose channel 0 values become Q-values for left/right action
- Each cell runs a shared spiking network: 54 input → 32 hidden → 12 output LIF neurons
- Output neurons are excitatory/inhibitory pairs (2 per channel) — delta = excitatory spikes minus inhibitory spikes
- MulticompartmentConnection with MCC_MSTDPET and built-in moving average smoothing
- Sparse random connectivity breaks synchronisation
- Lateral inhibition on output layer
- Target network updated every 20 episodes for stable Q-value estimates
- traces=True required — without it MSTDPET collapses to plain MSTDP
- Episode-boundary reset (`reset_state_variables()` + voltage reinit) added
  to prevent p_plus, p_minus, eligibility, and the moving-average buffer
  from leaking across episodes


