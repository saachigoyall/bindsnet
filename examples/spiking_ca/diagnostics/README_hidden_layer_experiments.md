# Hidden-layer experiments: five untested angles on bug 1

Everything tested against bug 1 so far (weight-scale sweep, sparser
fan-in, coincidence fraction, membrane decay, refractory period) either
failed outright or topped out around 0.54 decoder accuracy (fan-in=3),
well short of the ~0.82 input-layer ceiling. All of those changed *how
much or how fast input drive reaches a neuron*. These five scripts test
different mechanisms instead: competition between neurons, adaptive
thresholds, controlled total drive, a gentler encoding, and population
diversity.

Each script is independent and can be run in parallel. All follow the
same interface as the existing `run_weight_sweep.py`:

    python run_<name>.py            # saturation stats only (fast, seconds)
    python run_<name>.py --decode   # + full decoder, 3 seeds x 150 samples (slow)
    python analyze.py <name>_sweep.json

All five keep input->hidden weights at the ORIGINAL baseline (22
connections, uniform [0,20]) except where the experiment itself is
specifically about changing that (weight-norm, by design; fan-in/weight-
scale variants have already been fully explored separately).

## Fast-pass results already collected (saturation stats only, not yet
## decoded at full scale)

| script | most promising setting | hidden spikes | std | \|L-R\| |
|---|---|---|---|---|
| `run_lateral_hidden.py` | strength=5.0 | 3.06 | **1.951** | **0.531** |
| `run_weight_norm.py` | target_sum=13.0 | 1.03 | 0.175 | **0.469** |
| `run_encoder_baseline.py` | power=2.0 | 3.03 | 0.642 | 0.062 |
| `run_adaptation.py` | theta_plus=2.0 | 5.09 | 0.526 | 0.438 |
| `run_heterogeneous.py` | spread=20.0 | 5.31 | 0.639 | 0.094 |

The first two show notably higher variance-across-units and L/R
separation than anything in the earlier sweeps -- worth prioritizing if
running these one at a time rather than all in parallel.

**Known dead ends, already found in the fast pass, no need to spend
decode time on these specific settings:**
- `run_weight_norm.py --target_sum=20.0` -- std=0.000, a degenerate
  floor, same pattern as `refrac=20` in the earlier timing sweep.
- `run_encoder_baseline.py` at `power=3.0` and `4.0` -- goes fully
  silent (0.00 spikes), too aggressive a baseline reduction.

## 1. `run_lateral_hidden.py` -- lateral inhibition within the hidden layer

Only the output layer currently has lateral inhibition (neurons suppress
each other). The hidden layer is purely feedforward, nothing stops every
unit from responding almost identically. This adds the same inhibitory
mechanism already used downstream, to the hidden layer itself.

## 2. `run_adaptation.py` -- spike-frequency adaptation

Uses BindsNET's `AdaptiveLIFNodes` (confirmed present in this version:
`theta_plus`, `tc_theta_decay`). A neuron's own threshold rises after it
fires, then relaxes back down -- graded and history-dependent, unlike
`refrac` (a hard on/off cooldown, already tested, no effect).

## 3. `run_weight_norm.py` -- weight-sum normalization at fan-in=22

Keeps all 22 connections but normalizes their sum to a fixed target,
rather than each weight being drawn independently. Isolates
normalization as its own variable, separate from sparsity (which
earlier sweeps already covered).

## 4. `run_encoder_baseline.py` -- gentler "no signal" encoding

Currently a neutral grid value encodes to 50 Hz, a substantial constant
rate. Since only ~8 of 256 cells carry real state, the other ~248 are
constantly driving 50 Hz of noise. Tests a nonlinear rate mapping that
pushes the baseline down while keeping the extremes (0 Hz / 100 Hz)
unchanged.

## 5. `run_heterogeneous.py` -- randomized per-neuron thresholds

Every hidden neuron currently shares identical thresh/reset/rest/refrac.
Real neural populations vary. Tests whether population diversity alone
produces more informative collective behavior, independent of the other
four experiments.
