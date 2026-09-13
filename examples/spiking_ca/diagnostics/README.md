# Diagnostics: where does the CartPole state information go?

These scripts localize why the spiking-CA network never learned. Short version:
the state reaches the network fine and the hidden layer deletes it.

    decode pole direction from...      seed1    seed2    seed3    mean
    INPUT layer spike counts           0.813    0.847    0.800    0.820
    HIDDEN layer spike counts          0.540    0.547    0.480    0.522
                                       chance = 0.500, n=150, 5-fold CV

## What these measure — and what they don't

They measure whether pole angle is **linearly recoverable** at a given layer.
They say nothing about learning, the reward signal, eligibility traces, or
whether this architecture can solve CartPole. They only answer "where does the
information disappear", which is the question that has to be settled first.

`run_layer_decoder.py` decodes from the two pole-angle input cells only.
The notebook's `probe_layer_responses` summed the hidden response over all 256 CA
cells, so the 248 cells carrying no state diluted the 8 that do — a perfect
encoder would still have looked flat. `probe.py` keeps the per-cell tensor
instead.

Accuracy is 5-fold stratified CV rather than a single split: with n=150 a 30%
holdout leaves ~45 test samples, whose confidence interval (~±15 points) is wide
enough to hide the effect entirely.

Note that even a working input layer tops out around 0.82, not 1.0, because
states are labelled by the sign of the pole angle and near-zero angles are
genuinely unlabelable.

## Setup

    pip install git+https://github.com/saachigoyall/bindsnet.git@spiking-ca-cluster
    pip install gymnasium scikit-learn

## The three findings, in the order they have to be fixed

### 1. The hidden layer is saturated — this is the blocker

    python run_weight_sweep.py            # saturation stats, fast
    python run_weight_sweep.py --decode   # + decoder per weight scale, slow
    python analyze.py weight_sweep.json

`build_network` gives each hidden unit 22 connections with weights uniform on
[0, 20] mV, mean 10.1. The LIF sits at rest −65 with threshold −52 — a 13 mV gap.
A single presynaptic spike is already over threshold, so with 22 inputs firing at
50–100 Hz the units are refractory-limited, not input-limited.

Measured: 4–7 spikes of a ~8 ceiling for *every* cell in the grid, std across the
32 units of 0.55 spikes, whole-grid mean (5.38) equal to the pole-cell mean
(5.56). Close to a constant function.

The sweep shows there is no scalar rescaling that fixes it — saturated at w ≥ 5,
silent at w ≤ 0.3, decoder at chance throughout. The input drive needs
normalization, or a much sparser fan-in, so a unit integrates rather than firing
on the first spike it sees.

### 2. The observation is written once and immediately erased

    python run_channel0_trace.py

`set_inputs` writes state into channel 0. `restore_fixed_channels` restores
channels 1, 2, 4 and 5 — never 0. `ca_step` adds unbounded deltas to all six.

    hard-LEFT    step  0:  [-0.8  -0.8]   <- the observation
                 step  1:  [-4.8  -0.8]
                 step 12:  [-5.0  -4.8]   <- probe / Q-value read here
    hard-RIGHT   step  0:  [ 0.8   0.8]
                 step 12:  [-5.0  -1.2]

Also present in `spiking_ca_cluster.py`: the training loop calls `set_inputs`
once, then reads Q-values three CA steps later. Fix is one line — add channel 0
to `restore_fixed_channels`, in both files.

### 3. The CA update is unbounded, so the grid pins to the clip

    python run_grid_saturation.py

Deltas are raw spike-count differences added to a grid clipped at ±5. Fraction of
grid at |v| ≥ 1, where the encoder clips to a binary 0 or 100 Hz: 2.3% at step 0,
92.7% by step 12. About 85% of what the network sees is saturated.

### Ordering

Fixing 2 and 3 without 1 leaves the decoder at chance (0.37–0.55 measured). The
saturated hidden layer dominates. Fix the parameterization first.

## Standing test

The input-vs-hidden gap is the acceptance criterion: 0.82 → 0.52 today. When the
hidden number approaches the input number, the architecture is sound and the
learning rule becomes worth looking at again.

## Files

| file | what it is |
|---|---|
| `ca_model.py` | CA + network code, verbatim from `encoder_decoder_check_fully_stripped.ipynb` (only three Python-3.12-only f-string prints replaced with `pass`) |
| `probe.py` | per-cell layer probing, state sampler |
| `run_layer_decoder.py` | the input-vs-hidden control |
| `run_channel0_trace.py` | channel-0 erasure trace |
| `run_grid_saturation.py` | grid/encoder saturation stats |
| `run_weight_sweep.py` | hidden-layer saturation vs `w_ih` scale |
| `analyze.py` | cross-validated accuracy for any run output |
