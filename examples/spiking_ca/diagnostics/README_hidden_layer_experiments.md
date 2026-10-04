# Hidden-layer experiments: five approaches to bug 1

Everything tested against bug 1 so far (weight-scale sweep, sparser
fan-in, coincidence fraction, membrane decay, refractory period) either
failed outright or topped out around 0.54 decoder accuracy (fan-in=3),
well short of the ~0.82 input-layer ceiling. All of those changed *how
much or how fast input drive reaches a neuron*. These five test
different mechanisms instead: competition between neurons, adaptive
thresholds, controlled total drive, a gentler encoding, and population
diversity.

Each script is independent and can be run in parallel. All follow the
same interface as the existing `run_weight_sweep.py`:

    python run_<name>.py            # saturation stats only (fast, seconds)
    python run_<name>.py --decode   # + full decoder, 3 seeds x 150 samples (slow)
    python analyze.py <name>_sweep.json

Input->hidden weights stay at the original baseline (22 connections,
uniform [0,20]) throughout, except where the experiment is specifically
about changing that.

## A note on how closely these follow the literature

An earlier pass at these five turned out to only loosely resemble the
papers they were nominally grounded in. Four have been corrected to
actually test what the cited work validated; the fifth is honestly
relabeled rather than forced into a citation it doesn't really match.

## 1. `run_lateral_adaptive_combined.py` -- lateral inhibition + adaptive threshold, combined

**Grounded in:** Diehl & Cook (2015), "Unsupervised learning of digit
recognition using spike-timing-dependent plasticity." Their validated
approach uses lateral inhibition AND a per-spike threshold increase
TOGETHER, not either alone. Tests the combination, plus each mechanism
alone for comparison (both already had preliminary fast-stat results
from an earlier pass).

## 2. `run_double_adaptation.py` -- double-exponential spike-frequency adaptation

**Grounded in:** Bellec et al. (2018, 2020), whose validated model uses
a threshold that rises after each spike and decays back via TWO
separate time constants (fast + slow), not one. BindsNET's built-in
`AdaptiveLIFNodes` only supports a single exponential, so this
implements a proper double-exponential adaptive neuron directly
(subclassing `Nodes`, following the same pattern as BindsNET's own
`AdaptiveLIFNodes` source). Verified to reduce to standard LIF behavior
when both adaptation components are off, matching the original
baseline's saturation stats exactly.

## 3. `run_ongoing_weight_norm.py` -- ongoing (not one-time) weight-sum normalization

**Grounded in:** Turrigiano et al. (1998), the foundational synaptic
scaling paper. Synaptic scaling is a continuous homeostatic process, a
neuron keeps adjusting its total weight based on its own recent
activity. An earlier version of this experiment only set the weight sum
once at network creation, which is really just initialization, not a
test of the mechanism. This version renormalizes each hidden unit's
incoming weight after every probe, over several rounds, pulling firing
rate toward a target the way the real mechanism does.

## 4. `run_heterogeneous_decay.py` -- heterogeneous membrane time constants (Gamma distribution)

**Grounded in:** Perez-Nieves et al. (2021), "Neural heterogeneity
promotes robust learning." Their validated result is specifically about
diversity in membrane time constant (`tc_decay`), drawn from a Gamma
distribution, not threshold diversity. An earlier version of this
experiment tested threshold jitter instead, a different parameter than
the one the paper actually validated. This version draws each hidden
unit's own `tc_decay` from a Gamma distribution as the original work
did. Note: fast-stat results show almost no change across a wide range
of Gamma parameters, consistent with the earlier finding that `tc_decay`
doesn't matter much while weights are strong enough to cross threshold
on a single spike, worth keeping in mind when interpreting this one.

## 5. `run_encoder_baseline.py` -- gentler "no signal" encoding

**Not a direct literature reproduction** -- worth being upfront about
this rather than overclaiming. Bohte (2018) raises the general concern
that most spiking networks use unrealistically high Poisson rates (up
to hundreds of Hz) compared to how real sensory neurons adaptively keep
spike counts sparse. No specific paper validates the particular
power-law remapping used here; this is our own construction motivated by
that general concern, not a reproduction of a specific finding. Included
because the underlying concern (50 Hz constant baseline noise from ~248
irrelevant grid cells) is real and worth testing, just flagged honestly
as our own idea rather than literature-validated.

## Fast-pass results collected so far (saturation stats only, not yet
## decoded at full scale -- treat all of this as preliminary, see the
## refrac=15 false-positive from earlier work as a caution against
## trusting small/fast signals)

| script | notable setting | hidden spikes | std | \|L-R\| |
|---|---|---|---|---|
| `run_lateral_adaptive_combined.py` | lat=5.0 alone | 3.06 | **1.951** | **0.531** |
| `run_lateral_adaptive_combined.py` | theta=2.0 alone | 5.09 | 0.526 | 0.438 |
| `run_lateral_adaptive_combined.py` | lat=5.0 + theta=2.0 (combined) | 2.81 | **1.521** | 0.031 |
| `run_lateral_adaptive_combined.py` | lat=2.5 + theta=1.0 (gentler combined) | 4.00 | 1.127 | 0.281 |
| `run_double_adaptation.py` | fast only (theta_fast=1.0) | 5.25 | 0.563 | 0.156 |
| `run_double_adaptation.py` | slow only (theta_slow=0.3) | 5.31 | 0.531 | 0.156 |
| `run_double_adaptation.py` | fast+slow combined | 5.22 | 0.548 | **0.000** |
| `run_ongoing_weight_norm.py` | target_rate=1.0 | 3.09 | 0.635 | 0.031 |
| `run_heterogeneous_decay.py` | any Gamma setting | ~5.3-5.4 | ~0.54 | ~0.19 |
| `run_encoder_baseline.py` | power=2.0 | 3.03 | 0.642 | 0.062 |

Worth flagging: the combined lateral+adaptive setting (row 3) shows
lower `|L-R|` separation than lateral inhibition alone (row 1), even
though it has high variance across units. That's a reminder that std
and `|L-R|` don't always move together, and neither one alone predicts
decoder accuracy (see refrac=15's earlier false positive) -- these are
navigation aids for picking what to `--decode` first, not results in
themselves.
