# Hidden layer experiments: five approaches to the saturated hidden layer

This is a follow up to `README.md` in this folder, which found that pole
direction can be decoded from the input layer (about 0.82) but not from
the hidden layer (about chance, 0.5), because the hidden layer fires
near its ceiling for every input.

Five things were tried first, and none closed the gap. Weight scale
alone stayed at chance. Sparser fan in produced the best result so far,
about 0.54 to 0.55 at fan in 3, but that peak is small and could be
within noise. Varying the coincidence fraction, the membrane decay
(`tc_decay`) and the refractory period did not help. The refractory
period looked promising on a small sample, close to the input ceiling,
and went back to chance at full scale.

All of those changed how strongly or how long input drives a neuron.
The five scripts here test different mechanisms instead: competition
between neurons, adaptive thresholds, controlled total drive, a gentler
encoding, and variation between neurons.

## How these relate to the literature

All five are inspired by published work, and none is a replication. The
cited papers trained networks or studied real neurons, while these
scripts decode pole direction from an untrained network on a single
static grid snapshot. Each section below says what the paper did and how
this differs. Experiment 5 is our own idea.

## What has not been done

* The five scripts have fast saturation statistics only. Full decoder
  results have not been collected.
* None of the five have been combined with each other. (Experiment 1
  does combine two mechanisms, as described below.)
* None of them involve the CA loop, so the channel 0 and CA delta fixes
  from `README.md` have not been applied.

## How to run

Each script is independent, so all five can run at the same time on
separate processes. From this folder:

    python run_lateral_adaptive_combined.py --decode
    python run_double_adaptation.py --decode
    python run_ongoing_weight_norm.py --decode
    python run_heterogeneous_decay.py --decode
    python run_encoder_baseline.py --decode

Without `--decode`, each script prints saturation statistics in seconds
(hidden spikes per unit, spread across units, and the left versus right
difference). With `--decode`, it also fits the decoder on 150 sampled
states for each of 3 seeds, which is slow, and writes a json file that
`analyze.py` reads:

| script | json file |
|---|---|
| `run_lateral_adaptive_combined.py` | `lateral_adaptive_combined_sweep.json` |
| `run_double_adaptation.py` | `double_adaptation_sweep.json` |
| `run_ongoing_weight_norm.py` | `ongoing_weight_norm_sweep.json` |
| `run_heterogeneous_decay.py` | `heterogeneous_decay_sweep.json` |
| `run_encoder_baseline.py` | `encoder_baseline_sweep.json` |

    python analyze.py lateral_adaptive_combined_sweep.json

`analyze.py` labels its first column `w_ih max` for every script, but it
is just the setting for that script.

## What to look for

Chance is 0.5, the best result so far is about 0.54 to 0.55, and the
input layer sits around 0.82. A setting that stays clearly above 0.55 on
all three seeds is worth following up. If all five sit at chance, no
single mechanism is the fix, and the next step would be combining the
ones that moved the hidden layer's firing the most.

## 1. `run_lateral_adaptive_combined.py`: lateral inhibition and an adaptive threshold

Only the output layer currently has lateral inhibition, so hidden units
respond independently and nothing makes them differentiate.

Inspired by Diehl and Cook (2015), whose network combined lateral
inhibition with an adaptive spiking threshold that rises after each
spike. They trained a network with STDP on MNIST. This script tests each
mechanism alone and then both together, using BindsNET's
`AdaptiveLIFNodes` for the threshold.

## 2. `run_double_adaptation.py`: two timescale spike frequency adaptation

Spike frequency adaptation, a threshold that rises after each spike and
decays back, is used in Bellec et al. (2018), with a single exponential
per neuron and different time constants across neurons. The two
timescale version here, with a fast and a slow component, is inspired by
the multi timescale adaptive threshold model of Kobayashi, Tsubo and
Shinomoto (2009).

Differences from that work: the MAT model was built to predict spike
times of single cortical neurons and uses a non resetting leaky
integrator, while this neuron resets after each spike. The time
constants here are our own. BindsNET's built in `AdaptiveLIFNodes` only
supports one timescale, so this script defines a custom neuron class. It
was checked to reproduce the baseline firing statistics when both
adaptation components are off.

## 3. `run_ongoing_weight_norm.py`: ongoing weight normalization

Inspired by Turrigiano et al. (1998), who found that neurons scale all
of their synaptic inputs up or down with activity, which may keep firing
rates from saturating. That was experimental work on cultured cortical
neurons, so this is a loose analogue.

Synaptic scaling is continuous, so the script renormalizes each hidden
unit's incoming weights over several rounds, pulling its firing toward a
target rate. Ten rounds move the firing toward the target without
reaching it, which is intentional, since the biological process is slow.

## 4. `run_heterogeneous_decay.py`: variation in membrane time constants

Inspired by Perez-Nieves et al. (2021), who found that heterogeneous
membrane and synaptic time constants, with heterogeneous initialisation
drawn from a gamma distribution, improved learning in trained networks.

Differences: this only varies the membrane time constant (`tc_decay`)
and measures decodability without training. Little change is expected.
Earlier results showed that `tc_decay` barely matters while a single
spike already crosses threshold, and the fast statistics here are
nearly identical across all settings.

## 5. `run_encoder_baseline.py`: a gentler "no signal" encoding

A neutral grid value currently encodes to 50 Hz, so the roughly 248
grid cells that carry no state inject constant noise.

This is our own idea, not a reproduction of a specific finding. It is
motivated by the general point in Zambrano et al. (2018) that spiking
networks using Poisson encoding run at exceedingly high firing rates.
No paper validates the particular power law remapping used here. At
`power` 3 and 4 the hidden layer goes completely silent, so those
settings are likely not useful.

## Fast pass results

Saturation statistics only, from one seed. These are for choosing what
to decode first. They are not results, and they have already been
misleading once: the refractory period looked very promising on a small
sample and did not hold up at full scale.

| script | setting | hidden spikes | spread across units | left vs right difference |
|---|---|---|---|---|
| `run_lateral_adaptive_combined.py` | lateral 5.0 alone | 3.06 | 1.951 | 0.531 |
| `run_lateral_adaptive_combined.py` | adaptive threshold 2.0 alone | 5.09 | 0.526 | 0.438 |
| `run_lateral_adaptive_combined.py` | both together (5.0 and 2.0) | 2.81 | 1.521 | 0.031 |
| `run_lateral_adaptive_combined.py` | both, gentler (2.5 and 1.0) | 4.00 | 1.127 | 0.281 |
| `run_double_adaptation.py` | fast component only | 5.25 | 0.563 | 0.156 |
| `run_double_adaptation.py` | slow component only | 5.31 | 0.531 | 0.156 |
| `run_double_adaptation.py` | both components | 5.22 | 0.548 | 0.000 |
| `run_ongoing_weight_norm.py` | target rate 1.0 | 3.09 | 0.635 | 0.031 |
| `run_heterogeneous_decay.py` | any gamma setting | about 5.3 to 5.4 | about 0.54 | about 0.19 |
| `run_encoder_baseline.py` | power 2.0 | 3.03 | 0.642 | 0.062 |

The unmodified baseline gives 5.38 hidden spikes, a spread of 0.549 and
a left versus right difference of 0.188. Note that the combined setting
in experiment 1 has high spread but low left versus right difference, so
the two measures do not always move together, and neither one predicts
decoder accuracy on its own.

## References

Bellec G, Salaj D, Subramoney A, Legenstein R, Maass W (2018). Long
short-term memory and learning-to-learn in networks of spiking neurons.
NeurIPS 2018.

Diehl PU, Cook M (2015). Unsupervised learning of digit recognition
using spike-timing-dependent plasticity. Frontiers in Computational
Neuroscience 9:99.

Kobayashi R, Tsubo Y, Shinomoto S (2009). Made-to-order spiking neuron
model equipped with a multi-timescale adaptive threshold. Frontiers in
Computational Neuroscience 3:9.

Perez-Nieves N, Leung VCH, Dragotti PL, Goodman DFM (2021). Neural
heterogeneity promotes robust learning. Nature Communications 12:5791.

Turrigiano GG, Leslie KR, Desai NS, Rutherford LC, Nelson SB (1998).
Activity-dependent scaling of quantal amplitude in neocortical neurons.
Nature 391:892 to 896.

Zambrano D, Nusselder R, Scholte HS, Bohté SM (2018). Sparse computation
in adaptive spiking neural networks. Frontiers in Neuroscience 12:987.
