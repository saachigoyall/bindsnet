"""CORRECTED EXPERIMENT 3: ONGOING weight-sum normalization.

Turrigiano et al. (1998) -- the paper this experiment is grounded in --
describes synaptic scaling as a CONTINUOUS process: a neuron constantly
adjusts its total synaptic weight based on its own recent activity. The
earlier version of this experiment set the weight sum once at network
creation and never touched it again -- that is weight initialization,
not a test of the homeostatic mechanism. This version renormalizes each
hidden neuron's total incoming weight after every probe, based on how
much it fired, matching the actual mechanism: neurons that fired a lot
get scaled down, neurons that fired little get scaled up, pulling
everyone toward a target activity level over repeated exposure.

Since this needs state that persists and updates ACROSS probes (not a
single-shot property of one network), it runs a short sequence of probes
per configuration, renormalizing between each one, and reports the
final decoder accuracy after normalization has had a chance to act.

Input->hidden weight COUNT (fan-in=22) and initial structure stay at the
original baseline; only the ongoing rescaling is new.

    python run_ongoing_weight_norm.py            # saturation stats (fast)
    python run_ongoing_weight_norm.py --decode   # + decoder (slow)
"""
import json, sys
import numpy as np
import torch
from ca_model import build_network
from probe import POLE_ANGLE_CELLS, cell_rows, fresh_grid, probe_layers, sample_state

DECODE = '--decode' in sys.argv
rows = cell_rows(POLE_ANGLE_CELLS)

# target average spike count per hidden neuron per probe. Neurons firing
# above this get scaled down before the next probe; below it, scaled up.
TARGET_RATES = [1.0, 2.0, 4.0]
N_NORM_STEPS = 10          # how many renormalization rounds to run before evaluating
SCALING_RATE = 0.2         # how aggressively to correct toward target each round


def renormalize(net, target_rate, current_spikes_per_unit, scaling_rate=SCALING_RATE):
    """Turrigiano-style multiplicative scaling: each hidden unit's
    incoming weights get scaled by a factor that pulls its recent firing
    rate toward target_rate, applied gradually (scaling_rate controls how
    much of the correction is applied per round, not all at once)."""
    conn = net.connections[('input', 'hidden')]
    w = conn.pipeline[0].value  # shape (54, 32)
    for j in range(32):
        rate = max(current_spikes_per_unit[j].item(), 0.01)
        correction = 1.0 + scaling_rate * ((target_rate / rate) - 1.0)
        correction = max(0.1, min(correction, 3.0))  # bound the per-step change
        w[:, j] *= correction


def run_with_ongoing_norm(seed, target_rate, n_samples_eval, decode=False):
    net = build_network(nu_ho=0.0, seed=seed)
    rng = np.random.default_rng(seed)

    # normalization phase: probe on random states, renormalize between each
    for _ in range(N_NORM_STEPS):
        obs, _ = sample_state(rng)
        _, hid = probe_layers(fresh_grid(obs), net)
        # hid[rows] is (2, 32): 2 pole-angle cells x 32 hidden units.
        # Average over the 2 cells to get one rate per hidden unit.
        spikes_per_unit = hid[rows].mean(dim=0)
        renormalize(net, target_rate, spikes_per_unit)

    # evaluation phase: measure saturation stats and (optionally) decode
    resp = {}
    for tag, obs in [('L', np.array([0, 0, -0.2, 0], dtype=np.float32)),
                      ('R', np.array([0, 0, 0.2, 0], dtype=np.float32))]:
        np.random.seed(7); torch.manual_seed(7)
        _, hid = probe_layers(fresh_grid(obs), net)
        resp[tag] = hid[rows]

    if not decode:
        return resp, None

    X, y = [], []
    for _ in range(n_samples_eval):
        obs, label = sample_state(rng)
        _, hid = probe_layers(fresh_grid(obs), net)
        X.append(hid[rows].flatten().numpy())
        y.append(label)
    return resp, (X, y)


print(f"{'target_rate':<14}{'hidden spikes (post-norm)':<28}{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for target_rate in TARGET_RATES:
    resp, _ = run_with_ongoing_norm(1, target_rate, 0, decode=False)
    print(f"{target_rate:<14}{resp['L'].mean():<28.2f}{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            _, (X, y) = run_with_ongoing_norm(seed, target_rate, 150, decode=True)
            per_seed[seed] = {'Xhid': X, 'y': y}
        results[str(target_rate)] = per_seed

if DECODE:
    json.dump(results, open('ongoing_weight_norm_sweep.json', 'w'))
    print('\nwrote ongoing_weight_norm_sweep.json -> '
          'python analyze.py ongoing_weight_norm_sweep.json')
