"""BUG 3 (the blocker): input->hidden weights drive the hidden layer into
refractory saturation, so it responds almost identically to every cell.

  w_ih: 22 connections/unit, uniform [0, 20] mV, mean 10.1
  LIF : rest -65, thresh -52  -> 13 mV gap; refrac 5, T=50 -> ceiling ~8 spikes

One presynaptic spike already crosses threshold, so units are refractory-limited
rather than input-limited. This sweep reports, per weight scale, how much the
hidden layer actually varies -- and the decoder column (from run_weight_sweep
--decode) shows that no scalar rescaling rescues it.

    python run_weight_sweep.py            # saturation stats only (fast)
    python run_weight_sweep.py --decode   # also fits the decoder per scale (slow)
"""
import json, sys
import numpy as np
import torch
from ca_model import build_network
from probe import POLE_ANGLE_CELLS, cell_rows, fresh_grid, probe_layers, sample_state

SCALES = [20.0, 5.0, 2.0, 1.0, 0.6, 0.3]
DECODE = '--decode' in sys.argv
rows = cell_rows(POLE_ANGLE_CELLS)


def scaled_network(seed, w_max):
    net = build_network(nu_ho=0.0, seed=seed)
    net.connections[('input', 'hidden')].pipeline[0].value.mul_(w_max / 20.0)
    return net


print(f"{'w_ih max':<12}{'hidden spikes':<16}{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for w in SCALES:
    net = scaled_network(1, w)
    resp = {}
    for tag, obs in [('L', np.array([0, 0, -0.2, 0], dtype=np.float32)),
                     ('R', np.array([0, 0,  0.2, 0], dtype=np.float32))]:
        np.random.seed(7); torch.manual_seed(7)
        _, hid = probe_layers(fresh_grid(obs), net)
        resp[tag] = hid[rows]
    print(f"{w:<12}{resp['L'].mean():<16.2f}{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            n = scaled_network(seed, w)
            rng = np.random.default_rng(seed)
            X, y = [], []
            for _ in range(150):
                obs, label = sample_state(rng)
                _, hid = probe_layers(fresh_grid(obs), n)
                X.append(hid[rows].flatten().numpy()); y.append(label)
            per_seed[seed] = {'Xhid': np.array(X).tolist(), 'y': y}
        results[str(w)] = per_seed

if DECODE:
    json.dump(results, open('weight_sweep.json', 'w'))
    print('\nwrote weight_sweep.json  ->  python analyze.py weight_sweep.json')
