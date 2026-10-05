"""EXPERIMENT 4: heterogeneous tc_decay (membrane time constant), drawn
from a Gamma distribution.

Inspired by Perez-Nieves et al. (2021), "Neural heterogeneity promotes
robust learning" (Nature Communications 12:5791), who found that
heterogeneous membrane and synaptic time constants, with heterogeneous
initialisation drawn from a Gamma distribution, improved learning in
trained networks. This is a loose analogue: it varies only the membrane
time constant and measures decodability without any training. Each
hidden neuron gets its own tc_decay sampled from a Gamma distribution.

Input->hidden weights stay at the original baseline throughout.

    python run_heterogeneous_decay.py            # saturation stats (fast)
    python run_heterogeneous_decay.py --decode   # + decoder (slow)
"""
import json, sys
import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDPET as MCC_MSTDPET
from ca_model import DEVICE
from probe import POLE_ANGLE_CELLS, cell_rows, fresh_grid, probe_layers, sample_state

DECODE = '--decode' in sys.argv
rows = cell_rows(POLE_ANGLE_CELLS)

# (gamma_shape, gamma_scale) pairs for tc_decay ~ Gamma(shape, scale),
# mean = shape*scale. First row: near-zero variance, ~= original uniform
# tc_decay=100 for all units (a sanity check against the baseline).
# Later rows increase the spread while keeping the mean near 100.
SETTINGS = [
    (10000.0, 0.01),   # ~= homogeneous, tc_decay ~= 100 for all units
    (4.0, 25.0),        # moderate spread, mean = 100
    (1.0, 100.0),        # wide spread (exponential-like), mean = 100
    (0.5, 200.0),        # very wide spread, mean = 100
]


def build_network_hetero_decay(seed, gamma_shape, gamma_scale, nu_ho=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = Network(dt=1.0)
    input_layer = Input(n=54, shape=(54,))

    tc_decay_per_unit = torch.tensor(
        np.random.gamma(gamma_shape, gamma_scale, size=32), dtype=torch.float32
    ).clamp(min=1.0)  # avoid degenerate near-zero time constants

    hidden_layer = LIFNodes(n=32, thresh=-52.0, reset=-65.0, rest=-65.0,
                              refrac=5, tc_decay=tc_decay_per_unit, traces=True)
    output_layer = LIFNodes(n=12, thresh=-52.0, refrac=5, traces=True)

    w_ih = torch.zeros(54, 32)
    for j in range(32):
        idx = torch.randperm(54)[:22]
        w_ih[idx, j] = torch.rand(22) * 20.0

    w_ho = torch.zeros(32, 12)
    for j in range(12):
        idx = torch.randperm(32)[:13]
        w_ho[idx, j] = torch.rand(13) * 20.0

    input_to_hidden = MulticompartmentConnection(
        source=input_layer, target=hidden_layer, device=DEVICE,
        pipeline=[Weight('weight', w_ih, range=[0.0, 30.0])])
    hidden_to_output = MulticompartmentConnection(
        source=hidden_layer, target=output_layer, device=DEVICE,
        pipeline=[Weight('weight', w_ho, range=[0.0, 30.0], nu=(nu_ho, nu_ho),
                          learning_rule=MCC_MSTDPET, enforce_polarity=True)],
        tc_plus=100.0, tc_minus=100.0, average_update=10, continues_update=True)

    w_lateral = -torch.ones(12, 12) * 5.0
    for i in range(12):
        w_lateral[i, i] = 0.0
    lateral_inhibition = MulticompartmentConnection(
        source=output_layer, target=output_layer, device=DEVICE,
        pipeline=[Weight('weight', w_lateral, range=[-10.0, 0.0])])

    net.add_layer(input_layer, name='input')
    net.add_layer(hidden_layer, name='hidden')
    net.add_layer(output_layer, name='output')
    net.add_connection(input_to_hidden, source='input', target='hidden')
    net.add_connection(hidden_to_output, source='hidden', target='output')
    net.add_connection(lateral_inhibition, source='output', target='output')

    dummy = torch.zeros(1, 54)
    net.run(inputs={'input': dummy}, time=1, reward=0.0)
    with torch.no_grad():
        net.layers['hidden'].v = torch.rand(1, 32) * 13.0 - 6.5 - 65.0
        net.layers['output'].v = torch.rand(1, 12) * 13.0 - 6.5 - 65.0
    return net


def saturation_stats(net):
    resp = {}
    for tag, obs in [('L', np.array([0, 0, -0.2, 0], dtype=np.float32)),
                      ('R', np.array([0, 0, 0.2, 0], dtype=np.float32))]:
        np.random.seed(7); torch.manual_seed(7)
        _, hid = probe_layers(fresh_grid(obs), net)
        resp[tag] = hid[rows]
    return resp


print(f"{'gamma_shape':<14}{'gamma_scale':<14}{'hidden spikes':<16}"
      f"{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for gamma_shape, gamma_scale in SETTINGS:
    net = build_network_hetero_decay(1, gamma_shape, gamma_scale)
    resp = saturation_stats(net)
    key = f"shape{gamma_shape}_scale{gamma_scale}"
    print(f"{gamma_shape:<14}{gamma_scale:<14}{resp['L'].mean():<16.2f}"
          f"{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            net = build_network_hetero_decay(seed, gamma_shape, gamma_scale)
            rng = np.random.default_rng(seed)
            X, y = [], []
            for _ in range(150):
                obs, label = sample_state(rng)
                _, hid = probe_layers(fresh_grid(obs), net)
                X.append(hid[rows].flatten().numpy())
                y.append(label)
            per_seed[seed] = {'Xhid': np.array(X).tolist(), 'y': y}
        results[key] = per_seed

if DECODE:
    json.dump(results, open('heterogeneous_decay_sweep.json', 'w'))
    print('\nwrote heterogeneous_decay_sweep.json -> '
          'python analyze.py heterogeneous_decay_sweep.json')
