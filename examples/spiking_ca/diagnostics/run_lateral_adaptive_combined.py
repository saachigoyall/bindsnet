"""EXPERIMENT 1: lateral inhibition and an adaptive threshold in the hidden layer.

Inspired by Diehl and Cook (2015), "Unsupervised learning of digit
recognition using spike-timing-dependent plasticity" (Frontiers in
Computational Neuroscience 9:99), whose network combined lateral
inhibition with an adaptive spiking threshold that rises after each
spike. This is a loose analogue, not a replication: they trained a
network with STDP on MNIST, while this only measures whether a hidden
layer with these mechanisms carries more decodable information about
pole direction. The script tests each mechanism alone and then both
together, using BindsNET's AdaptiveLIFNodes for the threshold half.

Input->hidden weights stay at the original baseline throughout.

    python run_lateral_adaptive_combined.py            # saturation stats (fast)
    python run_lateral_adaptive_combined.py --decode   # + decoder (slow)
"""
import json, sys
import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDPET as MCC_MSTDPET
from ca_model import DEVICE
from probe import POLE_ANGLE_CELLS, cell_rows, fresh_grid, probe_layers, sample_state

DECODE = '--decode' in sys.argv
rows = cell_rows(POLE_ANGLE_CELLS)

# (lateral_strength, theta_plus) pairs. First row is the original
# baseline (neither mechanism). Later rows combine both, at the
# strengths that looked most promising individually.
SETTINGS = [
    (0.0, 0.0),    # baseline
    (5.0, 0.0),    # lateral inhibition alone
    (0.0, 2.0),    # adaptive threshold alone
    (5.0, 2.0),    # both together, in the spirit of Diehl and Cook
    (2.5, 1.0),    # combined, gentler
]


def build_network_combined(seed, lateral_strength, theta_plus, nu_ho=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = Network(dt=1.0)
    input_layer = Input(n=54, shape=(54,))

    if theta_plus > 0:
        hidden_layer = AdaptiveLIFNodes(n=32, thresh=-52.0, reset=-65.0, rest=-65.0,
                                          refrac=5, traces=True,
                                          theta_plus=theta_plus, tc_theta_decay=100.0)
    else:
        hidden_layer = LIFNodes(n=32, thresh=-52.0, reset=-65.0, rest=-65.0,
                                  refrac=5, traces=True)
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

    w_lateral_out = -torch.ones(12, 12) * 5.0
    for i in range(12):
        w_lateral_out[i, i] = 0.0
    lateral_out = MulticompartmentConnection(
        source=output_layer, target=output_layer, device=DEVICE,
        pipeline=[Weight('weight', w_lateral_out, range=[-10.0, 0.0])])

    net.add_layer(input_layer, name='input')
    net.add_layer(hidden_layer, name='hidden')
    net.add_layer(output_layer, name='output')
    net.add_connection(input_to_hidden, source='input', target='hidden')
    net.add_connection(hidden_to_output, source='hidden', target='output')
    net.add_connection(lateral_out, source='output', target='output')

    if lateral_strength > 0:
        w_lateral_hid = -torch.ones(32, 32) * lateral_strength
        for i in range(32):
            w_lateral_hid[i, i] = 0.0
        lateral_hid = MulticompartmentConnection(
            source=hidden_layer, target=hidden_layer, device=DEVICE,
            pipeline=[Weight('weight', w_lateral_hid,
                              range=[-lateral_strength * 2, 0.0])])
        net.add_connection(lateral_hid, source='hidden', target='hidden')

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


print(f"{'lateral':<10}{'theta_plus':<12}{'hidden spikes':<16}"
      f"{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for lateral_strength, theta_plus in SETTINGS:
    net = build_network_combined(1, lateral_strength, theta_plus)
    resp = saturation_stats(net)
    key = f"lat{lateral_strength}_theta{theta_plus}"
    print(f"{lateral_strength:<10}{theta_plus:<12}{resp['L'].mean():<16.2f}"
          f"{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            net = build_network_combined(seed, lateral_strength, theta_plus)
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
    json.dump(results, open('lateral_adaptive_combined_sweep.json', 'w'))
    print('\nwrote lateral_adaptive_combined_sweep.json -> '
          'python analyze.py lateral_adaptive_combined_sweep.json')
