"""EXPERIMENT 1: lateral inhibition within the hidden layer itself.

Currently only the OUTPUT layer has lateral inhibition (neurons suppress
each other). The hidden layer is purely feedforward, so nothing stops
every hidden neuron from responding nearly identically to any input.
This adds the same kind of inhibitory competition already used on the
output layer, to the hidden layer, to see if it sharpens contrast between
neurons the way it does downstream.

Input->hidden weights stay at the original baseline (22 conn/unit,
uniform [0,20]) -- isolates this as its own variable.

    python run_lateral_hidden.py            # saturation stats (fast)
    python run_lateral_hidden.py --decode   # + decoder (slow)
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

LATERAL_STRENGTHS = [0.0, 1.0, 2.5, 5.0]  # 0.0 = no inhibition (baseline)


def build_network_lateral_hidden(seed, lateral_strength, nu_ho=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = Network(dt=1.0)
    input_layer = Input(n=54, shape=(54,))
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


print(f"{'lateral_strength':<18}{'hidden spikes':<16}{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for strength in LATERAL_STRENGTHS:
    net = build_network_lateral_hidden(1, strength)
    resp = saturation_stats(net)
    print(f"{strength:<18}{resp['L'].mean():<16.2f}{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            net = build_network_lateral_hidden(seed, strength)
            rng = np.random.default_rng(seed)
            X, y = [], []
            for _ in range(150):
                obs, label = sample_state(rng)
                _, hid = probe_layers(fresh_grid(obs), net)
                X.append(hid[rows].flatten().numpy())
                y.append(label)
            per_seed[seed] = {'Xhid': np.array(X).tolist(), 'y': y}
        results[str(strength)] = per_seed

if DECODE:
    json.dump(results, open('lateral_hidden_sweep.json', 'w'))
    print('\nwrote lateral_hidden_sweep.json -> python analyze.py lateral_hidden_sweep.json')
