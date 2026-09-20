"""EXPERIMENT 3: weight-sum normalization at the ORIGINAL fan-in (22).

The fan-in sweep and coincidence-fraction sweep both changed fan-in
together with weight scale. This isolates normalization as its own
variable: keep all 22 connections, but constrain their TOTAL sum to a
fixed target, so the neuron's maximum possible drive is controlled
directly, rather than each of the 22 weights being drawn independently
up to some max (which lets total drive vary a lot by chance).

    python run_weight_norm.py            # saturation stats (fast)
    python run_weight_norm.py --decode   # + decoder (slow)
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
FAN_IN = 22  # original, unchanged

# target sum of the 22 weights for a given hidden unit. Original code's
# weights (uniform [0,20], 22 of them) sum to ~220 on average -- wildly
# over threshold. These targets are chosen around and below the 13 mV
# threshold gap.
TARGET_SUMS = [13.0, 20.0, 30.0, 50.0, 100.0]


def build_network_normalized(seed, target_sum, nu_ho=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = Network(dt=1.0)
    input_layer = Input(n=54, shape=(54,))
    hidden_layer = LIFNodes(n=32, thresh=-52.0, reset=-65.0, rest=-65.0,
                              refrac=5, traces=True)
    output_layer = LIFNodes(n=12, thresh=-52.0, refrac=5, traces=True)

    w_ih = torch.zeros(54, 32)
    for j in range(32):
        idx = torch.randperm(54)[:FAN_IN]
        raw = torch.rand(FAN_IN)
        raw = raw / raw.sum() * target_sum  # ← normalize to target_sum
        w_ih[idx, j] = raw

    w_ho = torch.zeros(32, 12)
    for j in range(12):
        idx = torch.randperm(32)[:13]
        w_ho[idx, j] = torch.rand(13) * 20.0

    input_to_hidden = MulticompartmentConnection(
        source=input_layer, target=hidden_layer, device=DEVICE,
        pipeline=[Weight('weight', w_ih, range=[0.0, max(30.0, target_sum)])])
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


print(f"{'target_sum':<12}{'hidden spikes':<16}{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for target_sum in TARGET_SUMS:
    net = build_network_normalized(1, target_sum)
    resp = saturation_stats(net)
    print(f"{target_sum:<12}{resp['L'].mean():<16.2f}{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            net = build_network_normalized(seed, target_sum)
            rng = np.random.default_rng(seed)
            X, y = [], []
            for _ in range(150):
                obs, label = sample_state(rng)
                _, hid = probe_layers(fresh_grid(obs), net)
                X.append(hid[rows].flatten().numpy())
                y.append(label)
            per_seed[seed] = {'Xhid': np.array(X).tolist(), 'y': y}
        results[str(target_sum)] = per_seed

if DECODE:
    json.dump(results, open('weight_norm_sweep.json', 'w'))
    print('\nwrote weight_norm_sweep.json -> python analyze.py weight_norm_sweep.json')
