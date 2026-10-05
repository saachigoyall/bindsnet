"""EXPERIMENT 5: lower the encoder's "no signal" baseline firing rate.

This is our own idea, not a reproduction of a specific finding. It is
motivated by the general point in Zambrano, Nusselder, Scholte and Bohte
(2018), "Sparse Computation in Adaptive Spiking Neural Networks"
(Frontiers in Neuroscience 12:987), that spiking networks using Poisson
encoding run at exceedingly high firing rates.

Currently a neutral value (clipped=0) encodes to 50 Hz, a substantial,
constant firing rate for "nothing is happening." Since only about 8 of
256 grid cells carry real state, the other 248 are constantly driving
50 Hz of noise into the hidden layer. This tests a nonlinear encoding
that pushes the baseline down toward 0 Hz while keeping the extremes
(clipped=-1 and clipped=+1) at 0 Hz and 100 Hz, same as before.

    rate = ((clipped + 1) / 2) ** power * 100

power=1 (original): 0 -> 50 Hz (linear)
power=2:             0 -> 25 Hz
power=4:             0 -> ~6 Hz
Higher power = quieter baseline, same extremes.

This changes ENCODING, not the network -- input->hidden weights and
fan-in stay at the original baseline throughout.

    python run_encoder_baseline.py            # saturation stats (fast)
    python run_encoder_baseline.py --decode   # + decoder (slow)
"""
import json, sys
import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDPET as MCC_MSTDPET
from bindsnet.encoding import PoissonEncoder
from ca_model import DEVICE, GRID_SIZE, get_neighbourhood
from probe import POLE_ANGLE_CELLS, cell_rows, fresh_grid, reset_network

DECODE = '--decode' in sys.argv
rows = cell_rows(POLE_ANGLE_CELLS)
N_CELLS = GRID_SIZE * GRID_SIZE

POWERS = [1.0, 2.0, 3.0, 4.0]  # 1.0 = original linear encoding

_encoder = PoissonEncoder(time=50, dt=1.0)


def encode_grid_power(grid, power):
    """Same as probe.encode_grid, but with the nonlinear rate mapping."""
    flat = np.stack([get_neighbourhood(grid, r, c).flatten()
                      for r in range(GRID_SIZE) for c in range(GRID_SIZE)])
    clipped = torch.clamp(torch.tensor(flat, dtype=torch.float32), -1.0, 1.0)
    rates = (((clipped + 1.0) / 2.0) ** power) * 100.0
    return torch.stack([_encoder(rates[i]) for i in range(N_CELLS)]).permute(1, 0, 2)


def probe_layers_power(grid, network, power, n_timesteps=50):
    spikes = encode_grid_power(grid, power)
    reset_network(network)
    inp = torch.zeros(N_CELLS, 54)
    hid = torch.zeros(N_CELLS, network.layers['hidden'].n)
    for t in range(n_timesteps):
        inp += spikes[t]
        network.run(inputs={'input': spikes[t]}, time=1, reward=0.0)
        hid += network.layers['hidden'].s.float()
    return inp, hid


def build_network_baseline(seed, nu_ho=0.0):
    """Original architecture, unchanged -- only the encoding function
    used to probe it changes, via probe_layers_power above."""
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


def sample_state(rng):
    obs = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0),
                     rng.uniform(-0.2, 0.2), rng.uniform(-2.0, 2.0)], dtype=np.float32)
    return obs, int(obs[2] > 0)


print(f"{'power':<10}{'hidden spikes':<16}{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for power in POWERS:
    net = build_network_baseline(1)
    resp = {}
    for tag, obs in [('L', np.array([0, 0, -0.2, 0], dtype=np.float32)),
                      ('R', np.array([0, 0, 0.2, 0], dtype=np.float32))]:
        np.random.seed(7); torch.manual_seed(7)
        _, hid = probe_layers_power(fresh_grid(obs), net, power)
        resp[tag] = hid[rows]
    print(f"{power:<10}{resp['L'].mean():<16.2f}{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            net = build_network_baseline(seed)
            rng = np.random.default_rng(seed)
            X, y = [], []
            for _ in range(150):
                obs, label = sample_state(rng)
                _, hid = probe_layers_power(fresh_grid(obs), net, power)
                X.append(hid[rows].flatten().numpy())
                y.append(label)
            per_seed[seed] = {'Xhid': np.array(X).tolist(), 'y': y}
        results[str(power)] = per_seed

if DECODE:
    json.dump(results, open('encoder_baseline_sweep.json', 'w'))
    print('\nwrote encoder_baseline_sweep.json -> python analyze.py encoder_baseline_sweep.json')
