"""
CA + spiking-network model, lifted verbatim from
    examples/spiking_ca/encoder_decoder_check_fully_stripped.ipynb
so the diagnostics import exactly the code that produced the original results.

Only change from the notebook: three f-string print() lines in build_network used
same-quote nesting (Python 3.12+ only) and were replaced with `pass`. No behaviour
is affected.
"""
import numpy as np
import torch
import time
import math
import copy
import json
import os
import matplotlib.pyplot as plt
import gymnasium as gym
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning.MCC_learning import MSTDPET as MCC_MSTDPET


def _patched_reset_state_variables(self) -> None:
    self.eligibility.zero_()
    self.eligibility_trace.zero_()
    self.p_plus.zero_()
    self.p_minus.zero_()
    if self.average_update > 0:
        self.average_buffer.zero_()
        self.average_buffer_index = 0
    return
MCC_MSTDPET.reset_state_variables = _patched_reset_state_variables
print('Patched MCC_MSTDPET.reset_state_variables')


GRID_SIZE = 16
NUM_CHANNELS = 6
TRACK_HALF_LEN = 2.4
inp_cell_pos = [(5, 13), (12, 10), (2, 10), (9, 3), (9, 13), (2, 6), (12, 6), (5, 3)]
out_cell_pos = [(6, 8), (9, 8)]
OBS_SCALE = [2.0, 0.25, 4.0, 0.15]
DEVICE = torch.device('cpu')


def make_grid():
    grid = (np.random.random((GRID_SIZE, GRID_SIZE, NUM_CHANNELS)) - 0.5) * 0.2
    return grid.astype(np.float32)

def restore_fixed_channels(grid):
    for r, c in inp_cell_pos:
        grid[r, c, 2] = 1.0
        grid[r, c, 1] = 1.0
        grid[r, c, 4] = 1.0
        grid[r, c, 5] = 1.0
    for r, c in out_cell_pos:
        grid[r, c, 3] = 1.0
    return grid

def set_inputs(grid, obs):
    for i, (r, c) in enumerate(inp_cell_pos):
        obs_idx = i // 2
        grid[r, c, 0] = float(obs[obs_idx]) * OBS_SCALE[obs_idx]
        grid[r, c, 1] = 1.0
        grid[r, c, 2] = 1.0
        grid[r, c, 4] = 1.0
        grid[r, c, 5] = 1.0
    return grid

def get_neighbourhood(grid, r, c):
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode='constant')
    return padded[r:r + 3, c:c + 3, :]

def get_outputs(grid):
    r0, c0 = out_cell_pos[0]
    r1, c1 = out_cell_pos[1]
    q = np.array([grid[r0, c0, 0], grid[r1, c1, 0]])
    return q * 100.0


encoder = PoissonEncoder(time=50, dt=1.0)

def encode_neighbourhood(neighbourhood):
    flat = torch.tensor(neighbourhood.flatten(), dtype=torch.float32)
    flat_clipped = torch.clamp(flat, -1.0, 1.0)
    flat_rates = (flat_clipped + 1.0) / 2.0 * 100.0
    return encoder(flat_rates)


def build_network(nu_ho=0.01, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    net = Network(dt=1.0)
    input_layer = Input(n=54, shape=(54,))
    hidden_layer = LIFNodes(n=32, thresh=-52.0, reset=-65.0, rest=-65.0, refrac=5, traces=True)
    output_layer = LIFNodes(n=12, thresh=-52.0, refrac=5, traces=True)
    w_ih = torch.zeros(54, 32)
    for j in range(32):
        idx = torch.randperm(54)[:22]
        w_ih[idx, j] = torch.rand(22) * 20.0
    w_ho = torch.zeros(32, 12)
    for j in range(12):
        idx = torch.randperm(32)[:13]
        w_ho[idx, j] = torch.rand(13) * 20.0
    input_to_hidden = MulticompartmentConnection(source=input_layer, target=hidden_layer, device=DEVICE, pipeline=[Weight('weight', w_ih, range=[0.0, 30.0])])
    hidden_to_output = MulticompartmentConnection(source=hidden_layer, target=output_layer, device=DEVICE, pipeline=[Weight('weight', w_ho, range=[0.0, 30.0], nu=(nu_ho, nu_ho), learning_rule=MCC_MSTDPET, enforce_polarity=True)], tc_plus=100.0, tc_minus=100.0, average_update=10, continues_update=True)
    w_lateral = -torch.ones(12, 12) * 5.0
    for i in range(12):
        w_lateral[i, i] = 0.0
    lateral_inhibition = MulticompartmentConnection(source=output_layer, target=output_layer, device=DEVICE, pipeline=[Weight('weight', w_lateral, range=[-10.0, 0.0])])
    net.add_layer(input_layer, name='input')
    net.add_layer(hidden_layer, name='hidden')
    net.add_layer(output_layer, name='output')
    net.add_connection(input_to_hidden, source='input', target='hidden')
    net.add_connection(hidden_to_output, source='hidden', target='output')
    net.add_connection(lateral_inhibition, source='output', target='output')
    monitor = Monitor(output_layer, state_vars=['s'], time=50)
    net.add_monitor(monitor, name='output_monitor')
    dummy = torch.zeros(1, 54)
    net.run(inputs={'input': dummy}, time=1, reward=0.0)
    with torch.no_grad():
        net.layers['hidden'].v = torch.rand(1, 32) * 13.0 - 6.5 - 65.0
        net.layers['output'].v = torch.rand(1, 12) * 13.0 - 6.5 - 65.0
    return net

def update_target_network(main_net, target_net):
    for conn_key in main_net.connections:
        try:
            for i, feature in enumerate(main_net.connections[conn_key].pipeline):
                if hasattr(feature, 'value'):
                    target_net.connections[conn_key].pipeline[i].value.data.copy_(feature.value.data)
        except:
            pass
    print('  [target network updated]')

def weight_zero_fraction(network):
    w = network.connections['hidden', 'output'].pipeline[0].value
    return (w == 0).float().mean().item()
update_network = build_network(nu_ho=0.01)
target_network = copy.deepcopy(update_network)
print('Network built with MulticompartmentConnection + MCC_MSTDPET')
pass
pass
pass


def reset_network(network):
    for name in ['hidden', 'output']:
        layer = network.layers[name]
        layer.s = torch.zeros_like(layer.s)

def ca_step(grid, network, reward=0.0, learn=False):
    n_cells = GRID_SIZE * GRID_SIZE
    all_neighbourhoods = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            n = get_neighbourhood(grid, r, c)
            all_neighbourhoods.append(n.flatten())
    all_n = np.stack(all_neighbourhoods)
    all_t = torch.tensor(all_n, dtype=torch.float32)
    flat_clipped = torch.clamp(all_t, -1.0, 1.0)
    flat_rates = (flat_clipped + 1.0) / 2.0 * 100.0
    all_spikes = torch.stack([encoder(flat_rates[i]) for i in range(n_cells)])
    all_spikes_t = all_spikes.permute(1, 0, 2)
    reset_network(network)
    output_counts = torch.zeros(n_cells, 12)
    output_spike_sum = torch.zeros(12)
    hidden_spike_sum = 0.0
    for t in range(50):
        spike_t = all_spikes_t[t]
        r = reward if learn and t == 49 else 0.0
        network.run(inputs={'input': spike_t}, time=1, reward=r)
        step_out = network.layers['output'].s.float()
        output_counts += step_out
        output_spike_sum += step_out.sum(dim=0)
        hidden_spike_sum += network.layers['hidden'].s.float().sum().item()
    deltas = torch.zeros(n_cells, 6)
    for ch in range(6):
        deltas[:, ch] = output_counts[:, ch * 2] - output_counts[:, ch * 2 + 1]
    mask = torch.rand(n_cells) < 0.5
    new_grid = grid.copy()
    idx = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if mask[idx]:
                new_grid[r, c, :] += deltas[idx].numpy()
            idx += 1
    new_grid = np.clip(new_grid, -5.0, 5.0)
    new_grid = restore_fixed_channels(new_grid)
    return (new_grid, output_spike_sum, hidden_spike_sum)


def compute_reward(cart_pos, terminated):
    if terminated:
        return -1.0
    return float(np.cos(cart_pos * np.pi / (2.0 * TRACK_HALF_LEN)))


def probe_layer_responses(grid, network, n_timesteps=50):
    n_cells = GRID_SIZE * GRID_SIZE
    all_neighbourhoods = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            n = get_neighbourhood(grid, r, c)
            all_neighbourhoods.append(n.flatten())
    all_n = np.stack(all_neighbourhoods)
    all_t = torch.tensor(all_n, dtype=torch.float32)
    flat_clipped = torch.clamp(all_t, -1.0, 1.0)
    flat_rates = (flat_clipped + 1.0) / 2.0 * 100.0
    all_spikes = torch.stack([encoder(flat_rates[i]) for i in range(n_cells)])
    all_spikes_t = all_spikes.permute(1, 0, 2)
    reset_network(network)
    input_total = torch.zeros(54)
    hidden_total = torch.zeros(32)
    output_total = torch.zeros(12)
    for t in range(n_timesteps):
        spike_t = all_spikes_t[t]
        input_total += spike_t.sum(dim=0)
        network.run(inputs={'input': spike_t}, time=1, reward=0.0)
        hidden_total += network.layers['hidden'].s.float().sum(dim=0)
        output_total += network.layers['output'].s.float().sum(dim=0)
    return (input_total, hidden_total, output_total)

def run_state_response_probe(net, obs, n_developmental=10, n_inference=3):
    grid = make_grid()
    grid = restore_fixed_channels(grid)
    grid = set_inputs(grid, obs)
    for _ in range(n_developmental):
        grid, _, _ = ca_step(grid, net, reward=0.0, learn=False)
    for i in range(n_inference):
        if i < n_inference - 1:
            grid, _, _ = ca_step(grid, net, reward=0.0, learn=False)
        else:
            in_vec, hid_vec, out_vec = probe_layer_responses(grid, net)
    return (in_vec, hid_vec, out_vec, get_outputs(grid))

def cosine_sim(a, b):
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float('nan')
    return (a @ b / denom).item()

def sample_state(rng):
    cart_pos = rng.uniform(-2.0, 2.0)
    cart_vel = rng.uniform(-2.0, 2.0)
    pole_angle = rng.uniform(-0.2, 0.2)
    pole_angular_vel = rng.uniform(-2.0, 2.0)
    obs = np.array([cart_pos, cart_vel, pole_angle, pole_angular_vel], dtype=np.float32)
    label = 1 if pole_angle > 0 else 0
    return (obs, label)
