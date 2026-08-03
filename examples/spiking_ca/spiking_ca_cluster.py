"""
Spiking Neural CA for CartPole — Cluster-ready training script
==============================================================
Combines Neural Cellular Automata (Variengien et al. 2021) with
spiking neurons and reward-modulated STDP via BindsNET.

Usage:
    python spiking_ca_cluster.py [options]

Options:
    --n_episodes        Number of training episodes (default: 500)
    --max_steps         Max steps per episode (default: 200)
    --tc_plus           Pre-synaptic trace time constant (default: 100.0)
    --tc_minus          Post-synaptic trace time constant (default: 100.0)
    --nu_ih             Learning rate input->hidden (default: 0.01)
    --nu_ho             Learning rate hidden->output (default: 0.01)
    --average_update    Moving average window for eligibility trace (default: 10)
    --target_update     Target network update frequency in episodes (default: 20)
    --epsilon_decay     Epsilon decay rate — episodes to reach 0.1 (default: 300)
    --n_inference_steps CA steps per action for inference (default: 3)
    --grid_size         CA grid size (default: 16)
    --hidden_n          Number of hidden neurons (default: 32)
    --sparse_ih         Sparsity of input->hidden connections as fraction (default: 0.4)
    --sparse_ho         Sparsity of hidden->output connections as fraction (default: 0.4)
    --lateral_w         Lateral inhibition weight (default: 5.0)
    --output_file       File to save results to (default: results.json)

Install:
    pip install git+https://github.com/BindsNET/bindsnet.git
    pip install gymnasium

Example cluster sweep:
    python spiking_ca_cluster.py --tc_plus 50  --output_file results_tc50.json
    python spiking_ca_cluster.py --tc_plus 100 --output_file results_tc100.json
    python spiking_ca_cluster.py --tc_plus 200 --output_file results_tc200.json
"""

import argparse
import json
import math
import copy
import numpy as np
import torch
import gymnasium as gym
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning.MCC_learning import MSTDPET as MCC_MSTDPET

# ── Patch MSTDPET.reset_state_variables ─────────────────────────────
# The installed MSTDPET.reset_state_variables only zeros eligibility and
# eligibility_trace. p_plus, p_minus, and the moving-average buffer are
# left untouched, so they silently carry over between calls (including
# across episode boundaries once the reset call below is added). This
# patches the class method in place so every MCC_MSTDPET instance
# picks up the fix regardless of which bindsnet build is installed.
# Permanent fix: https://github.com/saachigoyall/bindsnet PR (pending).

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
print("Patched MCC_MSTDPET.reset_state_variables")

# ── Parse arguments ──────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Spiking CA CartPole training")
parser.add_argument("--n_episodes",        type=int,   default=500)
parser.add_argument("--max_steps",         type=int,   default=200)
parser.add_argument("--tc_plus",           type=float, default=100.0)
parser.add_argument("--tc_minus",          type=float, default=100.0)
parser.add_argument("--nu_ih",             type=float, default=0.01)
parser.add_argument("--nu_ho",             type=float, default=0.01)
parser.add_argument("--average_update",    type=int,   default=10)
parser.add_argument("--target_update",     type=int,   default=20)
parser.add_argument("--epsilon_decay",     type=float, default=300.0)
parser.add_argument("--n_inference_steps", type=int,   default=3)
parser.add_argument("--grid_size",         type=int,   default=16)
parser.add_argument("--hidden_n",          type=int,   default=32)
parser.add_argument("--sparse_ih",         type=float, default=0.4)
parser.add_argument("--sparse_ho",         type=float, default=0.4)
parser.add_argument("--lateral_w",         type=float, default=5.0)
parser.add_argument("--output_file",       type=str,   default="results.json")
args = parser.parse_args()

print("Configuration:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

# ── Constants ────────────────────────────────────────────────────────

GRID_SIZE      = args.grid_size
NUM_CHANNELS   = 6
TRACK_HALF_LEN = 2.4
DEVICE         = torch.device("cpu")

# input and output cell positions scaled for grid size
# these are from the original CA paper scaled to 16x16
inp_cell_pos = [(5,13),(12,10),(2,10),(9,3),(9,13),(2,6),(12,6),(5,3)]
out_cell_pos = [(6,8),(9,8)]

# observation scaling confirmed from CA paper library source
OBS_SCALE = [2.0, 0.25, 4.0, 0.15]

# ── Grid functions ───────────────────────────────────────────────────

def make_grid():
    """Initialise grid with small random values — same as CA paper."""
    grid = (np.random.random((GRID_SIZE, GRID_SIZE, NUM_CHANNELS)) - 0.5) * 0.2
    return grid.astype(np.float32)

def restore_fixed_channels(grid):
    """
    Restore input and output cell identifiers after every update.
    input cells:  ch2=1 (identifier), ch1/ch4/ch5=1 (hidden)
    output cells: ch3=1 (identifier)
    """
    for (r, c) in inp_cell_pos:
        grid[r, c, 2] = 1.0
        grid[r, c, 1] = 1.0
        grid[r, c, 4] = 1.0
        grid[r, c, 5] = 1.0
    for (r, c) in out_cell_pos:
        grid[r, c, 3] = 1.0
    return grid

def set_inputs(grid, obs):
    """Write scaled CartPole observations into input cell channels."""
    for i, (r, c) in enumerate(inp_cell_pos):
        obs_idx = i // 2
        grid[r, c, 0] = float(obs[obs_idx]) * OBS_SCALE[obs_idx]
        grid[r, c, 1] = 1.0
        grid[r, c, 2] = 1.0
        grid[r, c, 4] = 1.0
        grid[r, c, 5] = 1.0
    return grid

def get_neighbourhood(grid, r, c):
    """Get 3x3x6 neighbourhood with zero padding at boundaries."""
    padded = np.pad(grid, ((1,1),(1,1),(0,0)), mode='constant')
    return padded[r:r+3, c:c+3, :]

def get_outputs(grid):
    """Read Q-values from output cell info channels, scale by 100."""
    r0, c0 = out_cell_pos[0]
    r1, c1 = out_cell_pos[1]
    q = np.array([grid[r0, c0, 0], grid[r1, c1, 0]])
    return q * 100.0

# ── Encoding ─────────────────────────────────────────────────────────

encoder = PoissonEncoder(time=50, dt=1.0)

def encode_neighbourhood(neighbourhood):
    """
    Convert 3x3x6 neighbourhood into Poisson spike trains.
    - flatten 3x3x6 → 54 numbers
    - clip to [-1, 1]
    - shift to [0, 1]: (val + 1) / 2
    - scale to firing rates [0, 100]
    - neutral value (0) maps to rate 50
    """
    flat         = torch.tensor(neighbourhood.flatten(), dtype=torch.float32)
    flat_clipped = torch.clamp(flat, -1.0, 1.0)
    flat_rates   = (flat_clipped + 1.0) / 2.0 * 100.0
    return encoder(flat_rates)

# ── Network ───────────────────────────────────────────────────────────
#
# Architecture:
#   Input:  54 neurons (3x3x6 neighbourhood)
#   Hidden: hidden_n LIF neurons, sparse connectivity, traces=True
#   Output: 12 LIF neurons — 2 per channel (excitatory + inhibitory), traces=True
#
# MulticompartmentConnection — as requested by Professor Hazan
# MCC_MSTDPET with average_update — built-in moving average smoothing
# enforce_polarity=True — Dale's law (excitatory weights stay non-negative)
# Sparse connectivity — breaks lockstep synchronisation
# Lateral inhibition — output neurons inhibit each other
# traces=True — required for MSTDPET eligibility trace to work correctly
#   (without this MSTDPET silently collapses to plain MSTDP)
# Voltage carries forward between CA steps (not reset) — preserves
#   temporal continuity for eligibility trace
#
# Key findings during development:
# - traces=True was the critical missing ingredient
# - voltage reset was destroying temporal continuity
# - reward applied on final timestep only (not all 50)
# - target network needed for stable Q-value estimates

update_network = Network(dt=1.0)

input_layer  = Input(n=54, shape=(54,))
hidden_layer = LIFNodes(
    n=args.hidden_n, thresh=-52.0, reset=-65.0, rest=-65.0, refrac=5, traces=True
)
output_layer = LIFNodes(n=12, thresh=-52.0, refrac=5, traces=True)

# sparse weights
n_ih = int(54 * args.sparse_ih)
n_ho = int(args.hidden_n * args.sparse_ho)

w_ih = torch.zeros(54, args.hidden_n)
for j in range(args.hidden_n):
    idx = torch.randperm(54)[:n_ih]
    w_ih[idx, j] = torch.rand(n_ih) * 20.0

w_ho = torch.zeros(args.hidden_n, 12)
for j in range(12):
    idx = torch.randperm(args.hidden_n)[:n_ho]
    w_ho[idx, j] = torch.rand(n_ho) * 20.0

input_to_hidden = MulticompartmentConnection(
    source=input_layer,
    target=hidden_layer,
    device=DEVICE,
    pipeline=[
        Weight(
            "weight",
            w_ih,
            range=[0.0, 30.0],
            nu=(args.nu_ih, args.nu_ih * 0.1),
            learning_rule=MCC_MSTDPET,
            enforce_polarity=True,
        )
    ],
    tc_plus=args.tc_plus,
    tc_minus=args.tc_minus,
    average_update=args.average_update,
    continues_update=True,
)

hidden_to_output = MulticompartmentConnection(
    source=hidden_layer,
    target=output_layer,
    device=DEVICE,
    pipeline=[
        Weight(
            "weight",
            w_ho,
            range=[0.0, 30.0],
            nu=(args.nu_ho, args.nu_ho),
            learning_rule=MCC_MSTDPET,
            enforce_polarity=True,
        )
    ],
    tc_plus=args.tc_plus,
    tc_minus=args.tc_minus,
    average_update=args.average_update,
    continues_update=True,
)

# lateral inhibition — fixed negative weights, no learning rule
w_lateral = -torch.ones(12, 12) * args.lateral_w
for i in range(12):
    w_lateral[i, i] = 0.0

lateral_inhibition = MulticompartmentConnection(
    source=output_layer,
    target=output_layer,
    device=DEVICE,
    pipeline=[
        Weight(
            "weight",
            w_lateral,
            range=[-args.lateral_w * 2, 0.0],
        )
    ],
)

update_network.add_layer(input_layer,  name="input")
update_network.add_layer(hidden_layer, name="hidden")
update_network.add_layer(output_layer, name="output")
update_network.add_connection(input_to_hidden,    source="input",  target="hidden")
update_network.add_connection(hidden_to_output,   source="hidden", target="output")
update_network.add_connection(lateral_inhibition, source="output", target="output")

monitor = Monitor(output_layer, state_vars=["s"], time=50)
update_network.add_monitor(monitor, name="output_monitor")

# warm up to build internal tensors with correct batch dimension
dummy = torch.zeros(1, 54)
update_network.run(inputs={"input": dummy}, time=1, reward=0.0)

# set initial voltages after warm up
with torch.no_grad():
    update_network.layers["hidden"].v = torch.rand(1, args.hidden_n) * 13.0 - 6.5 - 65.0
    update_network.layers["output"].v = torch.rand(1, 12) * 13.0 - 6.5 - 65.0

# target network — frozen copy for stable action selection
target_network = copy.deepcopy(update_network)

def update_target_network(main_net, target_net):
    """Copy weights from main network into target network."""
    for conn_key in main_net.connections:
        try:
            for i, feature in enumerate(main_net.connections[conn_key].pipeline):
                if hasattr(feature, 'value'):
                    target_net.connections[conn_key].pipeline[i].value.data.copy_(
                        feature.value.data
                    )
        except:
            pass

# ── Reset and CA step ────────────────────────────────────────────────

def reset_network(network):
    """
    Reset only spikes between CA steps.
    Voltage and refractory count carry forward to preserve temporal
    continuity — required for MSTDPET eligibility trace to work correctly.
    """
    for name in ["hidden", "output"]:
        layer = network.layers[name]
        layer.s = torch.zeros_like(layer.s)

def ca_step(grid, network, reward=0.0, learn=False):
    """
    One CA step — runs shared spiking network on all grid cells simultaneously.

    learn=False: inference only, reward always 0
    learn=True:  reward applied on final timestep only (t=49)
                 applying on all 50 timesteps dilutes the learning signal
    """
    n_cells = GRID_SIZE * GRID_SIZE

    all_neighbourhoods = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            n = get_neighbourhood(grid, r, c)
            all_neighbourhoods.append(n.flatten())

    all_n        = np.stack(all_neighbourhoods)
    all_t        = torch.tensor(all_n, dtype=torch.float32)
    flat_clipped = torch.clamp(all_t, -1.0, 1.0)
    flat_rates   = (flat_clipped + 1.0) / 2.0 * 100.0
    all_spikes   = torch.stack([encoder(flat_rates[i]) for i in range(n_cells)])
    all_spikes_t = all_spikes.permute(1, 0, 2)

    reset_network(network)
    output_counts = torch.zeros(n_cells, 12)

    for t in range(50):
        spike_t = all_spikes_t[t]
        r = reward if (learn and t == 49) else 0.0
        network.run(inputs={"input": spike_t}, time=1, reward=r)
        output_counts += network.layers["output"].s.float()

    deltas = torch.zeros(n_cells, 6)
    for ch in range(6):
        deltas[:, ch] = output_counts[:, ch*2] - output_counts[:, ch*2+1]

    mask     = torch.rand(n_cells) < 0.5
    new_grid = grid.copy()
    idx = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if mask[idx]:
                new_grid[r, c, :] += deltas[idx].numpy()
            idx += 1

    new_grid = np.clip(new_grid, -5.0, 5.0)
    new_grid = restore_fixed_channels(new_grid)
    return new_grid

# ── Reward function ───────────────────────────────────────────────────

def compute_reward(cart_pos, terminated):
    """
    Cosine reward from original CA paper (equation 1).
    Normalised to [-1, 1].
    cos(x * pi / 2L) when alive — rewards staying near centre of track.
    -1.0 when terminated.
    """
    if terminated:
        return -1.0
    return float(np.cos(cart_pos * math.pi / (2.0 * TRACK_HALF_LEN)))

# ── Training loop ─────────────────────────────────────────────────────

def train():
    """
    Train spiking CA with MCC_MSTDPET and target network.

    Each CartPole step:
    1. Developmental phase: 10 CA steps at episode start (no learning)
    2. n_inference_steps CA steps using TARGET network (stable Q-values)
    3. Epsilon greedy action selection
    4. Step environment, get reward
    5. One learning CA step using MAIN network (reward on t=49 only)
    Every target_update episodes: copy main weights to target network
    """
    env    = gym.make("CartPole-v1")
    scores = []

    for episode in range(args.n_episodes):
        obs, _ = env.reset()

        # reset network state between episodes
        # clears voltage, traces, p_plus, p_minus, eligibility traces,
        # and (with the patch above) the moving-average buffer + index
        # prevents stale state from previous episodes poisoning new ones
        update_network.reset_state_variables()
        target_network.reset_state_variables()

        # re-initialise voltages after reset to preserve batch dimension
        # BindsNET reset squashes voltage shape from (1,n) to (n,)
        with torch.no_grad():
            update_network.layers["hidden"].v = (
                torch.rand(1, args.hidden_n) * 13.0 - 6.5 - 65.0
            )
            update_network.layers["output"].v = torch.rand(1, 12) * 13.0 - 6.5 - 65.0
            target_network.layers["hidden"].v = (
                torch.rand(1, args.hidden_n) * 13.0 - 6.5 - 65.0
            )
            target_network.layers["output"].v = torch.rand(1, 12) * 13.0 - 6.5 - 65.0

        grid = make_grid()
        grid = restore_fixed_channels(grid)

        # developmental phase — let grid self-organise, no learning
        grid = set_inputs(grid, obs)
        for _ in range(10):
            grid = ca_step(grid, target_network, reward=0.0, learn=False)

        total = 0
        done  = False

        while not done and total < args.max_steps:
            grid = set_inputs(grid, obs)

            # inference steps using target network for stable Q-values
            for _ in range(args.n_inference_steps):
                grid = ca_step(grid, target_network, reward=0.0, learn=False)

            # epsilon greedy action selection
            epsilon = max(0.1, 1.0 - episode / args.epsilon_decay)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = get_outputs(grid)
                action   = int(np.argmax(q_values))

            obs, _, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            reward = compute_reward(float(obs[0]), terminated)

            # learning step — reward applied on final timestep only
            grid = ca_step(grid, update_network, reward=reward, learn=True)

            total += 1

        scores.append(total)

        # update target network
        if episode % args.target_update == 0 and episode > 0:
            update_target_network(update_network, target_network)

        if episode % 10 == 0:
            avg = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f"  episode {episode:4d}  steps={total:4d}  avg10={avg:.1f}")

    env.close()
    return scores

# ── Run and save results ──────────────────────────────────────────────

print(f"\nStarting training for {args.n_episodes} episodes...")
scores = train()

results = {
    'config': vars(args),
    'scores': scores,
    'final_mean_20': float(np.mean(scores[-20:])),
    'best_episode': int(max(scores)),
    'mean_all': float(np.mean(scores)),
}

with open(args.output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {args.output_file}")
print(f"Final mean (last 20): {results['final_mean_20']:.1f}")
print(f"Best episode: {results['best_episode']} steps")
print(f"Random baseline: ~21 steps")
