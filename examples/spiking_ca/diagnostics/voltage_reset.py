"""Checks 3, 2 and 1 from Dr. Hazan's Nov 9 email (power 3 encoder, batch fix, channel 0 restore, delta scaling).

  python drift_checks.py <seed> A [n_states]   delta 0.02, 10 CA steps:
                                               check 3 (scale hidden weights so hidden spikes per unit stay at the step 0 level)
                                               check 1 (is the drift the same in every cell?)
                                               plus spread / mean rate numbers for comparison with the Friday update
  python drift_checks.py <seed> B [n_states]   delta 0.005, 24 CA steps: check 2 (accuracy against mean input change)

Run it from anywhere; DIAG below points at the diagnostics folder of the repo (examples/spiking_ca/diagnostics).
"""
import sys, os, json
DIAG = os.environ.get('DIAG', '/home/claude/bindsnet/examples/spiking_ca/diagnostics')
sys.path.insert(0, DIAG)
import numpy as np, torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bindsnet.encoding import PoissonEncoder
from ca_model import (GRID_SIZE, NUM_CHANNELS, build_network, make_grid, inp_cell_pos, OBS_SCALE,
                      restore_fixed_channels, set_inputs, reset_network)
from probe import POLE_ANGLE_CELLS, cell_rows, assert_cells_independent

torch.set_num_threads(1)
seed, phase = int(sys.argv[1]), 'V'
MODE = sys.argv[4]   # none | probe | traj
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150
POWER, NC = 3, GRID_SIZE * GRID_SIZE
ROWS = cell_rows(POLE_ANGLE_CELLS)                     # the two pole angle cells
ANGLE = 24                                             # centre cell, channel 0 (restored every step)
FIXED = [25, 26, 28, 29]                               # centre cell channels 1, 2, 4, 5 (restored every step)
CHANGEABLE = [i for i in range(54) if i != ANGLE and i not in FIXED]      # the 49 inputs that can change
enc = PoissonEncoder(time=50, dt=1.0)

def acc(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return float(cross_val_score(clf, np.array(X), np.array(y), cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())

def rates_of(grid):
    """(256, 54) firing rate in Hz that each cell's 54 inputs get (before Poisson sampling)."""
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)))
    nb = sliding_window_view(padded, (3, 3, NUM_CHANNELS), axis=(0, 1, 2)).reshape(NC, 54)
    return (((np.clip(nb, -1.0, 1.0) + 1.0) / 2.0) ** POWER) * 100.0

net = build_network(nu_ho=0.0, seed=seed); net.learning = False
W = net.connections[('input', 'hidden')].pipeline[0].value
W0 = W.clone()
def set_scale(s): 
    with torch.no_grad(): W.copy_(W0 * s)

def run(grid, first=False):
    r = torch.tensor(rates_of(grid), dtype=torch.float32)
    r[r < 0.05] = 0.0                                   # BindsNET's Poisson encoder overflows on tiny nonzero rates
    sp = enc(r)                                          # (50, 256, 54)
    reset_network(net)
    if MODE == 'probe' or (MODE == 'traj' and first):
        for nm in ('hidden', 'output'): net.layers[nm].reset_state_variables()
    hid = torch.zeros(NC, 32); out = torch.zeros(NC, 12)
    for t in range(50):
        net.run(inputs={'input': sp[t].unsqueeze(0)}, time=1, reward=0.0)
        hid += net.layers['hidden'].s.float().reshape(NC, -1)
        out += net.layers['output'].s.float().reshape(NC, -1)
    return hid, out

def restore_ch0(grid, obs):
    for i, (r, c) in enumerate(inp_cell_pos):
        grid[r, c, 0] = float(obs[i // 2]) * OBS_SCALE[i // 2]
    return grid

def ca_update(grid, out, obs, delta):
    d = torch.zeros(NC, 6)
    for ch in range(6): d[:, ch] = out[:, ch * 2] - out[:, ch * 2 + 1]
    d = (d * delta).numpy().reshape(GRID_SIZE, GRID_SIZE, 6)
    mask = (torch.rand(NC) < 0.5).numpy().reshape(GRID_SIZE, GRID_SIZE, 1)
    new = np.clip(grid + d * mask, -5.0, 5.0)
    return restore_ch0(restore_fixed_channels(new), obs)

rng = np.random.default_rng(seed); np.random.seed(seed); torch.manual_seed(seed)
def sample():
    obs = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-0.2, 0.2), rng.uniform(-2, 2)], dtype=np.float32)
    return obs, int(obs[2] > 0)
def restore_ch0(grid, obs):
    for i, (r, c) in enumerate(inp_cell_pos):
        grid[r, c, 0] = float(obs[i // 2]) * OBS_SCALE[i // 2]
    return grid
def ca_update(grid, out, obs, delta):
    d = torch.zeros(NC, 6)
    for ch in range(6): d[:, ch] = out[:, ch * 2] - out[:, ch * 2 + 1]
    d = (d * delta).numpy().reshape(GRID_SIZE, GRID_SIZE, 6)
    mask = (torch.rand(NC) < 0.5).numpy().reshape(GRID_SIZE, GRID_SIZE, 1)
    new = np.clip(grid + d * mask, -5.0, 5.0)
    return restore_ch0(restore_fixed_channels(new), obs)
K = 10; y = []; H = {k: [] for k in (0, 1, 3, 10)}
for i in range(N):
    obs, lab = sample(); y.append(lab)
    grid = set_inputs(restore_fixed_channels(make_grid()), obs)
    for step in range(K + 1):
        hid, out = run(grid, first=(step == 0))
        if step in H: H[step].append(hid[ROWS].flatten().numpy())
        if step == K: break
        grid = ca_update(grid, out, obs, 0.02)
res = {'seed': seed, 'mode': MODE, 'n': N, 'acc': {str(k): acc(H[k], y) for k in H}, 'hid_per_unit': {str(k): float(np.mean(H[k])) for k in H}}
json.dump(res, open(f'vreset_{MODE}_seed{seed}.json', 'w')); print(res)
