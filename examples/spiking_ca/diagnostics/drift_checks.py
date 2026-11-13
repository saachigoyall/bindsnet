"""Checks 3, 2 and 1 from Dr. Hazan's Nov 9 email (power 3 encoder, batch fix, channel 0 restore, delta scaling).

  python drift_checks.py <seed> A [n_states]   delta 0.02, 10 CA steps:
                                               check 3 (scale hidden weights so hidden spikes per unit stay at the step 0 level)
                                               check 1 (is the drift the same in every cell?)
                                               plus spread / mean rate numbers for comparison with the Friday update
  python drift_checks.py <seed> B [n_states]   delta 0.005, 24 CA steps: check 2 (accuracy against mean input change)

Run it from anywhere; DIAG below points at the diagnostics folder of the repo (examples/spiking_ca/diagnostics).
"""
import sys, os, json
DIAG = os.environ.get("DIAG", os.path.dirname(os.path.abspath(__file__)))
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
seed, phase = int(sys.argv[1]), sys.argv[2]
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

def run(grid):
    r = torch.tensor(rates_of(grid), dtype=torch.float32)
    r[r < 0.05] = 0.0                                   # BindsNET's Poisson encoder overflows on tiny nonzero rates
    sp = enc(r)                                          # (50, 256, 54)
    reset_network(net)
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

def trajectory(delta, K, keep_grids=()):
    y, hid_steps, rate_steps, grids = [], {k: [] for k in range(K + 1)}, {k: [] for k in range(K + 1)}, {k: [] for k in keep_grids}
    for i in range(N):
        obs, lab = sample(); y.append(lab)
        grid = set_inputs(restore_fixed_channels(make_grid()), obs)
        for step in range(K + 1):
            hid, out = run(grid)
            if step == 0 and i == 0: print('distinct hidden rows:', assert_cells_independent(hid), flush=True)
            hid_steps[step].append(hid[ROWS].flatten().numpy())
            rate_steps[step].append(rates_of(grid)[ROWS])
            if step in grids: grids[step].append(grid.copy())
            if step == K: break
            grid = ca_update(grid, out, obs, delta)
        if i % 25 == 0: print(f'seed {seed} {phase}: state {i}/{N}', flush=True)
    return y, {k: np.array(v) for k, v in hid_steps.items()}, {k: np.array(v) for k, v in rate_steps.items()}, grids

result = {'seed': seed, 'phase': phase, 'n_states': N}

if phase == 'A':
    DELTA, K = 0.02, 10
    y, H, R, G = trajectory(DELTA, K, keep_grids=(0, K))
    result['acc_step'] = {str(k): acc(H[k], y) for k in (0, 1, 3, 10)}
    # numbers to compare with the Friday update
    ch = R[0][:, :, CHANGEABLE], R[K][:, :, CHANGEABLE]
    result['mean_rate_hz'] = [float(c.mean()) for c in ch]
    result['spread_hz'] = [float(c.std(axis=2).mean()) for c in ch]           # std across the 49 inputs, averaged over states and pole cells
    result['hidden_spikes_per_unit'] = [float(H[0].mean()), float(H[K].mean())]
    result['mean_input_change_hz'] = float(np.abs(R[K][:, :, CHANGEABLE] - R[0][:, :, CHANGEABLE]).mean())
    # CHECK 3: scale input to hidden weights at the step 10 probe so hidden spikes per unit match step 0
    target = float(H[0].mean())
    def probe_grids(grids, s):
        set_scale(s)
        X = [run(g)[0][ROWS].flatten().numpy() for g in grids]
        set_scale(1.0)
        return np.array(X)
    lo, hi = 0.2, 1.0
    for _ in range(8):
        mid = (lo + hi) / 2
        f = probe_grids(G[K][:40], mid).mean()
        lo, hi = (mid, hi) if f < target else (lo, mid)
    s = (lo + hi) / 2
    X10s, X0s = probe_grids(G[K], s), probe_grids(G[0], s)
    result['check3'] = {'target_spikes_per_unit': target, 'scale': s,
                        'step10_unscaled_acc': acc(H[K], y), 'step10_scaled_acc': acc(X10s, y),
                        'step10_scaled_spikes_per_unit': float(X10s.mean()),
                        'step0_unscaled_acc': acc(H[0], y), 'step0_scaled_acc_control': acc(X0s, y)}
    # CHECK 1: is the drift the same in every cell?  Use interior cells only (border cells see zero padding).
    num = den = 0.0; common_g = []; fam = []; per_ch = np.zeros((2, 6)); 
    for g0, g1 in zip(G[0], G[K]):
        dG = (g1 - g0)[1:-1, 1:-1].reshape(-1, 6)                      # (196 interior cells, 6 channels)
        mu, sd = dG.mean(0), dG.std(0)
        common_g.append(float((mu ** 2).sum() / ((mu ** 2).sum() + (sd ** 2).sum())))
        per_ch += np.stack([mu, sd])
        dR = (rates_of(g1) - rates_of(g0)).reshape(GRID_SIZE, GRID_SIZE, 54)[1:-1, 1:-1].reshape(-1, 54)[:, CHANGEABLE]
        m2, s2 = dR.mean(0), dR.std(0)
        fam.append(float((m2 ** 2).sum() / ((m2 ** 2).sum() + (s2 ** 2).sum())))
    result['check1'] = {'common_fraction_grid_channels': float(np.mean(common_g)), 'common_fraction_input_rates': float(np.mean(fam)),
                        'per_channel_mean_change': (per_ch[0] / N).tolist(), 'per_channel_std_across_cells': (per_ch[1] / N).tolist()}
else:
    DELTA, K = 0.005, 24
    y, H, R, _ = trajectory(DELTA, K)
    result['steps'] = list(range(K + 1))
    result['acc'] = [acc(H[k], y) for k in range(K + 1)]
    result['mean_input_change_hz'] = [float(np.abs(R[k][:, :, CHANGEABLE] - R[0][:, :, CHANGEABLE]).mean()) for k in range(K + 1)]
    result['hidden_spikes_per_unit'] = [float(H[k].mean()) for k in range(K + 1)]

json.dump(result, open(f'checks_{phase}_seed{seed}.json', 'w'))
print('done', seed, phase, flush=True)
