"""Hidden layer variants tested inside the CA loop (power 3 encoder, batch fix, channel 0 restore, delta 0.02, 10 CA steps).

  python step10_variants.py <variant> <seed> [n_states]

variants: base, lateral5, shared_theta, percell_theta, percell_theta_small, hetero_decay, percell_double, bal0.5, bal1.0
Voltage and adaptive thresholds are reset to rest at the start of every state (Dr. Hazan's rule), and carry over between CA steps.
Writes variant_<variant>_seed<seed>.json
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
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from ca_model import (GRID_SIZE, NUM_CHANNELS, make_grid, inp_cell_pos, OBS_SCALE, restore_fixed_channels, set_inputs, reset_network, DEVICE)
from probe import POLE_ANGLE_CELLS, cell_rows

torch.set_num_threads(1)
variant, seed = sys.argv[1], int(sys.argv[2])
PARTS = variant.split('+')
BALP = next((p for p in PARTS if p.startswith('bal')), None)
WNP = next((p for p in PARTS if p.startswith('wnorm')), None)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150
POWER, NC, DELTA, K = 3, GRID_SIZE * GRID_SIZE, 0.02, 10
ROWS = cell_rows(POLE_ANGLE_CELLS); ANGLE = 24
enc = PoissonEncoder(time=50, dt=1.0)

class PerCellAdaptiveLIFNodes(AdaptiveLIFNodes):
    """Adaptive threshold with one theta per cell and neuron ([batch, n]); BindsNET's version sums spikes over the batch,
    so a spike in any of the 256 cells raised the threshold of all of them. Optional second, slower component."""
    def __init__(self, *a, theta_plus2=0.0, tc_theta2=1000.0, **k):
        super().__init__(*a, **k)
        self.theta_plus2, self.tc_theta2 = theta_plus2, tc_theta2
        self.theta2 = torch.zeros(0)
    def forward(self, x):
        if self.theta.shape != self.v.shape:
            self.theta = torch.zeros_like(self.v); self.theta2 = torch.zeros_like(self.v)
        self.v.sub_(self.rest).mul_(self.decay).add_(self.rest)
        if self.learning:
            self.theta.mul_(self.theta_decay); self.theta2.mul_(float(np.exp(-1.0 / self.tc_theta2)))
        self.v.add_((self.refrac_count <= 0).float() * x)
        self.refrac_count.sub_(self.dt)
        self.s = self.v >= self.thresh + self.theta + self.theta2
        self.refrac_count.masked_fill_(self.s, self.refrac); self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta.add_(self.theta_plus * self.s.float()); self.theta2.add_(self.theta_plus2 * self.s.float())
        Nodes_forward(self, x)
    def reset_state_variables(self):
        super().reset_state_variables()
        self.theta = torch.zeros_like(self.v); self.theta2 = torch.zeros_like(self.v)
from bindsnet.network.nodes import Nodes
def Nodes_forward(self, x): Nodes.forward(self, x)

def build(variant, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Network(dt=1.0); inp = Input(n=54, shape=(54,))
    hid_kw = dict(n=32, thresh=-52.0, reset=-65.0, rest=-65.0, refrac=5, traces=True)
    if variant == 'shared_theta': hid = AdaptiveLIFNodes(theta_plus=2.0, tc_theta_decay=100.0, **hid_kw)
    elif variant == 'percell_theta': hid = PerCellAdaptiveLIFNodes(theta_plus=2.0, tc_theta_decay=100.0, **hid_kw)
    elif variant == 'percell_theta_small': hid = PerCellAdaptiveLIFNodes(theta_plus=0.5, tc_theta_decay=100.0, **hid_kw)
    elif variant == 'percell_double': hid = PerCellAdaptiveLIFNodes(theta_plus=1.0, tc_theta_decay=20.0, theta_plus2=0.2, tc_theta2=200.0, **hid_kw)
    elif variant == 'hetero_decay':
        g = torch.distributions.Gamma(4.0, 1 / 25.0).sample((32,)).clamp(10, 400)       # mean 100 ms
        hid = LIFNodes(tc_decay=g, **hid_kw)
    else: hid = LIFNodes(**hid_kw)
    out = LIFNodes(n=12, thresh=-52.0, refrac=5, traces=True)
    w_ih = torch.zeros(54, 32)
    for j in range(32):
        idx = torch.randperm(54)[:22]; w_ih[idx, j] = torch.rand(22) * 20.0
    w_ho = torch.zeros(32, 12)
    for j in range(12):
        idx = torch.randperm(32)[:13]; w_ho[idx, j] = torch.rand(13) * 20.0
    net.add_layer(inp, 'input'); net.add_layer(hid, 'hidden'); net.add_layer(out, 'output')
    net.add_connection(MulticompartmentConnection(source=inp, target=hid, device=DEVICE, pipeline=[Weight('weight', w_ih, range=[0.0, 30.0])]), 'input', 'hidden')
    net.add_connection(MulticompartmentConnection(source=hid, target=out, device=DEVICE, pipeline=[Weight('weight', w_ho, range=[0.0, 30.0])]), 'hidden', 'output')
    wl = -torch.ones(12, 12) * 5.0; wl.fill_diagonal_(0.0)
    net.add_connection(MulticompartmentConnection(source=out, target=out, device=DEVICE, pipeline=[Weight('weight', wl, range=[-10.0, 0.0])]), 'output', 'output')
    if variant == 'lateral5':
        wlh = -torch.ones(32, 32) * 5.0; wlh.fill_diagonal_(0.0)
        net.add_connection(MulticompartmentConnection(source=hid, target=hid, device=DEVICE, pipeline=[Weight('weight', wlh, range=[-10.0, 0.0])]), 'hidden', 'hidden')
    if BALP is not None:
        NI = 16
        inh = LIFNodes(n=NI, thresh=-52.0, reset=-65.0, rest=-65.0, refrac=5, traces=True)
        w_ie = torch.zeros(54, NI)
        for k in range(NI):
            idx = torch.tensor([i for i in torch.randperm(54).tolist() if i != ANGLE][:22]); w_ie[idx, k] = torch.rand(22) * 20.0   # same 53 inputs, never the angle
        net.add_layer(inh, 'inh')
        net.add_connection(MulticompartmentConnection(source=inp, target=inh, device=DEVICE, pipeline=[Weight('weight', w_ie, range=[0.0, 30.0])]), 'input', 'inh')
        w_ei = -torch.ones(NI, 32)                                         # scaled later by calibrate()
        net.add_connection(MulticompartmentConnection(source=inh, target=hid, device=DEVICE, pipeline=[Weight('weight', w_ei, range=[-1000.0, 0.0])]), 'inh', 'hidden')
        net._w_ih = w_ih
    net.learning = False
    for nm in ('hidden',): net.layers[nm].learning = True        # thresholds adapt, weights do not
    net.run(inputs={'input': torch.zeros(1, 1, 54)}, time=1, reward=0.0)
    return net

def rates_of(grid):
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)))
    nb = sliding_window_view(padded, (3, 3, NUM_CHANNELS), axis=(0, 1, 2)).reshape(NC, 54)
    return (((np.clip(nb, -1.0, 1.0) + 1.0) / 2.0) ** POWER) * 100.0

net = build(variant, seed)
def reset_state():
    for nm in net.layers:
        if nm != 'input': net.layers[nm].reset_state_variables()
def run(grid):
    r = torch.tensor(rates_of(grid), dtype=torch.float32); r[r < 0.05] = 0.0
    sp = enc(r); reset_network(net)
    hid = torch.zeros(NC, 32); out = torch.zeros(NC, 12); inh = torch.zeros(NC, 16)
    for t in range(50):
        net.run(inputs={'input': sp[t].unsqueeze(0)}, time=1, reward=0.0)
        hid += net.layers['hidden'].s.float().reshape(NC, -1); out += net.layers['output'].s.float().reshape(NC, -1)
        if 'inh' in net.layers: inh += net.layers['inh'].s.float().reshape(NC, -1)
    return hid, out, inh

if BALP is not None:
    g = float(BALP[3:])
    neutral = set_inputs(restore_fixed_channels(make_grid()), np.zeros(4, dtype=np.float32))
    reset_state(); _, _, inh0 = run(neutral)
    p = rates_of(neutral) / 1000.0
    w = net._w_ih.clone(); w[ANGLE, :] = 0.0
    E = float((torch.tensor(p, dtype=torch.float32) @ w).mean() * 50)             # mean mV per window from the 53 inputs into one hidden unit
    f = float(inh0.sum(1).mean())                                                  # inhibitory spikes per window per cell
    s = g * E / max(f, 1e-6)
    with torch.no_grad(): net.connections[('inh', 'hidden')].pipeline[0].value.fill_(-s)
    print(f'balance: excitation {E:.1f} mV, inhibitory spikes {f:.1f}, weight -{s:.3f} mV per spike (g={g})', flush=True)

W_IH = net.connections[('input', 'hidden')].pipeline[0].value
W_IH0 = W_IH.clone()
def renormalize(per_unit_rate, target, scaling_rate=0.2):
    for j in range(32):
        corr = 1.0 + scaling_rate * (target / max(float(per_unit_rate[j]), 0.01) - 1.0)
        W_IH[:, j] *= max(0.1, min(corr, 3.0))
WN = float(WNP[5:].split('@')[0]) if WNP else None
WN_RATE = float(WNP.split('@')[1]) if WNP and '@' in WNP else 0.2

def restore_ch0(grid, obs):
    for i, (r, c) in enumerate(inp_cell_pos): grid[r, c, 0] = float(obs[i // 2]) * OBS_SCALE[i // 2]
    return grid
def ca_update(grid, out, obs):
    d = torch.zeros(NC, 6)
    for ch in range(6): d[:, ch] = out[:, ch * 2] - out[:, ch * 2 + 1]
    d = (d * DELTA).numpy().reshape(GRID_SIZE, GRID_SIZE, 6)
    mask = (torch.rand(NC) < 0.5).numpy().reshape(GRID_SIZE, GRID_SIZE, 1)
    return restore_ch0(restore_fixed_channels(np.clip(grid + d * mask, -5.0, 5.0)), obs)
def acc(X, y):
    return float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), np.array(X), np.array(y), cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())

rng = np.random.default_rng(seed); np.random.seed(seed); torch.manual_seed(seed)
y, H, OUT = [], {k: [] for k in (0, 1, 3, 10)}, []
for i in range(N):
    obs = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-0.2, 0.2), rng.uniform(-2, 2)], dtype=np.float32); y.append(int(obs[2] > 0))
    grid = set_inputs(restore_fixed_channels(make_grid()), obs); reset_state()
    with torch.no_grad(): W_IH.copy_(W_IH0)
    for step in range(K + 1):
        hid, out, _ = run(grid)
        if step in H: H[step].append(hid[ROWS].flatten().numpy())
        if WN is not None:
            with torch.no_grad(): renormalize(hid.mean(0), WN, WN_RATE)
        if step == 0: OUT.append(float(out.sum(1).mean()))
        if step == K: break
        grid = ca_update(grid, out, obs)
    if i % 25 == 0: print(f'{variant} seed {seed}: {i}/{N}', flush=True)
res = {'variant': variant, 'seed': seed, 'n': N, 'acc': {str(k): acc(H[k], y) for k in H}, 'hid_per_unit': {str(k): float(np.mean(H[k])) for k in H}, 'out_spikes_step0': float(np.mean(OUT))}
json.dump(res, open(f'variant_{variant}_seed{seed}.json', 'w')); print(res, flush=True)
