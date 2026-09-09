"""THE CONTROL: decode pole direction from the input layer and from the hidden
layer, on the same states, in the same run.

Answers "where does the state information disappear?". It does NOT say anything
about learning, the reward, or whether the architecture can solve CartPole.

    python run_layer_decoder.py [n_samples] [seeds...]
"""
import json, sys
import numpy as np
from ca_model import build_network
from probe import POLE_ANGLE_CELLS, cell_rows, fresh_grid, probe_layers, sample_state

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEEDS = [int(s) for s in sys.argv[2:]] or [1, 2, 3]
rows = cell_rows(POLE_ANGLE_CELLS)

out = {}
for seed in SEEDS:
    net = build_network(nu_ho=0.0, seed=seed)
    rng = np.random.default_rng(seed)
    Xin, Xhid, y = [], [], []
    for _ in range(N):
        obs, label = sample_state(rng)
        inp, hid = probe_layers(fresh_grid(obs), net)
        Xin.append(inp[rows].flatten().numpy())
        Xhid.append(hid[rows].flatten().numpy())
        y.append(label)
    out[seed] = {'Xin': np.array(Xin).tolist(), 'Xhid': np.array(Xhid).tolist(), 'y': y}
    print(f'seed {seed}: {N} states done', flush=True)

json.dump(out, open('layer_decoder.json', 'w'))
print('wrote layer_decoder.json  ->  now run: python analyze.py layer_decoder.json')
