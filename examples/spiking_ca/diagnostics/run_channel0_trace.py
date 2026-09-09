"""BUG 1: the observation is written into channel 0 once and never re-injected.

`set_inputs` writes state into channel 0.
`restore_fixed_channels` restores channels 1, 2, 4, 5 -- never 0.
`ca_step` adds unbounded deltas to all six channels.

So the state is overwritten by the first CA step, while the probe (and the
training loop's Q-value read) happens several steps later.

    python run_channel0_trace.py
"""
import numpy as np
import torch
from ca_model import ca_step, build_network
from probe import POLE_ANGLE_CELLS, fresh_grid

net = build_network(nu_ho=0.0, seed=1)
for name, obs in [('hard-LEFT',  np.array([0, 0, -0.2, 0], dtype=np.float32)),
                  ('hard-RIGHT', np.array([0, 0,  0.2, 0], dtype=np.float32))]:
    np.random.seed(1); torch.manual_seed(1)   # exact values are seed-dependent;
    grid = fresh_grid(obs)                    # the collapse is not
    print(f'\n{name}: channel-0 value at the two pole-angle input cells')
    for step in range(13):
        if step:
            grid, _, _ = ca_step(grid, net, reward=0.0, learn=False)
        note = ('  <- set_inputs wrote the observation' if step == 0 else
                '  <- probe / Q-value read here'        if step == 12 else '')
        print(f'  after {step:2d} CA steps: '
              f'{np.array([grid[r, c, 0] for r, c in POLE_ANGLE_CELLS])}{note}')
