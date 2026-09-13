"""BUG 2: the CA update is unbounded, so the grid pins to the +-5 clip.

Deltas are raw spike-count differences added to a grid clipped at +-5. The
encoder then clips at +-1, so once the grid saturates almost every Poisson rate
is a binary 0 or 100 Hz and the graded state value is lost in the crowd.

    python run_grid_saturation.py
"""
import numpy as np
from ca_model import ca_step, build_network, get_neighbourhood, set_inputs, GRID_SIZE
from probe import fresh_grid

net = build_network(nu_ho=0.0, seed=1)
obs = np.array([0.5, 1.2, -0.2, 0.3], dtype=np.float32)
np.random.seed(1)
grid = fresh_grid(obs)
print('fraction of grid values at |v| >= 1  (i.e. clipped to binary by the encoder)')
for step in range(13):
    if step:
        grid, _, _ = ca_step(grid, net, reward=0.0, learn=False)
        grid = set_inputs(grid, obs)          # even WITH the channel-0 fix
    if step in (0, 1, 2, 4, 8, 12):
        print(f'  step {step:2d}: {(np.abs(grid) >= 1.0).mean():.3f}   '
              f'(min {grid.min():.2f}, max {grid.max():.2f})')

flat = np.stack([get_neighbourhood(grid, r, c).flatten()
                 for r in range(GRID_SIZE) for c in range(GRID_SIZE)])
rates = (np.clip(flat, -1, 1) + 1) / 2 * 100
print(f'\nPoisson rates actually fed to the network at step 12:')
print(f'  at 0 Hz  : {(rates == 0).mean():.3f}')
print(f'  at 100 Hz: {(rates == 100).mean():.3f}')
print(f'  graded   : {((rates > 0) & (rates < 100)).mean():.3f}')
