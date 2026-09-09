"""Shared probing helpers.

The key difference from the notebook's `probe_layer_responses`: that function
returned `hidden_total` summed over all 256 CA cells, so the 248 cells that carry
no state diluted the 8 that do. Here we keep the per-cell tensor (256, n) and let
the caller decide which rows to use.
"""
import numpy as np
import torch
from ca_model import (GRID_SIZE, encoder, get_neighbourhood, inp_cell_pos,
                      make_grid, reset_network, restore_fixed_channels, set_inputs)

N_CELLS = GRID_SIZE * GRID_SIZE
POLE_ANGLE_CELLS = [inp_cell_pos[4], inp_cell_pos[5]]   # set_inputs: obs_idx = i//2


def encode_grid(grid):
    """Grid -> (T, 256, 54) Poisson spike trains, exactly as ca_step does it."""
    flat = np.stack([get_neighbourhood(grid, r, c).flatten()
                     for r in range(GRID_SIZE) for c in range(GRID_SIZE)])
    rates = (torch.clamp(torch.tensor(flat, dtype=torch.float32), -1.0, 1.0) + 1.0) / 2.0 * 100.0
    return torch.stack([encoder(rates[i]) for i in range(N_CELLS)]).permute(1, 0, 2)


def probe_layers(grid, network, n_timesteps=50):
    """Return per-cell spike counts at the input and hidden layers.

    input  -> (256, 54)
    hidden -> (256, 32)
    """
    spikes = encode_grid(grid)
    reset_network(network)
    inp = torch.zeros(N_CELLS, 54)
    hid = torch.zeros(N_CELLS, network.layers['hidden'].n)
    for t in range(n_timesteps):
        inp += spikes[t]
        network.run(inputs={'input': spikes[t]}, time=1, reward=0.0)
        hid += network.layers['hidden'].s.float()
    return inp, hid


def fresh_grid(obs):
    g = make_grid()
    g = restore_fixed_channels(g)
    return set_inputs(g, obs)


def cell_rows(positions):
    return [r * GRID_SIZE + c for r, c in positions]


def sample_state(rng):
    """Notebook's sampler, unchanged. Label = sign of pole angle."""
    obs = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0),
                    rng.uniform(-0.2, 0.2), rng.uniform(-2.0, 2.0)], dtype=np.float32)
    return obs, int(obs[2] > 0)
