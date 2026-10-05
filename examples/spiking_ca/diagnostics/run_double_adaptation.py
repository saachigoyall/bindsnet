"""EXPERIMENT 2: two-timescale spike-frequency adaptation.

Spike-frequency adaptation as a threshold that rises after each spike
and decays back down is the mechanism used by Bellec et al. (2018),
"Long short-term memory and learning-to-learn in networks of spiking
neurons" (NeurIPS), who use a single exponential decay per neuron with
different time constants across neurons. The two-timescale version
implemented here, a fast and a slow threshold component, follows the
multi-timescale adaptive threshold (MAT) model of Kobayashi et al.
(2009), "Made-to-order spiking neuron model equipped with a
multi-timescale adaptive threshold" (Frontiers in Computational
Neuroscience), which used fast (10 ms) and slow (200 ms) components.
This is a loose analogue, not a reproduction: those papers used
different neuron models and tasks, and the time constants here are our
own. BindsNET's built-in AdaptiveLIFNodes only supports a single
exponential, so this implements a two-component adaptive neuron
directly (subclassing Nodes, following the same pattern as BindsNET's
own AdaptiveLIFNodes source).

Input->hidden weights stay at the original baseline throughout.

    python run_double_adaptation.py            # saturation stats (fast)
    python run_double_adaptation.py --decode   # + decoder (slow)
"""
import json, sys
from typing import Iterable, Optional, Union
import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, Nodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDPET as MCC_MSTDPET
from ca_model import DEVICE
from probe import POLE_ANGLE_CELLS, cell_rows, fresh_grid, probe_layers, sample_state

DECODE = '--decode' in sys.argv
rows = cell_rows(POLE_ANGLE_CELLS)


class DoubleAdaptiveLIFNodes(Nodes):
    """LIF neurons with a two-component adaptive threshold: a fast
    component (large per-spike jump, decays quickly) and a slow
    component (small per-spike jump, decays slowly), in the spirit of
    the multi-timescale adaptive threshold model (Kobayashi et al.,
    2009) rather than BindsNET's built-in single-exponential
    AdaptiveLIFNodes. Implementation follows the same pattern as
    BindsNET's own AdaptiveLIFNodes."""

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        thresh: Union[float, torch.Tensor] = -52.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus_fast: Union[float, torch.Tensor] = 1.0,
        tc_theta_fast: Union[float, torch.Tensor] = 30.0,
        theta_plus_slow: Union[float, torch.Tensor] = 0.3,
        tc_theta_slow: Union[float, torch.Tensor] = 300.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        super().__init__(n=n, shape=shape, traces=traces,
                          traces_additive=traces_additive, tc_trace=tc_trace,
                          trace_scale=trace_scale, sum_input=sum_input)

        self.register_buffer("rest", torch.tensor(rest))
        self.register_buffer("reset", torch.tensor(reset))
        self.register_buffer("thresh", torch.tensor(thresh))
        self.register_buffer("refrac", torch.tensor(refrac))
        self.register_buffer("tc_decay", torch.tensor(tc_decay))
        self.register_buffer("decay", torch.empty_like(self.tc_decay, dtype=torch.float32))

        self.register_buffer("theta_plus_fast", torch.tensor(theta_plus_fast))
        self.register_buffer("tc_theta_fast", torch.tensor(tc_theta_fast))
        self.register_buffer("theta_decay_fast", torch.empty_like(self.tc_theta_fast))
        self.register_buffer("theta_plus_slow", torch.tensor(theta_plus_slow))
        self.register_buffer("tc_theta_slow", torch.tensor(tc_theta_slow))
        self.register_buffer("theta_decay_slow", torch.empty_like(self.tc_theta_slow))

        self.register_buffer("v", torch.FloatTensor())
        self.register_buffer("theta_fast", torch.zeros(*self.shape))
        self.register_buffer("theta_slow", torch.zeros(*self.shape))
        self.register_buffer("refrac_count", torch.FloatTensor())
        self.lbound = lbound

    def forward(self, x: torch.Tensor) -> None:
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta_fast *= self.theta_decay_fast
            self.theta_slow *= self.theta_decay_slow

        self.v += (self.refrac_count <= 0).float() * x
        self.refrac_count -= self.dt

        self.s = self.v >= self.thresh + self.theta_fast + self.theta_slow

        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta_fast += self.theta_plus_fast * self.s.float().sum(0)
            self.theta_slow += self.theta_plus_slow * self.s.float().sum(0)

        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.refrac_count.zero_()

    def compute_decays(self, dt) -> None:
        super().compute_decays(dt=dt)
        self.decay = torch.exp(-self.dt / self.tc_decay)
        self.theta_decay_fast = torch.exp(-self.dt / self.tc_theta_fast)
        self.theta_decay_slow = torch.exp(-self.dt / self.tc_theta_slow)

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


# (theta_plus_fast, tc_fast, theta_plus_slow, tc_slow). First row is
# single-exponential-equivalent (slow component effectively off) for
# comparison. The fast/slow ratio (about 10x) is similar in spirit to
# the fast (10 ms) and slow (200 ms) components in the MAT model, but
# the values here are our own choice.
SETTINGS = [
    (0.0, 30.0, 0.0, 300.0),      # baseline: no adaptation
    (1.0, 30.0, 0.0, 300.0),      # fast component only
    (0.0, 30.0, 0.3, 300.0),      # slow component only
    (1.0, 30.0, 0.3, 300.0),      # BOTH components, two-timescale adaptation
]


def build_network_double_adaptive(seed, tpf, tcf, tps, tcs, nu_ho=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = Network(dt=1.0)
    input_layer = Input(n=54, shape=(54,))
    hidden_layer = DoubleAdaptiveLIFNodes(
        n=32, thresh=-52.0, reset=-65.0, rest=-65.0, refrac=5, traces=True,
        theta_plus_fast=tpf, tc_theta_fast=tcf,
        theta_plus_slow=tps, tc_theta_slow=tcs)
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


def saturation_stats(net):
    resp = {}
    for tag, obs in [('L', np.array([0, 0, -0.2, 0], dtype=np.float32)),
                      ('R', np.array([0, 0, 0.2, 0], dtype=np.float32))]:
        np.random.seed(7); torch.manual_seed(7)
        _, hid = probe_layers(fresh_grid(obs), net)
        resp[tag] = hid[rows]
    return resp


print(f"{'theta_fast':<12}{'tc_fast':<10}{'theta_slow':<12}{'tc_slow':<10}"
      f"{'hidden spikes':<16}{'std across units':<20}{'|mean(L)-mean(R)|':<20}")
results = {}
for tpf, tcf, tps, tcs in SETTINGS:
    net = build_network_double_adaptive(1, tpf, tcf, tps, tcs)
    resp = saturation_stats(net)
    key = f"tpf{tpf}_tcf{tcf}_tps{tps}_tcs{tcs}"
    print(f"{tpf:<12}{tcf:<10}{tps:<12}{tcs:<10}{resp['L'].mean():<16.2f}"
          f"{resp['L'].std():<20.3f}"
          f"{abs(resp['L'].mean() - resp['R'].mean()):<20.3f}")

    if DECODE:
        per_seed = {}
        for seed in [1, 2, 3]:
            net = build_network_double_adaptive(seed, tpf, tcf, tps, tcs)
            rng = np.random.default_rng(seed)
            X, y = [], []
            for _ in range(150):
                obs, label = sample_state(rng)
                _, hid = probe_layers(fresh_grid(obs), net)
                X.append(hid[rows].flatten().numpy())
                y.append(label)
            per_seed[seed] = {'Xhid': np.array(X).tolist(), 'y': y}
        results[key] = per_seed

if DECODE:
    json.dump(results, open('double_adaptation_sweep.json', 'w'))
    print('\nwrote double_adaptation_sweep.json -> '
          'python analyze.py double_adaptation_sweep.json')
