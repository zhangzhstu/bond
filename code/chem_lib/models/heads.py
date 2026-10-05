"""
Feature-map heads for the ablation study.

Every head maps R^{in} -> R^{out}, so any of them can replace the others without changes to the
encoder, the GP or the meta-learning loop. They differ in boundedness and branch aggregation:

    bounded,   branched (M>1):  'dnm' (BOND, default; the module used for the main results)
    bounded,   unbranched:      'tanh_linear', 'l2norm'
    unbounded, branched (M>1):  'branch_linear'
    unbounded, unbranched:      'mlp'

'identity' applies no feature map.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DNM_models2 import ADNM_ifm3


class BranchLinear(nn.Module):
    """Branched, unbounded head.

    Same fan-out (out_size * M) and branch summation as ADNM_ifm3, without the sigmoid
    activation and the tanh gate on the branch weights.
    """

    def __init__(self, input_size, out_size, M):
        super().__init__()
        self.M, self.input_size, self.out_size = M, input_size, out_size
        self.mlp = nn.Linear(input_size, out_size * M)
        self.m = nn.Parameter(torch.empty(out_size, M))
        nn.init.uniform_(self.m, -1.0, 1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(*x.shape[:-1], self.out_size, self.M)
        return torch.sum(x * self.m, -1)


class TanhLinear(nn.Module):
    """Bounded, unbranched head: tanh(Linear(x))."""

    def __init__(self, input_size, out_size):
        super().__init__()
        self.fc = nn.Linear(input_size, out_size)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class L2NormLinear(nn.Module):
    """Bounded, unbranched head: Linear(x) normalised to unit L2 norm.

    TanhLinear bounds each coordinate; this head bounds the whole vector and discards its
    magnitude.
    """

    def __init__(self, input_size, out_size, eps=1e-8):
        super().__init__()
        self.fc = nn.Linear(input_size, out_size)
        self.eps = eps

    def forward(self, x):
        z = self.fc(x)
        return z / (z.norm(dim=-1, keepdim=True) + self.eps)


class MLPHead(nn.Module):
    """Unbounded, unbranched head: a two-layer ReLU MLP.

    `hidden` defaults to out_size. Set it with matched_hidden() to keep the parameter count
    comparable to the DNM head, so the comparison does not mix structure with capacity.
    """

    def __init__(self, input_size, out_size, hidden=None):
        super().__init__()
        hidden = hidden if hidden is not None else out_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, out_size)
        )

    def forward(self, x):
        return self.net(x)


def build_head(head_type, input_size, out_size, M=30, mlp_hidden=None):
    """Return the head named by `head_type`; 'dnm' is the module used for the main results."""
    if head_type == "dnm":
        return ADNM_ifm3(input_size, out_size, M=M)
    if head_type == "branch_linear":
        return BranchLinear(input_size, out_size, M=M)
    if head_type == "tanh_linear":
        return TanhLinear(input_size, out_size)
    if head_type == "l2norm":
        return L2NormLinear(input_size, out_size)
    if head_type == "mlp":
        return MLPHead(input_size, out_size, hidden=mlp_hidden)
    if head_type == "identity":
        # No feature map: the GP sees the encoder output (and the fingerprint, if concatenated)
        # directly. With --use_fingerprints 0 this is the ADKF-IFT configuration.
        return nn.Identity()
    raise ValueError(f"unknown head_type {head_type!r}")


def matched_hidden(input_size, out_size, M):
    """Hidden width that gives an MLPHead roughly the parameter count of ADNM_ifm3(., ., M).

    ADNM_ifm3:  input_size*out_size*M + out_size*M + out_size*M  (weights, bias, gates)
    MLPHead  :  input_size*h + h + h*out_size + out_size
    """
    target = input_size * out_size * M + 2 * out_size * M
    h = (target - out_size) / (input_size + 1 + out_size)
    return max(1, int(round(h)))
