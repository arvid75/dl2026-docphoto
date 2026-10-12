"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3, prior: float = 0.5):
        super().__init__()

        # One (Linear -> ReLU -> Dropout) block per hidden width. Dropout after
        # every hidden layer (not after the output): the net has ~1.07M params
        # for ~1500 training photos, so it overfits easily and needs the
        # regularisation on every hidden representation.
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = width
        self.backbone = nn.Sequential(*layers)

        # Final linear mapping to a single logit; a sigmoid squashes it to [0, 1].
        self.head = nn.Linear(prev, 1)

        # The output must start at the training mean (`prior`). With a sigmoid
        # output, set the head's weights to 0 and its bias to logit(prior), so
        # an untrained net predicts exactly `prior` for every input. That gives
        # training a sensible starting point (the constant predictor) instead of
        # 0.5, so early epochs refine the mean rather than discovering it.
        prior = min(max(float(prior), 1e-6), 1 - 1e-6)
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            self.head.bias.fill_(float(torch.logit(torch.tensor(prior))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        out = torch.sigmoid(self.head(h))   # one number in [0, 1] per row
        return out.squeeze(-1)              # shape (batch,), not (batch, 1)
