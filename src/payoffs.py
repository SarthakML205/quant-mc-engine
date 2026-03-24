"""
payoffs.py
----------
Payoff classes for option pricing.

All ``calculate`` methods operate entirely on PyTorch tensors — no Python
loops — and return a 1-D tensor of shape ``(n_paths,)`` representing the
(undiscounted) payoff for each simulated path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Payoff(ABC):
    """Abstract base class for option payoffs."""

    @abstractmethod
    def calculate(self, paths: Tensor) -> Tensor:
        """
        Compute the undiscounted payoff for each simulated path.

        Parameters
        ----------
        paths : Tensor
            Price paths of shape ``(n_paths, n_steps + 1)``.
            Column 0 is S0; the last column is the terminal price S_T.

        Returns
        -------
        Tensor
            Shape ``(n_paths,)`` - non-negative payoff value per path.
        """


class VanillaCall(Payoff):
    """
    European plain-vanilla call option.

    Payoff = max(S_T − K, 0)

    Parameters
    ----------
    K : float
        Strike price.
    """

    def __init__(self, K: float) -> None:
        self.K = K

    def calculate(self, paths: Tensor) -> Tensor:
        terminal = paths[:, -1]  # S_T for each path
        return torch.clamp(terminal - self.K, min=0.0)


class VanillaPut(Payoff):
    """
    European plain-vanilla put option.

    Payoff = max(K − S_T, 0)

    Parameters
    ----------
    K : float
        Strike price.
    """

    def __init__(self, K: float) -> None:
        self.K = K

    def calculate(self, paths: Tensor) -> Tensor:
        terminal = paths[:, -1]
        return torch.clamp(self.K - terminal, min=0.0)


class AsianArithmetic(Payoff):
    """
    Asian option with arithmetic-average underlying.

    Payoff = max(A − K, 0)

    where A = (1/n_steps) * Σ S_{i·Δt}  for i = 1 … n_steps
    (excluding the initial S0 column at index 0).

    Parameters
    ----------
    K : float
        Strike price.
    """

    def __init__(self, K: float) -> None:
        self.K = K

    def calculate(self, paths: Tensor) -> Tensor:
        # Average over all steps except the initial S0 (column 0)
        arithmetic_mean = paths[:, 1:].mean(dim=1)
        return torch.clamp(arithmetic_mean - self.K, min=0.0)
