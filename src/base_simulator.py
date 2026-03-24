"""
base_simulator.py
-----------------
Defines the SimulatorConfig dataclass and the abstract BaseSimulator class.

SimulatorConfig holds all market and simulation parameters.
BaseSimulator declares the interface that every concrete simulator must implement
and provides the discounted pricing routine.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from .payoffs import Payoff


MethodType = Literal["standard", "antithetic", "sobol"]


@dataclass
class SimulatorConfig:
    """
    All parameters required to configure a Monte Carlo simulation.

    Attributes
    ----------
    S0 : float
        Initial spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate (annualised).
    sigma : float
        Annualised volatility of the underlying asset.
    n_paths : int
        Number of Monte Carlo sample paths (must be even for antithetic method).
    n_steps : int
        Number of discrete time steps per path.
    seed : int | None
        Random seed for reproducibility. Use ``None`` for non-deterministic runs.
    device : str
        PyTorch device string (e.g. ``'cuda'``, ``'cpu'``).
        Defaults to ``'cuda'`` when a CUDA-capable GPU is available, else ``'cpu'``.
    lambda_j : float
        Merton jump-diffusion: annualised Poisson jump intensity. ``0.0`` disables jumps.
    mu_j : float
        Merton jump-diffusion: mean log-jump size.
    sigma_j : float
        Merton jump-diffusion: volatility of log-jump sizes.
    """

    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2

    n_paths: int = 10_000
    n_steps: int = 252

    seed: int | None = 42

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Merton Jump-Diffusion parameters (set lambda_j > 0 to enable)
    lambda_j: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.n_paths < 2:
            raise ValueError(f"n_paths must be >= 2, got {self.n_paths}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        # Ensure n_paths is even so antithetic method can split cleanly
        if self.n_paths % 2 != 0:
            self.n_paths += 1

    @property
    def dt(self) -> float:
        """Time increment per step."""
        return self.T / self.n_steps


class BaseSimulator(ABC):
    """
    Abstract Monte Carlo simulator.

    Concrete subclasses must implement :meth:`simulate_paths`, which returns
    a price matrix of shape ``(n_paths, n_steps + 1)`` — the first column
    is always the spot price S0.

    The :meth:`price` method handles the risk-neutral discounting and
    computes the standard error and 95 % confidence interval.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        if config.seed is not None:
            torch.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def simulate_paths(self, method: MethodType = "standard") -> Tensor:
        """
        Generate simulated asset-price paths.

        Parameters
        ----------
        method : {'standard', 'antithetic', 'sobol'}
            Sampling strategy for the random variates.

        Returns
        -------
        Tensor
            Shape ``(n_paths, n_steps + 1)`` on ``config.device``.
            Column 0 is S0 for every path; subsequent columns are S_{dt},
            S_{2dt}, …, S_T.
        """

    # ------------------------------------------------------------------
    # Pricing routine (shared by all simulators)
    # ------------------------------------------------------------------

    def price(
        self,
        payoff: Payoff,
        method: MethodType = "standard",
    ) -> dict[str, float]:
        """
        Price an option using Monte Carlo simulation.

        Parameters
        ----------
        payoff : Payoff
            Payoff object (e.g. :class:`~src.payoffs.VanillaCall`).
        method : {'standard', 'antithetic', 'sobol'}
            Variance-reduction strategy to use when generating paths.

        Returns
        -------
        dict
            Keys:
            ``'price'``        – discounted Monte Carlo price estimate,
            ``'std_error'``    – standard error of the mean,
            ``'conf_interval'``– 2-tuple (lower, upper) 95 % CI.
        """
        cfg = self.config
        paths = self.simulate_paths(method=method)          # (n_paths, n_steps+1)
        discounted_payoffs = math.exp(-cfg.r * cfg.T) * payoff.calculate(paths)

        price_est = float(discounted_payoffs.mean().item())
        std_err = float(
            (discounted_payoffs.std() / math.sqrt(cfg.n_paths)).item()
        )
        half_width = 1.96 * std_err

        return {
            "price": price_est,
            "std_error": std_err,
            "conf_interval": (price_est - half_width, price_est + half_width),
        }
