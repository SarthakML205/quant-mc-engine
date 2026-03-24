"""
stochastic_processes.py
-----------------------
Concrete Monte Carlo simulators.

GBMSimulator
    Geometric Brownian Motion via the Euler-Maruyama discretization.
    Supports three sampling methods: standard pseudo-random, antithetic
    variates, and Sobol quasi-random sequences.

MertonJumpDiffusionSimulator
    Extends GBM with a compound Poisson jump component (Merton, 1976).
    All tensor operations are fully vectorized — zero Python for-loops in
    the path-generation logic.
"""

from __future__ import annotations

import math

import torch
from scipy.special import ndtri  # numerically stable normal PPF
from torch import Tensor

from .base_simulator import BaseSimulator, MethodType, SimulatorConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gbm_log_returns(
    config: SimulatorConfig,
    Z: Tensor,
) -> Tensor:
    """
    Convert a standard-normal tensor Z of shape (n_paths, n_steps) into
    GBM log-returns using the Euler-Maruyama scheme.

        log-return = (r - 0.5·σ²)·dt + σ·√dt·Z

    Returns
    -------
    Tensor
        Shape ``(n_paths, n_steps)``.
    """
    dt = config.dt
    drift = (config.r - 0.5 * config.sigma**2) * dt
    diffusion = config.sigma * math.sqrt(dt)
    return drift + diffusion * Z


def _paths_from_log_returns(config: SimulatorConfig, log_returns: Tensor) -> Tensor:
    """
    Convert log-returns ``(n_paths, n_steps)`` → price paths
    ``(n_paths, n_steps + 1)`` with S0 prepended.

    Uses ``torch.cumsum`` for numerical stability and speed.
    """
    cum_log = torch.cumsum(log_returns, dim=1)             # (n_paths, n_steps)
    prices = config.S0 * torch.exp(cum_log)                # (n_paths, n_steps)

    # Prepend the initial price S0 to every path
    s0_col = torch.full(
        (config.n_paths, 1),
        config.S0,
        dtype=prices.dtype,
        device=prices.device,
    )
    return torch.cat([s0_col, prices], dim=1)              # (n_paths, n_steps+1)


def _standard_normals(config: SimulatorConfig) -> Tensor:
    """Draw iid N(0,1) variates of shape (n_paths, n_steps)."""
    return torch.randn(
        config.n_paths,
        config.n_steps,
        device=config.device,
        dtype=torch.float64,
    )


def _antithetic_normals(config: SimulatorConfig) -> Tensor:
    """
    Generate antithetic pairs: draw Z for n_paths//2, then stack [Z, -Z].

    Result shape: (n_paths, n_steps).
    """
    half = config.n_paths // 2
    Z_half = torch.randn(
        half,
        config.n_steps,
        device=config.device,
        dtype=torch.float64,
    )
    return torch.cat([Z_half, -Z_half], dim=0)


def _sobol_normals(config: SimulatorConfig) -> Tensor:
    """
    Generate quasi-random standard normals using a Sobol low-discrepancy
    sequence.

    Steps
    -----
    1. Sample ``n_paths`` points from a Sobol engine with ``n_steps``
       dimensions. Each row is a uniform draw in (0, 1)^{n_steps}.
    2. Apply the normal percent-point function (inverse CDF) via
       ``scipy.special.ndtri`` for numerical stability.
    3. Return a tensor of shape ``(n_paths, n_steps)`` on config.device.

    Note: ``SobolEngine`` scrambling (``scramble=True``) further reduces
    correlation artifacts and is recommended for production use.
    """
    engine = torch.quasirandom.SobolEngine(
        dimension=config.n_steps,
        scramble=True,
        seed=config.seed,
    )
    # Draw uniform samples; use float64 to avoid extreme PPF values near 0/1
    uniform = engine.draw(config.n_paths).to(dtype=torch.float64)

    # Clamp strictly inside (0, 1) to avoid ±inf from the PPF
    eps = torch.finfo(torch.float64).eps
    uniform = uniform.clamp(eps, 1.0 - eps)

    # Map uniform → standard normal via numerically stable inverse CDF
    import numpy as np  # noqa: PLC0415 – local import to keep top-level deps minimal
    normals_np = ndtri(uniform.cpu().numpy())  # shape (n_paths, n_steps)
    return torch.tensor(normals_np, dtype=torch.float64, device=config.device)


# ---------------------------------------------------------------------------
# GBM Simulator
# ---------------------------------------------------------------------------


class GBMSimulator(BaseSimulator):
    """
    Monte Carlo simulator based on Geometric Brownian Motion.

    The asset price follows the log-normal stochastic differential equation:

        dS = r·S·dt + σ·S·dW

    discretised via Euler-Maruyama:

        S_{t+Δt} = S_t · exp[(r − σ²/2)·Δt + σ·√Δt·Z],   Z ~ N(0, 1)

    Path generation is fully vectorized using ``torch.cumsum`` on the
    log-returns tensor — no Python for-loops.

    Parameters
    ----------
    config : SimulatorConfig
        Simulation parameters.
    """

    def simulate_paths(self, method: MethodType = "standard") -> Tensor:
        """
        Simulate GBM price paths.

        Parameters
        ----------
        method : {'standard', 'antithetic', 'sobol'}
            Sampling strategy. ``'antithetic'`` uses antithetic variates;
            ``'sobol'`` uses a quasi-random Sobol sequence.

        Returns
        -------
        Tensor
            Shape ``(n_paths, n_steps + 1)``, dtype float64.
        """
        if method == "standard":
            Z = _standard_normals(self.config)
        elif method == "antithetic":
            Z = _antithetic_normals(self.config)
        elif method == "sobol":
            Z = _sobol_normals(self.config)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'standard', 'antithetic', or 'sobol'."
            )

        log_returns = _gbm_log_returns(self.config, Z)
        return _paths_from_log_returns(self.config, log_returns)


# ---------------------------------------------------------------------------
# Merton Jump-Diffusion Simulator
# ---------------------------------------------------------------------------


class MertonJumpDiffusionSimulator(BaseSimulator):
    """
    Monte Carlo simulator for the Merton (1976) Jump-Diffusion model.

    The asset dynamics are:

        dS/S = (r − λ·κ)·dt + σ·dW + (e^J − 1)·dN

    where:
        - ``dW`` is a standard Brownian increment,
        - ``dN ~ Poisson(λ·dt)`` is the jump counting process,
        - ``J ~ N(μ_j, σ_j²)`` is the log-jump size per event,
        - ``κ = e^{μ_j + σ_j²/2} − 1`` is the mean percentage jump.

    Implementation notes
    --------------------
    * The GBM component uses the same ``torch.cumsum`` trick as
      :class:`GBMSimulator`.
    * Jump counts per path-step are drawn with ``torch.poisson`` (vectorized).
    * Jump sizes are accumulated with ``cumsum`` over Poisson-weighted log-jumps.
    * All operations are on ``config.device``; zero Python for-loops.

    Parameters
    ----------
    config : SimulatorConfig
        Must have ``lambda_j > 0`` to observe jumps. If ``lambda_j == 0``
        the simulator degenerates to pure GBM.
    """

    def simulate_paths(self, method: MethodType = "standard") -> Tensor:
        """
        Simulate Merton Jump-Diffusion price paths.

        Parameters
        ----------
        method : {'standard', 'antithetic', 'sobol'}
            Applies to the *diffusion* (GBM) component only.

        Returns
        -------
        Tensor
            Shape ``(n_paths, n_steps + 1)``, dtype float64.
        """
        cfg = self.config
        dt = cfg.dt

        # ---- Diffusion component ----
        if method == "standard":
            Z_diff = _standard_normals(cfg)
        elif method == "antithetic":
            Z_diff = _antithetic_normals(cfg)
        elif method == "sobol":
            Z_diff = _sobol_normals(cfg)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'standard', 'antithetic', or 'sobol'."
            )

        # Mean percentage jump: κ = exp(μ_j + σ_j²/2) − 1
        kappa = math.exp(cfg.mu_j + 0.5 * cfg.sigma_j**2) - 1.0

        # Drift correction for jumps (risk-neutral measure)
        jump_drift_correction = cfg.lambda_j * kappa * dt

        # GBM log-returns with jump-drift compensation
        drift = (cfg.r - 0.5 * cfg.sigma**2 - jump_drift_correction) * dt
        diffusion = cfg.sigma * math.sqrt(dt) * Z_diff
        gbm_log_returns = drift + diffusion             # (n_paths, n_steps)

        # ---- Jump component ----
        # Number of jumps per (path, step): Poisson(λ·dt)
        lam_dt = torch.full(
            (cfg.n_paths, cfg.n_steps),
            cfg.lambda_j * dt,
            device=cfg.device,
            dtype=torch.float64,
        )
        jump_counts = torch.poisson(lam_dt)             # (n_paths, n_steps)

        # Log-jump sizes per event: N(μ_j, σ_j²)
        Z_jump = torch.randn(
            cfg.n_paths,
            cfg.n_steps,
            device=cfg.device,
            dtype=torch.float64,
        )
        # Aggregate log-jumps: n_jumps × log-jump-size per step
        jump_log_returns = jump_counts * (cfg.mu_j + cfg.sigma_j * Z_jump)

        # ---- Combined log-returns → paths ----
        total_log_returns = gbm_log_returns + jump_log_returns
        return _paths_from_log_returns(cfg, total_log_returns)
