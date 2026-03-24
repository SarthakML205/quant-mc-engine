"""
analytical.py
-------------
Black-Scholes-Merton closed-form solution for European options and
helper functions for comparing Monte Carlo estimates against this baseline.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from .base_simulator import SimulatorConfig


def black_scholes(config: SimulatorConfig, option_type: str = "call") -> float:
    """
    Calculate the Black-Scholes-Merton price for a European option.

    Uses the standard risk-neutral formula:

        d1 = [ln(S0/K) + (r + σ²/2)T] / (σ√T)
        d2 = d1 − σ√T

        Call = S0·N(d1) − K·e^{−rT}·N(d2)
        Put  = K·e^{−rT}·N(−d2) − S0·N(−d1)

    Parameters
    ----------
    config : SimulatorConfig
        Simulation configuration carrying S0, K, T, r, sigma.
    option_type : {'call', 'put'}
        Type of option to price.

    Returns
    -------
    float
        The analytical Black-Scholes price.

    Raises
    ------
    ValueError
        If *option_type* is not ``'call'`` or ``'put'``.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    S0, K, T, r, sigma = config.S0, config.K, config.T, config.r, config.sigma

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = math.exp(-r * T)

    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * discount * norm.cdf(d2))

    # put
    return float(K * discount * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def compute_error_metrics(mc_price: float, bs_price: float) -> dict[str, float]:
    """
    Compute error metrics between a Monte Carlo estimate and the BS price.

    Parameters
    ----------
    mc_price : float
        Monte Carlo option price estimate.
    bs_price : float
        Black-Scholes analytical price (ground truth).

    Returns
    -------
    dict
        ``'mse'``  – Mean Squared Error  ``(mc − bs)²``
        ``'ape'``  – Absolute Percentage Error  ``|mc − bs| / |bs| × 100``
    """
    if bs_price == 0.0:
        raise ValueError("bs_price must be non-zero to compute APE.")

    error = mc_price - bs_price
    mse = error**2
    ape = abs(error) / abs(bs_price) * 100.0

    return {"mse": mse, "ape": ape}


def compute_greeks(
    config: SimulatorConfig,
    option_type: str = "call",
    bump_S: float = 0.01,
    bump_sigma: float = 0.001,
) -> dict[str, float]:
    """
    Compute Black-Scholes Delta and Vega via central finite differences.

    Parameters
    ----------
    config : SimulatorConfig
        Market parameters.
    option_type : {'call', 'put'}
        Option type to differentiate.
    bump_S : float
        Absolute spot-price bump for Delta.
    bump_sigma : float
        Absolute volatility bump for Vega.

    Returns
    -------
    dict
        ``'delta'`` – ∂C/∂S₀,
        ``'vega'``  – ∂C/∂σ scaled to per 1 % vol move.
    """
    import dataclasses

    option_type = option_type.lower()

    cfg_s_up = dataclasses.replace(config, S0=config.S0 + bump_S)
    cfg_s_dn = dataclasses.replace(config, S0=config.S0 - bump_S)
    delta = (
        black_scholes(cfg_s_up, option_type) - black_scholes(cfg_s_dn, option_type)
    ) / (2.0 * bump_S)

    cfg_v_up = dataclasses.replace(config, sigma=config.sigma + bump_sigma)
    cfg_v_dn = dataclasses.replace(config, sigma=config.sigma - bump_sigma)
    vega = (
        black_scholes(cfg_v_up, option_type) - black_scholes(cfg_v_dn, option_type)
    ) / (2.0 * bump_sigma) * 0.01

    return {"delta": delta, "vega": vega}
