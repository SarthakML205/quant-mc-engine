"""
variance_reduction.py
---------------------
Advanced variance-reduction wrappers and Monte Carlo Greek computation.

Functions
---------
control_variate_price   – CV-adjusted pricing using a correlated control option.
mc_greeks               – Monte Carlo finite-difference Delta and Vega using
                          Common Random Numbers (CRN).
"""

from __future__ import annotations

import dataclasses
import math

from .base_simulator import BaseSimulator, MethodType, SimulatorConfig
from .payoffs import Payoff


def control_variate_price(
    simulator: BaseSimulator,
    target_payoff: Payoff,
    control_payoff: Payoff,
    control_analytical_price: float,
    method: MethodType = "standard",
) -> dict[str, float]:
    """
    Price a target option using the Control Variate (CV) technique.

    The CV method exploits the correlation between the target payoff (e.g. an
    Asian arithmetic call) and a control payoff whose true price is known
    analytically (e.g. a European call via Black-Scholes).

    The minimum-variance adjusted estimator is:

        V_CV = V_target - beta * (V_control - E[V_control])

    where  beta = Cov(V_target, V_control) / Var(V_control).

    A single shared path-draw is used for both payoffs so the correlation
    structure that makes the technique work is preserved.

    Parameters
    ----------
    simulator : BaseSimulator
        A configured simulator instance (e.g. ``GBMSimulator``).
    target_payoff : Payoff
        The option being priced (e.g. ``AsianArithmetic``).
    control_payoff : Payoff
        The option used as control (e.g. ``VanillaCall`` with same K, T).
    control_analytical_price : float
        Known analytical price of the control option (e.g. from ``black_scholes``).
    method : {'standard', 'antithetic', 'sobol'}
        Sampling strategy applied to path generation.

    Returns
    -------
    dict
        ``'price'``                  – CV-adjusted price estimate,
        ``'std_error'``              – standard error of the adjusted estimator,
        ``'conf_interval'``          – 95 % CI (lower, upper),
        ``'beta'``                   – optimal CV coefficient beta,
        ``'variance_reduction_pct'`` – % variance removed vs raw Monte Carlo.
    """
    cfg = simulator.config
    discount = math.exp(-cfg.r * cfg.T)

    # One shared path draw — correlation between target and control preserved
    paths = simulator.simulate_paths(method=method)          # (n_paths, n_steps+1)

    target_raw  = discount * target_payoff.calculate(paths)  # (n_paths,)
    control_raw = discount * control_payoff.calculate(paths) # (n_paths,)

    # Optimal beta via minimum-variance closed form
    t_mean = target_raw.mean()
    c_mean = control_raw.mean()
    cov_tc = ((target_raw - t_mean) * (control_raw - c_mean)).mean()
    var_c  = ((control_raw - c_mean) ** 2).mean()

    beta = float((cov_tc / var_c).item()) if float(var_c.item()) > 1e-12 else 0.0

    # CV-adjusted payoff sample
    adjusted = target_raw - beta * (control_raw - control_analytical_price)

    price_est  = float(adjusted.mean().item())
    std_err    = float(
        (adjusted.std(unbiased=True) / math.sqrt(cfg.n_paths)).item()
    )
    half_width = 1.96 * std_err

    var_raw = float(target_raw.var(unbiased=True).item())
    var_adj = float(adjusted.var(unbiased=True).item())
    vr_pct  = (
        max(0.0, (1.0 - var_adj / var_raw) * 100.0) if var_raw > 1e-12 else 0.0
    )

    return {
        "price": price_est,
        "std_error": std_err,
        "conf_interval": (price_est - half_width, price_est + half_width),
        "beta": beta,
        "variance_reduction_pct": vr_pct,
    }


def mc_greeks(
    simulator: BaseSimulator,
    payoff: Payoff,
    method: MethodType = "standard",
    bump_S: float = 0.5,
    bump_sigma: float = 0.002,
) -> dict[str, float]:
    """
    Estimate option Greeks via Monte Carlo finite differences.

    Uses Common Random Numbers (CRN) — a fixed seed for all bump scenarios —
    to cancel re-sampling noise so that only the effect of the parameter
    change remains in the estimate.

    Parameters
    ----------
    simulator : BaseSimulator
        Configured simulator instance.
    payoff : Payoff
        Option payoff object.
    method : {'standard', 'antithetic', 'sobol'}
        Sampling method (applied consistently to all bumped configs).
    bump_S : float
        Absolute spot-price bump for Delta computation.
    bump_sigma : float
        Absolute volatility bump for Vega computation.

    Returns
    -------
    dict
        ``'delta'`` – central-difference Delta estimate (dC/dS0),
        ``'vega'``  – central-difference Vega per 1 % vol move.
    """
    cfg = simulator.config
    crn_seed = 0  # common random numbers — same seed for all bumped sims

    def _price_at(new_cfg: SimulatorConfig) -> float:
        return type(simulator)(new_cfg).price(payoff, method=method)["price"]

    # Delta: dC/dS0
    cfg_s_up = dataclasses.replace(cfg, S0=cfg.S0 + bump_S,       seed=crn_seed)
    cfg_s_dn = dataclasses.replace(cfg, S0=cfg.S0 - bump_S,       seed=crn_seed)
    delta = (_price_at(cfg_s_up) - _price_at(cfg_s_dn)) / (2.0 * bump_S)

    # Vega: dC/dsigma × 0.01 (per 1 % vol move)
    cfg_v_up = dataclasses.replace(cfg, sigma=cfg.sigma + bump_sigma, seed=crn_seed)
    cfg_v_dn = dataclasses.replace(cfg, sigma=cfg.sigma - bump_sigma, seed=crn_seed)
    vega = (_price_at(cfg_v_up) - _price_at(cfg_v_dn)) / (2.0 * bump_sigma) * 0.01

    return {"delta": delta, "vega": vega}
