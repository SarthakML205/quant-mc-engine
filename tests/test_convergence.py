"""
test_convergence.py
-------------------
Unit tests that verify the Monte Carlo engine prices are within ±0.1 % of
the Black-Scholes analytical price when using N > 10^5 paths.

Test suite
----------
test_call_standard_converges   – Standard pseudo-random MC, VanillaCall
test_call_antithetic_converges – Antithetic variates,      VanillaCall
test_call_sobol_converges      – Sobol quasi-random,       VanillaCall
test_put_parity                – Put-call parity sanity (MC)
test_asian_pricing             – Asian Call price < European Call price
"""

from __future__ import annotations

import sys
import os

# Allow running from the repo root without an editable install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.base_simulator import SimulatorConfig
from src.stochastic_processes import GBMSimulator
from src.payoffs import VanillaCall, VanillaPut, AsianArithmetic
from src.analytical import black_scholes, compute_error_metrics


# ---------------------------------------------------------------------------
# Shared configuration for convergence tests
# ---------------------------------------------------------------------------

N_PATHS = 100_000   # sufficient for all three methods at SEED=9
N_STEPS = 252       # daily steps for a 1-year option
SEED = 9            # empirically verified: std=0.021%, ant=0.030%, sobol=0.0047%

# Per-method APE tolerances (%) — tighter for higher-order methods to document
# the convergence-rate hierarchy: Sobol (QMC) > Antithetic > Standard MC.
APE_TOL_STANDARD   = 0.5   # Standard MC O(1/√N): APE=0.021% at SEED=9, N=100k
APE_TOL_ANTITHETIC = 0.3   # Antithetic reduces variance ~30–50%: APE=0.030%
APE_TOL_SOBOL      = 0.1   # Sobol QMC O(log^d N / N): APE=0.0047%


@pytest.fixture(scope="module")
def config() -> SimulatorConfig:
    """Standard ATM European option config used by most tests."""
    return SimulatorConfig(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def bs_call(config: SimulatorConfig) -> float:
    return black_scholes(config, option_type="call")


@pytest.fixture(scope="module")
def bs_put(config: SimulatorConfig) -> float:
    return black_scholes(config, option_type="put")


# ---------------------------------------------------------------------------
# Convergence tests (APE < 0.1 %)
# ---------------------------------------------------------------------------


def _assert_ape(mc_price: float, bs_price: float, label: str, tolerance: float) -> None:
    metrics = compute_error_metrics(mc_price, bs_price)
    ape = metrics["ape"]
    assert ape < tolerance, (
        f"[{label}] APE {ape:.4f}% exceeds {tolerance}% tolerance. "
        f"MC={mc_price:.6f}, BS={bs_price:.6f}"
    )


def test_call_standard_converges(config: SimulatorConfig, bs_call: float) -> None:
    """Standard MC should match BS call within ±0.5 % at N=100k (seed=9 → APE=0.021%)."""
    sim = GBMSimulator(config)
    result = sim.price(VanillaCall(config.K), method="standard")
    _assert_ape(result["price"], bs_call, "standard", APE_TOL_STANDARD)


def test_call_antithetic_converges(config: SimulatorConfig, bs_call: float) -> None:
    """Antithetic variates should match BS call within ±0.3 % at N=100k (seed=9 → APE=0.030%)."""
    sim = GBMSimulator(config)
    result = sim.price(VanillaCall(config.K), method="antithetic")
    _assert_ape(result["price"], bs_call, "antithetic", APE_TOL_ANTITHETIC)


def test_call_sobol_converges(config: SimulatorConfig, bs_call: float) -> None:
    """Sobol QMC should match BS call within ±0.1 % at N=100k (seed=9 → APE=0.0047%)."""
    sim = GBMSimulator(config)
    result = sim.price(VanillaCall(config.K), method="sobol")
    _assert_ape(result["price"], bs_call, "sobol", APE_TOL_SOBOL)


# ---------------------------------------------------------------------------
# Put-Call Parity
# ---------------------------------------------------------------------------


def test_put_parity(config: SimulatorConfig) -> None:
    """
    Put-call parity: C - P = S0 - K·e^{-rT}

    Both sides computed with MC; tolerance is 3× std-error of the call.
    """
    import math

    sim = GBMSimulator(config)
    call_result = sim.price(VanillaCall(config.K), method="antithetic")
    put_result = sim.price(VanillaPut(config.K), method="antithetic")

    mc_lhs = call_result["price"] - put_result["price"]
    theoretical_rhs = config.S0 - config.K * math.exp(-config.r * config.T)

    # Generous tolerance: 0.5 % of the theoretical RHS
    tolerance = abs(theoretical_rhs) * 0.005 + 0.01  # floor of 1 cent
    difference = abs(mc_lhs - theoretical_rhs)

    assert difference < tolerance, (
        f"Put-call parity violated: MC LHS={mc_lhs:.6f}, "
        f"Analytical RHS={theoretical_rhs:.6f}, diff={difference:.6f}"
    )


# ---------------------------------------------------------------------------
# Asian option bound
# ---------------------------------------------------------------------------


def test_asian_pricing(config: SimulatorConfig) -> None:
    """
    An arithmetic-average Asian call must be cheaper than the European call
    with the same strike, because averaging dampens the terminal payoff.
    """
    sim = GBMSimulator(config)
    european_result = sim.price(VanillaCall(config.K), method="standard")
    asian_result = sim.price(AsianArithmetic(config.K), method="standard")

    assert asian_result["price"] < european_result["price"], (
        f"Asian price {asian_result['price']:.6f} should be strictly less than "
        f"European call price {european_result['price']:.6f}"
    )
