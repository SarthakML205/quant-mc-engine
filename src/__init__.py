"""
quant-mc-engine: Production-grade Monte Carlo option pricing engine.
"""
from .base_simulator import SimulatorConfig
from .stochastic_processes import GBMSimulator, MertonJumpDiffusionSimulator
from .payoffs import VanillaCall, VanillaPut, AsianArithmetic
from .analytical import black_scholes, compute_error_metrics, compute_greeks
from .variance_reduction import control_variate_price, mc_greeks

__all__ = [
    "SimulatorConfig",
    "GBMSimulator",
    "MertonJumpDiffusionSimulator",
    "VanillaCall",
    "VanillaPut",
    "AsianArithmetic",
    "black_scholes",
    "compute_error_metrics",
    "compute_greeks",
    "control_variate_price",
    "mc_greeks",
]
