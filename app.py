"""
app.py
------
Quant Lab — Interactive Streamlit dashboard for the Monte Carlo
Option Pricing Engine.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.base_simulator import SimulatorConfig
from src.stochastic_processes import GBMSimulator
from src.payoffs import VanillaCall, AsianArithmetic
from src.analytical import black_scholes, compute_error_metrics, compute_greeks
from src.variance_reduction import control_variate_price

# ── Constants ─────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "Standard MC":      "#1f77b4",
    "Antithetic":       "#ff7f0e",
    "Control Variates": "#2ca02c",
    "Sobol QMC":        "#9467bd",
}
# Sampling-method key used by GBMSimulator for everything except Control Variates
SAMPLING_KEY = {
    "Standard MC":  "standard",
    "Antithetic":   "antithetic",
    "Sobol QMC":    "sobol",
}
N_SWEEP        = (500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000)
N_STEPS_SWEEP  = 50    # fast sweep — convergence behaviour is vs N, not steps
N_STEPS_MAIN   = 252   # daily steps for full pricing
N_DISPLAY      = 50    # paths shown in trajectory chart

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant Lab · MC Engine",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached computation helpers ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_display_paths(S0, K, T, sigma, r):
    cfg = SimulatorConfig(S0=S0, K=K, T=T, r=r, sigma=sigma,
                          n_paths=N_DISPLAY, n_steps=N_STEPS_MAIN, seed=7)
    return GBMSimulator(cfg).simulate_paths("standard").cpu().numpy()


@st.cache_data(show_spinner=False)
def _get_asian_reference(S0, K, T, sigma, r) -> float:
    """High-N standard MC Asian call price used as reference when comparing methods."""
    cfg = SimulatorConfig(S0=S0, K=K, T=T, r=r, sigma=sigma,
                          n_paths=200_000, n_steps=N_STEPS_MAIN, seed=0)
    return GBMSimulator(cfg).price(AsianArithmetic(K), method="standard")["price"]


@st.cache_data(show_spinner=False)
def _run_pricing(S0, K, T, sigma, r, n_paths, option_type, methods_tuple):
    """
    Price the selected option with every enabled method.
    Returns a dict {method_name: result_dict}.
    """
    cfg = SimulatorConfig(S0=S0, K=K, T=T, r=r, sigma=sigma,
                          n_paths=n_paths, n_steps=N_STEPS_MAIN, seed=None)
    sim     = GBMSimulator(cfg)
    pay_eur = VanillaCall(K)
    pay_tgt = VanillaCall(K) if option_type == "European Call" else AsianArithmetic(K)
    bs_ctrl = black_scholes(cfg, "call")

    out = {}
    for m in methods_tuple:
        if m == "Control Variates":
            out[m] = control_variate_price(sim, pay_tgt, pay_eur, bs_ctrl,
                                           method="standard")
        else:
            out[m] = sim.price(pay_tgt, method=SAMPLING_KEY[m])
    return out


@st.cache_data(show_spinner=False)
def _run_convergence_sweep(S0, K, T, sigma, r, option_type, methods_tuple):
    """
    N-sweep for all enabled methods.
    Returns {method_name: [price | None for each N in N_SWEEP]}.
    """
    results = {m: [] for m in methods_tuple}

    for n in N_SWEEP:
        cfg = SimulatorConfig(S0=S0, K=K, T=T, r=r, sigma=sigma,
                              n_paths=n, n_steps=N_STEPS_SWEEP, seed=42)
        sim     = GBMSimulator(cfg)
        pay_eur = VanillaCall(K)
        pay_tgt = VanillaCall(K) if option_type == "European Call" else AsianArithmetic(K)
        bs_ctrl = black_scholes(cfg, "call")

        for m in methods_tuple:
            if m == "Control Variates":
                if option_type == "Asian Arithmetic Call":
                    res = control_variate_price(sim, pay_tgt, pay_eur, bs_ctrl)
                    results[m].append(res["price"])
                else:
                    results[m].append(None)
            else:
                res = sim.price(pay_tgt, method=SAMPLING_KEY[m])
                results[m].append(res["price"])

    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚗️ Quant Lab")
    st.caption("Monte Carlo Option Pricing Engine")
    st.divider()

    st.subheader("Market Parameters")
    S0    = st.slider("Spot Price  S₀",    50.0,  200.0, 100.0, 1.0)
    K     = st.slider("Strike  K",         50.0,  200.0, 100.0, 1.0)
    T     = st.slider("Maturity  T (yr)",   0.1,    3.0,   1.0, 0.1)
    sigma = st.slider("Volatility  σ",      0.05,   0.80,  0.20, 0.01, format="%.2f")
    r     = st.slider("Risk-free Rate  r",  0.00,   0.15,  0.05, 0.005, format="%.3f")
    n_paths = st.slider("Paths  N",  1_000, 50_000, 10_000, 1_000,
                        help="Number of Monte Carlo simulation paths.")

    st.divider()
    st.subheader("Option Type")
    option_type = st.selectbox(
        "", ["European Call", "Asian Arithmetic Call"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Sampling Techniques")
    use_standard   = st.checkbox("Standard MC",         value=True)
    use_antithetic = st.checkbox("Antithetic Variates", value=True)
    cv_disabled    = (option_type == "European Call")
    use_cv         = st.checkbox(
        "Control Variates",
        value=(not cv_disabled),
        disabled=cv_disabled,
        help="Requires Asian option (uses European Call as the control).",
    )
    use_sobol = st.checkbox("Sobol QMC", value=False)

    st.divider()
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Session state init ────────────────────────────────────────────────────────

for key in ("pricing_results", "conv_results", "bs_price", "asian_ref", "option_type_run"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Trigger computation ───────────────────────────────────────────────────────

if run_btn:
    enabled = []
    if use_standard:   enabled.append("Standard MC")
    if use_antithetic: enabled.append("Antithetic")
    if use_cv and not cv_disabled: enabled.append("Control Variates")
    if use_sobol:      enabled.append("Sobol QMC")

    if not enabled:
        st.sidebar.warning("Enable at least one technique.")
    else:
        methods_t = tuple(enabled)
        bs_ref = black_scholes(
            SimulatorConfig(S0=S0, K=K, T=T, r=r, sigma=sigma,
                            n_paths=n_paths, n_steps=N_STEPS_MAIN, seed=None),
            "call",
        )
        with st.spinner("Running simulations…"):
            pricing = _run_pricing(S0, K, T, sigma, r, n_paths,
                                   option_type, methods_t)
        with st.spinner("Running convergence sweep…"):
            conv = _run_convergence_sweep(S0, K, T, sigma, r,
                                          option_type, methods_t)

        st.session_state.pricing_results = pricing
        st.session_state.conv_results    = conv
        st.session_state.bs_price        = bs_ref
        st.session_state.asian_ref       = asian_ref
        st.session_state.option_type_run = option_type

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Pricing Lab",
    "🏁 Convergence Race",
    "📊 Efficiency Scorecard",
    "📚 Executive Summary",
])

_GUIDE = (
    "Configure parameters in the sidebar and click **▶ Run Simulation** to begin."
)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 — Pricing Lab
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.pricing_results is None:
        st.info(_GUIDE)
    else:
        results      = st.session_state.pricing_results
        bs_price     = st.session_state.bs_price
        asian_ref    = st.session_state.asian_ref
        opt_type_run = st.session_state.option_type_run
        # Use the Asian reference price for Asian options; BS for European
        reference_price = asian_ref if (asian_ref is not None) else bs_price
        ref_label = (
            f"Asian Ref (high-N MC): ${asian_ref:.4f}"
            if asian_ref is not None
            else f"Black-Scholes: ${bs_price:.4f}"
        )

        col_chart, col_table = st.columns([3, 2], gap="large")

        # ── Path trajectories ──────────────────────────────────────────────
        with col_chart:
            st.subheader("Simulated Price Trajectories")
            paths_np = _get_display_paths(S0, K, T, sigma, r)
            t_grid   = np.linspace(0, T, N_STEPS_MAIN + 1)
            p_mean   = paths_np.mean(axis=0)
            p_std    = paths_np.std(axis=0)

            fig = go.Figure()
            for i in range(N_DISPLAY):
                fig.add_trace(go.Scatter(
                    x=t_grid, y=paths_np[i], mode="lines",
                    line=dict(color="steelblue", width=0.6),
                    opacity=0.25, showlegend=False,
                ))
            # 95% CI ribbon
            fig.add_trace(go.Scatter(
                x=np.concatenate([t_grid, t_grid[::-1]]),
                y=np.concatenate([p_mean + 1.96*p_std,
                                  (p_mean - 1.96*p_std)[::-1]]),
                fill="toself", fillcolor="rgba(220,50,50,0.12)",
                line=dict(color="rgba(220,50,50,0)"),
                name="±1.96σ  (95% CI)",
            ))
            fig.add_trace(go.Scatter(
                x=t_grid, y=p_mean, mode="lines",
                line=dict(color="crimson", width=2.5), name="Mean path",
            ))
            fig.add_hline(y=K, line_dash="dash", line_color="black",
                          annotation_text=f"Strike K = {K}",
                          annotation_position="top left")
            fig.update_layout(
                xaxis_title="Time (years)", yaxis_title="Asset Price S(t)",
                template="plotly_white", height=430,
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                margin=dict(t=50),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Price comparison table ─────────────────────────────────────────
        with col_table:
            st.subheader("Pricing Results")
            st.metric("Black-Scholes (Euro Call)", f"${bs_price:.4f}",
                      help="Analytical Black-Scholes European call price.")
            if asian_ref is not None:
                st.metric(
                    "Asian Reference (200k paths)", f"${asian_ref:.4f}",
                    help="High-N Standard MC price used as reference for Asian options.",
                )
            st.divider()

            rows = []
            for name, res in results.items():
                ci     = res["conf_interval"]
                ci_w   = ci[1] - ci[0]
                ape    = abs(res["price"] - reference_price) / (reference_price + 1e-12) * 100
                rows.append({
                    "Method":        name,
                    "Price":         f"${res['price']:.4f}",
                    "Std Error":     f"{res['std_error']:.5f}",
                    "CI Width":      f"{ci_w:.5f}",
                    "APE vs Ref":    f"{ape:.3f}%",
                })

            df = pd.DataFrame(rows).set_index("Method")
            st.dataframe(df, use_container_width=True)

            if results:
                best_name, best_res = min(
                    results.items(),
                    key=lambda kv: abs(kv[1]["price"] - reference_price),
                )
                best_ape = abs(best_res["price"] - reference_price) / (reference_price + 1e-12) * 100
                st.success(f"**Most accurate:** {best_name}  (APE {best_ape:.3f}%)")

        # ── Technique comparison bar chart ─────────────────────────────────
        st.divider()
        st.subheader("Method-to-Method Comparison")
        names   = list(results.keys())
        prices  = [results[n]["price"] for n in names]
        se_vals = [results[n]["std_error"] for n in names]
        colors  = [METHOD_COLORS.get(n, "#888") for n in names]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            x=names, y=prices,
            error_y=dict(type="data", array=[1.96*s for s in se_vals],
                         visible=True),
            marker_color=colors,
            text=[f"${p:.4f}" for p in prices],
            textposition="outside",
            name="Price ± 95% CI",
        ))
        fig_cmp.add_hline(
            y=reference_price, line_dash="dash", line_color="black",
            annotation_text=ref_label, annotation_position="top right",
        )
        fig_cmp.update_layout(
            yaxis_title="Option Price",
            template="plotly_white", height=380,
            showlegend=False, margin=dict(t=50),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 — Convergence Race
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.conv_results is None:
        st.info(_GUIDE)
    else:
        conv      = st.session_state.conv_results
        bs_price  = st.session_state.bs_price
        asian_ref = st.session_state.asian_ref
        # Reference line value for the convergence chart
        ref_val   = asian_ref if asian_ref is not None else bs_price
        ref_txt   = (
            f"Asian Ref = {ref_val:.4f}"
            if asian_ref is not None
            else f"BS = {ref_val:.4f}"
        )

        # ── Price vs N ────────────────────────────────────────────────────
        st.subheader("Option Price vs Number of Paths")
        st.caption(
            f"Each technique races toward the reference price ({ref_txt}). "
            f"Steps fixed at {N_STEPS_SWEEP} for sweep speed."
        )
        fig_conv = go.Figure()
        for name, prices in conv.items():
            valid = [(n, p) for n, p in zip(N_SWEEP, prices) if p is not None]
            if not valid:
                continue
            ns, ps = zip(*valid)
            fig_conv.add_trace(go.Scatter(
                x=list(ns), y=list(ps), mode="lines+markers",
                name=name,
                line=dict(color=METHOD_COLORS.get(name, "#333"), width=2.5),
                marker=dict(size=7),
            ))
        fig_conv.add_hline(y=ref_val, line_dash="dash", line_color="black",
                           annotation_text=ref_txt,
                           annotation_position="top right")
        fig_conv.update_layout(
            xaxis=dict(type="log", title="Number of Paths  N"),
            yaxis_title="Option Price Estimate",
            template="plotly_white", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
            margin=dict(t=50),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # ── log(Error) vs log(N) ──────────────────────────────────────────
        st.subheader("Convergence Rate  —  log₁₀(|Error|) vs log₁₀(N)")
        fig_rate = go.Figure()
        slope_notes = []
        for name, prices in conv.items():
            valid = [(n, p) for n, p in zip(N_SWEEP, prices) if p is not None]
            if len(valid) < 3:
                continue
            ns, ps = zip(*valid)
            log_n  = np.log10(np.array(ns, dtype=float))
            log_e  = np.log10([abs(p - ref_val) + 1e-8 for p in ps])
            slope, _ = np.polyfit(log_n, log_e, 1)
            slope_notes.append(f"**{name}**: slope ≈ {slope:.2f}")
            fig_rate.add_trace(go.Scatter(
                x=list(log_n), y=list(log_e), mode="lines+markers",
                name=f"{name}  (slope ≈ {slope:.2f})",
                line=dict(color=METHOD_COLORS.get(name, "#333"), width=2.5),
                marker=dict(size=7),
            ))
        fig_rate.update_layout(
            xaxis_title="log₁₀(N)",
            yaxis_title=f"log₁₀(|MC Price − {ref_txt}|)",
            template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.01),
            margin=dict(t=50),
        )
        st.plotly_chart(fig_rate, use_container_width=True)
        st.caption(
            "Standard MC slope ≈ −0.5  →  O(1/√N).  "
            "Antithetic has the same slope but smaller constant.  "
            "Control Variates & Sobol approach slope ≈ −1.0  →  O(1/N)."
        )
        if slope_notes:
            st.info("  |  ".join(slope_notes))

# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 — Efficiency Scorecard
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.pricing_results is None:
        st.info(_GUIDE)
    else:
        results   = st.session_state.pricing_results
        bs_price  = st.session_state.bs_price
        asian_ref = st.session_state.asian_ref
        conv      = st.session_state.conv_results or {}
        ref_price = asian_ref if asian_ref is not None else bs_price

        std_se = results.get("Standard MC", {}).get("std_error")

        # ── Top metric cards ──────────────────────────────────────────────
        st.subheader("At-a-Glance Metrics")
        metric_cols = st.columns(len(results))
        for col, (name, res) in zip(metric_cols, results.items()):
            ape = abs(res["price"] - ref_price) / (ref_price + 1e-12) * 100
            with col:
                st.metric(
                    label=name,
                    value=f"${res['price']:.4f}",
                    delta=f"APE {ape:.3f}%",
                    delta_color="inverse",
                )

        st.divider()

        # ── CI Width bar ──────────────────────────────────────────────────
        col_l, col_r = st.columns(2, gap="large")
        names   = list(results.keys())
        colors  = [METHOD_COLORS.get(n, "#888") for n in names]

        with col_l:
            st.subheader("CI Width Comparison")
            st.caption("95 % confidence interval width — lower = more precise.")
            widths   = [r["conf_interval"][1] - r["conf_interval"][0]
                        for r in results.values()]
            std_w    = next((w for n, w in zip(names, widths)
                              if n == "Standard MC"), None)
            ann_ci   = []
            for n, w in zip(names, widths):
                if n == "Standard MC" or std_w is None or std_w == 0:
                    ann_ci.append("baseline")
                else:
                    pct = (1 - w / std_w) * 100
                    ann_ci.append(f"{pct:.0f}% narrower")

            fig_ci = go.Figure(go.Bar(
                x=names, y=widths, marker_color=colors,
                text=ann_ci, textposition="outside",
            ))
            fig_ci.update_layout(
                yaxis_title="CI Width", template="plotly_white", height=380,
                showlegend=False, margin=dict(t=50),
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        # ── Variance reduction bar ─────────────────────────────────────────
        with col_r:
            st.subheader("Variance Reduction vs Standard MC")
            st.caption("% reduction in estimator variance relative to Standard MC.")
            vr_vals = []
            for name, res in results.items():
                if name == "Control Variates":
                    vr_vals.append(res.get("variance_reduction_pct", 0.0))
                elif std_se and std_se > 0:
                    vr = max(0.0, (1 - (res["std_error"] / std_se) ** 2) * 100)
                    vr_vals.append(vr)
                else:
                    vr_vals.append(0.0)

            fig_vr = go.Figure(go.Bar(
                x=names, y=vr_vals, marker_color=colors,
                text=[f"{v:.1f}%" for v in vr_vals], textposition="outside",
            ))
            fig_vr.update_layout(
                yaxis_title="Variance Reduction (%)",
                yaxis_range=[0, max(vr_vals + [5]) * 1.3],
                template="plotly_white", height=380,
                showlegend=False, margin=dict(t=50),
            )
            st.plotly_chart(fig_vr, use_container_width=True)

        # ── Scorecard table ────────────────────────────────────────────────
        st.subheader("Full Scorecard")
        sc_rows = []
        for name, res, vr in zip(names, results.values(), vr_vals):
            ci_w   = res["conf_interval"][1] - res["conf_interval"][0]
            ape    = abs(res["price"] - ref_price) / (ref_price + 1e-12) * 100

            paths_1pct = "—"
            if name in conv:
                for n, p in zip(N_SWEEP, conv[name]):
                    if p is not None and abs(p - ref_price) / (ref_price + 1e-12) * 100 < 1.0:
                        paths_1pct = f"{n:,}"
                        break
                else:
                    paths_1pct = f"> {N_SWEEP[-1]:,}"

            eff = f"{(std_se / res['std_error'])**2:.1f}×" if std_se and res["std_error"] > 0 else "—"
            sc_rows.append({
                "Method":               name,
                "Price":                f"${res['price']:.4f}",
                "APE vs BS":            f"{ape:.3f}%",
                "CI Width":             round(ci_w, 5),
                "Var. Reduction %":     f"{vr:.1f}%",
                "Efficiency vs Std":    eff,
                "Paths to 1% APE":      paths_1pct,
            })

        sc_df = pd.DataFrame(sc_rows).set_index("Method")
        st.dataframe(sc_df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 — Executive Summary
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
## What This Engine Does

This is not a simple option calculator. It is a **comparison engine** designed
to optimise the *signal-to-noise ratio* in Monte Carlo financial simulation.
Each technique attacks the same problem:

$$V_0 = e^{-rT}\\,\\mathbb{E}^{\\mathbb{Q}}[\\text{Payoff}(S_T)]$$

…but each one reduces the **variance** of that estimate in a fundamentally
different way, using fewer or "smarter" paths.

---
""")

    with st.expander("📘 Standard Monte Carlo — The Baseline"):
        st.markdown("""
Standard MC draws $N$ iid pseudo-random paths and averages their discounted
payoffs.  The RMSE shrinks at:

$$\\text{RMSE} = O\\!\\left(\\tfrac{1}{\\sqrt{N}}\\right)$$

Doubling accuracy requires **4× the paths**.  This is the reference against
which every other technique is measured.

**When to use:** Always — it is the universal baseline.
""")

    with st.expander("🔁 Antithetic Variates — Exploiting Symmetry"):
        st.markdown("""
Instead of drawing $N$ independent normals $Z_i$, antithetic generates $N/2$
draws and mirrors each with $-Z_i$.

Because the call payoff $\\max(S_T - K, 0)$ is convex and monotone in $Z$:

$$\\text{Var}\\!\\left(\\tfrac{f(Z) + f(-Z)}{2}\\right) < \\text{Var}(f(Z))$$

The **negative correlation** between $f(Z)$ and $f(-Z)$ cancels part of the
variance for free — no extra paths, no extra analytical work.

**Typical result:** 30–60 % variance reduction for calls and puts.

**When to use:** Any option where the payoff is monotone in the underlying.
""")

    with st.expander("🧮 Control Variates — Correcting with a Known Anchor"):
        st.markdown("""
Control Variates (CV) exploits a second option whose **analytical price is known**
(the European call via Black-Scholes) to correct the simulation of a harder option
(e.g. the Asian arithmetic call):

$$V_{\\text{CV}} = V_{\\text{target}} - \\beta\\,(V_{\\text{control}} - C_{\\text{BS}})$$

The coefficient $\\beta = \\text{Cov}(V_{\\text{target}}, V_{\\text{control}}) / \\text{Var}(V_{\\text{control}})$
is estimated from the **same simulation paths** so both payoffs share identical
randomness. This strong positive correlation is what drives variance down.

**Typical result:** 70–95 % variance reduction for Asian options.

**When to use:** Path-dependent options (Asian, barrier) whose pricing is hard,
paired with a European call as the control.
""")

    with st.expander("🌀 Sobol Quasi-Monte Carlo — Filling Space Uniformly"):
        st.markdown("""
Standard MC fills the probability space *randomly*, creating clustering and gaps.
Sobol sequences are *low-discrepancy*: they fill space as **uniformly as
mathematically possible**.

The error bound improves from $O(N^{-1/2})$ to approximately $O((\\log N)^d / N)$,
where $d$ is the number of time steps.  For practical $N$ this is close to $O(1/N)$:

| Method | Rate | Paths to reach $10^{-3}$ error |
|--------|------|-------------------------------|
| Standard MC | $O(N^{-1/2})$ | ≈ 1,000,000 |
| Antithetic  | $O(N^{-1/2})$, smaller constant | ≈ 250,000 |
| **Sobol QMC** | **$\\approx O(N^{-1})$ effective** | **≈ 10,000** |

**When to use:** High-dimensional pricing where each time step adds a dimension
($d > 20$) and maximum accuracy with minimum paths is the goal.
""")

    st.divider()
    st.markdown("""
### Why This Matters for AI/Quant Engineering

> **"I understand Noise"** — Variance reduction is the same discipline as
> regularisation in Machine Learning: you are reducing the signal-to-noise ratio
> of your estimator.

> **"I optimise for Efficiency"** — 1 000 Sobol paths can outperform 100 000
> pseudo-random paths. Resource efficiency is a first-class engineering concern.

> **"I build for Users"** — This dashboard translates complex stochastic maths
> into a tool a Senior Trader or Risk Manager can use without knowing any code.
""")
