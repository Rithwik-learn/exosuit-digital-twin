"""
Visualisation Module
====================
Generates publication-quality figures for the research paper.

Produces four key figures:
    1. Cumulative loading curves (with/without exosuit)
    2. Loading deficit convergence over time
    3. Controller torque response profiles
    4. Summary comparison bar chart

Usage:
    python visualise_results.py           # from saved results
    python simulation.py --plot           # inline after simulation
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

from config import TARGET_JOINTS, DAILY_LOAD_TARGETS, OUTPUT_DIR

# DAILY_LOAD_TARGETS already includes the restoration fraction
EFFECTIVE_TARGETS = DAILY_LOAD_TARGETS

# ── Publication style ──────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colour palette (colourblind-friendly)
COLORS = {
    "hip_r":   "#2176AE",  "hip_l":   "#2176AE",
    "knee_r":  "#57B894",  "knee_l":  "#57B894",
    "ankle_r": "#D4A843",  "ankle_l": "#D4A843",
}
JOINT_STYLES = {
    "hip_r":   {"color": "#2176AE", "ls": "-",  "label": "Hip (R)"},
    "hip_l":   {"color": "#2176AE", "ls": "--", "label": "Hip (L)"},
    "knee_r":  {"color": "#57B894", "ls": "-",  "label": "Knee (R)"},
    "knee_l":  {"color": "#57B894", "ls": "--", "label": "Knee (L)"},
    "ankle_r": {"color": "#D4A843", "ls": "-",  "label": "Ankle (R)"},
    "ankle_l": {"color": "#D4A843", "ls": "--", "label": "Ankle (L)"},
}


def plot_all(results: dict, save: bool = True):
    """Generate all figures from simulation results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig1 = plot_cumulative_loading(results, save=save)
    fig2 = plot_deficit_convergence(results, save=save)
    fig3 = plot_torque_profiles(results, save=save)
    fig4 = plot_summary_comparison(results, save=save)
    fig5 = plot_single_cycle_waveforms(results, save=save)
    fig6 = plot_controller_dynamics(results, save=save)

    plt.show()
    return fig1, fig2, fig3, fig4, fig5, fig6


# ── Figure 1: Cumulative Loading ──────────────────────

def plot_cumulative_loading(results: dict, save: bool = True):
    """
    Fig 1: Cumulative bone loading over the simulated day.
    Shows exosuit-augmented loading vs. daily targets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(
        "Cumulative bone loading with closed-loop exosuit",
        fontweight="bold", y=1.02
    )

    joint_pairs = [
        ("hip_r", "hip_l", "Hip joint"),
        ("knee_r", "knee_l", "Knee joint"),
        ("ankle_r", "ankle_l", "Ankle joint"),
    ]

    cycles = np.array(results["cycles"])

    for ax, (jr, jl, title) in zip(axes, joint_pairs):
        # Plot loading curves
        for jname in [jr, jl]:
            style = JOINT_STYLES[jname]
            loading = results["loading"][jname]
            ax.plot(
                cycles, loading,
                color=style["color"], ls=style["ls"],
                lw=2, label=style["label"]
            )

        # Target line
        target = EFFECTIVE_TARGETS[jr]
        ax.axhline(
            target, color="#E63946", ls=":", lw=1.5,
            alpha=0.7, label=f"Target ({target:.0f} N·s)"
        )

        # Baseline (no suit) endpoint
        bl_r = results.get("baseline_loading", {}).get(jr, 0)
        if bl_r > 0 and len(cycles) > 0:
            ax.axhline(
                bl_r, color="#999", ls="-.", lw=1,
                alpha=0.5, label=f"No suit ({bl_r:.0f} N·s)"
            )

        ax.set_title(title)
        ax.set_xlabel("Gait cycles")
        ax.set_ylabel("Cumulative load (N·s)")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
        ax.set_xlim(0, cycles[-1] if len(cycles) > 0 else 6000)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig1_cumulative_loading.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    return fig


# ── Figure 2: Deficit Convergence ─────────────────────

def plot_deficit_convergence(results: dict, save: bool = True):
    """
    Fig 2: Loading deficit (%) over time.
    Shows how the controller drives deficits toward zero.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "Loading deficit convergence under closed-loop control",
        fontweight="bold"
    )

    cycles = np.array(results["cycles"])

    # Plot only right-side joints to reduce clutter
    for jname in ["hip_r", "knee_r", "ankle_r"]:
        style = JOINT_STYLES[jname]
        deficit = results["deficits"][jname]
        ax.plot(
            cycles, deficit,
            color=style["color"], ls=style["ls"],
            lw=2.5, label=style["label"]
        )

    # Target zone
    ax.axhline(5.0, color="#E63946", ls=":", lw=1, alpha=0.5)
    ax.fill_between(
        cycles, 0, 5,
        color="#57B894", alpha=0.08, label="Target zone (≤5%)"
    )

    ax.set_xlabel("Gait cycles")
    ax.set_ylabel("Loading deficit (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig2_deficit_convergence.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    return fig


# ── Figure 3: Torque Profiles ─────────────────────────

def plot_torque_profiles(results: dict, save: bool = True):
    """
    Fig 3: Exosuit actuator torque commands over time.
    Shows the controller's adaptive response.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "Exosuit resistance torque — controller output",
        fontweight="bold"
    )

    cycles = np.array(results["cycles"])

    for jname in ["hip_r", "knee_r", "ankle_r"]:
        style = JOINT_STYLES[jname]
        torques = results["torques"][jname]
        ax.plot(
            cycles, torques,
            color=style["color"], ls=style["ls"],
            lw=2, label=style["label"]
        )

    ax.set_xlabel("Gait cycles")
    ax.set_ylabel("Resistance torque (N·m)")
    ax.set_ylim(bottom=-0.5)
    ax.legend(loc="upper right", framealpha=0.8)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig3_torque_profiles.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    return fig


# ── Figure 4: Summary Comparison ─────────────────────

def plot_summary_comparison(results: dict, save: bool = True):
    """
    Fig 4: Bar chart comparing end-of-day loading
    for no-suit vs. closed-loop exosuit vs. target.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        "Daily bone loading: microgravity baseline vs. exosuit",
        fontweight="bold"
    )

    joints_to_plot = ["hip_r", "knee_r", "ankle_r"]
    labels = [TARGET_JOINTS[j]["label"].replace(" (R)", "") for j in joints_to_plot]
    # Use short labels
    labels = ["Hip", "Knee", "Ankle"]
    x = np.arange(len(labels))
    width = 0.25

    # Baseline (no suit)
    baseline_vals = [
        results.get("baseline_loading", {}).get(j, 0)
        for j in joints_to_plot
    ]
    # With suit (final cumulative)
    suit_vals = [
        results["loading"][j][-1] if results["loading"][j] else 0
        for j in joints_to_plot
    ]
    # Targets
    target_vals = [EFFECTIVE_TARGETS[j] for j in joints_to_plot]

    bars1 = ax.bar(
        x - width, baseline_vals, width,
        label="Microgravity (no suit)",
        color="#CCCCCC", edgecolor="#999", linewidth=0.5
    )
    bars2 = ax.bar(
        x, suit_vals, width,
        label="Closed-loop exosuit",
        color="#2176AE", edgecolor="#185A8A", linewidth=0.5
    )
    bars3 = ax.bar(
        x + width, target_vals, width,
        label="Earth-baseline target",
        color="#E63946", edgecolor="#B52D38", linewidth=0.5,
        alpha=0.6
    )

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 30,
                    f"{h:.0f}", ha="center", va="bottom",
                    fontsize=8, color="#333"
                )

    ax.set_ylabel("Cumulative daily load (N·s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig4_summary_comparison.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    return fig


# ── Figure 5: Single-Cycle Force Waveforms ─────────────

def plot_single_cycle_waveforms(results: dict, save: bool = True):
    """
    Fig 5: Joint reaction force profiles within one gait cycle.
    Compares: microgravity baseline (no suit), microgravity + exosuit
    at three controller states (early, mid-day, near-target).

    This is the key biomechanics figure for the paper.
    """
    from digital_twin import DigitalTwin
    from config import MODEL_PATH, SIM_START_TIME, SIM_END_TIME, TIME_STEP

    twin = DigitalTwin(MODEL_PATH)
    t = np.arange(SIM_START_TIME, SIM_END_TIME, TIME_STEP)
    t_pct = (t / SIM_END_TIME) * 100  # gait cycle %

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(
        "Joint reaction forces within a single gait cycle",
        fontweight="bold", y=1.02
    )

    joints_to_plot = ["hip_r", "knee_r", "ankle_r"]
    titles = ["Hip joint", "Knee joint", "Ankle joint"]

    # Three scenarios: no suit, low torque (early day), high torque (late day)
    scenarios = [
        {"label": "Microgravity (no suit)", "torques": None,
         "color": "#999", "ls": "-.", "lw": 1.5},
        {"label": "Exosuit — early (8 N·m)", "torques": 8.0,
         "color": "#2176AE", "ls": "--", "lw": 1.8},
        {"label": "Exosuit — peak (20 N·m)", "torques": 20.0,
         "color": "#2176AE", "ls": "-", "lw": 2.2},
    ]

    for ax, joint, title in zip(axes, joints_to_plot, titles):
        for scenario in scenarios:
            if scenario["torques"] is None:
                loads = None
            else:
                loads = {j: scenario["torques"] for j in TARGET_JOINTS}

            forces = twin.compute_joint_reaction_forces(
                external_loads=loads
            )
            ax.plot(
                t_pct, forces[joint],
                color=scenario["color"],
                ls=scenario["ls"],
                lw=scenario["lw"],
                label=scenario["label"],
            )

        ax.set_title(title)
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Reaction force (N)")
        ax.set_xlim(0, 100)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig5_single_cycle_waveforms.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    return fig


# ── Figure 6: Controller Dynamics ──────────────────────

def plot_controller_dynamics(results: dict, save: bool = True):
    """
    Fig 6: Multi-panel showing the adaptive controller behaviour.
    Top: torque output per joint over time
    Bottom: normalised loading progress vs. scheduled target line

    Demonstrates that the controller adapts torque based on
    deficit rate rather than just deficit magnitude.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        "Adaptive closed-loop controller dynamics",
        fontweight="bold", y=1.01
    )

    cycles = np.array(results["cycles"])

    # ── Top panel: torque profiles ──
    for jname in ["hip_r", "knee_r", "ankle_r"]:
        style = JOINT_STYLES[jname]
        torques = results["torques"][jname]
        ax1.plot(
            cycles, torques,
            color=style["color"], ls=style["ls"],
            lw=2, label=style["label"]
        )

    ax1.set_ylabel("Resistance torque (N·m)")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax1.set_ylim(bottom=-0.5)

    # Annotate phases
    if len(cycles) > 10:
        mid = len(cycles) // 2
        ax1.annotate(
            "Ramp-up phase", xy=(cycles[3], 0),
            fontsize=9, color="#666", ha="left", va="bottom"
        )

    # ── Bottom panel: loading progress vs schedule ──
    for jname in ["hip_r", "knee_r", "ankle_r"]:
        style = JOINT_STYLES[jname]
        loading = np.array(results["loading"][jname])
        target = EFFECTIVE_TARGETS[jname]
        progress = (loading / target) * 100
        ax2.plot(
            cycles, progress,
            color=style["color"], ls=style["ls"],
            lw=2, label=style["label"]
        )

    # Ideal linear schedule line
    total_cyc = results["metadata"]["total_cycles"]
    ax2.plot(
        [0, total_cyc], [0, 100],
        color="#E63946", ls=":", lw=1.5, alpha=0.6,
        label="Ideal schedule"
    )

    # 95% threshold
    ax2.axhline(95, color="#57B894", ls="--", lw=1, alpha=0.4)
    ax2.text(
        cycles[1], 96.5, "95% convergence threshold",
        fontsize=8, color="#57B894", alpha=0.7
    )

    ax2.set_xlabel("Gait cycles")
    ax2.set_ylabel("Loading progress (%)")
    ax2.set_ylim(-2, 110)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.8)

    fig.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "fig6_controller_dynamics.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    return fig


# ── Standalone entry ────────────────────────────────────

if __name__ == "__main__":
    results_path = os.path.join(OUTPUT_DIR, "simulation_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        plot_all(results, save=True)
    else:
        print(f"No results found at {results_path}")
        print("Run simulation.py first, or pass --plot flag.")
