"""
Earth-Gravity Baseline: Joint Reaction Analysis
=================================================
Runs OpenSim's AnalyzeTool with JointReaction analysis
on the Gait2392 model using real walking data.

This produces the baseline bone-loading forces at 1g
that the exosuit controller will try to restore in
microgravity.

Usage:
    python run_baseline.py

Output:
    - Joint reaction force plots for one gait cycle
    - Calibrated daily loading targets (printed + saved)
    - .sto file with raw joint reaction forces
"""

import os
import sys
import json
import numpy as np

# ── OpenSim setup ─────────────────────────────────────
opensim_bin = r"D:\Program Files\OpenSim\OpenSim 4.5\bin"
os.add_dll_directory(opensim_bin)
os.environ['PATH'] = opensim_bin + ';' + os.environ['PATH']
sys.path.insert(0, r"D:\Program Files\OpenSim\OpenSim 4.5\sdk\Python")

import opensim as osim

# ── Paths ─────────────────────────────────────────────
PROJECT_DIR = r"D:\Program Files\Opensim-using_DT_Paper"
MODEL_PATH = os.path.join(PROJECT_DIR, "Models", "gait2392_simbody.osim")
MOTION_FILE = os.path.join(PROJECT_DIR, "subject01_walk1.mot")
GRF_FILE = os.path.join(PROJECT_DIR, "subject01_walk1_grf.xml")
RESULTS_DIR = os.path.join(PROJECT_DIR, "baseline_results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Joints we care about for bone loading ─────────────
TARGET_JOINTS = ["hip_r", "hip_l", "knee_r", "knee_l", "ankle_r", "ankle_l"]

# ── Time range: one full gait cycle ───────────────────
# The full file is 0-15s. We'll use a single gait cycle.
# A typical gait cycle at normal speed is ~1.0-1.2s.
# We use 0.6s to 1.83s (one complete right-leg cycle from
# the subject01 data, heel-strike to heel-strike).
T_START = 0.6
T_END = 1.83


def run_joint_reaction_analysis():
    """Run JointReaction analysis using OpenSim's AnalyzeTool."""

    print("\n" + "=" * 60)
    print("  EARTH-GRAVITY BASELINE ANALYSIS")
    print("  Model: Gait2392 | Data: subject01_walk1")
    print("=" * 60)

    # Load model
    model = osim.Model(MODEL_PATH)
    model.initSystem()
    print(f"\n  Model loaded: {model.getName()}")
    print(f"  Bodies: {model.getBodySet().getSize()}")
    print(f"  Muscles: {model.getMuscles().getSize()}")

    # ── Set up the Analyze tool ───────────────────────
    analyze = osim.AnalyzeTool()
    analyze.setModel(model)
    analyze.setModelFilename(MODEL_PATH)
    analyze.setCoordinatesFileName(MOTION_FILE)
    analyze.setInitialTime(T_START)
    analyze.setFinalTime(T_END)
    analyze.setLowpassCutoffFrequency(6.0)
    analyze.setResultsDir(RESULTS_DIR)

    # ── Add external loads (ground reaction forces) ───
    # This is critical: without GRF, the forces are wrong
    if os.path.exists(GRF_FILE):
        analyze.setExternalLoadsFileName(GRF_FILE)
        print(f"  Ground reaction forces: loaded")
    else:
        print(f"  [WARN] No GRF file found at {GRF_FILE}")
        print(f"  Results will be less accurate without GRF.")

    # ── Create JointReaction analysis ─────────────────
    jr = osim.JointReaction()
    jr.setName("JointReaction")
    jr.setOn(True)
    jr.setStartTime(T_START)
    jr.setEndTime(T_END)

    # Specify which joints to report
    joint_names = osim.ArrayStr()
    for jname in TARGET_JOINTS:
        joint_names.append(jname)
    jr.setJointNames(joint_names)

    # Express forces in the child body frame (bone frame)
    on_body = osim.ArrayStr()
    on_body.append("child")
    jr.setOnBody(on_body)

    in_frame = osim.ArrayStr()
    in_frame.append("child")
    jr.setInFrame(in_frame)

    # Add analysis to model
    model.addAnalysis(jr)
    model.initSystem()

    # ── Run the analysis ──────────────────────────────
    print(f"\n  Running Joint Reaction Analysis...")
    print(f"  Time range: {T_START:.2f}s to {T_END:.2f}s")
    print(f"  (one gait cycle)")

    analyze.setModel(model)

    try:
        analyze.run()
        print(f"  Analysis complete!")
    except Exception as e:
        print(f"\n  [ERROR] Analysis failed: {e}")
        print(f"  Trying alternative approach...")
        return run_alternative_analysis()

    return parse_results()


def run_alternative_analysis():
    """
    Fallback: use InverseDynamicsTool if AnalyzeTool has issues.
    Then estimate joint reaction forces from joint torques.
    """
    print("\n  Running Inverse Dynamics as fallback...")

    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(MODEL_PATH)
    id_tool.setCoordinatesFileName(MOTION_FILE)
    id_tool.setStartTime(T_START)
    id_tool.setEndTime(T_END)
    id_tool.setLowpassCutoffFrequency(6.0)
    id_tool.setOutputGenForceFileName("inverse_dynamics.sto")
    id_tool.setResultsDir(RESULTS_DIR)

    if os.path.exists(GRF_FILE):
        id_tool.setExternalLoadsFileName(GRF_FILE)

    try:
        id_tool.run()
        print("  Inverse Dynamics complete!")
    except Exception as e:
        print(f"  [ERROR] {e}")
        print("\n  Using synthetic baseline instead.")
        return generate_synthetic_baseline()

    return parse_id_results()


def parse_results():
    """Parse JointReaction .sto output files."""
    print(f"\n  Parsing results from {RESULTS_DIR}...")

    # JointReaction outputs: _JointReaction_ReactionLoads.sto
    sto_pattern = "_JointReaction_ReactionLoads.sto"
    sto_file = None

    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(sto_pattern) or "JointReaction" in fname:
            sto_file = os.path.join(RESULTS_DIR, fname)
            break

    if sto_file is None:
        print("  [WARN] JointReaction output not found.")
        print(f"  Files in {RESULTS_DIR}:")
        for f in os.listdir(RESULTS_DIR):
            print(f"    {f}")
        return generate_synthetic_baseline()

    print(f"  Found: {os.path.basename(sto_file)}")

    # Read the .sto file
    storage = osim.Storage(sto_file)
    n_rows = storage.getSize()
    dt = (T_END - T_START) / n_rows

    print(f"  Time steps: {n_rows}")

    # Extract force magnitudes for each target joint
    results = {}
    col_labels = [storage.getColumnLabels().get(i)
                  for i in range(storage.getColumnLabels().getSize())]

    for joint_name in TARGET_JOINTS:
        body_name = get_child_body(joint_name)

        # Look for force columns: joint_on_body_in_child_fx/fy/fz
        fx_col = find_column(col_labels, joint_name, "fx")
        fy_col = find_column(col_labels, joint_name, "fy")
        fz_col = find_column(col_labels, joint_name, "fz")

        if fx_col is not None and fy_col is not None and fz_col is not None:
            forces = np.zeros(n_rows)
            for i in range(n_rows):
                sv = storage.getStateVector(i)
                data = sv.getData()
                fx = data.get(fx_col)
                fy = data.get(fy_col)
                fz = data.get(fz_col)
                forces[i] = np.sqrt(fx**2 + fy**2 + fz**2)
            results[joint_name] = forces
            print(f"  {joint_name}: peak={np.max(forces):.1f}N, "
                  f"mean={np.mean(forces):.1f}N")
        else:
            print(f"  [WARN] Columns not found for {joint_name}")
            print(f"  Available columns: {col_labels[:10]}...")

    if not results:
        print("  No joint reaction data extracted. Using synthetic baseline.")
        return generate_synthetic_baseline()

    return process_baseline(results, dt, n_rows)


def find_column(col_labels, joint_name, component):
    """Find the column index for a joint force component."""
    patterns = [
        f"{joint_name}_on_",    # e.g. hip_r_on_femur_r_in_femur_r_fx
        f"{joint_name}_",       # simpler pattern
    ]
    for i, label in enumerate(col_labels):
        if label == "time":
            continue
        for pat in patterns:
            if pat in label.lower() and component in label.lower():
                return i - 1  # subtract 1 because time is index 0
    return None


def get_child_body(joint_name):
    """Map joint name to child body name."""
    body_map = {
        "hip_r": "femur_r", "hip_l": "femur_l",
        "knee_r": "tibia_r", "knee_l": "tibia_l",
        "ankle_r": "talus_r", "ankle_l": "talus_l",
    }
    return body_map.get(joint_name, joint_name)


def parse_id_results():
    """Parse Inverse Dynamics results and estimate JRF."""
    sto_file = os.path.join(RESULTS_DIR, "inverse_dynamics.sto")
    if not os.path.exists(sto_file):
        return generate_synthetic_baseline()

    storage = osim.Storage(sto_file)
    n_rows = storage.getSize()
    dt = (T_END - T_START) / n_rows

    # Map joint names to coordinate names for ID output
    coord_map = {
        "hip_r": "hip_flexion_r", "hip_l": "hip_flexion_l",
        "knee_r": "knee_angle_r", "knee_l": "knee_angle_l",
        "ankle_r": "ankle_angle_r", "ankle_l": "ankle_angle_l",
    }

    col_labels = [storage.getColumnLabels().get(i)
                  for i in range(storage.getColumnLabels().getSize())]

    results = {}
    for joint_name, coord_name in coord_map.items():
        # Find the torque column
        col_idx = None
        for i, label in enumerate(col_labels):
            if coord_name in label.lower():
                col_idx = i - 1
                break

        if col_idx is not None:
            torques = np.zeros(n_rows)
            for i in range(n_rows):
                sv = storage.getStateVector(i)
                torques[i] = abs(sv.getData().get(col_idx))

            # Estimate JRF from torque:
            # JRF ≈ torque / moment_arm × cocontraction_factor
            moment_arm = 0.05  # ~5cm
            cocontraction = 2.5
            forces = torques / moment_arm * cocontraction
            results[joint_name] = forces
            print(f"  {joint_name}: peak torque={np.max(torques):.1f}N·m, "
                  f"estimated peak JRF={np.max(forces):.1f}N")

    return process_baseline(results, dt, n_rows)


def generate_synthetic_baseline():
    """
    Generate realistic synthetic baseline from literature values.
    Used as fallback if OpenSim analysis encounters issues.
    """
    print("\n  Generating literature-based baseline forces...")

    cycle_time = T_END - T_START
    dt = 0.01
    n = int(cycle_time / dt)
    t = np.linspace(0, cycle_time, n)
    phase = 2 * np.pi * t / cycle_time

    bw = 700  # ~70kg person

    results = {}
    # Literature values (Heller 2001, Bergmann 2001):
    # Hip: 2.5-3.5 BW peak, double-bump pattern
    # Knee: 2-3 BW peak
    # Ankle: 3-5 BW peak
    profiles = {
        "hip_r":   {"peak_bw": 3.0, "phase1": 0.8, "phase2": 2.2},
        "hip_l":   {"peak_bw": 3.0, "phase1": 0.8, "phase2": 2.2},
        "knee_r":  {"peak_bw": 2.5, "phase1": 0.6, "phase2": 2.0},
        "knee_l":  {"peak_bw": 2.5, "phase1": 0.6, "phase2": 2.0},
        "ankle_r": {"peak_bw": 4.0, "phase1": 0.9, "phase2": 2.5},
        "ankle_l": {"peak_bw": 4.0, "phase1": 0.9, "phase2": 2.5},
    }

    for joint_name, p in profiles.items():
        forces = bw * p["peak_bw"] * (
            0.6 * np.exp(-((phase - p["phase1"])**2) / 0.12)
            + 0.45 * np.exp(-((phase - p["phase2"])**2) / 0.12)
            + 0.05
        )
        results[joint_name] = forces
        print(f"  {joint_name}: peak={np.max(forces):.0f}N ({p['peak_bw']} BW)")

    return process_baseline(results, dt, n)


def process_baseline(results, dt, n_steps):
    """
    Process baseline forces → compute daily loading targets.
    """
    print("\n" + "=" * 60)
    print("  BASELINE LOADING ANALYSIS")
    print("=" * 60)

    cycle_time = T_END - T_START
    cycles_per_day = 6000  # ~2 hours of walking equivalent

    targets = {}
    for joint_name, forces in results.items():
        # Force-time integral for one cycle (trapezoidal integration)
        impulse_per_cycle = np.trapz(np.abs(forces), dx=dt)

        # Daily target = impulse per cycle × cycles per day
        daily_target = impulse_per_cycle * cycles_per_day

        # Peak and mean force
        peak_force = np.max(forces)
        mean_force = np.mean(forces)

        targets[joint_name] = {
            "impulse_per_cycle_Ns": float(impulse_per_cycle),
            "daily_target_Ns": float(daily_target),
            "peak_force_N": float(peak_force),
            "mean_force_N": float(mean_force),
        }

        print(f"  {joint_name:>10}: peak={peak_force:7.1f}N  "
              f"impulse/cycle={impulse_per_cycle:7.1f}N·s  "
              f"daily target={daily_target:>12,.0f}N·s")

    # Save targets
    output = {
        "description": "Earth-gravity baseline loading targets",
        "model": "Gait2392",
        "gait_cycle_time_s": cycle_time,
        "cycles_per_day": cycles_per_day,
        "joints": targets,
    }

    output_path = os.path.join(PROJECT_DIR, "baseline_targets.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Targets saved to: {output_path}")

    # ── Plot the baseline force profiles ──────────────
    plot_baseline(results, dt)

    print("\n" + "=" * 60)
    print("  NEXT STEP: Update config.py with these targets,")
    print("  then run the closed-loop simulation.")
    print("=" * 60 + "\n")

    return targets


def plot_baseline(results, dt):
    """Plot baseline joint reaction force profiles."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots.")
        return

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Earth-gravity baseline: joint reaction forces (1 gait cycle)",
                 fontweight="bold", y=1.02)

    pairs = [
        (["hip_r", "hip_l"], "Hip joint", "#2176AE"),
        (["knee_r", "knee_l"], "Knee joint", "#57B894"),
        (["ankle_r", "ankle_l"], "Ankle joint", "#D4A843"),
    ]

    for ax, (joints, title, color) in zip(axes, pairs):
        for jname in joints:
            if jname not in results:
                continue
            forces = results[jname]
            n = len(forces)
            t_pct = np.linspace(0, 100, n)
            ls = "-" if "_r" in jname else "--"
            label = "Right" if "_r" in jname else "Left"
            ax.plot(t_pct, forces, color=color, ls=ls, lw=2, label=label)

        ax.set_title(title)
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Reaction force (N)")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 100)

    fig.tight_layout()
    save_path = os.path.join(PROJECT_DIR, "fig_baseline_forces.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n  Plot saved to: {save_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────
if __name__ == "__main__":
    targets = run_joint_reaction_analysis()
