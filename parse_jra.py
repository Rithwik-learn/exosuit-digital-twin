"""
Parse OpenSim Joint Reaction Analysis Results
==============================================
Reads the real JRA output from the Gait2354 model,
extracts bone-loading forces, and produces baseline
calibration data for the closed-loop controller.
"""

import os
import sys
import json
import numpy as np

# OpenSim setup
opensim_bin = r"D:\Program Files\OpenSim\OpenSim 4.5\bin"
os.add_dll_directory(opensim_bin)
os.environ['PATH'] = opensim_bin + ';' + os.environ['PATH']
sys.path.insert(0, r"D:\Program Files\OpenSim\OpenSim 4.5\sdk\Python")

PROJECT_DIR = r"D:\Program Files\Opensim-using_DT_Paper"
JRA_FILE = os.path.join(PROJECT_DIR, "JRA_Results",
                        "subject01_JointReaction_ReactionLoads.sto")

# ── Read the .sto file ────────────────────────────────
print("=" * 60)
print("  PARSING OPENSIM JOINT REACTION ANALYSIS")
print("=" * 60)

# Read header to find column names
with open(JRA_FILE, 'r') as f:
    header_lines = []
    for line in f:
        header_lines.append(line.strip())
        if line.strip() == 'endheader':
            break
    # Next line is column headers
    col_line = f.readline().strip()
    columns = col_line.split('\t')

print(f"\n  File: {os.path.basename(JRA_FILE)}")
print(f"  Columns: {len(columns)}")
print(f"  First 10 columns: {columns[:10]}")

# Load numerical data
data = np.loadtxt(JRA_FILE, skiprows=len(header_lines) + 1)
time = data[:, 0]
print(f"  Time steps: {len(time)}")
print(f"  Time range: {time[0]:.3f}s to {time[-1]:.3f}s")

# ── Extract joint reaction force magnitudes ───────────
# Column naming convention:
#   joint_on_body_in_frame_fx, _fy, _fz (forces)
#   joint_on_body_in_frame_mx, _my, _mz (moments)

target_joints = {
    "hip_r":   "hip_r",
    "hip_l":   "hip_l",
    "knee_r":  "knee_r",
    "knee_l":  "knee_l",
    "ankle_r": "ankle_r",
    "ankle_l": "ankle_l",
}

# Find force columns for each joint
results = {}
print(f"\n  Joint reaction forces (from OpenSim):")
print(f"  {'Joint':<12} {'Peak (N)':>10} {'Mean (N)':>10} {'Peak (BW)':>10}")
print(f"  {'-'*44}")

body_weight = 72.6 * 9.81  # subject mass × gravity

for joint_label, joint_name in target_joints.items():
    # Search for force columns matching this joint
    fx_idx = fy_idx = fz_idx = None
    for i, col in enumerate(columns):
        col_lower = col.lower()
        if joint_name in col_lower and '_fx' in col_lower:
            fx_idx = i
        elif joint_name in col_lower and '_fy' in col_lower:
            fy_idx = i
        elif joint_name in col_lower and '_fz' in col_lower:
            fz_idx = i

    if fx_idx and fy_idx and fz_idx:
        fx = data[:, fx_idx]
        fy = data[:, fy_idx]
        fz = data[:, fz_idx]
        force_mag = np.sqrt(fx**2 + fy**2 + fz**2)
        results[joint_label] = force_mag

        peak = np.max(force_mag)
        mean = np.mean(force_mag)
        print(f"  {joint_label:<12} {peak:>10.1f} {mean:>10.1f} {peak/body_weight:>10.2f}")
    else:
        print(f"  {joint_label:<12} — columns not found")
        # Try to list matching columns
        matches = [c for c in columns if joint_name in c.lower()]
        if matches:
            print(f"    Found: {matches[:6]}")

# ── Compute loading impulse per cycle ─────────────────
cycle_time = time[-1] - time[0]
dt = cycle_time / (len(time) - 1)
cycles_per_day = 6000

print(f"\n  Cycle duration: {cycle_time:.3f}s")
print(f"  Time step: {dt*1000:.2f}ms")

print(f"\n  Daily loading targets (at 50% restoration):")
print(f"  {'Joint':<12} {'Impulse/cycle':>14} {'Daily (Earth)':>14} {'Target (50%)':>14}")
print(f"  {'-'*56}")

calibration = {}
for joint_name, forces in results.items():
    impulse = np.trapz(forces, dx=dt)
    daily_earth = impulse * cycles_per_day
    target_50 = daily_earth * 0.50
    calibration[joint_name] = {
        "impulse_per_cycle": float(impulse),
        "daily_earth": float(daily_earth),
        "target_50pct": float(target_50),
        "peak_N": float(np.max(forces)),
        "mean_N": float(np.mean(forces)),
    }
    print(f"  {joint_name:<12} {impulse:>12.1f}N·s {daily_earth:>12,.0f}N·s {target_50:>12,.0f}N·s")

# Save calibration
cal_path = os.path.join(PROJECT_DIR, "opensim_calibration.json")
with open(cal_path, 'w') as f:
    json.dump(calibration, f, indent=2)
print(f"\n  Calibration saved to: {cal_path}")

# ── Plot the force profiles ───────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 150, "savefig.dpi": 300,
    })

    # Normalise time to gait cycle percentage
    t_pct = ((time - time[0]) / cycle_time) * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Joint reaction forces from OpenSim (Gait2354, 1g baseline)",
        fontweight="bold", y=1.02
    )

    pairs = [
        (["hip_r", "hip_l"], "Hip joint", "#2176AE"),
        (["knee_r", "knee_l"], "Knee joint", "#57B894"),
        (["ankle_r", "ankle_l"], "Ankle joint", "#D4A843"),
    ]

    for ax, (joints, title, color) in zip(axes, pairs):
        for jname in joints:
            if jname not in results:
                continue
            ls = "-" if "_r" in jname else "--"
            label = "Right" if "_r" in jname else "Left"
            ax.plot(t_pct, results[jname], color=color, ls=ls, lw=2, label=label)

        ax.set_title(title)
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Reaction force (N)")
        ax.set_xlim(0, 100)
        ax.legend(fontsize=9)

    fig.tight_layout()
    plot_path = os.path.join(PROJECT_DIR, "fig_opensim_baseline_forces.png")
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"  Plot saved to: {plot_path}")
    plt.close()

except Exception as e:
    print(f"  [WARN] Plotting failed: {e}")

print("\n" + "=" * 60)
print("  REAL OPENSIM FORCES EXTRACTED SUCCESSFULLY")
print("  These replace the literature-calibrated mock data.")
print("=" * 60)
