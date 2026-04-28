"""
Closed-Loop Exosuit Digital Twin — Main Simulation
===================================================
Orchestrates the full control loop:

    Motion → Digital Twin → Bone Loading → Deficit Monitor
       ↑                                        ↓
       └──── Exosuit Actuators ← Controller ────┘

Usage:
    python simulation.py [--cycles 6000] [--plot]

The simulation models one day of astronaut activity in
microgravity, showing how the closed-loop exosuit
dynamically adjusts resistance to meet bone-loading targets.
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime

from config import (
    MODEL_PATH, IK_MOTION_FILE, OUTPUT_DIR,
    CYCLES_PER_DAY, CONTROL_UPDATE_INTERVAL,
    TARGET_JOINTS, DAILY_LOAD_TARGETS,
    MICROGRAVITY_FORCE_FRACTION, TARGET_RESTORATION_FRACTION
)
from digital_twin import DigitalTwin
from bone_loading_monitor import BoneLoadingMonitor
from controller import ResistanceController


def run_simulation(
    total_cycles: int = CYCLES_PER_DAY,
    update_interval: int = CONTROL_UPDATE_INTERVAL,
    verbose: bool = True,
) -> dict:
    """
    Run the full closed-loop simulation for one day.

    Parameters
    ----------
    total_cycles : int
        Total gait cycles to simulate (~6000 for 2 hrs walking).
    update_interval : int
        How often the controller re-evaluates (in cycles).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict : Complete simulation results for analysis/plotting.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Initialise components ───────────────────────────
    twin = DigitalTwin(MODEL_PATH)
    monitor = BoneLoadingMonitor(microgravity=True)
    controller = ResistanceController(total_cycles=total_cycles)

    if verbose:
        print("\n" + "=" * 60)
        print("  CLOSED-LOOP EXOSUIT DIGITAL TWIN SIMULATION")
        print("=" * 60)
        info = twin.get_model_info()
        print(f"  Model: {info.get('name', 'Gait2392')}")
        print(f"  Total cycles: {total_cycles}")
        print(f"  Control interval: {update_interval} cycles")
        print(f"  Microgravity baseline: {MICROGRAVITY_FORCE_FRACTION*100:.0f}% of Earth JRF")
        print(f"  Restoration target: {TARGET_RESTORATION_FRACTION*100:.0f}% of Earth loading")
        print("=" * 60 + "\n")

    # ── Data collection ─────────────────────────────────
    results = {
        "cycles": [],
        "loading": {j: [] for j in TARGET_JOINTS},
        "deficits": {j: [] for j in TARGET_JOINTS},
        "torques": {j: [] for j in TARGET_JOINTS},
        "targets": DAILY_LOAD_TARGETS.copy(),
        "metadata": {
            "total_cycles": total_cycles,
            "update_interval": update_interval,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Also track a "no exosuit" baseline for comparison
    baseline_monitor = BoneLoadingMonitor(microgravity=True)

    # ── Main control loop ───────────────────────────────
    current_torques = {j: 0.0 for j in TARGET_JOINTS}
    n_updates = total_cycles // update_interval

    for step in range(n_updates):
        cycle_num = (step + 1) * update_interval

        # 1. DIGITAL TWIN: Compute joint reaction forces
        #    with current exosuit resistance applied
        joint_forces_with_suit = twin.compute_joint_reaction_forces(
            kinematics_file=IK_MOTION_FILE,
            external_loads=current_torques,
        )

        # Also compute baseline (no exosuit) for comparison
        joint_forces_baseline = twin.compute_joint_reaction_forces(
            kinematics_file=IK_MOTION_FILE,
            external_loads=None,
        )

        # 2. MONITOR: Update cumulative loading
        monitor.update(joint_forces_with_suit, n_cycles=update_interval)
        baseline_monitor.update(
            joint_forces_baseline, n_cycles=update_interval
        )

        # 3. MONITOR: Get deficit report
        deficit_report = monitor.get_deficit_report()

        # 4. CONTROLLER: Compute new resistance torques
        current_torques = controller.compute_resistance(
            deficit_report, current_cycle=cycle_num
        )

        # 5. RECORD: Store data for analysis
        results["cycles"].append(cycle_num)
        for joint_name in TARGET_JOINTS:
            state = monitor.joints[joint_name]
            results["loading"][joint_name].append(state.cumulative_load)
            results["deficits"][joint_name].append(
                state.deficit_fraction * 100
            )
            results["torques"][joint_name].append(
                current_torques[joint_name]
            )

        # 6. LOG: Print progress
        if verbose and (step % 5 == 0 or step == n_updates - 1):
            print(monitor.summary_string())
            print(controller.summary_string(current_torques))

        # Early termination if all targets met
        if monitor.all_targets_met():
            if verbose:
                print(f"\n  ✓ All loading targets met at cycle {cycle_num}!")
            break

    # ── Add baseline results ────────────────────────────
    results["baseline_loading"] = {
        j: s.cumulative_load
        for j, s in baseline_monitor.joints.items()
    }

    # ── Final report ────────────────────────────────────
    if verbose:
        print("\n" + "=" * 60)
        print("  FINAL RESULTS")
        print("=" * 60)
        print(monitor.summary_string())
        print("\n  Comparison (end of day):")
        print(f"  {'Joint':<16} {'No Suit (N·s)':>14} {'With Suit (N·s)':>16} {'Improvement':>12}")
        print(f"  {'-'*58}")
        for j in TARGET_JOINTS:
            bl = results["baseline_loading"][j]
            ws = monitor.joints[j].cumulative_load
            tgt = DAILY_LOAD_TARGETS[j]
            improvement = ((ws - bl) / tgt) * 100 if tgt > 0 else 0
            label = TARGET_JOINTS[j]["label"]
            print(f"  {label:<16} {bl:>14.0f} {ws:>16.0f} {improvement:>+11.1f}%")
        print("=" * 60)

    # ── Save results ────────────────────────────────────
    # Convert numpy arrays for JSON serialisation
    serialisable = _make_serialisable(results)
    results_path = os.path.join(OUTPUT_DIR, "simulation_results.json")
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    if verbose:
        print(f"\n  Results saved to: {results_path}")

    return results


def _make_serialisable(obj):
    """Convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# ─── Entry point ────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Closed-Loop Exosuit Digital Twin Simulation"
    )
    parser.add_argument(
        "--cycles", type=int, default=CYCLES_PER_DAY,
        help=f"Total gait cycles to simulate (default: {CYCLES_PER_DAY})"
    )
    parser.add_argument(
        "--interval", type=int, default=CONTROL_UPDATE_INTERVAL,
        help=f"Control update interval in cycles (default: {CONTROL_UPDATE_INTERVAL})"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots after simulation"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    results = run_simulation(
        total_cycles=args.cycles,
        update_interval=args.interval,
        verbose=not args.quiet,
    )

    if args.plot:
        from visualise_results import plot_all
        plot_all(results)
