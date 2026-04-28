"""
Bone Loading Monitor
====================
Tracks cumulative mechanical loading at each target joint
and computes deficits relative to Earth-baseline thresholds.

The key metric is the force-time integral (N·s):
    L(t) = ∫₀ᵗ |F_reaction(τ)| dτ

This is a proxy for the mechanical stimulus that drives
bone remodelling via Wolff's Law.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

# numpy compatibility: trapz was renamed to trapezoid in numpy 2.0
_trapz = getattr(np, 'trapz', None) or getattr(np, 'trapezoid')

from config import (
    TARGET_JOINTS, DAILY_LOAD_TARGETS,
    TIME_STEP,
    SIM_START_TIME, SIM_END_TIME,
    CONVERGENCE_THRESHOLD,
)


@dataclass
class JointLoadingState:
    """Tracks the loading state for a single joint."""
    joint_name: str
    cumulative_load: float = 0.0        # N·s accumulated so far
    daily_target: float = 0.0           # N·s target for the day
    deficit: float = 0.0                # N·s remaining to target
    deficit_fraction: float = 1.0       # 0 = fully loaded, 1 = no loading
    is_loaded: bool = False             # True if target met
    history: List[float] = field(default_factory=list)


class BoneLoadingMonitor:
    """
    Monitors cumulative bone loading across all target joints.

    At each control update, it:
        1. Integrates the latest joint reaction forces over time
        2. Adds the integral to the running cumulative total
        3. Computes the deficit relative to the daily target
        4. Reports which joints need more loading
    """

    def __init__(self, microgravity: bool = True):
        self.microgravity = microgravity
        self.joints: Dict[str, JointLoadingState] = {}
        self.cycle_count = 0

        # Initialise tracking for each joint
        for joint_name, info in TARGET_JOINTS.items():
            # DAILY_LOAD_TARGETS already includes restoration fraction
            target = DAILY_LOAD_TARGETS[joint_name]
            self.joints[joint_name] = JointLoadingState(
                joint_name=joint_name,
                daily_target=target,
                deficit=target,
                deficit_fraction=1.0,
            )

    def update(
        self, joint_forces: Dict[str, np.ndarray], n_cycles: int = 1
    ) -> Dict[str, JointLoadingState]:
        """
        Update cumulative loading from new joint reaction force data.

        Parameters
        ----------
        joint_forces : dict
            {joint_name: np.ndarray of force magnitudes (N)} for one cycle.
        n_cycles : int
            How many gait cycles this data represents.

        Returns
        -------
        dict : Updated JointLoadingState for each joint.
        """
        self.cycle_count += n_cycles

        for joint_name, forces in joint_forces.items():
            if joint_name not in self.joints:
                continue

            state = self.joints[joint_name]

            # Compute force-time integral for this cycle (trapezoidal rule)
            dt = TIME_STEP
            cycle_load = _trapz(np.abs(forces), dx=dt)

            # Note: microgravity scaling is applied upstream in the
            # force computation (DigitalTwin), not here. The monitor
            # integrates whatever forces it receives.

            # Scale by number of cycles
            total_new_load = cycle_load * n_cycles

            # Update cumulative state
            state.cumulative_load += total_new_load
            state.deficit = max(0, state.daily_target - state.cumulative_load)
            state.deficit_fraction = state.deficit / state.daily_target
            state.is_loaded = (
                state.cumulative_load >= CONVERGENCE_THRESHOLD * state.daily_target
            )
            state.history.append(state.cumulative_load)

        return self.joints

    def get_deficit_report(self) -> dict:
        """
        Return a summary of loading deficits for the controller.

        Returns
        -------
        dict : {joint_name: {
            'deficit_Ns': float,
            'deficit_pct': float,
            'cumulative_Ns': float,
            'target_Ns': float,
            'is_loaded': bool,
        }}
        """
        report = {}
        for name, state in self.joints.items():
            report[name] = {
                "deficit_Ns": state.deficit,
                "deficit_pct": state.deficit_fraction * 100,
                "cumulative_Ns": state.cumulative_load,
                "target_Ns": state.daily_target,
                "is_loaded": state.is_loaded,
            }
        return report

    def get_priority_joints(self) -> List[str]:
        """
        Return joints sorted by deficit (worst first).
        Only returns joints that haven't met their target.
        """
        unloaded = [
            (name, state.deficit_fraction)
            for name, state in self.joints.items()
            if not state.is_loaded
        ]
        unloaded.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in unloaded]

    def all_targets_met(self) -> bool:
        """Check if all joints have met their daily loading targets."""
        return all(s.is_loaded for s in self.joints.values())

    def reset_daily(self):
        """Reset cumulative loading for a new simulated day."""
        for state in self.joints.values():
            state.cumulative_load = 0.0
            state.deficit = state.daily_target
            state.deficit_fraction = 1.0
            state.is_loaded = False
            state.history.clear()
        self.cycle_count = 0

    def summary_string(self) -> str:
        """Human-readable summary for logging."""
        lines = [f"\n{'='*60}"]
        lines.append(f"  Bone Loading Report — Cycle {self.cycle_count}")
        lines.append(f"{'='*60}")
        for name, state in self.joints.items():
            label = TARGET_JOINTS[name]["label"]
            bar_len = 30
            filled = int((1 - state.deficit_fraction) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            pct = (1 - state.deficit_fraction) * 100
            status = "✓" if state.is_loaded else "○"
            lines.append(
                f"  {status} {label:<14} [{bar}] "
                f"{pct:5.1f}%  ({state.cumulative_load:.0f}/{state.daily_target:.0f} N·s)"
            )
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)
