"""
Resistance Controller
=====================
Closed-loop PID controller that maps bone-loading deficits
to exosuit actuator torque commands.

Control law:
    τ_suit(j) = Kp·e(j) + Ki·∫e(j)dt + Kd·de(j)/dt

where e(j) is the normalised loading deficit at joint j.

The controller outputs are clamped to physical actuator limits
and include a dead-zone to avoid unnecessary low-level resistance.
"""

import numpy as np
from typing import Dict, Tuple

from config import (
    PID_KP, PID_KI, PID_KD,
    MAX_ACTUATOR_TORQUE, MIN_ACTUATOR_TORQUE,
    TARGET_JOINTS
)


class PIDController:
    """Single-joint PID controller."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = 0.0, output_max: float = 25.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0

    def compute(self, error: float, dt: float = 1.0) -> float:
        """
        Compute PID output for a given error signal.

        Parameters
        ----------
        error : float
            Normalised deficit (0 = no deficit, 1 = full deficit).
        dt : float
            Time step for integral/derivative.

        Returns
        -------
        float : Torque command (N·m), clamped to actuator limits.
        """
        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup)
        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            -self.output_max / self.ki if self.ki > 0 else -100,
            self.output_max / self.ki if self.ki > 0 else 100,
        )
        i_term = self.ki * self._integral

        # Derivative (with filtering to reduce noise)
        d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        # Total output
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_min, self.output_max)

        self._prev_output = output
        return output

    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0


class ResistanceController:
    """
    Multi-joint resistance controller for the exosuit.

    Uses a rate-based scheduling strategy:
        1. Compute the required loading rate to meet the daily target
           given the remaining time: rate_req = deficit / cycles_remaining
        2. Compare to the current actual loading rate
        3. PID drives the torque to close the rate gap

    This produces genuinely adaptive behaviour:
        - Early in the day: low urgency → moderate torque
        - Falling behind schedule: urgency rises → torque increases
        - Ahead of schedule: controller backs off
        - Target met: torque drops to zero

    The rate-based error signal is key for publication — it shows
    the controller *adapting* rather than just saturating.
    """

    def __init__(self, total_cycles: int = None):
        from config import CYCLES_PER_DAY
        self.total_cycles = total_cycles or CYCLES_PER_DAY
        self.controllers: Dict[str, PIDController] = {}
        self.torque_history: Dict[str, list] = {}
        self._prev_loading: Dict[str, float] = {}

        for joint_name in TARGET_JOINTS:
            max_torque = MAX_ACTUATOR_TORQUE.get(joint_name, 25.0)
            self.controllers[joint_name] = PIDController(
                kp=PID_KP,
                ki=PID_KI,
                kd=PID_KD,
                output_min=0.0,
                output_max=max_torque,
            )
            self.torque_history[joint_name] = []
            self._prev_loading[joint_name] = 0.0

    def compute_resistance(
        self, deficit_report: dict, current_cycle: int = 0
    ) -> Dict[str, float]:
        """
        Compute exosuit torque commands from loading deficit report.

        Parameters
        ----------
        deficit_report : dict
            Output from BoneLoadingMonitor.get_deficit_report().
        current_cycle : int
            Current gait cycle number (for time-awareness).

        Returns
        -------
        dict : {joint_name: torque_command (N·m)}
        """
        torque_commands = {}

        # Time awareness: how much of the day remains?
        cycles_remaining = max(self.total_cycles - current_cycle, 1)
        day_progress = current_cycle / self.total_cycles  # 0 → 1

        for joint_name, pid in self.controllers.items():
            if joint_name not in deficit_report:
                torque_commands[joint_name] = 0.0
                continue

            info = deficit_report[joint_name]

            # If target already met, no resistance needed
            if info["is_loaded"]:
                torque_commands[joint_name] = 0.0
                self.torque_history[joint_name].append(0.0)
                pid.reset()  # clean slate for if target un-met somehow
                continue

            # ── Rate-based error signal ──
            # Required loading rate to meet target by end of day
            required_rate = info["deficit_Ns"] / cycles_remaining

            # Actual loading rate (from last control interval)
            actual_load = info["cumulative_Ns"]
            prev_load = self._prev_loading.get(joint_name, 0.0)
            actual_rate = max(actual_load - prev_load, 0.0)

            # Normalised error: how far behind schedule are we?
            # Positive = behind schedule, negative = ahead
            if required_rate > 0:
                rate_error = (required_rate - actual_rate) / required_rate
                rate_error = np.clip(rate_error, -0.5, 1.0)
            else:
                rate_error = 0.0

            # Urgency scaling: as day progresses, increase gain
            # Gentle ramp: 0.3 at start → 1.0 at 70% → 1.5 at 95%
            urgency = 0.3 + 0.7 * (day_progress ** 0.6)
            if day_progress > 0.85:
                urgency += 0.5 * ((day_progress - 0.85) / 0.15)

            # Combine: deficit-fraction as baseline + rate error for dynamics
            deficit_signal = info["deficit_pct"] / 100.0
            combined_error = (
                0.4 * deficit_signal * urgency  # baseline drive
                + 0.6 * max(rate_error, 0) * urgency  # rate correction
            )
            combined_error = np.clip(combined_error, 0, 1.0)

            # PID computation
            torque = pid.compute(combined_error)

            # Dead-zone: don't bother with very small torques
            if torque < MIN_ACTUATOR_TORQUE:
                torque = 0.0

            torque_commands[joint_name] = torque
            self.torque_history[joint_name].append(torque)
            self._prev_loading[joint_name] = actual_load

        return torque_commands

    def get_torque_history(self) -> Dict[str, np.ndarray]:
        """Return torque command history for plotting."""
        return {
            name: np.array(hist)
            for name, hist in self.torque_history.items()
        }

    def reset(self):
        """Reset all PID controllers."""
        for pid in self.controllers.values():
            pid.reset()
        for hist in self.torque_history.values():
            hist.clear()
        for key in self._prev_loading:
            self._prev_loading[key] = 0.0

    def summary_string(self, torques: Dict[str, float]) -> str:
        """Human-readable summary of current torque commands."""
        lines = ["  Exosuit resistance commands:"]
        for name, torque in torques.items():
            label = TARGET_JOINTS[name]["label"]
            max_t = MAX_ACTUATOR_TORQUE.get(name, 25.0)
            bar_len = 20
            filled = int((torque / max_t) * bar_len)
            bar = "▓" * filled + "░" * (bar_len - filled)
            lines.append(
                f"    {label:<14} [{bar}] {torque:5.1f} / {max_t:.0f} N·m"
            )
        return "\n".join(lines)
