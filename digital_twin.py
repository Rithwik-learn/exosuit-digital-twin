"""
Digital Twin Module (OpenSim-Driven)
=====================================
Wraps the Gait2354 musculoskeletal model and uses REAL
joint reaction force profiles from OpenSim's Joint Reaction
Analysis as the baseline for microgravity simulation.

Pipeline feeding this module:
    1. Scale Gait2354 to subject (done)
    2. RRA to reduce residuals (done, pre-computed in OutputReference)
    3. CMC to compute muscle forces (done, pre-computed)
    4. Joint Reaction Analysis (done, via Tools→Analyze)
    5. parse_jra.py → opensim_calibration.json + .sto templates

This replaces the literature-calibrated mock with actual
OpenSim-computed bone-loading forces.
"""

import os
import numpy as np

try:
    import opensim as osim
    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False

from config import (
    MODEL_PATH, JRA_FILE, TARGET_JOINTS,
    SIM_START_TIME, SIM_END_TIME, TIME_STEP,
    BODY_WEIGHT_N, PEAK_FORCE_EARTH,
    MICROGRAVITY_FORCE_FRACTION
)


def load_jra_force_profiles():
    """
    Load real joint reaction force profiles from the
    OpenSim JRA .sto output file.

    Returns
    -------
    dict : {joint_name: np.ndarray of force magnitudes (N)}
    """
    if not os.path.exists(JRA_FILE):
        print(f"[WARN] JRA file not found: {JRA_FILE}")
        return None

    # Read header
    with open(JRA_FILE, 'r') as f:
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.strip() == 'endheader':
                break
        col_line = f.readline().strip()
        columns = col_line.split('\t')

    # Load data
    data = np.loadtxt(JRA_FILE, skiprows=header_lines + 1)

    # Extract force magnitudes for each target joint
    profiles = {}
    for joint_name in TARGET_JOINTS:
        fx_idx = fy_idx = fz_idx = None
        for i, col in enumerate(columns):
            cl = col.lower()
            if joint_name in cl and '_fx' in cl:
                fx_idx = i
            elif joint_name in cl and '_fy' in cl:
                fy_idx = i
            elif joint_name in cl and '_fz' in cl:
                fz_idx = i
        if fx_idx and fy_idx and fz_idx:
            mag = np.sqrt(data[:, fx_idx]**2
                          + data[:, fy_idx]**2
                          + data[:, fz_idx]**2)
            profiles[joint_name] = mag

    # The CMC window only covers the right-leg stance phase,
    # so mirror right-leg profiles to the left for simulation.
    for r, l in [("hip_r", "hip_l"), ("knee_r", "knee_l"), ("ankle_r", "ankle_l")]:
        if r in profiles and (l not in profiles or np.max(profiles[l]) < 100):
            # Phase-shift the right-leg profile by 50% for the left
            n = len(profiles[r])
            profiles[l] = np.roll(profiles[r], n // 2)

    return profiles


class DigitalTwin:
    """
    Real-time digital twin driven by OpenSim JRA output.

    The Earth-gravity force profile is the exact waveform
    computed by OpenSim from the Gait2354 model + CMC muscle
    forces. Microgravity scaling and exosuit contributions
    are applied on top of this validated baseline.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.state = None
        self.earth_profiles = load_jra_force_profiles()

        if self.earth_profiles:
            print(f"[DigitalTwin] Loaded OpenSim force profiles for "
                  f"{len(self.earth_profiles)} joints")
        else:
            print(f"[DigitalTwin] OpenSim profiles unavailable — "
                  f"falling back to literature-calibrated model")

        if OPENSIM_AVAILABLE and os.path.exists(model_path):
            self._load_model()

    def _load_model(self):
        """Load Gait2354 and initialise the system."""
        try:
            self.model = osim.Model(self.model_path)
            self.model.setName("ExosuitDigitalTwin_Gait2354")
            self.state = self.model.initSystem()
            print(f"[DigitalTwin] Model: {self.model.getName()}")
            print(f"[DigitalTwin] DOF: {self.model.getNumCoordinates()}, "
                  f"Muscles: {self.model.getMuscles().getSize()}")
        except Exception as e:
            print(f"[DigitalTwin] Model load failed: {e}")

    def compute_joint_reaction_forces(self, kinematics_file=None,
                                       external_loads=None) -> dict:
        """
        Compute joint reaction forces for one gait cycle.

        The baseline is the real OpenSim JRA output, resampled
        to the simulation time step. Microgravity reduces this
        to 6%, and exosuit torques add muscle-driven load.

        Parameters
        ----------
        external_loads : dict, optional
            {joint_name: torque_magnitude_Nm} from the exosuit.

        Returns
        -------
        dict : {joint_name: np.ndarray of force magnitudes (N)}
        """
        t = np.arange(SIM_START_TIME, SIM_END_TIME, TIME_STEP)
        n = len(t)
        results = {}

        for joint_name in TARGET_JOINTS:
            # Earth-gravity template from OpenSim
            if self.earth_profiles and joint_name in self.earth_profiles:
                earth_profile = self.earth_profiles[joint_name]
                # Resample to match simulation length
                if len(earth_profile) != n:
                    idx = np.linspace(0, len(earth_profile) - 1, n)
                    earth_profile = np.interp(idx,
                                              np.arange(len(earth_profile)),
                                              earth_profile)
            else:
                # Fallback literature waveform
                earth_profile = self._synthetic_profile(joint_name, t)

            # Microgravity baseline: 6% of Earth, same temporal shape
            micro_forces = earth_profile * MICROGRAVITY_FORCE_FRACTION
            micro_forces += 0.01 * BODY_WEIGHT_N * np.random.randn(n)
            micro_forces = np.maximum(micro_forces, 0)

            forces = micro_forces.copy()

            # Exosuit contribution scales with torque
            if external_loads and joint_name in external_loads:
                torque = external_loads[joint_name]
                if torque > 0:
                    moment_arms = {
                        "hip_r": 0.055, "hip_l": 0.055,
                        "knee_r": 0.045, "knee_l": 0.045,
                        "ankle_r": 0.050, "ankle_l": 0.050,
                    }
                    ma = moment_arms.get(joint_name, 0.05)
                    cocontraction = 1.8

                    muscle_force = (torque / ma) * cocontraction
                    shape = earth_profile / (np.max(earth_profile) + 1e-6)
                    forces += muscle_force * shape

            results[joint_name] = forces

        return results

    def _synthetic_profile(self, joint_name, t):
        """Fallback if OpenSim profiles unavailable."""
        cycle_time = t[-1] - t[0] if len(t) > 1 else 1.0
        phase = 2 * np.pi * (t - t[0]) / cycle_time
        peak = PEAK_FORCE_EARTH.get(joint_name, 2000.0)
        profile = (
            0.55 * np.exp(-((phase - 0.75)**2) / 0.12)
            + 0.45 * np.exp(-((phase - 2.25)**2) / 0.10)
            + 0.05
        )
        profile *= peak / np.max(profile)
        return profile

    def get_model_info(self) -> dict:
        """Return summary information about the loaded model."""
        if OPENSIM_AVAILABLE and self.model:
            return {
                "name": self.model.getName(),
                "n_bodies": self.model.getBodySet().getSize(),
                "n_coordinates": self.model.getNumCoordinates(),
                "n_muscles": self.model.getMuscles().getSize(),
                "data_source": "OpenSim JRA (Gait2354)",
            }
        return {
            "name": "Gait2354 (OpenSim-calibrated)",
            "data_source": "JRA .sto file",
        }
