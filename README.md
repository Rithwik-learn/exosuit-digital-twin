# Closed-Loop Robotic Exosuit Digital Twin for Bone Loading Restoration in Microgravity
A digital twin system that integrates an OpenSim musculoskeletal model with a closed-loop PID controller
to dynamically adjust exosuit resistance torques, restoring bone loading during routine microgravity
activities.
## Research Summary
Astronauts lose 1–2% bone density per month in microgravity. Current countermeasures (ARED) require
2.5 hours of daily exercise. This project proposes a wearable exosuit that turns routine orbital movements
into bone-loading events using real-time feedback from a musculoskeletal digital twin.
**Key Results:**- Hip loading restored to **64%** of Earth-gravity target- Knee loading restored to **50%**- Ankle loading restored to **36%**- Improvement of **24–52%** over unassisted microgravity baseline
## Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  OpenSim Model   │────▶│  Bone Loading     │────▶│  PID Controller  │
│  (Gait2354 JRA)  │     │  Monitor          
│     │  (Rate-based)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
│
┌─────────────────┐         
│  Exosuit         
│  Actuators       
│◀────────┘
│
└─────────────────┘
```
## Project Structure
```
│
├── config.py                  # All tunable parameters, loads OpenSim calibration
├── digital_twin.py            # OpenSim-driven force model (reads JRA .sto file)
├── bone_loading_monitor.py    # Tracks cumulative force-time integral per joint
├── controller.py              # Rate-based PID with time-aware urgency
├── simulation.py              # Main simulation loop (6000 gait cycles)
├── visualise_results.py       # Generates 6 publication figures
├── parse_jra.py               # Parses OpenSim JRA output → calibration JSON
├── testing.py                 # OpenSim Python API verification
├── run_baseline.py            # Baseline force analysis script
│
├── opensim_calibration.json   # Real OpenSim-derived force calibration
├── Models/
│   └── subject01_simbody.osim # Scaled Gait2392 model
│
├── JRA_Results/
│   └── subject01_JointReaction_ReactionLoads.sto  # OpenSim JRA output (7MB)
│
├── results/                   # Generated figures and simulation output
│   ├── fig1_cumulative_loading.png
│   ├── fig2_deficit_convergence.png
│   ├── fig3_torque_profiles.png
│   ├── fig4_summary_comparison.png
│   ├── fig5_single_cycle_waveforms.png
│   ├── fig6_controller_dynamics.png
│   └── simulation_results.json
│
└── paper/
└── Exosuit_Digital_Twin_Paper.docx
```
## OpenSim Pipeline
The digital twin uses the validated Gait2354 model from OpenSim 4.5:
1. **Model Scaling** — Gait2354 scaled to subject (72.6 kg) using static trial markers
2. **Residual Reduction (RRA)** — Adjusted kinematics and mass properties
3. **Computed Muscle Control (CMC)** — 54 individual muscle force trajectories
4. **Joint Reaction Analysis (JRA)** — Bone-loading forces at hip, knee, ankle
Peak forces from the digital twin:
| Joint | Peak Force (N) | Peak (BW) |
|-------|---------------|-----------|
| Hip   | 2,460         | 3.45      |
| Knee  | 3,548         | 4.98      |
| Ankle | 3,995         | 5.61      |
## Requirements- Python 3.8 (required for OpenSim bindings)- OpenSim 4.5- NumPy, Matplotlib
## Usage
### 1. Parse OpenSim results (generates calibration)
```bash
python parse_jra.py
```
### 2. Run the closed-loop simulation
```bash
python simulation.py --cycles 6000 --interval 100 --plot
```
### 3. Output- 6 publication-quality figures in `results/`- `simulation_results.json` with full numerical data- Console output with real-time loading progress
## Key Finding
Current soft exosuit actuators (15–25 N·m) are insufficient for full bone loading restoration. The
simulation quantifies the required torques: **40–60 N·m per joint**. This gap can be addressed through
passive elastic elements, pulsed loading protocols, or hybrid exosuit-ARED approaches.
## References- Bergmann et al. (2001) — In-vivo hip contact forces- Delp et al. (2007) — OpenSim platform- D'Lima et al. (2006) — In-vivo knee forces- Taylor et al. (2004) — Tibio-femoral loading- Stansfield et al. (2003) — Ankle JRF validation