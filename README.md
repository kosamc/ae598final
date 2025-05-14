# Mars EDL Trajectory Optimization

This MATLAB code optimizes the Entry Flight Path Angle (EFPA) for Mars Entry, Descent, and Landing (EDL) trajectories using a three-phase simulation approach.

## Key Features
- EFPA optimization using `fmincon`
- -EFPA optimization using the Hamiltonian
- Three-phase EDL simulation:
  - Hypersonic entry
  - Parachute descent
  - Powered landing
- Hamiltonian validation for optimal control
- Monte Carlo robustness analysis
- Comprehensive visualization of results

## Requirements
- MATLAB (R2020a or later)
- Optimization Toolbox

## Usage
Run `Optimal_EFPA_Mars_EDL_Simulation()` to:
1. Optimize EFPA for minimum heat flux and acceleration
2. Validate with Hamiltonian approach
3. Perform Monte Carlo analysis
4. Generate trajectory plots

## Outputs
- Optimal EFPA angle
- Heat flux and acceleration profiles
- Landing accuracy metrics
- Phase transition analysis
- Robustness evaluation

Developed for Mars landing mission analysis and optimization studies.
