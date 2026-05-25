# Quantum VQE Simulation of Nickel Complex

## Overview
This project implements a **Variational Quantum Eigensolver (VQE)** simulation to estimate the ground-state energy of a nickel-based coordination complex using **Qiskit Nature** and **PySCF**.

The workflow combines classical quantum chemistry (DFT) with quantum-inspired optimization (VQE), making it a hybrid quantum-classical simulation.

## Features
- Density Functional Theory (DFT) baseline using B3LYP
- Active space reduction for computational feasibility
- Fermion-to-qubit mapping using Parity Mapper
- UCC (Unitary Coupled Cluster) ansatz
- Optimization via COBYLA
- Energy comparison between DFT and VQE results
- Checkpoint saving for long-running VQE optimization

## Project Structure
quantum-vqe-nickel/
├── ni_complex_vqe.py
├── README.md
├── requirements.txt
└── .gitignore
text## Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
Dependencies

numpy
pyscf
qiskit
qiskit-nature
qiskit-algorithms


## Methodology
1. Molecular System
A nickel coordination complex is defined with custom geometry:

Atoms: 10 atoms (1 Ni, 4 N, 2 C, 2 O, 1 additional C axial ligand)
Total electrons: 90
Charge: 0
Spin (2S): 2 (triplet state)
Basis set: STO-3G (minimal basis)

2. Classical Baseline (DFT)

Method: B3LYP functional
Auxiliary basis: def2-svp-jkfit for density fitting
Grid: Treutler-Ahlrichs radial grids, Becke partitioning
SCF convergence: Not fully converged after 300 cycles (oscillatory behavior observed)

3. Active Space Reduction

Core orbitals frozen to reduce system size
Active space: (5, 1) → 5 active electrons in 8 spatial orbitals
Qubits required: 16 qubits

4. Quantum Simulation (VQE)

Ansatz: UCC (Unitary Coupled Cluster Singles + Doubles)
Mapper: Parity Mapping
Optimizer: COBYLA
Backend: Qiskit Estimator primitive (statevector simulator)
Parameters: 157 variational parameters
Checkpointing: Enabled for long runs

Results
DFT Calculation
DFT energy (B3LYP): -1967.42190073 Hartree (-53536.30 eV)
Note: SCF did not fully converge after 300 cycles (oscillations observed between -1966.8 and -1967.4 Hartree). The final reported value is from cycle 300.
VQE Calculation

Active space: (5, 1) electrons in 8 orbitals → 16 qubits
Ansatz parameters: 157
Initial VQE energy: -2.52766 Hartree

VQE optimization progressed through ~5500+ iterations:



































StageIterationsEnergy (Hartree)Start0-2.52766Intermediate1000~-2.5309Intermediate3000~-2.53145Intermediate5000~-2.53145Final (interrupted)5500+~-2.53228
Final active-space VQE energy (at interruption): -2.53228 Hartree
Note: The VQE run was interrupted manually after ~5500 iterations. The optimization showed slow but progressive improvement.
Energy Comparison

























QuantityEnergy (Hartree)Energy (eV)DFT (full system)-1967.42190-53536.30VQE (active space)-2.53228-68.91Difference-1964.88962-53467.39
Important: Direct comparison between full-system DFT and active-space VQE is not meaningful as they operate on different Hilbert spaces.
Performance Notes

Runtime: >2 hours (VQE interrupted at 5500+ iterations)
Qubits used: 16
DFT cycles: 300 (not fully converged)
Platform: WSL2 (Ubuntu on Windows), 12 threads

Technical Observations
DFT Convergence Issues

Persistent HOMO > LUMO warnings
Oscillatory convergence behavior
Failed to converge within 300 cycles
Spin state appeared as singlet despite initial triplet setting

VQE Behavior

Extremely slow convergence (~0.0001 Ha per 1000 iterations)
Many flat plateaus in energy
157 parameters create a very challenging optimization landscape

Limitations

Minimal basis set (STO-3G) → limited chemical accuracy
Active space approximation may miss important correlation
DFT not fully converged
Ideal statevector simulation (no noise)
Not executed on real quantum hardware

Future Improvements

Use larger basis sets (def2-SVP, cc-pVDZ, etc.)
CASSCF-based active space selection
Convergence plots for VQE
Comparison with CCSD, CASSCF, DMRG
Try different optimizers (SLSQP, L-BFGS-B, SPSA)
Better initial point guessing
Qubit tapering / parameter reduction

Applications

Quantum chemistry simulations of transition metal complexes
Benchmarking hybrid quantum-classical algorithms
Educational demonstration of VQE workflow

Author
Aditya Singh
M.Sc. Applied Physics
Amity University, Lucknow
Environment Information

System: Linux (WSL2)
Python: 3.13.11
PySCF: 2.12.1
numpy: 2.4.3
scipy: 1.17.1
