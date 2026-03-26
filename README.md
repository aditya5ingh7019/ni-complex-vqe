# Quantum VQE Simulation of Nickel Complex

## Overview
This project implements a **Variational Quantum Eigensolver (VQE)** simulation to estimate the ground-state energy of a nickel-based coordination complex using **Qiskit Nature** and **PySCF**.

The workflow combines **classical quantum chemistry (DFT)** with **quantum-inspired optimization (VQE)**, making it a hybrid quantum-classical simulation.

---

## Features
- Density Functional Theory (DFT) baseline using **B3LYP**
- Active space reduction for computational feasibility
- Fermion-to-qubit mapping using **Parity Mapper**
- **UCC (Unitary Coupled Cluster)** ansatz
- Optimization via **COBYLA**
- Energy comparison between DFT and VQE results

---

## Project Structure

quantum-vqe-nickel/
│── ni_complex_vqe.py
│── README.md
│── requirements.txt
│── .gitignore


---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
Dependencies
numpy
pyscf
qiskit
qiskit-nature
qiskit-algorithms

Methodology
1. Molecular System

A nickel complex is defined with a custom geometry and simulated using the STO-3G basis set.

2. Classical Baseline (DFT)
Method: B3LYP functional
Used to compute reference ground-state energy
3. Active Space Reduction
Core orbitals frozen
Reduced system:
6 electrons
8 spatial orbitals
16 qubits
4. Quantum Simulation (VQE)
Ansatz: UCC (Singles + Doubles)
Mapper: Parity Mapping
Optimizer: COBYLA
Backend: Qiskit Estimator primitive


The script prints:

DFT Energy (Hartree & eV)
VQE Ground State Energy
Energy Difference (DFT - VQE)
Optimizer statistics (iterations, final value)

Results (Expected)

Running DFT (B3LYP)...
DFT energy: -1234.56789012 Hartree (-33600.00 eV)

Active space: 6 electrons in 8 orbitals → 16 qubits

Starting VQE on active space (this may take 10-30+ min)...

============================================================
RESULTS
============================================================
Active-space VQE energy : -1234.56789012 Hartree
Full-system DFT energy : -1234.56789012 Hartree
Difference (DFT - VQE) : 0.00000000 Hartree


Performance Notes
Runtime: 30 minutes to 1 hour (CPU, depending on system)
Qubits used: 16
Optimizer iterations: up to 500
Limitations
Uses minimal basis set (STO-3G) → limited accuracy
No noise model (ideal simulation)
Small active space approximation
Not executed on real quantum hardware
Future Improvements
Use larger basis sets (e.g., cc-pVDZ)
Implement noise models / hardware backend
Add convergence plots for VQE
Compare with classical methods (CCSD, FCI)
Parameter optimization tuning
Jupyter Notebook visualization
Applications
Quantum chemistry simulations
Transition metal complex modeling
Benchmarking hybrid quantum algorithms
Educational demonstration of VQE workflow
Author

Aditya Singh
M.Sc. Applied Physics
Amity University, Lucknow
