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
- Checkpoint saving for long-running VQE optimization

---

## Project Structure

quantum-vqe-nickel/
│── ni_complex_vqe.py
│── README.md
│── requirements.txt
│── .gitignore
text


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

---

## Methodology
1. Molecular System

A nickel coordination complex is defined with custom geometry:

    10 atoms: 1 Ni, 4 N, 2 C, 2 O, 1 additional C (axial ligand)

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

    Active space: (5, 1) electrons in 8 spatial orbitals

        5 active electrons

        1 active orbital (spatial)

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
text

DFT energy (B3LYP): -1967.42190073 Hartree (-53536.30 eV)

Note: SCF did not fully converge after 300 cycles (oscillations observed between -1966.8 and -1967.4 Hartree). The final reported value is from cycle 300.
VQE Calculation
text

Active space: (5, 1) electrons in 8 orbitals → 16 qubits
Ansatz parameters: 157
Initial VQE energy: -2.52766 Hartree

VQE optimization progressed through ~5500+ iterations with gradual energy improvement:
Stage	Iterations	Energy (Hartree)
Start	0	-2.52766
Intermediate	1000	~-2.5309
Intermediate	3000	~-2.53145
Intermediate	5000	~-2.53145
Final (interrupted)	5500+	~-2.53228

Final active-space VQE energy (at interruption): -2.53228 Hartree

    Note: The VQE run was interrupted manually after ~5500 iterations. The optimization was showing slow convergence but progressive improvement.

Energy Comparison
Quantity	Energy (Hartree)	Energy (eV)
DFT (full system)	-1967.42190	-53536.30
VQE (active space)	-2.53228	-68.91
Difference	-1964.88962	-53467.39

Important: Direct comparison between DFT (full system) and VQE (active space) is not meaningful, as they operate on different Hilbert spaces:

    DFT includes all 90 electrons in full basis

    VQE operates on reduced active space (5 electrons, 8 orbitals)

The VQE calculation is designed to solve the active-space Hamiltonian derived from the DFT mean-field. The large energy difference reflects the frozen core approximation.
Performance Notes

    Runtime: >2 hours (VQE interrupted at 5500+ iterations)

    Qubits used: 16

    DFT cycles: 300 (not fully converged)

    Platform: WSL2 (Ubuntu on Windows), 12 threads

Technical Observations
DFT Convergence Issues

    Persistent HOMO > LUMO warnings throughout optimization

    Oscillatory convergence behavior

    Failed to converge within 300 cycles

    Spin state appears to be singlet (2S+1 = 1.00) despite initial spin=2 setting

VQE Behavior

    Extremely slow convergence (~0.0001 Ha per 1000 iterations)

    Many iterations show no energy improvement (flat plateaus)

    Energy range explored: -2.527 to -2.532 Hartree

    157 parameters create a challenging optimization landscape

Limitations

    Minimal basis set (STO-3G) → limited chemical accuracy

    Active space approximation may miss important correlation

    DFT not fully converged → reference state may not be optimal

    No noise model (ideal statevector simulation)

    Not executed on real quantum hardware

    VQE convergence extremely slow for this system size

Future Improvements

    Use larger basis sets (e.g., def2-SVP, cc-pVDZ)

    Implement alternative active space selection (CASSCF-based)

    Add convergence plots for VQE trajectory

    Compare with classical methods (CCSD, CASSCF, DMRG)

    Experiment with different optimizers (SLSQP, L-BFGS-B, SPSA)

    Add early stopping criteria for VQE

    Implement better initial point guessing (e.g., HF orbitals)

    Jupyter Notebook visualization of results

    Reduce parameter count via qubit tapering or smarter ansatz

Applications

    Quantum chemistry simulations of transition metal complexes

    Benchmarking hybrid quantum-classical algorithms

    Educational demonstration of VQE workflow

    Exploration of active space approximations

Author

Aditya Singh
M.Sc. Applied Physics
Amity University, Lucknow
Environment Information
text

System: Linux (WSL2)
Python: 3.13.11
PySCF: 2.12.1
numpy: 2.4.3
scipy: 1.17.1
