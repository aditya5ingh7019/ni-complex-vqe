# Quantum VQE Simulation of Nickel Complex

## Overview
This project implements a **Variational Quantum Eigensolver (VQE)** simulation to estimate the ground-state energy of a nickel-based coordination complex using **Qiskit Nature** and **PySCF**.

The workflow combines classical quantum chemistry (DFT) with quantum-inspired optimization (VQE), making it a hybrid quantum-classical simulation.

---

## Features
- Density Functional Theory (DFT) baseline using B3LYP
- Active space reduction for computational feasibility
- Fermion-to-qubit mapping using Parity Mapper
- UCC (Unitary Coupled Cluster) ansatz
- Optimization via COBYLA
- Energy comparison between DFT and VQE results
- Checkpoint saving for long-running VQE optimization

---

## Project Structure

```bash
quantum-vqe-nickel/
├── ni_complex_vqe.py
├── README.md
├── requirements.txt
└── .gitignore

```

## Dependencies

- numpy
- pyscf
- qiskit
- qiskit-nature
- qiskit-algorithms

---

# Methodology

## 1. Molecular System

A nickel coordination complex is defined with custom geometry.

### System Details

- Atoms: 10 atoms
  - 1 Ni
  - 4 N
  - 2 C
  - 2 O
  - 1 additional axial ligand atom
- Total electrons: 90
- Charge: 0
- Spin (2S): 2 (triplet state)
- Basis set: STO-3G (minimal basis)

---

## 2. Classical Baseline (DFT)

### DFT Configuration

- Method: B3LYP functional
- Auxiliary basis: `def2-svp-jkfit`
- Density fitting enabled

### Grid Configuration

- Treutler-Ahlrichs radial grids
- Becke partitioning

### SCF Settings

- SCF cycles: 300

### Observations

- SCF did not fully converge
- Oscillatory convergence behavior observed
- Energy oscillated approximately between:
  - `-1966.8 Ha`
  - `-1967.4 Ha`

Final reported energy corresponds to the last completed SCF cycle.

---

## 3. Active Space Reduction

To make the quantum simulation computationally feasible:

- Core orbitals were frozen

### Active Space Selected

- 5 active electrons
- 8 spatial orbitals

### Qubit Requirement

- 16 qubits

---

## 4. Quantum Simulation (VQE)

### Configuration

- Ansatz: UCC (Unitary Coupled Cluster Singles + Doubles)
- Mapper: Parity Mapper
- Optimizer: COBYLA
- Backend: Qiskit Estimator primitive
- Simulation type: Statevector simulation
- Variational parameters: 157
- Checkpointing: Enabled

---

# Results

## DFT Calculation

| Quantity | Value |
|---|---|
| Method | B3LYP |
| Energy (Hartree) | -1967.42190073 |
| Energy (eV) | -53536.30 |

> **Note:**  
> SCF did not fully converge after 300 cycles.  
> Final value corresponds to the last completed cycle.

---

## VQE Calculation

### Active Space

- Electrons: 5
- Orbitals: 8
- Qubits: 16

### Ansatz Information

- Parameters: 157

### Initial Energy

```text
-2.52766 Hartree
```

---

## Optimization Progress

| Stage | Iterations | Energy (Hartree) |
|---|---|---|
| Start | 0 | -2.52766 |
| Intermediate | ~1000 | ~-2.53090 |
| Intermediate | ~3000 | ~-2.53145 |
| Intermediate | ~5000 | ~-2.53145 |
| Final (Interrupted) | 5500+ | ~-2.53228 |

---

## Final Active-Space Energy

```text
-2.53228 Hartree
```

> **Note:**  
> The VQE optimization was manually interrupted after approximately 5500 iterations.  
> Optimization showed slow but consistent improvement.

---

# Energy Comparison

| Quantity | Energy (Hartree) | Energy (eV) |
|---|---|---|
| DFT (full system) | -1967.42190 | -53536.30 |
| VQE (active space) | -2.53228 | -68.91 |
| Difference | -1964.88962 | -53467.39 |

> **Important:**  
> Direct comparison between full-system DFT energy and active-space VQE energy is not physically meaningful because they correspond to different Hilbert spaces and Hamiltonians.

---

# Performance Notes

| Parameter | Value |
|---|---|
| Runtime | >2 hours |
| VQE iterations | 5500+ |
| Qubits used | 16 |
| DFT cycles | 300 |
| Platform | WSL2 (Ubuntu on Windows) |
| Threads | 12 |

---

# Technical Observations

## DFT Convergence Issues

- Persistent HOMO > LUMO warnings
- Oscillatory convergence behavior
- Failed to converge within 300 cycles
- Spin state appeared as singlet despite triplet initialization

---

## VQE Optimization Behavior

- Extremely slow convergence
- Improvement rate approximately:
  - `~0.0001 Ha per 1000 iterations`
- Multiple flat optimization plateaus observed
- Large parameter count created a difficult optimization landscape

---

# Limitations

- Minimal STO-3G basis limits chemical accuracy
- Active-space approximation may neglect important correlations
- DFT not fully converged
- Ideal noiseless simulation only
- Not executed on real quantum hardware

---

# Future Improvements

Potential future enhancements include:

## Larger Basis Sets

- `def2-SVP`
- `cc-pVDZ`

## Methodological Improvements

- CASSCF-based active space selection
- VQE convergence visualization

## Benchmark Comparisons

- CCSD
- CASSCF
- DMRG

## Alternative Optimizers

- SLSQP
- L-BFGS-B
- SPSA

## Additional Optimizations

- Improved initial parameter guessing
- Qubit tapering
- Parameter reduction strategies

---

# Applications

- Quantum chemistry simulation of transition-metal complexes
- Benchmarking hybrid quantum-classical algorithms
- Educational demonstrations of VQE workflows
- Exploration of NISQ-era quantum chemistry methods

---

# Environment Information

| Component | Version |
|---|---|
| System | Linux (WSL2) |
| Python | 3.13.11 |
| PySCF | 2.12.1 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |

---

# Author

**Aditya Singh**  
M.Sc. Applied Physics  
Amity University, Lucknow
