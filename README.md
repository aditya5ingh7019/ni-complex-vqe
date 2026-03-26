# Nickel Complex VQE Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.45+-purple.svg)](https://qiskit.org/)
[![PySCF](https://img.shields.io/badge/PySCF-2.0+-green.svg)](https://pyscf.org/)

Quantum computing simulation of a nickel complex using VQE (Variational Quantum Eigensolver) with Qiskit Nature and PySCF.

## Overview

This project demonstrates the application of quantum computing algorithms to computational chemistry problems, specifically calculating the ground state energy of a transition metal complex using the Variational Quantum Eigensolver (VQE). The nickel complex serves as an example of a system with significant electron correlation, making it suitable for quantum computational methods.

## Features

- **Hybrid quantum-classical approach**: Combines classical DFT with quantum VQE
- **Active space selection**: Reduces problem size to 16 qubits for practical simulation
- **UCCSD ansatz**: Uses unitary coupled cluster with single and double excitations
- **DFT baseline**: Includes B3LYP calculations for comparison
- **Modular code structure**: Easy to modify for different molecules or active spaces

## Molecular Structure

The complex features a nickel center with equatorial nitrogen ligands and axial oxygen ligands:

Ni 0.0 0.0 0.0 # Central nickel atom
N 1.9 0.0 0.0 # Equatorial nitrogens
N -1.9 0.0 0.0
N 0.0 1.9 0.0
N 0.0 -1.9 0.0
C 2.6 2.6 0.0 # Carbon atoms
C -2.6 2.6 0.0
C 0.0 0.0 2.3 # Axial carbon
O 0.0 1.3 2.9 # Axial oxygens
O 0.0 -1.3 2.9

**Key properties:**
- Total electrons: ~70 (full system)
- Spin multiplicity: Triplet (S=1, 2 unpaired electrons)
- Point group: Approximate D4h symmetry

## Methodology

### 1. Classical DFT Baseline
- **Method**: Restricted Open-Shell Kohn-Sham (ROKS)
- **Functional**: B3LYP with density fitting
- **Basis set**: STO-3G (minimal basis for demonstration)

### 2. Quantum VQE Calculation
- **Active space**: 6 electrons in 8 spatial orbitals
- **Qubit reduction**: Parity mapping → 16 qubits
- **Ansatz**: UCCSD (Unitary Coupled Cluster with Singles and Doubles)
- **Optimizer**: COBYLA (Constrained Optimization BY Linear Approximation)

### 3. Computational Workflow

PySCF Driver → Active Space Selection → Qubit Mapping → VQE Optimization → Energy Results
text


## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ni-complex-vqe.git
cd ni-complex-vqe

Step 2: Install dependencies
bash

pip install -r requirements.txt

Step 3: Verify installation
bash

python -c "import qiskit; print(qiskit.__version__)"
python -c "import pyscf; print(pyscf.__version__)"

Usage
Quick Start

Run the main simulation:
bash

python src/ni_complex_vqe.py

Interactive Notebook

For detailed exploration, use the Jupyter notebook:
bash

jupyter notebook notebooks/demo.ipynb

Customizing Parameters

Modify the main function in src/ni_complex_vqe.py to change:

    Active space size (num_electrons, num_orbitals)

    Basis set (basis='sto3g', basis='6-31g', etc.)

    Optimizer parameters (maxiter, tol)

    Molecular geometry

Example custom run:
python

from src.ni_complex_vqe import NickelComplexVQE

# Customize solver
solver = NickelComplexVQE(basis='6-31g')
solver.run_dft()
solver.setup_vqe(num_electrons=8, num_orbitals=10, maxiter=1000)
energy = solver.run_vqe()

Results
Expected Output
text

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


Interpretation

    VQE energy: Electronic energy of the active space

    DFT energy: Full system energy for comparison

    Energy difference: Correlation energy captured by VQE

Performance Notes
Computational Requirements

    Time: 10-30 minutes on a typical laptop

    Memory: ~2-4 GB RAM

    Storage: ~100 MB for temporary files

Optimization Tips

    Reduce active space: For faster runs, use 4 electrons in 6 orbitals

    Change optimizer: Try SLSQP for faster convergence (less robust)

    Use statevector simulator: For exact simulation (no noise)

    Parallel execution: Qiskit supports parallel circuit evaluation

Limitations

    STO-3G basis set is minimal; use larger basis for accuracy

    VQE with 16 qubits is simulated classically

    No error mitigation included

    Single repetition of UCCSD (reps=1)
