#!/usr/bin/env python
# coding: utf-8

# In[8]:


"""
Quantum VQE Simulation of Nickel Complex
=========================================
This script performs VQE (Variational Quantum Eigensolver) calculations
on a nickel complex using Qiskit Nature and PySCF.
"""


import numpy as np
from pyscf import gto, scf
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA  
from qiskit.primitives import Estimator

# ====================== 1. Molecule (your improved geometry) ======================
mol = gto.M(
    atom="""
    Ni 0.0 0.0 0.0
    N 1.9 0.0 0.0
    N -1.9 0.0 0.0
    N 0.0 1.9 0.0
    N 0.0 -1.9 0.0
    C 2.6 2.6 0.0
    C -2.6 2.6 0.0
    C 0.0 0.0 2.3
    O 0.0 1.3 2.9
    O 0.0 -1.3 2.9
    """,
    basis='sto3g',
    charge=0,
    spin=2,
    verbose=4
)

# ====================== 2. DFT Baseline (B3LYP) ======================
print("Running DFT (B3LYP)...")
mf = scf.ROKS(mol)
mf.xc = 'b3lyp'
mf = mf.density_fit()
mf.conv_tol = 1e-7
mf.max_cycle = 300
mf.level_shift = 0.4
mf.kernel()

print(f"\nDFT energy: {mf.e_tot:.8f} Hartree ({mf.e_tot*27.2114:.2f} eV)")

# ====================== 3. Driver & Problem ======================
atom_string = """Ni 0.0 0.0 0.0
N 1.9 0.0 0.0
N -1.9 0.0 0.0
N 0.0 1.9 0.0
N 0.0 -1.9 0.0
C 2.6 2.6 0.0
C -2.6 2.6 0.0
C 0.0 0.0 2.3
O 0.0 1.3 2.9
O 0.0 -1.3 2.9"""

driver = PySCFDriver(atom=atom_string, basis="sto3g", charge=0, spin=2)
problem = driver.run()

# ====================== 4. Active Space (16 qubits - good for laptop) ======================
freeze = FreezeCoreTransformer(freeze_core=True)
active = ActiveSpaceTransformer(num_electrons=6, num_spatial_orbitals=8)

problem_reduced = freeze.transform(problem)
problem_reduced = active.transform(problem_reduced)

print(f"Active space: {problem_reduced.num_particles} electrons in {problem_reduced.num_spatial_orbitals} orbitals → {2*problem_reduced.num_spatial_orbitals} qubits")

# ====================== 5. VQE Setup ======================
mapper = ParityMapper()

ansatz = UCC(
    num_spatial_orbitals=problem_reduced.num_spatial_orbitals,
    num_particles=problem_reduced.num_particles,
    qubit_mapper=mapper,
    excitations='sd',
    reps=1
)

optimizer = COBYLA(maxiter=500, tol=1e-6)   # Better for noisy/flat landscapes

estimator = Estimator()

vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=np.zeros(ansatz.num_parameters))

solver = GroundStateEigensolver(mapper, vqe)

# ====================== 6. Run VQE ======================
print("\nStarting VQE on active space (this may take 10-30+ min)...")
result = solver.solve(problem_reduced)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Active-space VQE energy : {result.total_energies[0]:.8f} Hartree")
print(f"Full-system DFT energy  : {mf.e_tot:.8f} Hartree")
print(f"Difference (DFT - VQE)  : {mf.e_tot - result.total_energies[0]:.8f} Hartree")

# Safe way to get optimizer info
if hasattr(result, 'optimizer_result') and result.optimizer_result is not None:
    opt = result.optimizer_result
    print(f"\nOptimizer iterations : {getattr(opt, 'nit', 'N/A')}")
    print(f"Final optimizer value: {getattr(opt, 'fun', 'N/A')}")
else:
    print("\nOptimizer details not directly available in this Qiskit version.")

print("\n✓ Run completed! This is a successful prototype run.")


# In[ ]:




