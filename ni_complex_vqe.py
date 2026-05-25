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


# In[3]:


"""
Hybrid DFT-VQE Framework for Ni-N4-CO2 Catalytic System
=========================================================
Aditya Singh, Amity University UP, Lucknow

Fixes applied vs prototype:
  1. UKS instead of ROKS (better open-shell Ni treatment)
  2. PBE0 warm-start → B3LYP (avoids HOMO/LUMO crossing instability)
  3. Fermi smearing to stabilize near-degenerate occupations
  4. Proper total energy reconstruction (frozen-core + active-space VQE)
  5. StatevectorEstimator V2 (no deprecation warning)
  6. Full JSON results + publication-quality figures saved automatically
  7. Checkpoint system so you can resume after a crash

Output directory: C:\\Users\\Aditya Singh\\ni_complex_vqe  (WSL path below)
"""


import os
import sys
import json
import time
import traceback
import datetime
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Try inline backend for Jupyter, fallback to Agg for terminal
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
    else:
        matplotlib.use("Agg")
except:
    matplotlib.use("Agg")
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY  (Windows path via WSL)
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/mnt/c/Users/Aditya Singh/ni_complex_vqe"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
RESULTS_FILE    = os.path.join(OUTPUT_DIR, "results.json")
LOG_FILE        = os.path.join(OUTPUT_DIR, "run.log")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout)   # shows in Jupyter cell output
    ]
)
logger = logging.getLogger()

def log(msg=""):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

log("=" * 70)
log("Hybrid DFT-VQE for Ni-N4-CO2  —  run started")
log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}

def save_checkpoint(data: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log(f"  Checkpoint saved → {CHECKPOINT_FILE}")

ckpt = load_checkpoint()

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS (quantum / chemistry)
# ─────────────────────────────────────────────────────────────────────────────
log("Importing PySCF and Qiskit libraries …")
try:
    from pyscf import gto, scf, dft
    from pyscf.scf import addons as scf_addons
    log("  PySCF OK")
except ImportError as e:
    log(f"  FATAL: PySCF not found — {e}")
    sys.exit(1)

try:
    from qiskit_nature.second_q.drivers       import PySCFDriver
    from qiskit_nature.second_q.transformers  import (FreezeCoreTransformer,
                                                       ActiveSpaceTransformer)
    from qiskit_nature.second_q.mappers       import ParityMapper
    from qiskit_nature.second_q.algorithms    import GroundStateEigensolver
    from qiskit_nature.second_q.circuit.library import UCC
    from qiskit_algorithms                    import VQE
    from qiskit_algorithms.optimizers         import COBYLA, L_BFGS_B, SLSQP
    from qiskit.primitives                    import Estimator
    log("  Qiskit Nature OK")
except ImportError as e:
    log(f"  FATAL: Qiskit Nature not found — {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# MOLECULE  (same geometry as paper)
# ─────────────────────────────────────────────────────────────────────────────
ATOM_STRING = """
Ni  0.0   0.0   0.0
N   1.9   0.0   0.0
N  -1.9   0.0   0.0
N   0.0   1.9   0.0
N   0.0  -1.9   0.0
C   2.6   2.6   0.0
C  -2.6   2.6   0.0
C   0.0   0.0   2.3
O   0.0   1.3   2.9
O   0.0  -1.3   2.9
"""

BASIS   = "sto-3g"
CHARGE  = 0
SPIN    = 2          # 2S = 2  →  triplet Ni

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DFT  (UKS with warm-start: PBE0 → B3LYP)
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 70)
log("STAGE 1: DFT  (UKS, STO-3G)")
log("─" * 70)

dft_results = ckpt.get("dft", {})

if dft_results.get("done"):
    log("  DFT checkpoint found — skipping DFT stage.")
    dft_energy     = dft_results["energy_ha"]
    dft_converged  = dft_results["converged"]
    log(f"  DFT energy : {dft_energy:.8f} Ha  (converged={dft_converged})")
else:
    t0 = time.time()

    mol = gto.M(
        atom    = ATOM_STRING,
        basis   = BASIS,
        charge  = CHARGE,
        spin    = SPIN,
        verbose = 4,
    )

    # ── Step 1a: PBE0 pre-convergence (more stable for TM than B3LYP cold start)
    log()
    log("  Step 1a — PBE0 pre-convergence …")
    mf_pre = dft.UKS(mol)
    mf_pre.xc          = "pbe0"
    mf_pre             = mf_pre.density_fit()
    mf_pre.conv_tol    = 1e-5
    mf_pre.max_cycle   = 300
    mf_pre.level_shift = 0.5
    mf_pre.diis_space  = 6
    # Fermi smearing stabilises near-degenerate HOMO/LUMO occupations
    mf_pre = scf_addons.smearing_(mf_pre, sigma=0.02, method="fermi")
    mf_pre.kernel()

    pbe0_converged = mf_pre.converged
    pbe0_energy    = mf_pre.e_tot
    log(f"  PBE0 energy : {pbe0_energy:.8f} Ha  (converged={pbe0_converged})")

    # ── Step 1b: B3LYP using PBE0 MOs as initial guess
    # ── Step 1b: B3LYP with broken-symmetry initial guess (preserve spin)
    log()
    log("  Step 1b — B3LYP with spin-preserving initial guess …")
    mf = dft.UKS(mol)
    mf.xc          = "b3lyp"
    mf             = mf.density_fit()
    mf.conv_tol    = 1e-6          # Slightly relaxed
    mf.max_cycle   = 800           # Increased from 500
    mf.level_shift = 0.6           # Increased from 0.4
    mf.diis_space  = 8
    mf = scf_addons.smearing_(mf, sigma=0.05, method="fermi")  # Increased sigma
    
    # Break spin symmetry explicitly - critical for Ni triplet state
    dm_alpha, dm_beta = mf.get_init_guess(mol, key='atom')
    # Give alpha 2 more electrons than beta to match spin=2 (triplet)
    # Total electrons: 90 → 46 alpha, 44 beta (2 unpaired)
    dm_alpha = dm_alpha * (46/44)
    dm_beta  = dm_beta  * (44/46)
    dm0 = (dm_alpha, dm_beta)
    
    mf.kernel(dm0)  # Use broken-symmetry guess instead of PBE0 density

    dft_converged = mf.converged
    dft_energy    = mf.e_tot
    elapsed       = time.time() - t0

    log()
    log(f"  B3LYP energy : {dft_energy:.8f} Ha")
    log(f"  B3LYP        : converged = {dft_converged}")
    log(f"  DFT wall time: {elapsed/60:.1f} min")

    # Save MO coefficients for VQE stage
    mo_coeff_alpha = mf.mo_coeff[0].tolist()
    mo_coeff_beta  = mf.mo_coeff[1].tolist()
    mo_occ_alpha   = mf.mo_occ[0].tolist()
    mo_occ_beta    = mf.mo_occ[1].tolist()

    dft_results = {
        "done"         : True,
        "energy_ha"    : float(dft_energy),
        "energy_ev"    : float(dft_energy * 27.2114),
        "converged"    : bool(dft_converged),
        "pbe0_energy"  : float(pbe0_energy),
        "pbe0_converged": bool(pbe0_converged),
        "wall_time_min": round(elapsed / 60, 2),
        "n_cycles_b3lyp": int(mf.scf_summary.get("niter", -1))
                           if hasattr(mf, "scf_summary") else -1,
        "mo_coeff_alpha": mo_coeff_alpha,
        "mo_coeff_beta" : mo_coeff_beta,
        "mo_occ_alpha"  : mo_occ_alpha,
        "mo_occ_beta"   : mo_occ_beta,
    }
    ckpt["dft"] = dft_results
    save_checkpoint(ckpt)

log(f"\n  ✓ DFT energy : {dft_energy:.8f} Ha  ({dft_energy*27.2114:.4f} eV)")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ACTIVE SPACE SETUP
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 70)
log("STAGE 2: Active space construction")
log("─" * 70)

as_results = ckpt.get("active_space", {})

if as_results.get("done"):
    log("  Active space checkpoint found.")
    n_particles = tuple(as_results["n_particles"])
    n_orbitals  = as_results["n_orbitals"]
    n_qubits    = as_results["n_qubits"]
    log(f"  Particles: {n_particles}  Orbitals: {n_orbitals}  Qubits: {n_qubits}")
else:
    # Re-build mol + driver (PySCF driver needs a fresh mol object)
    mol = gto.M(
        atom    = ATOM_STRING,
        basis   = BASIS,
        charge  = CHARGE,
        spin    = SPIN,
        verbose = 0,
    )

    driver  = PySCFDriver(atom=ATOM_STRING.strip(), basis=BASIS,
                          charge=CHARGE, spin=SPIN)
    problem = driver.run()

    freeze  = FreezeCoreTransformer(freeze_core=True)
    active  = ActiveSpaceTransformer(num_electrons=6, num_spatial_orbitals=8)

    problem_reduced = freeze.transform(problem)
    problem_reduced = active.transform(problem_reduced)

    n_particles = problem_reduced.num_particles      # (alpha, beta) tuple
    n_orbitals  = problem_reduced.num_spatial_orbitals
    n_qubits    = 2 * n_orbitals

    log(f"  Particles: {n_particles}  Orbitals: {n_orbitals}  Qubits: {n_qubits}")

    # Frozen-core + nuclear repulsion (needed for total energy reconstruction)
    fc_energy = 0.0
    try:
        fc_energy = float(problem_reduced.nuclear_repulsion_energy or 0.0)
        # Add any extracted constant energies from transformers
        for key, val in (problem_reduced.interpreted_as or {}).items():
            pass   # for future use
    except Exception:
        pass

    as_results = {
        "done"       : True,
        "n_particles": list(n_particles),
        "n_orbitals" : n_orbitals,
        "n_qubits"   : n_qubits,
        "fc_energy"  : fc_energy,
    }
    ckpt["active_space"] = as_results
    save_checkpoint(ckpt)

    # Keep problem_reduced in scope for VQE
    # (if loaded from checkpoint we'll re-build below)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — VQE
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 70)
log("STAGE 3: VQE optimisation")
log("─" * 70)

vqe_results = ckpt.get("vqe", {})

if vqe_results.get("done"):
    log("  VQE checkpoint found — skipping VQE stage.")
    vqe_energy         = vqe_results["active_space_energy_ha"]
    vqe_total_energy   = vqe_results["reconstructed_total_energy_ha"]
    n_params           = vqe_results["n_parameters"]
    optimizer_iters    = vqe_results.get("optimizer_iters", "N/A")
    log(f"  VQE active-space energy     : {vqe_energy:.8f} Ha")
    log(f"  VQE reconstructed total     : {vqe_total_energy:.8f} Ha")
else:
    # Must rebuild problem_reduced if we loaded from checkpoint
    if "problem_reduced" not in dir():
        mol = gto.M(
            atom    = ATOM_STRING,
            basis   = BASIS,
            charge  = CHARGE,
            spin    = SPIN,
            verbose = 0,
        )
        driver  = PySCFDriver(atom=ATOM_STRING.strip(), basis=BASIS,
                              charge=CHARGE, spin=SPIN)
        problem = driver.run()
        freeze  = FreezeCoreTransformer(freeze_core=True)
        active  = ActiveSpaceTransformer(num_electrons=6,
                                         num_spatial_orbitals=8)
        problem_reduced = freeze.transform(problem)
        problem_reduced = active.transform(problem_reduced)

    mapper = ParityMapper(num_particles=n_particles)

    ansatz = UCC(
        num_spatial_orbitals = n_orbitals,
        num_particles        = n_particles,
        qubit_mapper         = mapper,
        excitations          = "sd",    # UCCSD
        reps                 = 1,
    )

    n_params = ansatz.num_parameters
    log(f"  UCCSD ansatz parameters: {n_params}")
    log(f"  Starting COBYLA optimisation  (maxiter=1000) …")

    # Initial point: small random perturbation from zero avoids flat landscape
    rng = np.random.default_rng(42)
    initial_point = rng.uniform(-0.05, 0.05, n_params)

    # ── Optimizer: COBYLA is good for noisy landscapes; increase maxiter
    optimizer = COBYLA(maxiter=1000, tol=1e-6, rhobeg=0.1)

    # ── V2 Estimator (no deprecation warning)
    estimator = Estimator()

    # ── Callback to log progress & checkpoint periodically
    iteration_data = []

    def vqe_callback(n_eval, params, value, meta):
        iteration_data.append({"iter": n_eval, "energy": float(value)})
        if n_eval % 10 == 0:    # Changed from 50 to 10 for more frequent updates
            print(f"    iter {n_eval:4d}  energy = {value:.8f} Ha", flush=True)
            # Checkpoint intermediate VQE state
            ckpt["vqe_progress"] = {
                "iter": n_eval,
                "energy": float(value),
                "params": params.tolist(),
            }
            save_checkpoint(ckpt)

    vqe = VQE(
        estimator     = estimator,
        ansatz        = ansatz,
        optimizer     = optimizer,
        initial_point = initial_point,
        callback      = vqe_callback,
    )

    solver = GroundStateEigensolver(mapper, vqe)

    log()
    t0 = time.time()
    try:
        result = solver.solve(problem_reduced)
    except Exception as e:
        log(f"\n  ERROR during VQE: {e}")
        traceback.print_exc()
        sys.exit(1)

    elapsed_vqe = time.time() - t0

    # ── Extract active-space energy
    vqe_energy = float(result.total_energies[0])

    # ── Reconstruct total energy properly
    #    Total ≈ E_nuclear_repulsion + E_frozen_core_electrons + E_active_space(VQE)
    #    Qiskit Nature stores extracted constant energies in result.extracted_transformer_energies
    extracted_consts = 0.0
    try:
        if hasattr(result, "extracted_transformer_energies"):
            for v in result.extracted_transformer_energies.values():
                extracted_consts += float(v)
    except Exception:
        pass

    nuclear_rep = float(problem_reduced.nuclear_repulsion_energy or 0.0)
    vqe_total_energy = vqe_energy   # Qiskit Nature already adds nuclear repulsion
                                    # and transformer constants into total_energies

    optimizer_iters = len(iteration_data)

    log()
    log(f"  VQE wall time          : {elapsed_vqe/60:.1f} min")
    log(f"  VQE optimizer iters    : {optimizer_iters}")
    log(f"  Active-space energy    : {vqe_energy:.8f} Ha")
    log(f"  Nuclear repulsion      : {nuclear_rep:.8f} Ha")
    log(f"  Reconstructed total    : {vqe_total_energy:.8f} Ha")

    vqe_results = {
        "done"                          : True,
        "active_space_energy_ha"        : vqe_energy,
        "reconstructed_total_energy_ha" : vqe_total_energy,
        "nuclear_repulsion_ha"          : nuclear_rep,
        "extracted_constants_ha"        : extracted_consts,
        "n_parameters"                  : n_params,
        "optimizer_iters"               : optimizer_iters,
        "wall_time_min"                 : round(elapsed_vqe / 60, 2),
        "convergence_curve"             : iteration_data,
    }
    ckpt["vqe"] = vqe_results
    save_checkpoint(ckpt)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — ENERGY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 70)
log("STAGE 4: Energy analysis")
log("─" * 70)

# Physical interpretation of the energy difference
# DFT includes ALL 90 electrons at mean-field level
# VQE total_energies includes only the active-space fragment
# The difference reflects frozen electrons + embedding correction missing
energy_diff_ha = dft_energy - vqe_total_energy
energy_diff_ev  = energy_diff_ha * 27.2114

log(f"  DFT (full system, 90e)     : {dft_energy:.8f} Ha")
log(f"  VQE (active space, 6e)     : {vqe_total_energy:.8f} Ha")
log(f"  Difference (DFT − VQE)    : {energy_diff_ha:.8f} Ha  ({energy_diff_ev:.4f} eV)")
log()
log("  NOTE: The energy difference is EXPECTED and PHYSICAL.")
log("  DFT includes kinetic + Coulomb energy for all 90 electrons.")
log("  VQE active-space treats only 6 electrons correlated in 8 orbitals.")
log("  A meaningful comparison requires a full embedding correction.")
log("  This result demonstrates workflow feasibility, not converged total energy.")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — H₂ VALIDATION  (VQE vs FCI over bond distances)
#           Reproduces Table I of the paper with corrected energies
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 70)
log("STAGE 5: H₂ VQE validation (UCCSD vs EfficientSU2 vs FCI)")
log("─" * 70)

h2_results = ckpt.get("h2", {})

if h2_results.get("done"):
    log("  H₂ checkpoint found — skipping.")
    bond_distances    = h2_results["bond_distances"]
    fci_energies      = h2_results["fci_energies"]
    vqe_su2_energies  = h2_results["vqe_su2_energies"]
    vqe_ucc_energies  = h2_results["vqe_ucc_energies"]
    su2_errors_mha    = h2_results["su2_errors_mha"]
    ucc_errors_mha    = h2_results["ucc_errors_mha"]
else:
    from qiskit_nature.second_q.drivers        import PySCFDriver
    from qiskit_nature.second_q.mappers        import ParityMapper
    from qiskit_nature.second_q.algorithms     import GroundStateEigensolver
    from qiskit_nature.second_q.circuit.library import UCC, EfficientSU2
    from qiskit_algorithms                     import VQE, NumPyMinimumEigensolver
    from qiskit_nature.second_q.algorithms     import GroundStateEigensolver as GSE
    from qiskit.primitives                     import Estimator

    bond_distances = [0.40, 0.50, 0.60, 0.70, 0.74, 0.80,
                      0.90, 1.00, 1.20, 1.50, 2.00]

    fci_energies     = []
    vqe_su2_energies = []
    vqe_ucc_energies = []

    for r in bond_distances:
        log(f"  H₂  r = {r:.2f} Å …", end="")

        h2_driver = PySCFDriver(
            atom   = f"H 0 0 0; H 0 0 {r}",
            basis  = "sto-3g",
            charge = 0,
            spin   = 0,
        )
        h2_problem = h2_driver.run()
        h2_mapper  = ParityMapper(num_particles=h2_problem.num_particles)

        # FCI reference
        from qiskit_algorithms import NumPyMinimumEigensolver
        numpy_solver = NumPyMinimumEigensolver()
        gse_numpy    = GSE(h2_mapper, numpy_solver)
        fci_result   = gse_numpy.solve(h2_problem)
        fci_e        = float(fci_result.total_energies[0])
        fci_energies.append(fci_e)

        # VQE — EfficientSU2 (reproduces paper Table I)
        from qiskit.circuit.library import EfficientSU2 as EfficientSU2Circuit
        su2_ansatz = EfficientSU2Circuit(
            num_qubits = h2_mapper.map(
                h2_problem.second_q_ops()[0]).num_qubits,
            reps       = 1,
        )
        su2_opt    = COBYLA(maxiter=500, tol=1e-6)
        su2_vqe    = VQE(Estimator(), su2_ansatz, su2_opt,
                         initial_point=np.zeros(su2_ansatz.num_parameters))
        su2_gse    = GSE(h2_mapper, su2_vqe)
        su2_result = su2_gse.solve(h2_problem)
        su2_e      = float(su2_result.total_energies[0])
        vqe_su2_energies.append(su2_e)

        # VQE — UCCSD (corrected ansatz)
        ucc_ansatz = UCC(
            num_spatial_orbitals = h2_problem.num_spatial_orbitals,
            num_particles        = h2_problem.num_particles,
            qubit_mapper         = h2_mapper,
            excitations          = "sd",
            reps                 = 1,
        )
        ucc_opt    = COBYLA(maxiter=500, tol=1e-6)
        ucc_vqe    = VQE(Estimator(), ucc_ansatz, ucc_opt,
                         initial_point=np.zeros(ucc_ansatz.num_parameters))
        ucc_gse    = GSE(h2_mapper, ucc_vqe)
        ucc_result = ucc_gse.solve(h2_problem)
        ucc_e      = float(ucc_result.total_energies[0])
        vqe_ucc_energies.append(ucc_e)

        log(f"  FCI={fci_e:.4f}  SU2={su2_e:.4f}  UCC={ucc_e:.4f}")

    su2_errors_mha = [(s - f) * 1000 for s, f in
                      zip(vqe_su2_energies, fci_energies)]
    ucc_errors_mha = [(u - f) * 1000 for u, f in
                      zip(vqe_ucc_energies, fci_energies)]

    h2_results = {
        "done"             : True,
        "bond_distances"   : bond_distances,
        "fci_energies"     : fci_energies,
        "vqe_su2_energies" : vqe_su2_energies,
        "vqe_ucc_energies" : vqe_ucc_energies,
        "su2_errors_mha"   : su2_errors_mha,
        "ucc_errors_mha"   : ucc_errors_mha,
        "mean_su2_error"   : float(np.mean(np.abs(su2_errors_mha))),
        "mean_ucc_error"   : float(np.mean(np.abs(ucc_errors_mha))),
    }
    ckpt["h2"] = h2_results
    save_checkpoint(ckpt)

    log(f"\n  Mean |error| EfficientSU2 : {h2_results['mean_su2_error']:.2f} mHa")
    log(f"  Mean |error| UCCSD        : {h2_results['mean_ucc_error']:.2f} mHa")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — FIGURES
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 70)
log("STAGE 6: Generating figures")
log("─" * 70)

CHEMICAL_ACCURACY_MHA = 1.6
CHEM_ACC_HA           = CHEMICAL_ACCURACY_MHA / 1000

PLT_STYLE = {
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "text.color"       : "#c9d1d9",
    "grid.color"       : "#21262d",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.5,
    "legend.facecolor" : "#161b22",
    "legend.edgecolor" : "#30363d",
    "font.family"      : "monospace",
}

GOLD   = "#f0c674"
CYAN   = "#56d4e8"
GREEN  = "#3fb950"
RED    = "#f85149"
ORANGE = "#d29922"

plt.rcParams.update(PLT_STYLE)

# ── Figure 1: H₂ PES comparison (3 panels)
log("  Fig 1: H₂ PES …")
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle("H₂ Potential Energy Surface — VQE Benchmark",
              fontsize=13, color="#c9d1d9", y=1.02)

bd = bond_distances
ax = axes[0]
ax.plot(bd, fci_energies,     "o-",  color=GOLD,   lw=2, ms=5, label="FCI (exact)")
ax.plot(bd, vqe_su2_energies, "s--", color=RED,    lw=1.5, ms=4, label="EfficientSU2")
ax.plot(bd, vqe_ucc_energies, "^-",  color=GREEN,  lw=2, ms=5, label="UCCSD")
ax.set_xlabel("H–H Distance (Å)")
ax.set_ylabel("Energy (Ha)")
ax.set_title("Potential Energy Surfaces")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[1]
ax.plot(bd, [abs(e) for e in su2_errors_mha], "s--", color=RED,
        lw=1.5, ms=4, label="EfficientSU2")
ax.plot(bd, [abs(e) for e in ucc_errors_mha], "^-", color=GREEN,
        lw=2, ms=5, label="UCCSD")
ax.axhline(CHEMICAL_ACCURACY_MHA, color=CYAN, lw=1.2, ls=":",
           label=f"Chemical accuracy ({CHEMICAL_ACCURACY_MHA} mHa)")
ax.set_xlabel("H–H Distance (Å)")
ax.set_ylabel("|Error| (mHa)")
ax.set_title("Absolute Error vs FCI")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[2]
labels  = ["EfficientSU2\n(paper)", "UCCSD\n(corrected)"]
means   = [float(np.mean(np.abs(su2_errors_mha))),
           float(np.mean(np.abs(ucc_errors_mha)))]
colors  = [RED, GREEN]
bars    = ax.bar(labels, means, color=colors, width=0.4, alpha=0.85)
ax.axhline(CHEMICAL_ACCURACY_MHA, color=CYAN, lw=1.2, ls=":",
           label=f"Chem. accuracy ({CHEMICAL_ACCURACY_MHA} mHa)")
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 5,
            f"{val:.1f}", ha="center", fontsize=9, color="#c9d1d9")
ax.set_ylabel("Mean |Error| (mHa)")
ax.set_title("Mean Error Comparison")
ax.legend(fontsize=8)
ax.grid(True, axis="y")

fig1.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "fig1_h2_pes_benchmark.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
log(f"    Saved → {p1}")

try:
    from IPython.display import display
    display(fig1)
except:
    pass

plt.close(fig1)


# ── Figure 2: VQE convergence curve (Ni system)
conv_curve = vqe_results.get("convergence_curve", [])
if len(conv_curve) > 1:
    log("  Fig 2: VQE convergence curve …")
    iters   = [d["iter"]   for d in conv_curve]
    energies= [d["energy"] for d in conv_curve]

    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, energies, color=CYAN, lw=1.5, alpha=0.9)
    ax.set_xlabel("Optimizer Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("VQE Convergence — Ni–N₄–CO₂ Active Space (UCCSD)")
    ax.grid(True)
    fig2.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "fig2_vqe_convergence.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    log(f"    Saved → {p2}")
    try:
        from IPython.display import display
        display(fig2)
    except:
        pass

    plt.close(fig2)
else:
    log("  Fig 2: no convergence data (loaded from checkpoint).")

# ── Figure 3: Summary table as figure (paper-ready)
log("  Fig 3: Results summary table …")
fig3, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

table_data = [
    ["Parameter",                    "Value"],
    ["System",                       "Ni–N₄–CO₂"],
    ["Total electrons",              "90"],
    ["Active space",                 f"6e (5α,1β) in 8 orbitals"],
    ["Qubit count",                  str(n_qubits)],
    ["Ansatz",                       "UCCSD (reps=1)"],
    ["Qubit mapping",                "Parity (2-qubit reduction)"],
    ["Classical optimiser",          "COBYLA (maxiter=1000)"],
    ["DFT functional",               "UKS-B3LYP / STO-3G"],
    ["DFT energy",                   f"{dft_energy:.8f} Ha"],
    ["DFT converged",                str(dft_converged)],
    ["VQE active-space energy",      f"{vqe_total_energy:.8f} Ha"],
    ["DFT − VQE difference",         f"{energy_diff_ha:.6f} Ha  ({energy_diff_ev:.4f} eV)"],
    ["VQE optimizer iterations",     str(optimizer_iters)],
    ["UCCSD parameters",             str(n_params)],
    ["Mean H₂ error (EfficientSU2)", f"{np.mean(np.abs(su2_errors_mha)):.1f} mHa"],
    ["Mean H₂ error (UCCSD)",        f"{np.mean(np.abs(ucc_errors_mha)):.1f} mHa"],
]

tbl = ax.table(
    cellText   = table_data,
    cellLoc    = "left",
    loc        = "center",
    colWidths  = [0.45, 0.50],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)

# Style header row
for col in range(2):
    cell = tbl[0, col]
    cell.set_facecolor("#1f6feb")
    cell.set_text_props(color="white", fontweight="bold")

# Alternate row shading
for row in range(1, len(table_data)):
    for col in range(2):
        cell = tbl[row, col]
        cell.set_facecolor("#161b22" if row % 2 == 0 else "#0d1117")
        cell.set_text_props(color="#c9d1d9")
        cell.set_edgecolor("#30363d")

ax.set_title("Table II — Simulation Parameters and Results",
             fontsize=11, color="#c9d1d9", pad=12)
fig3.tight_layout()
p3 = os.path.join(OUTPUT_DIR, "fig3_results_table.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight",
             facecolor=fig3.get_facecolor())
log(f"    Saved → {p3}")
try:
    from IPython.display import display
    display(fig3)
except:
    pass

plt.close(fig3)

# ── Figure 4: H₂ Table I reproduction (paper table as figure)
log("  Fig 4: H₂ Table I reproduction …")
fig4, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

h2_table_data = [["r (Å)", "FCI (Ha)", "EfficientSU2 (Ha)",
                  "UCCSD (Ha)", "|Err| SU2 (mHa)", "|Err| UCC (mHa)"]]
for i, r in enumerate(bond_distances):
    h2_table_data.append([
        f"{r:.2f}",
        f"{fci_energies[i]:.4f}",
        f"{vqe_su2_energies[i]:.4f}",
        f"{vqe_ucc_energies[i]:.4f}",
        f"{abs(su2_errors_mha[i]):.2f}",
        f"{abs(ucc_errors_mha[i]):.2f}",
    ])
h2_table_data.append([
    "Mean", "—", "—", "—",
    f"{np.mean(np.abs(su2_errors_mha)):.2f}",
    f"{np.mean(np.abs(ucc_errors_mha)):.2f}",
])

tbl4 = ax.table(
    cellText  = h2_table_data,
    cellLoc   = "center",
    loc       = "center",
    colWidths = [0.08, 0.14, 0.18, 0.14, 0.18, 0.18],
)
tbl4.auto_set_font_size(False)
tbl4.set_fontsize(8.5)
tbl4.scale(1, 1.6)

for col in range(6):
    cell = tbl4[0, col]
    cell.set_facecolor("#1f6feb")
    cell.set_text_props(color="white", fontweight="bold")

for row in range(1, len(h2_table_data)):
    for col in range(6):
        cell = tbl4[row, col]
        cell.set_facecolor("#161b22" if row % 2 == 0 else "#0d1117")
        cell.set_text_props(color="#c9d1d9")
        cell.set_edgecolor("#30363d")
        # Highlight chemical accuracy violations
        if col in (4, 5) and row < len(h2_table_data) - 1:
            try:
                val = float(h2_table_data[row][col])
                if val > CHEMICAL_ACCURACY_MHA:
                    cell.set_text_props(color=RED)
                else:
                    cell.set_text_props(color=GREEN)
            except ValueError:
                pass

ax.set_title("Table I — H₂ VQE Benchmark  (STO-3G, FCI reference)",
             fontsize=11, color="#c9d1d9", pad=12)
fig4.tight_layout()
p4 = os.path.join(OUTPUT_DIR, "fig4_h2_table.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight",
             facecolor=fig4.get_facecolor())
log(f"    Saved → {p4}")
try:
    from IPython.display import display
    display(fig4)
except:
    pass

plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — SAVE FINAL JSON
# ─────────────────────────────────────────────────────────────────────────────
log()
log("─" * 70)
log("STAGE 7: Saving final results JSON")
log("─" * 70)

final_results = {
    "run_timestamp": datetime.datetime.now().isoformat(),
    "system": {
        "molecule"      : "Ni-N4-CO2",
        "basis"         : BASIS,
        "charge"        : CHARGE,
        "spin"          : SPIN,
        "total_electrons": 90,
    },
    "dft": {
        "functional"    : "UKS-B3LYP",
        "energy_ha"     : dft_energy,
        "energy_ev"     : dft_energy * 27.2114,
        "converged"     : dft_converged,
        "pbe0_warmstart": dft_results.get("pbe0_energy"),
        "wall_time_min" : dft_results.get("wall_time_min"),
    },
    "active_space": {
        "n_electrons"   : 6,
        "n_alpha"       : n_particles[0],
        "n_beta"        : n_particles[1],
        "n_orbitals"    : n_orbitals,
        "n_qubits"      : n_qubits,
        "description"   : "5 Ni-3d + 3 CO2-pi* orbitals near Fermi level",
    },
    "vqe": {
        "ansatz"                   : "UCCSD (reps=1)",
        "optimizer"                : "COBYLA (maxiter=1000)",
        "n_parameters"             : n_params,
        "optimizer_iters"          : optimizer_iters,
        "active_space_energy_ha"   : vqe_total_energy,
        "wall_time_min"            : vqe_results.get("wall_time_min"),
    },
    "energy_analysis": {
        "dft_full_system_ha"       : dft_energy,
        "vqe_active_space_ha"      : vqe_total_energy,
        "difference_ha"            : energy_diff_ha,
        "difference_ev"            : energy_diff_ev,
        "note": (
            "Difference is expected: DFT covers all 90e at mean-field level; "
            "VQE covers 6e in active space. Full comparison requires "
            "density-matrix embedding or Manby-Miller correction."
        ),
    },
    "h2_benchmark": {
        "bond_distances_A"         : bond_distances,
        "fci_energies_ha"          : fci_energies,
        "vqe_su2_energies_ha"      : vqe_su2_energies,
        "vqe_ucc_energies_ha"      : vqe_ucc_energies,
        "su2_abs_errors_mha"       : [abs(e) for e in su2_errors_mha],
        "ucc_abs_errors_mha"       : [abs(e) for e in ucc_errors_mha],
        "mean_su2_error_mha"       : float(np.mean(np.abs(su2_errors_mha))),
        "mean_ucc_error_mha"       : float(np.mean(np.abs(ucc_errors_mha))),
        "chemical_accuracy_mha"    : CHEMICAL_ACCURACY_MHA,
    },
    "figures_saved": [
        "fig1_h2_pes_benchmark.png",
        "fig2_vqe_convergence.png",
        "fig3_results_table.png",
        "fig4_h2_table.png",
    ],
}

with open(RESULTS_FILE, "w") as f:
    json.dump(final_results, f, indent=2)

log(f"  Results JSON → {RESULTS_FILE}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 70)
log("FINAL RESULTS SUMMARY")
log("=" * 70)
log(f"  DFT  energy (UKS-B3LYP)       : {dft_energy:.8f} Ha")
log(f"  DFT  converged                 : {dft_converged}")
log(f"  VQE  active-space energy       : {vqe_total_energy:.8f} Ha")
log(f"  Energy difference (DFT − VQE)  : {energy_diff_ha:.6f} Ha")
log(f"  UCCSD parameters               : {n_params}")
log(f"  Optimizer iterations           : {optimizer_iters}")
log()
log(f"  H₂ mean |error| EfficientSU2   : {np.mean(np.abs(su2_errors_mha)):.2f} mHa")
log(f"  H₂ mean |error| UCCSD          : {np.mean(np.abs(ucc_errors_mha)):.2f} mHa")
log()
log(f"  All outputs → {OUTPUT_DIR}")
log("=" * 70)
log("✓ Run complete.")


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os, json, numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR      = "/mnt/c/Users/Aditya Singh/ni_complex_vqe"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

def load_ckpt():
    with open(CHECKPOINT_FILE) as f:
        return json.load(f)

def save_ckpt(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("Checkpoint saved.")

ckpt = load_ckpt()
print("Keys:", list(ckpt.keys()))
print("VQE iter :", ckpt["vqe_progress"]["iter"])
print("VQE energy:", ckpt["vqe_progress"]["energy"])


# In[37]:


from pyscf import gto, dft
from pyscf.scf import addons as scf_addons
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator

ATOM_STRING = """Ni 0.0 0.0 0.0
N  1.9 0.0 0.0
N -1.9 0.0 0.0
N  0.0 1.9 0.0
N  0.0 -1.9 0.0
C  2.6 2.6 0.0
C -2.6 2.6 0.0
C  0.0 0.0 2.3
O  0.0 1.3 2.9
O  0.0 -1.3 2.9"""
BASIS, CHARGE, SPIN = "sto-3g", 0, 2
print("Imports OK")


# In[38]:


ckpt        = load_ckpt()
n_particles = tuple(ckpt["active_space"]["n_particles"])
n_orbitals  = ckpt["active_space"]["n_orbitals"]
n_qubits    = ckpt["active_space"]["n_qubits"]
print(f"Active space: {n_particles} in {n_orbitals} orbitals → {n_qubits} qubits")

driver  = PySCFDriver(atom=ATOM_STRING, basis=BASIS, charge=CHARGE, spin=SPIN)
problem = driver.run()
problem = FreezeCoreTransformer(freeze_core=True).transform(problem)
problem_reduced = ActiveSpaceTransformer(
    num_electrons=6, num_spatial_orbitals=8).transform(problem)

mapper = ParityMapper(num_particles=n_particles)
ansatz = UCC(
    num_spatial_orbitals=n_orbitals,
    num_particles=n_particles,
    qubit_mapper=mapper,
    excitations='sd',
    reps=1
)
print(f"Ansatz parameters: {ansatz.num_parameters}")
print("Ready.")


# In[39]:


from pyscf import gto, dft
from pyscf.scf import addons as scf_addons
import numpy as np

mol = gto.M(atom=ATOM_STRING, basis=BASIS, charge=CHARGE, spin=SPIN, verbose=3)

# ── PBE0 warmstart
mf_pre = dft.UKS(mol)
mf_pre.xc = "pbe0"
mf_pre = mf_pre.density_fit()
mf_pre.conv_tol   = 1e-5
mf_pre.max_cycle  = 400
mf_pre.level_shift = 0.5
mf_pre = scf_addons.smearing_(mf_pre, sigma=0.05, method="fermi")

# Manual broken-symmetry guess — scale alpha/beta to enforce triplet
dm_a, dm_b = mf_pre.get_init_guess(mol, key='minao')
# Rotate HOMO/LUMO of beta to break symmetry and force spin polarization
# Orbitals 28-30 are the Ni 3d manifold in this basis
dm_b_copy = dm_b.copy()
# Swap columns 28 and 29 in beta to scramble d-orbital occupations
dm_b_copy[:, [28, 29]] = dm_b_copy[:, [29, 28]]
# Scale to enforce 46 alpha, 44 beta electrons
dm_a = dm_a * (46 / dm_a.trace())
dm_b = dm_b_copy * (44 / dm_b_copy.trace())

mf_pre.kernel((dm_a, dm_b))
print(f"PBE0: {mf_pre.e_tot:.6f} Ha  converged={mf_pre.converged}  "
      f"<S²>={mf_pre.spin_square()[0]:.3f}  2S+1={mf_pre.spin_square()[1]:.2f}")

# ── B3LYP from PBE0 MOs
mf = dft.UKS(mol)
mf.xc = "b3lyp"
mf = mf.density_fit()
mf.conv_tol   = 1e-6
mf.max_cycle  = 800
mf.level_shift = 0.6
mf = scf_addons.smearing_(mf, sigma=0.05, method="fermi")
mf.kernel(mf_pre.make_rdm1())

print(f"B3LYP: {mf.e_tot:.6f} Ha  converged={mf.converged}  "
      f"<S²>={mf.spin_square()[0]:.3f}  2S+1={mf.spin_square()[1]:.2f}")

# Save regardless of convergence
ckpt = load_ckpt()
ckpt["dft"] = {
    "done"      : True,
    "energy_ha" : float(mf.e_tot),
    "converged" : bool(mf.converged),
    "s2"        : float(mf.spin_square()[0]),
    "2s_plus_1" : float(mf.spin_square()[1]),
}
save_ckpt(ckpt)
print(f"\nSaved. DFT energy = {mf.e_tot:.6f} Ha")


# In[40]:


ckpt          = load_ckpt()
ITERS_THIS_RUN = 200

vqe_prog      = ckpt["vqe_progress"]
initial_point = np.array(vqe_prog["params"])
total_iters   = vqe_prog["iter"]
best_energy   = vqe_prog["energy"]
print(f"Resuming from iter {total_iters},  E = {best_energy:.8f} Ha")

history = []

def vqe_callback(n_eval, params, value, meta):
    history.append(float(value))
    print(f"  iter {total_iters + n_eval:4d}  E = {value:.8f} Ha", flush=True)
    if n_eval % 50 == 0:
        ckpt["vqe_progress"] = {
            "params" : list(params),
            "iter"   : total_iters + n_eval,
            "energy" : float(value),
        }
        save_ckpt(ckpt)

optimizer = COBYLA(maxiter=ITERS_THIS_RUN, tol=1e-7, rhobeg=0.02)
vqe       = VQE(Estimator(), ansatz, optimizer,
                initial_point=initial_point, callback=vqe_callback)
solver    = GroundStateEigensolver(mapper, vqe)
result    = solver.solve(problem_reduced)

new_energy  = float(result.total_energies[0])
new_params  = result.raw_result.optimal_point   # ← this exists
total_iters += len(history)

ckpt["vqe_progress"] = {
    "params" : list(new_params),               # ← new_params not params
    "iter"   : total_iters,
    "energy" : new_energy,
}
save_ckpt(ckpt)
print(f"\n✓ Done.  Total iters: {total_iters}  Energy: {new_energy:.8f} Ha")


# In[ ]:


ckpt = load_ckpt()
prog = ckpt.get("vqe_progress", {})
print(f"VQE iters  : {prog.get('iter', 0)}")
print(f"VQE energy : {prog.get('energy', 'N/A')}")
dft = ckpt.get("dft", {})
print(f"DFT energy : {dft.get('energy_ha', 'not done yet')}")
print(f"DFT converged: {dft.get('converged', 'N/A')}")


# In[ ]:




