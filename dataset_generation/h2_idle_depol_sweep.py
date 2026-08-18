#!/usr/bin/env python3
"""
H2 VQE - idle_depol sensitivity (CORRECTED)
실제 density_matrix로 noisy 에너지를 계산합니다.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit.library import UCCSD, HartreeFock
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from scipy.optimize import minimize

# ==================================================================
DIST = 0.735
BASIS = "sto-3g"
T1_NS = 100_000  # 100 us

# ==================================================================
print("[1] H2 정의 및 FCI 참값...")
driver = PySCFDriver(atom=f"H 0 0 0; H 0 0 {DIST}", basis=BASIS)
problem = driver.run()

mapper = JordanWignerMapper()
H_op = mapper.map(problem.hamiltonian.second_q_op())
e_nuc = problem.nuclear_repulsion_energy

solver = NumPyMinimumEigensolver()
result = solver.compute_minimum_eigenvalue(H_op)
e_fci_elec = float(result.eigenvalue.real)
e_fci = e_fci_elec + e_nuc

print(f"  FCI (total): {e_fci:+.6f} Ha")

# ==================================================================
print("[2] UCCSD ansatz...")
hf_state = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
               initial_state=hf_state)
n_params = ansatz.num_parameters
print(f"  params: {n_params}")

# ==================================================================
print("[3] Noiseless VQE optimize...")
H_matrix = H_op.to_matrix()

def compute_energy(params):
    bound_circuit = ansatz.assign_parameters(params)
    sv = Statevector(bound_circuit)
    sv_array = sv.data
    return float(np.real(np.conj(sv_array) @ H_matrix @ sv_array)) + e_nuc

rng = np.random.default_rng(42)
best_energy = np.inf
best_params = None

for restart in range(3):
    if restart == 0:
        x0 = np.zeros(n_params)
    else:
        x0 = rng.normal(0, 0.1, n_params)
    
    result = minimize(compute_energy, x0, method="SLSQP",
                     options={"maxiter": 500, "ftol": 1e-10})
    if result.fun < best_energy:
        best_energy = result.fun
        best_params = result.x

e_opt = best_energy
gap = abs(e_opt - e_fci)
print(f"  최적 에너지 (noiseless): {e_opt:+.6f} hartree")
print(f"  FCI 대비 차이: {gap*1000:.2f} mHa")

# ==================================================================
print("[4] 회로 준비...")
bound_circuit = ansatz.assign_parameters(best_params)
hw_circuit = transpile(bound_circuit.decompose(reps=4),
                       basis_gates=['rz', 'sx', 'x', 'cx'],
                       optimization_level=1)
print(f"  circuit depth: {hw_circuit.depth()}")

# ==================================================================
print("[5] Idle_depol sweep...")

idle_depol_values = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0023, 0.003, 0.004, 0.005]
results = []

for idle_p in idle_depol_values:
    # Noise model
    nm = NoiseModel(basis_gates=['rz', 'sx', 'x', 'cx', 'id'])
    
    nm.add_all_qubit_quantum_error(depolarizing_error(0.0008, 1), ['sx', 'x'])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.008, 2), ['cx'])
    nm.add_all_qubit_quantum_error(depolarizing_error(idle_p, 1), ['id'])
    
    # Density matrix simulator
    sim = AerSimulator(method='density_matrix', noise_model=nm,
                       basis_gates=['rz', 'sx', 'x', 'cx', 'id'])
    
    qc = hw_circuit.copy()
    qc.save_density_matrix()
    
    qc_t = transpile(qc, sim, basis_gates=['rz', 'sx', 'x', 'cx', 'id'],
                     optimization_level=0)
    
    job = sim.run(qc_t, shots=1)
    rho = np.asarray(job.result().data(0)['density_matrix'])
    
    # E = Tr(rho * H)
    e_noisy = np.real(np.trace(rho @ H_matrix)) + e_nuc
    delta_e = abs(e_noisy - e_fci)
    
    results.append({
        'idle_depol': idle_p,
        'Energy (hartree)': e_noisy,
        'Error ΔE (hartree)': delta_e,
        'Error %': delta_e / abs(e_fci) * 100
    })
    
    print(f"  idle_depol = {idle_p:.4f}... E = {e_noisy:+.6f}, ΔE = {delta_e:.6f}")

df = pd.DataFrame(results)
print("\n" + "="*90)
print(df.to_string(index=False))
print("="*90)

df.to_csv("h2_idle_depol_sweep.csv", index=False)
print("h2_idle_depol_sweep.csv 저장됨")

# Sanity check
errors = df['Error ΔE (hartree)'].values
is_monotonic = np.all(np.diff(errors) >= -1e-7)
if is_monotonic:
    print("✓ Error는 단조 증가합니다 (정상)")
else:
    print("⚠️  Error가 단조 증가하지 않습니다 (버그 의심)")

# 그래프
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['idle_depol'], df['Error %'], 'o-', lw=2, ms=8, color='C0')
ax.set_xlabel('idle_depol', fontsize=12)
ax.set_ylabel('|E - E_FCI| / |E_FCI| (%)', fontsize=12)
ax.set_title('H2 VQE: Idle Noise Sensitivity (Density Matrix)', fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('h2_idle_depol_sweep.png', dpi=150)
print("h2_idle_depol_sweep.png 저장됨")
