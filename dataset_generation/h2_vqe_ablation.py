"""
2-1단계: H2 VQE 회로에서 Ablation (물리 큐빗)

목표: H2 화학 회로에서 5개 노이즈 파라미터가 에너지 오차에 미치는 영향 평가

실행: python dataset_generation/h2_vqe_ablation.py
출력: h2_vqe_ablation_results.csv
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import efficient_su2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, reset_error
from qiskit.quantum_info import DensityMatrix
from scipy.optimize import minimize
import pandas as pd
import numpy as np

# ============ Step 1: H2 분자 정의 + FCI 참값 계산 ============
print("Step 1: H2 분자 정의 및 FCI 참값 계산...")
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
problem = driver.run()

# Mapper로 Fermionic -> Qubit Hamiltonian 변환
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.second_q_ops()[0])

# FCI 참값 계산
solver = NumPyMinimumEigensolver()
result = solver.compute_minimum_eigenvalue(hamiltonian)
e_fci = float(result.eigenvalue.real)
print(f"FCI 참값: {e_fci:.6f} hartree\n")

# ============ Step 2: Ansatz 설정 ============
print("Step 2: Ansatz 설정...")
ansatz = efficient_su2(num_qubits=hamiltonian.num_qubits, reps=2)
print(f"Ansatz 파라미터 수: {ansatz.num_parameters}")
print(f"Ansatz depth: {ansatz.decompose().depth()}\n")

# ============ Step 3: Hamiltonian 정보 ============
print("Step 3: Hamiltonian 정보...")
print(f"Hamiltonian 큐빗: {hamiltonian.num_qubits}")
print(f"Pauli term 개수: {len(hamiltonian)}\n")

# ============ Step 4: 노이즈 모델 생성 함수 ============
def create_noise_model(data_depol, meas_flip, reset_flip, gate_depol):
    """주어진 노이즈 파라미터로 NoiseModel 생성"""
    
    noise_model = NoiseModel()
    
    # 1-qubit gate depolarizing
    if gate_depol > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(gate_depol, 1), 
            ['u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's', 't']
        )
    
    # 2-qubit gate depolarizing (CX)
    if gate_depol > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(gate_depol, 2), ['cx']
        )
    
    # Reset error
    if reset_flip > 0:
        noise_model.add_all_qubit_quantum_error(
            reset_error(reset_flip), ['reset']
        )
    
    # Measurement error (readout)
    if meas_flip > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(2 * meas_flip, 1), ['measure']
        )
    
    # Data qubit depolarizing (idle 동안)
    if data_depol > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(data_depol, 1), ['id']
        )
    
    return noise_model

# ============ Step 5: VQE 최적화 함수 ============
def optimize_vqe(ansatz, hamiltonian, noise_model, experiment_name):
    """VQE를 돌려서 최적 파라미터 찾기"""
    print(f"  최적화 중: {experiment_name}...")
    
    def objective(params):
        """에너지 기댓값 계산"""
        qc = ansatz.assign_parameters(params)
        dm = DensityMatrix.from_instruction(qc)
        energy = float(np.real(np.vdot(dm.data, hamiltonian.to_matrix() @ dm.data)))
        return energy
    
    # 초기 파라미터
    x0 = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)
    
    # COBYLA로 최적화
    result = minimize(objective, x0, method='COBYLA', options={'maxiter': 100})
    
    optimal_params = result.x
    optimal_energy = result.fun
    
    return optimal_params, optimal_energy

# ============ Step 6: 에너지 평가 함수 ============
def evaluate_energy(ansatz, hamiltonian, params):
    """주어진 파라미터로 에너지 계산"""
    qc = ansatz.assign_parameters(params)
    dm = DensityMatrix.from_instruction(qc)
    energy = float(np.real(np.vdot(dm.data, hamiltonian.to_matrix() @ dm.data)))
    return energy

# ============ Step 7: 5가지 설정 실행 ============
print("Step 7: Ablation 실행 (5가지 설정)...\n")

configs = {
    "Baseline": {
        "data_depol": 0.001,
        "meas_flip": 0.01,
        "reset_flip": 0.01,
        "gate_depol": 0.008
    },
    "No data_depol": {
        "data_depol": 0.0,
        "meas_flip": 0.01,
        "reset_flip": 0.01,
        "gate_depol": 0.008
    },
    "No meas_flip": {
        "data_depol": 0.001,
        "meas_flip": 0.0,
        "reset_flip": 0.01,
        "gate_depol": 0.008
    },
    "No reset_flip": {
        "data_depol": 0.001,
        "meas_flip": 0.01,
        "reset_flip": 0.0,
        "gate_depol": 0.008
    },
    "No gate_depol": {
        "data_depol": 0.001,
        "meas_flip": 0.01,
        "reset_flip": 0.01,
        "gate_depol": 0.0
    },
}

results = {}
baseline_error = None

for name, params in configs.items():
    noise_model = create_noise_model(**params)
    optimal_params, optimal_energy = optimize_vqe(ansatz, hamiltonian, noise_model, name)
    
    error = abs(optimal_energy - e_fci)
    error_percent = (error / abs(e_fci)) * 100
    
    if name == "Baseline":
        baseline_error = error
    
    error_reduction = ((baseline_error - error) / baseline_error * 100) if baseline_error else 0
    
    results[name] = {
        "Energy (hartree)": optimal_energy,
        "Error ΔE (hartree)": error,
        "Error %": error_percent,
        "Error reduction %": error_reduction
    }
    
    print(f"  {name}: E={optimal_energy:.6f}, ΔE={error:.6f}, reduction={error_reduction:.1f}%")

print()

# ============ Step 8: 결과 표 정리 ============
print("Step 8: 결과 표 정리...")

df = pd.DataFrame([
    {
        "Experiment": name,
        "Energy (hartree)": results[name]["Energy (hartree)"],
        "Error ΔE (hartree)": results[name]["Error ΔE (hartree)"],
        "Error %": results[name]["Error %"],
        "Error reduction %": results[name]["Error reduction %"]
    }
    for name in results
])

print("\n" + "="*100)
print(df.to_string(index=False))
print("="*100 + "\n")

df.to_csv("h2_vqe_ablation_results.csv", index=False)
print("결과 저장: h2_vqe_ablation_results.csv")

print("\n결론:")
print("어떤 노이즈 파라미터가 가장 큰 영향을 미치는지 위 표의 'Error reduction %' 열을 보면 된다.")
print("가장 큰 reduction %를 가진 파라미터가 dominant한 노이즈다.")