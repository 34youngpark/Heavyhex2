from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

# H2 Trotter 회로
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
problem = driver.run()
fermionic_op = problem.hamiltonian.second_q_op()

mapper = JordanWignerMapper()
qubit_op = mapper.map(fermionic_op)

evo_gate = PauliEvolutionGate(
    qubit_op,
    time=0.5,
    synthesis=LieTrotter(reps=1),
)

qc = QuantumCircuit(qubit_op.num_qubits)
num_alpha, num_beta = problem.num_particles
for i in range(num_alpha):
    qc.x(i)
for i in range(num_beta):
    qc.x(i + qubit_op.num_qubits // 2)

qc.append(evo_gate, range(qubit_op.num_qubits))
qc_decomposed = qc.decompose(reps=4)

print(f"큐빗 수: {qc_decomposed.num_qubits}")
print(f"회로 depth: {qc_decomposed.depth()}\n")

# Gate schedule 분석 + Idle time 계산
GATE_DURATIONS_NS = {
    "cx": 500.0,
    "cz": 500.0,
    "x": 35.0,
    "sx": 35.0,
    "rz": 0.0,
    "u": 35.0,
    "u1": 0.0,
    "u2": 35.0,
    "u3": 35.0,
    "id": 35.0,
    "measure": 1000.0,
    "reset": 1000.0,
    "barrier": 0.0,
}

def gate_duration(instr_name: str) -> float:
    return GATE_DURATIONS_NS.get(instr_name.lower(), 35.0)

dag = circuit_to_dag(qc_decomposed)
n_qubits = qc_decomposed.num_qubits

idle_total_ns = [0.0] * n_qubits
idle_events = [0] * n_qubits
n_layers = 0

for layer in dag.layers():
    layer_circuit = layer["graph"]
    ops = layer_circuit.op_nodes()
    if not ops:
        continue

    qubit_busy_ns = {i: 0.0 for i in range(n_qubits)}
    for node in ops:
        dur = gate_duration(node.op.name)
        for qubit in node.qargs:
            qidx = qc_decomposed.find_bit(qubit).index
            qubit_busy_ns[qidx] = max(qubit_busy_ns[qidx], dur)

    t_max = max(qubit_busy_ns.values())
    if t_max == 0.0:
        continue

    n_layers += 1
    for i in range(n_qubits):
        idle = t_max - qubit_busy_ns[i]
        if idle > 0:
            idle_total_ns[i] += idle
            idle_events[i] += 1

print(f"물리적 시간이 소모되는 레이어 수: {n_layers}\n")
print(f"{'큐빗':<6}{'총 Idle(ns)':<15}{'평균 Idle/cycle(ns)':<22}{'Idle event 횟수':<15}")
for i in range(n_qubits):
    avg = idle_total_ns[i] / n_layers if n_layers else 0
    print(f"Q{i:<5}{idle_total_ns[i]:<15.1f}{avg:<22.2f}{idle_events[i]:<15}")

# ============ Step 3-4: idle_depol 값 도출 ============
T1_NS = 100_000.0  # 100 us
avg_idle_per_layer = sum(idle_total_ns) / n_qubits / n_layers if n_layers else 0
idle_depol_candidate = avg_idle_per_layer / T1_NS

print(f"\n전체 평균 idle time/layer: {avg_idle_per_layer:.2f} ns")
print(f"가정: T1 = {T1_NS/1000:.0f} us")
print(f"도출된 idle_depol 후보값: {idle_depol_candidate:.5f}")
print("-> noise_profiles.json에 'h2_idle_adapted'로 추가할 것")