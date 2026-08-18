from heavyhex33_stim import build_stim_circuit, LOGICAL_Z_IDX
import numpy as np

# 노이즈 0 설정으로 회로 생성
c = build_stim_circuit(num_cycles=3, noise_type="X", p=0.0,
                       noise_profile="ideal/dp0_mf0_rf0_gd0")

# 샘플링
sampler = c.compile_sampler()
samples = sampler.sample(shots=100)

# 마지막 17개 bit는 data measurement
data_bits = samples[:, -17:]

# logical Z parity 계산 (LOGICAL_Z_IDX에 해당하는 bit들의 XOR)
logical_z_parity = np.zeros(100, dtype=np.uint8)
for idx in LOGICAL_Z_IDX:
    logical_z_parity ^= data_bits[:, idx]

# 검증: 모두 0이어야 함
print(f"Logical Z parity (should all be 0): {logical_z_parity}")
print(f"All zero? {np.all(logical_z_parity == 0)}")