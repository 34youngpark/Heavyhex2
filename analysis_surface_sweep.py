"""
2-2단계: Surface Code CNN 결과 분석 (수정)
"""

import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

print("Step 1: 결과 파일 수집...\n")

result_files = glob.glob("results/train/CNN_*.csv")
results = {}

for filepath in sorted(result_files):
    match = re.search(r'id([\d.]+)\.csv', filepath)
    if match:
        idle_depol = float(match.group(1))
        
        df = pd.read_csv(filepath)
        
        # 마지막 행의 Val_LER (올바른 컬럼)
        if len(df) > 0:
            last_row = df.iloc[-1]
            ler = last_row['Val_LER']  # ← 올바른 컬럼
            
            results[idle_depol] = ler
            print(f"idle_depol = {idle_depol:.4f} | Val_LER = {ler:.4f}")

print()

sorted_idle_depol = sorted(results.keys())
sorted_ler = [results[x] for x in sorted_idle_depol]

df_result = pd.DataFrame({
    "idle_depol": sorted_idle_depol,
    "Val_LER": sorted_ler
})

print("="*60)
print(df_result.to_string(index=False))
print("="*60 + "\n")

df_result.to_csv("surface_code_idle_depol_sweep.csv", index=False)

# 그래프
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sorted_idle_depol, sorted_ler, 'o-', linewidth=2, markersize=8, 
        color='green', label='Surface Code CNN')
ax.axvline(x=0.002, color='red', linestyle='--', linewidth=2, 
           label='Optimal from VQE (≈0.002)')
ax.set_xlabel('idle_depol', fontsize=12)
ax.set_ylabel('Logical Error Rate (Val_LER)', fontsize=12)
ax.set_title('Surface Code: idle_depol vs LER', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('surface_code_idle_depol_sweep.png', dpi=150)
print("그래프 저장: surface_code_idle_depol_sweep.png")