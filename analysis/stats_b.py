"""
stats_b.py - 제조장비 이상탐지 통계 검정
가설 1~5에 대한 Welch t-검정 및 카이제곱 검정
"""
import pandas as pd
import numpy as np
from scipy import stats

# 데이터 로드
df = pd.read_csv('/Users/limjung/Projects/equipment-analysis/data/equipment_anomaly_data.csv')
df['faulty'] = df['faulty'].astype(int)

normal = df[df['faulty'] == 0]
faulty = df[df['faulty'] == 1]

print(f"전체 데이터: {len(df)}건")
print(f"정상: {len(normal)}건 / 고장: {len(faulty)}건")
print("=" * 60)

# ──────────────────────────────────────────────
# 가설 1. 진동 — Welch t-검정
# ──────────────────────────────────────────────
print("\n[가설 1] 진동 Welch t-검정")
t1, p1 = stats.ttest_ind(faulty['vibration'], normal['vibration'], equal_var=False)
print(f"  정상 평균: {normal['vibration'].mean():.4f}")
print(f"  고장 평균: {faulty['vibration'].mean():.4f}")
print(f"  t = {t1:.4f}, p = {p1:.6e}")
if p1 < 0.001:
    print(f"  → p < 0.001 (유의)")
else:
    print(f"  → p = {p1:.3f} ({'유의' if p1 < 0.05 else '유의하지 않음'})")

# ──────────────────────────────────────────────
# 가설 2. 온도 & 압력 — Welch t-검정 각각
# ──────────────────────────────────────────────
print("\n[가설 2a] 온도 Welch t-검정")
t2t, p2t = stats.ttest_ind(faulty['temperature'], normal['temperature'], equal_var=False)
print(f"  정상 평균: {normal['temperature'].mean():.4f}")
print(f"  고장 평균: {faulty['temperature'].mean():.4f}")
print(f"  t = {t2t:.4f}, p = {p2t:.6e}")
if p2t < 0.001:
    print(f"  → p < 0.001 (유의)")
else:
    print(f"  → p = {p2t:.3f} ({'유의' if p2t < 0.05 else '유의하지 않음'})")

print("\n[가설 2b] 압력 Welch t-검정")
t2p, p2p = stats.ttest_ind(faulty['pressure'], normal['pressure'], equal_var=False)
print(f"  정상 평균: {normal['pressure'].mean():.4f}")
print(f"  고장 평균: {faulty['pressure'].mean():.4f}")
print(f"  t = {t2p:.4f}, p = {p2p:.6e}")
if p2p < 0.001:
    print(f"  → p < 0.001 (유의)")
else:
    print(f"  → p = {p2p:.3f} ({'유의' if p2p < 0.05 else '유의하지 않음'})")

# ──────────────────────────────────────────────
# 가설 3. 습도 — Welch t-검정
# ──────────────────────────────────────────────
print("\n[가설 3] 습도 Welch t-검정")
t3, p3 = stats.ttest_ind(faulty['humidity'], normal['humidity'], equal_var=False)
print(f"  정상 평균: {normal['humidity'].mean():.4f}")
print(f"  고장 평균: {faulty['humidity'].mean():.4f}")
print(f"  t = {t3:.4f}, p = {p3:.6e}")
if p3 < 0.001:
    print(f"  → p < 0.001 (유의)")
else:
    print(f"  → p = {p3:.3f} ({'유의' if p3 < 0.05 else '유의하지 않음'})")

# ──────────────────────────────────────────────
# 가설 4. 장비종류 & 지역 — 카이제곱 독립성 검정
# ──────────────────────────────────────────────
print("\n[가설 4a] 장비종류 × faulty 카이제곱 독립성 검정")
ct_equip = pd.crosstab(df['equipment'], df['faulty'])
chi2_e, p_e, dof_e, expected_e = stats.chi2_contingency(ct_equip)
print(f"  분할표:\n{ct_equip}")
print(f"  χ² = {chi2_e:.4f}, 자유도 = {dof_e}, p = {p_e:.6e}")
if p_e < 0.001:
    print(f"  → p < 0.001 (유의)")
else:
    print(f"  → p = {p_e:.3f} ({'유의' if p_e < 0.05 else '유의하지 않음'})")
# 장비별 고장률
print("  장비별 고장률:")
for equip in df['equipment'].unique():
    sub = df[df['equipment'] == equip]
    rate = sub['faulty'].mean() * 100
    print(f"    {equip}: {rate:.1f}%")

print("\n[가설 4b] 지역 × faulty 카이제곱 독립성 검정")
ct_loc = pd.crosstab(df['location'], df['faulty'])
chi2_l, p_l, dof_l, expected_l = stats.chi2_contingency(ct_loc)
print(f"  분할표:\n{ct_loc}")
print(f"  χ² = {chi2_l:.4f}, 자유도 = {dof_l}, p = {p_l:.6e}")
if p_l < 0.001:
    print(f"  → p < 0.001 (유의)")
else:
    print(f"  → p = {p_l:.3f} ({'유의' if p_l < 0.05 else '유의하지 않음'})")
# 지역별 고장률
print("  지역별 고장률:")
loc_rates = {}
for loc in df['location'].unique():
    sub = df[df['location'] == loc]
    rate = sub['faulty'].mean() * 100
    loc_rates[loc] = rate
    print(f"    {loc}: {rate:.1f}%")
max_rate = max(loc_rates.values())
min_rate = min(loc_rates.values())
print(f"  최대-최소 차이: {max_rate - min_rate:.1f}%p")

# ──────────────────────────────────────────────
# 가설 5. 복합조건(진동 2.0~2.6 & 온도 >=100) × faulty — 카이제곱
# ──────────────────────────────────────────────
print("\n[가설 5] 복합조건 × faulty 카이제곱 검정")
print("  복합조건: 진동 2.0~2.6 AND 온도 >= 100°C")
df['complex_cond'] = ((df['vibration'] >= 2.0) & (df['vibration'] < 2.6) & (df['temperature'] >= 100)).astype(int)
ct_cplx = pd.crosstab(df['complex_cond'], df['faulty'])
print(f"  분할표 (행: 복합조건 충족여부 0/1, 열: faulty 0/1):\n{ct_cplx}")
chi2_5, p_5, dof_5, expected_5 = stats.chi2_contingency(ct_cplx)
print(f"  χ² = {chi2_5:.4f}, 자유도 = {dof_5}, p = {p_5:.6e}")
if p_5 < 0.001:
    print(f"  → p < 0.001 (유의)")
else:
    print(f"  → p = {p_5:.3f} ({'유의' if p_5 < 0.05 else '유의하지 않음'})")
# 복합조건 충족 그룹 고장률
cond_met = df[df['complex_cond'] == 1]
cond_not = df[df['complex_cond'] == 0]
print(f"  복합조건 충족 {len(cond_met)}건 중 고장: {cond_met['faulty'].sum()}건 ({cond_met['faulty'].mean()*100:.1f}%)")
print(f"  복합조건 미충족 {len(cond_not)}건 중 고장: {cond_not['faulty'].sum()}건 ({cond_not['faulty'].mean()*100:.1f}%)")

# ──────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 요약 (HTML 박스 삽입용)")
print("=" * 60)

def fmt_p(p):
    if p < 0.001:
        return "p &lt; 0.001"
    else:
        return f"p = {p:.3f}"

print(f"\n가설1  vibration t검정: t = {t1:.2f}, {fmt_p(p1)}, {'유의' if p1 < 0.05 else '유의하지 않음'}")
print(f"가설2a temperature t검정: t = {t2t:.2f}, {fmt_p(p2t)}, {'유의' if p2t < 0.05 else '유의하지 않음'}")
print(f"가설2b pressure t검정:    t = {t2p:.2f}, {fmt_p(p2p)}, {'유의' if p2p < 0.05 else '유의하지 않음'}")
print(f"가설3  humidity t검정:    t = {t3:.2f}, {fmt_p(p3)}, {'유의' if p3 < 0.05 else '유의하지 않음'}")
print(f"가설4a equipment χ²검정: χ² = {chi2_e:.2f}, df={dof_e}, {fmt_p(p_e)}, {'유의' if p_e < 0.05 else '유의하지 않음'}")
print(f"가설4b location  χ²검정: χ² = {chi2_l:.2f}, df={dof_l}, {fmt_p(p_l)}, {'유의' if p_l < 0.05 else '유의하지 않음'}")
print(f"가설5  복합조건 χ²검정:  χ² = {chi2_5:.2f}, df={dof_5}, {fmt_p(p_5)}, {'유의' if p_5 < 0.05 else '유의하지 않음'}")
