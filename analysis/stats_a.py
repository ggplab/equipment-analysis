"""
stats_a.py
Mission A 건설장비 — 가설 1~5 통계검정
유의수준 5% (alpha = 0.05)
"""
import pandas as pd
from scipy import stats

DATA_PATH = "/Users/limjung/Projects/equipment-analysis/data/construction_machine_data.csv"

df = pd.read_csv(DATA_PATH)

# 컬럼 이름 공백 제거 (BOM 등 대비)
df.columns = df.columns.str.strip()

RUL = df["Remaining_Useful_Life"]
HOURS = df["Operating_Hours"]
VIB = df["Vibration"]
TEMP = df["Temperature"]
PRES = df["Pressure"]
CTYPE = df["Component_Type"]

print("=" * 60)
print("Mission A — 통계검정 결과 (유의수준 α = 0.05)")
print("=" * 60)
print(f"총 샘플 수: {len(df)}")
print()

# ── 가설 1: Pearson 상관 — Operating_Hours vs RUL ──────────
r1, p1 = stats.pearsonr(HOURS, RUL)
sig1 = "유의함 (p < 0.05)" if p1 < 0.05 else "유의하지 않음"
print("[가설 1] Pearson 상관 — Operating_Hours vs RUL")
print(f"  r = {r1:.4f},  p = {p1:.4e}")
print(f"  → {sig1}")
print()

# ── 가설 2: Pearson 상관 — Vibration vs RUL ──────────────
r2, p2 = stats.pearsonr(VIB, RUL)
sig2 = "유의함 (p < 0.05)" if p2 < 0.05 else "유의하지 않음"
print("[가설 2] Pearson 상관 — Vibration vs RUL")
print(f"  r = {r2:.4f},  p = {p2:.4e}")
print(f"  → {sig2}")
print()

# ── 가설 3: One-way ANOVA — Component_Type ───────────────
grp_engine = RUL[CTYPE == "Engine"]
grp_hyd    = RUL[CTYPE == "Hydraulic Cylinder"]
grp_gear   = RUL[CTYPE == "Gear"]
F3, p3 = stats.f_oneway(grp_engine, grp_hyd, grp_gear)
sig3 = "유의함 (p < 0.05)" if p3 < 0.05 else "유의하지 않음"
print("[가설 3] 일원분산분석(ANOVA) — Component_Type × RUL")
print(f"  Engine n={len(grp_engine)}, mean={grp_engine.mean():.1f}h")
print(f"  Hydraulic Cylinder n={len(grp_hyd)}, mean={grp_hyd.mean():.1f}h")
print(f"  Gear   n={len(grp_gear)}, mean={grp_gear.mean():.1f}h")
print(f"  F = {F3:.4f},  p = {p3:.4e}")
print(f"  → {sig3}")
print()

# ── 가설 4: Pearson 상관 — Temperature & Pressure vs RUL ──
r4t, p4t = stats.pearsonr(TEMP, RUL)
r4p, p4p = stats.pearsonr(PRES, RUL)
sig4t = "유의함" if p4t < 0.05 else "유의하지 않음"
sig4p = "유의함" if p4p < 0.05 else "유의하지 않음"
print("[가설 4a] Pearson 상관 — Temperature vs RUL")
print(f"  r = {r4t:.4f},  p = {p4t:.4e}  → {sig4t}")
print()
print("[가설 4b] Pearson 상관 — Pressure vs RUL")
print(f"  r = {r4p:.4f},  p = {p4p:.4e}  → {sig4p}")
print()

# ── 가설 5: Welch t-검정 — 복합위험군 vs 나머지 ─────────
q75_hours = HOURS.quantile(0.75)
q75_vib   = VIB.quantile(0.75)
mask_both  = (HOURS >= q75_hours) & (VIB >= q75_vib)
mask_other = ~mask_both

grp_both  = RUL[mask_both]
grp_other = RUL[mask_other]
t5, p5    = stats.ttest_ind(grp_both, grp_other, equal_var=False)   # Welch
mean_diff = grp_both.mean() - grp_other.mean()

sig5 = "유의함 (p < 0.05)" if p5 < 0.05 else "유의하지 않음"
print("[가설 5] Welch t-검정 — 복합위험군 vs 나머지")
print(f"  복합위험군 (n={len(grp_both)}): mean = {grp_both.mean():.1f}h")
print(f"  나머지     (n={len(grp_other)}): mean = {grp_other.mean():.1f}h")
print(f"  평균 차이: {mean_diff:.1f}h")
print(f"  t = {t5:.4f},  p = {p5:.4e}")
print(f"  Q75 기준 — 가동시간 ≥ {q75_hours:.0f}h, 진동 ≥ {q75_vib:.4f}")
print(f"  → {sig5}")
print()

print("=" * 60)
print("판정 요약")
print("=" * 60)
for i, (r_val, p_val, label) in enumerate([
    (r1, p1, "가설1 Pearson(Hours vs RUL)"),
    (r2, p2, "가설2 Pearson(Vibration vs RUL)"),
    (None, p3, "가설3 ANOVA(Component_Type)"),
    (r4t, p4t, "가설4a Pearson(Temp vs RUL)"),
    (r4p, p4p, "가설4b Pearson(Pressure vs RUL)"),
    (None, p5, "가설5 Welch-t(복합위험)"),
], 1):
    judg = "채택" if p_val < 0.05 else "기각"
    r_str = f"r={r_val:.4f}, " if r_val is not None else ""
    print(f"  {label}: {r_str}p={p_val:.4e} → {judg}")
