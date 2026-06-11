"""
Mission B — 제조장비 이상탐지 분석
입문자 수준: 개수 세기, 비율(%), 그룹별 평균, 히스토그램, 막대그래프, 산점도만 사용
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── 한글 폰트 설정 ──
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'data', 'equipment_anomaly_data.csv')
CHART_DIR = os.path.join(BASE, 'docs', 'charts', 'b')
os.makedirs(CHART_DIR, exist_ok=True)

# ── 데이터 로드 ──
df = pd.read_csv(DATA_PATH)
df['faulty'] = df['faulty'].astype(int)

total = len(df)
fault_total = df['faulty'].sum()
normal_total = total - fault_total
fault_rate = fault_total / total * 100

print(f"전체 행수: {total}")
print(f"고장(faulty=1): {fault_total}건  ({fault_rate:.1f}%)")
print(f"정상(faulty=0): {normal_total}건")

normal_df = df[df['faulty'] == 0]
fault_df  = df[df['faulty'] == 1]

# ── 그룹별 평균 ──
for col in ['temperature', 'pressure', 'vibration', 'humidity']:
    nm = normal_df[col].mean()
    fm = fault_df[col].mean()
    print(f"{col:12s}  정상 평균: {nm:.3f}   고장 평균: {fm:.3f}   차이: {fm-nm:+.3f}")

# =============================================================================
# Chart 1: 가설1 — 진동 히스토그램 (정상 vs 고장 겹침)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax = axes[0]
bins = np.linspace(df['vibration'].min(), df['vibration'].max(), 35)
ax.hist(normal_df['vibration'], bins=bins, alpha=0.65, color='#2563eb', label='정상', edgecolor='white', linewidth=0.5)
ax.hist(fault_df['vibration'],  bins=bins, alpha=0.65, color='#dc2626', label='고장', edgecolor='white', linewidth=0.5)
ax.axvline(fault_df['vibration'].mean(), color='#dc2626', linestyle='--', linewidth=1.5, label=f'고장 평균 {fault_df["vibration"].mean():.2f}')
ax.axvline(normal_df['vibration'].mean(), color='#2563eb', linestyle='--', linewidth=1.5, label=f'정상 평균 {normal_df["vibration"].mean():.2f}')
ax.set_xlabel('진동 수치')
ax.set_ylabel('장비 수(건)')
ax.set_title('진동 분포: 정상 vs 고장')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = axes[1]
labels = ['정상', '고장']
means  = [normal_df['vibration'].mean(), fault_df['vibration'].mean()]
colors = ['#2563eb', '#dc2626']
bars = ax2.bar(labels, means, color=colors, width=0.45, edgecolor='white')
for bar, val in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel('진동 평균')
ax2.set_title('그룹별 진동 평균 비교')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, max(means) * 1.25)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'b1_vibration.png'), dpi=150, bbox_inches='tight')
plt.close()
print("저장: b1_vibration.png")

# =============================================================================
# Chart 2: 가설2 — 온도·압력 히스토그램 (정상 vs 고장)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for idx, (col, label) in enumerate([('temperature', '온도'), ('pressure', '압력')]):
    ax = axes[idx]
    bins = np.linspace(df[col].min(), df[col].max(), 30)
    ax.hist(normal_df[col], bins=bins, alpha=0.65, color='#2563eb', label='정상', edgecolor='white', linewidth=0.5)
    ax.hist(fault_df[col],  bins=bins, alpha=0.65, color='#dc2626', label='고장', edgecolor='white', linewidth=0.5)
    nm = normal_df[col].mean()
    fm = fault_df[col].mean()
    ax.axvline(fm, color='#dc2626', linestyle='--', linewidth=1.5, label=f'고장 평균 {fm:.1f}')
    ax.axvline(nm, color='#2563eb', linestyle='--', linewidth=1.5, label=f'정상 평균 {nm:.1f}')
    ax.set_xlabel(label)
    ax.set_ylabel('장비 수(건)')
    ax.set_title(f'{label} 분포: 정상 vs 고장')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'b2_temp_pressure.png'), dpi=150, bbox_inches='tight')
plt.close()
print("저장: b2_temp_pressure.png")

# =============================================================================
# Chart 3: 가설3 — 습도 히스토그램 (정상 vs 고장)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax = axes[0]
bins = np.linspace(df['humidity'].min(), df['humidity'].max(), 25)
ax.hist(normal_df['humidity'], bins=bins, alpha=0.65, color='#2563eb', label='정상', edgecolor='white', linewidth=0.5)
ax.hist(fault_df['humidity'],  bins=bins, alpha=0.65, color='#dc2626', label='고장', edgecolor='white', linewidth=0.5)
nm = normal_df['humidity'].mean()
fm = fault_df['humidity'].mean()
ax.axvline(fm, color='#dc2626', linestyle='--', linewidth=1.5, label=f'고장 평균 {fm:.1f}')
ax.axvline(nm, color='#2563eb', linestyle='--', linewidth=1.5, label=f'정상 평균 {nm:.1f}')
ax.set_xlabel('습도')
ax.set_ylabel('장비 수(건)')
ax.set_title('습도 분포: 정상 vs 고장')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = axes[1]
sensor_cols = ['temperature', 'pressure', 'vibration', 'humidity']
sensor_labels = ['온도', '압력', '진동', '습도']
diffs = [(fault_df[c].mean() - normal_df[c].mean()) / normal_df[c].mean() * 100 for c in sensor_cols]
bar_colors = ['#f59e0b' if abs(d) > 5 else '#94a3b8' for d in diffs]
bars = ax2.bar(sensor_labels, diffs, color=bar_colors, edgecolor='white')
for bar, val in zip(bars, diffs):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + (0.3 if val >= 0 else -0.8),
             f'{val:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.axhline(0, color='#1f2933', linewidth=1)
ax2.set_ylabel('고장 그룹 평균이 정상 대비 차이(%)')
ax2.set_title('센서별 고장/정상 평균 차이 비교')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'b3_humidity_sensor_diff.png'), dpi=150, bbox_inches='tight')
plt.close()
print("저장: b3_humidity_sensor_diff.png")

# =============================================================================
# Chart 4: 가설4 — 장비별·지역별 고장률 막대그래프
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 장비별 고장률
equip_stats = df.groupby('equipment').agg(
    total=('faulty', 'count'),
    faults=('faulty', 'sum')
).reset_index()
equip_stats['rate'] = equip_stats['faults'] / equip_stats['total'] * 100
equip_stats = equip_stats.sort_values('rate', ascending=False)

ax = axes[0]
bars = ax.bar(equip_stats['equipment'], equip_stats['rate'],
              color='#3b82f6', width=0.45, edgecolor='white')
ax.axhline(fault_rate, color='#dc2626', linestyle='--', linewidth=1.5, label=f'전체 평균 {fault_rate:.1f}%')
for bar, val in zip(bars, equip_stats['rate']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('고장률(%)')
ax.set_title('장비 종류별 고장률')
ax.legend(fontsize=10)
ax.set_ylim(0, max(equip_stats['rate']) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 지역별 고장률
loc_stats = df.groupby('location').agg(
    total=('faulty', 'count'),
    faults=('faulty', 'sum')
).reset_index()
loc_stats['rate'] = loc_stats['faults'] / loc_stats['total'] * 100
loc_stats = loc_stats.sort_values('rate', ascending=False)

ax2 = axes[1]
bars2 = ax2.bar(loc_stats['location'], loc_stats['rate'],
                color='#8b5cf6', width=0.45, edgecolor='white')
ax2.axhline(fault_rate, color='#dc2626', linestyle='--', linewidth=1.5, label=f'전체 평균 {fault_rate:.1f}%')
for bar, val in zip(bars2, loc_stats['rate']):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylabel('고장률(%)')
ax2.set_title('지역별 고장률')
ax2.legend(fontsize=10)
ax2.set_ylim(0, max(loc_stats['rate']) * 1.3)
ax2.tick_params(axis='x', labelrotation=15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'b4_equip_location.png'), dpi=150, bbox_inches='tight')
plt.close()
print("저장: b4_equip_location.png")

# 콘솔 출력
print("\n장비별 고장률:")
for _, row in equip_stats.iterrows():
    print(f"  {row['equipment']:12s}: {row['faults']}건 / {row['total']}건 = {row['rate']:.1f}%")
print("지역별 고장률:")
for _, row in loc_stats.iterrows():
    print(f"  {row['location']:15s}: {row['faults']}건 / {row['total']}건 = {row['rate']:.1f}%")

# =============================================================================
# Chart 5: 가설5 — 진동×온도 구간별 고장률 히트맵 (색칠한 표 형태)
# =============================================================================
# 구간 정의
vib_bins  = [df['vibration'].min(), 1.0, 1.5, 2.0, 2.5, 3.0, df['vibration'].max()]
vib_labels = ['~1.0', '1.0~1.5', '1.5~2.0', '2.0~2.5', '2.5~3.0', '3.0~']
temp_bins  = [df['temperature'].min(), 60, 80, 100, 120, df['temperature'].max()]
temp_labels = ['~60°C', '60~80°C', '80~100°C', '100~120°C', '120°C~']

df['vib_bin']  = pd.cut(df['vibration'], bins=vib_bins, labels=vib_labels, include_lowest=True)
df['temp_bin'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_labels, include_lowest=True)

pivot = df.groupby(['vib_bin', 'temp_bin'], observed=True).agg(
    total=('faulty','count'),
    faults=('faulty','sum')
).reset_index()
pivot['rate'] = pivot['faults'] / pivot['total'] * 100

rate_matrix = pivot.pivot(index='vib_bin', columns='temp_bin', values='rate')
count_matrix = pivot.pivot(index='vib_bin', columns='temp_bin', values='total')

fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.cm.RdYlGn_r
im = ax.imshow(rate_matrix.values, cmap=cmap, aspect='auto', vmin=0, vmax=60)
plt.colorbar(im, ax=ax, label='고장률(%)', shrink=0.85)

ax.set_xticks(range(len(rate_matrix.columns)))
ax.set_xticklabels(rate_matrix.columns, fontsize=10)
ax.set_yticks(range(len(rate_matrix.index)))
ax.set_yticklabels(rate_matrix.index, fontsize=10)
ax.set_xlabel('온도 구간', fontsize=11)
ax.set_ylabel('진동 구간', fontsize=11)
ax.set_title('진동×온도 구간별 고장률(%) — 빨갈수록 위험', fontsize=13)

for i in range(rate_matrix.shape[0]):
    for j in range(rate_matrix.shape[1]):
        val = rate_matrix.values[i, j]
        cnt = count_matrix.values[i, j]
        if not np.isnan(val):
            text_color = 'white' if val > 40 else '#1f2933'
            ax.text(j, i, f'{val:.0f}%\n(n={int(cnt)})',
                    ha='center', va='center', fontsize=8.5, color=text_color)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'b5_vib_temp_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("저장: b5_vib_temp_heatmap.png")

# =============================================================================
# Chart 6: 경보 기준 — 진동 구간별 고장률 막대그래프 + 두 가지 비율
# =============================================================================
alert_bins   = [df['vibration'].min(), 1.5, 2.0, 2.5, 2.6, 3.0, df['vibration'].max()]
alert_labels = ['~1.5', '1.5~2.0', '2.0~2.5', '2.5~2.6', '2.6~3.0', '3.0~']

df['alert_bin'] = pd.cut(df['vibration'], bins=alert_bins, labels=alert_labels, include_lowest=True)

alert_stats = df.groupby('alert_bin', observed=True).agg(
    total=('faulty','count'),
    faults=('faulty','sum')
).reset_index()
alert_stats['rate'] = alert_stats['faults'] / alert_stats['total'] * 100

print("\n진동 구간별 고장률:")
for _, row in alert_stats.iterrows():
    print(f"  진동 {row['alert_bin']:8s}: 전체 {row['total']:4d}건 중 고장 {row['faults']:3d}건 → 고장률 {row['rate']:.1f}%")

# 경보선: 진동 2.6 이상
THRESHOLD = 2.6
alarm_df    = df[df['vibration'] >= THRESHOLD]
alarm_total = len(alarm_df)
alarm_fault = alarm_df['faulty'].sum()
precision   = alarm_fault / alarm_total * 100 if alarm_total > 0 else 0
recall      = alarm_fault / fault_total * 100

print(f"\n경보선 진동 ≥ {THRESHOLD}")
print(f"  알람 울린 장비: {alarm_total}건")
print(f"  그 중 실제 고장: {alarm_fault}건")
print(f"  알람 중 실제 고장 비율(정밀도): {precision:.1f}%")
print(f"  전체 고장 중 알람으로 잡아낸 비율(재현율): {recall:.1f}%")

# 보조 경보: 진동 2.0~2.6 이면서 온도 100 이상
aux_df    = df[(df['vibration'] >= 2.0) & (df['vibration'] < THRESHOLD) & (df['temperature'] >= 100)]
aux_total = len(aux_df)
aux_fault = aux_df['faulty'].sum()
aux_prec  = aux_fault / aux_total * 100 if aux_total > 0 else 0
print(f"\n보조 경보: 진동 2.0~2.6 + 온도 ≥ 100°C")
print(f"  해당 장비: {aux_total}건, 고장: {aux_fault}건, 고장률: {aux_prec:.1f}%")

# 차트
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
bar_colors = ['#22c55e' if r < 20 else ('#f59e0b' if r < 50 else '#dc2626') for r in alert_stats['rate']]
bars = ax.bar(alert_stats['alert_bin'], alert_stats['rate'], color=bar_colors, edgecolor='white', width=0.6)
ax.axhline(fault_rate, color='#64748b', linestyle=':', linewidth=1.5, label=f'전체 평균 {fault_rate:.1f}%')
ax.axvline(x=3.5, color='#dc2626', linestyle='--', linewidth=2, label='경보선 (진동 2.6)')
for bar, val in zip(bars, alert_stats['rate']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.0f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xlabel('진동 구간')
ax.set_ylabel('고장률(%)')
ax.set_title('진동 구간별 고장률 — 경보선 설정')
ax.legend(fontsize=10)
ax.set_ylim(0, max(alert_stats['rate']) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 오른쪽: 두 가지 비율 시각화
ax2 = axes[1]
metrics = ['알람 중\n실제 고장 비율', '전체 고장 중\n알람으로 잡아낸 비율']
values  = [precision, recall]
colors  = ['#dc2626', '#2563eb']
bars2 = ax2.barh(metrics, values, color=colors, height=0.45, edgecolor='white')
for bar, val in zip(bars2, values):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=13, fontweight='bold')
ax2.set_xlim(0, 110)
ax2.set_xlabel('비율(%)')
ax2.set_title(f'경보선(진동≥{THRESHOLD}) 성능 확인')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'b6_alert_threshold.png'), dpi=150, bbox_inches='tight')
plt.close()
print("저장: b6_alert_threshold.png")

# =============================================================================
# 최종 수치 요약
# =============================================================================
print("\n" + "="*60)
print("최종 수치 요약")
print("="*60)
print(f"전체: {total}건, 고장: {fault_total}건({fault_rate:.1f}%)")
print(f"진동 평균 — 정상: {normal_df['vibration'].mean():.2f}, 고장: {fault_df['vibration'].mean():.2f}")
print(f"온도 평균 — 정상: {normal_df['temperature'].mean():.1f}, 고장: {fault_df['temperature'].mean():.1f}")
print(f"압력 평균 — 정상: {normal_df['pressure'].mean():.1f}, 고장: {fault_df['pressure'].mean():.1f}")
print(f"습도 평균 — 정상: {normal_df['humidity'].mean():.1f}, 고장: {fault_df['humidity'].mean():.1f}")
print(f"경보선(진동≥{THRESHOLD}): 알람 {alarm_total}건, 고장적중 {alarm_fault}건")
print(f"  → 알람 중 실제 고장 비율: {precision:.1f}%")
print(f"  → 전체 고장 중 알람으로 잡아낸 비율: {recall:.1f}%")
print(f"보조경보(진동 2.0~2.6 + 온도≥100°C): {aux_total}건 중 고장 {aux_fault}건({aux_prec:.1f}%)")
print(f"\n생성된 차트:")
for f in sorted(os.listdir(CHART_DIR)):
    print(f"  docs/charts/b/{f}")
