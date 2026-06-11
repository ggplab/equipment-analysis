"""
Mission A — 건설장비 예지보전 분석
입문자 수준 분석: 개수 세기, 비율, 그룹별 평균, 히스토그램, 막대그래프, 산점도, 박스플롯
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── 설정 ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = '/Users/limjung/Projects/alice-equipment-analysis/data/construction_machine_data.csv'
CHART_DIR = '/Users/limjung/Projects/alice-equipment-analysis/docs/charts/a'
os.makedirs(CHART_DIR, exist_ok=True)

ACCENT  = '#2563eb'
OK      = '#16a34a'
WARN    = '#d97706'
DANGER  = '#dc2626'
COLORS  = ['#2563eb', '#16a34a', '#d97706', '#dc2626']
BG      = '#f8f9fa'

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
print(df.dtypes)

# ── 편의 함수 ─────────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  저장: {path}")

def bar_chart(ax, labels, values, colors=None, title='', ylabel='평균 잔존수명 (시간)'):
    if colors is None:
        colors = COLORS[:len(labels)]
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}h', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    return bars

# ════════════════════════════════════════════════════════════════════════════════
# 가설 1 — 오래 사용할수록 남은 수명이 짧다
# ════════════════════════════════════════════════════════════════════════════════
print("\n[가설 1] 가동시간 vs 잔존수명")

# 산점도
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.scatter(df['Operating_Hours'], df['Remaining_Useful_Life'],
           alpha=0.35, color=ACCENT, s=18, edgecolors='none')
ax.set_title('가동시간이 길수록 남은 수명이 줄어드나?', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('지금까지 가동한 시간 (시간)', fontsize=11)
ax.set_ylabel('남은 수명 (시간)', fontsize=11)
ax.set_facecolor(BG)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save(fig, 'h1_scatter_hours_rul.png')

# 구간별 평균 RUL 막대
q_labels = ['하위 25%\n(0~1,325h)', '25~50%\n(1,326~2,516h)',
            '50~75%\n(2,517~3,689h)', '상위 25%\n(3,690h~)']
bins_h = [0, df['Operating_Hours'].quantile(0.25),
             df['Operating_Hours'].quantile(0.50),
             df['Operating_Hours'].quantile(0.75),
             df['Operating_Hours'].max() + 1]
df['hours_q'] = pd.cut(df['Operating_Hours'], bins=bins_h, labels=q_labels, right=False)
avg_rul_by_hq = df.groupby('hours_q', observed=True)['Remaining_Useful_Life'].mean()
print(f"  구간별 평균 RUL:\n{avg_rul_by_hq.round(1)}")

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
bar_chart(ax, avg_rul_by_hq.index.tolist(), avg_rul_by_hq.values,
          colors=COLORS,
          title='가동시간 구간별 평균 잔존수명 비교')
save(fig, 'h1_bar_hours_rul.png')

# ════════════════════════════════════════════════════════════════════════════════
# 가설 2 — 진동이 심할수록 남은 수명이 짧다
# ════════════════════════════════════════════════════════════════════════════════
print("\n[가설 2] 진동 vs 잔존수명")

v_labels = ['낮음\n(0~1.8)', '중간\n(1.8~3.0)', '높음\n(3.0~4.1)', '매우 높음\n(4.1~5.0)']
bins_v = [0, df['Vibration'].quantile(0.25),
             df['Vibration'].quantile(0.50),
             df['Vibration'].quantile(0.75),
             df['Vibration'].max() + 0.01]
df['vib_q'] = pd.cut(df['Vibration'], bins=bins_v, labels=v_labels, right=False)
avg_rul_by_vq = df.groupby('vib_q', observed=True)['Remaining_Useful_Life'].mean()
print(f"  구간별 평균 RUL:\n{avg_rul_by_vq.round(1)}")

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
bar_chart(ax, avg_rul_by_vq.index.tolist(), avg_rul_by_vq.values,
          colors=COLORS,
          title='진동 세기 구간별 평균 잔존수명 비교')
save(fig, 'h2_bar_vib_rul.png')

# ════════════════════════════════════════════════════════════════════════════════
# 가설 3 — 부품 종류에 따라 수명이 다르다
# ════════════════════════════════════════════════════════════════════════════════
print("\n[가설 3] 부품 타입별 수명")

avg_rul_by_type = df.groupby('Component_Type')['Remaining_Useful_Life'].mean().sort_values()
print(f"  부품 타입별 평균 RUL:\n{avg_rul_by_type.round(1)}")

type_colors = {'Engine': ACCENT, 'Gear': DANGER, 'Hydraulic Cylinder': OK}
c_list = [type_colors.get(t, ACCENT) for t in avg_rul_by_type.index]

# 막대그래프
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
bar_chart(ax, avg_rul_by_type.index.tolist(), avg_rul_by_type.values,
          colors=c_list,
          title='부품 종류별 평균 잔존수명 비교')
save(fig, 'h3_bar_type_rul.png')

# 박스플롯
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
type_order = avg_rul_by_type.index.tolist()
data_by_type = [df[df['Component_Type'] == t]['Remaining_Useful_Life'].values for t in type_order]
bp = ax.boxplot(data_by_type, labels=type_order, patch_artist=True, widths=0.45,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=4, alpha=0.4))
for patch, t in zip(bp['boxes'], type_order):
    patch.set_facecolor(type_colors.get(t, ACCENT))
    patch.set_alpha(0.85)
ax.set_title('부품 종류별 잔존수명 분포 (각 박스: 중간 50% 범위)', fontsize=13, fontweight='bold', pad=10)
ax.set_ylabel('남은 수명 (시간)', fontsize=11)
ax.set_xlabel('부품 종류', fontsize=11)
ax.set_facecolor(BG)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 중앙값 표시
for i, t in enumerate(type_order, 1):
    med = df[df['Component_Type'] == t]['Remaining_Useful_Life'].median()
    ax.text(i, med + 15, f'{med:.0f}h', ha='center', fontsize=10, fontweight='bold', color='#333')

save(fig, 'h3_box_type_rul.png')

# ════════════════════════════════════════════════════════════════════════════════
# 가설 4 — 온도·압력이 높으면 수명이 짧다
# ════════════════════════════════════════════════════════════════════════════════
print("\n[가설 4] 온도·압력 vs 잔존수명")

t_labels = ['낮음', '중간', '높음', '매우 높음']
bins_t = [df['Temperature'].min() - 1,
          df['Temperature'].quantile(0.25),
          df['Temperature'].quantile(0.50),
          df['Temperature'].quantile(0.75),
          df['Temperature'].max() + 1]
df['temp_q'] = pd.cut(df['Temperature'], bins=bins_t, labels=t_labels, right=True)
avg_rul_by_tq = df.groupby('temp_q', observed=True)['Remaining_Useful_Life'].mean()

p_labels = ['낮음', '중간', '높음', '매우 높음']
bins_p = [df['Pressure'].min() - 1,
          df['Pressure'].quantile(0.25),
          df['Pressure'].quantile(0.50),
          df['Pressure'].quantile(0.75),
          df['Pressure'].max() + 1]
df['pres_q'] = pd.cut(df['Pressure'], bins=bins_p, labels=p_labels, right=True)
avg_rul_by_pq = df.groupby('pres_q', observed=True)['Remaining_Useful_Life'].mean()

print(f"  온도 구간별 평균 RUL:\n{avg_rul_by_tq.round(1)}")
print(f"  압력 구간별 평균 RUL:\n{avg_rul_by_pq.round(1)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.patch.set_facecolor(BG)

bar_chart(axes[0], avg_rul_by_tq.index.tolist(), avg_rul_by_tq.values,
          colors=COLORS, title='온도 구간별 평균 잔존수명')
bar_chart(axes[1], avg_rul_by_pq.index.tolist(), avg_rul_by_pq.values,
          colors=COLORS, title='압력 구간별 평균 잔존수명')
plt.tight_layout(pad=2)
save(fig, 'h4_bar_temp_pressure_rul.png')

# ════════════════════════════════════════════════════════════════════════════════
# 가설 5 — 가동시간도 많고 진동도 심하면 훨씬 위험하다 (복합 조건)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[가설 5] 복합 위험 그룹")

hours_thresh = df['Operating_Hours'].quantile(0.75)
vib_thresh   = df['Vibration'].quantile(0.75)
print(f"  가동시간 상위 25% 기준: {hours_thresh:.0f}h")
print(f"  진동 상위 25% 기준: {vib_thresh:.2f}")

mask_h = df['Operating_Hours'] >= hours_thresh
mask_v = df['Vibration'] >= vib_thresh

g_both   = df[mask_h & mask_v]['Remaining_Useful_Life'].mean()
g_h_only = df[mask_h & ~mask_v]['Remaining_Useful_Life'].mean()
g_v_only = df[~mask_h & mask_v]['Remaining_Useful_Life'].mean()
g_none   = df[~mask_h & ~mask_v]['Remaining_Useful_Life'].mean()
n_both   = (mask_h & mask_v).sum()
n_none   = (~mask_h & ~mask_v).sum()

print(f"  그룹별 평균 RUL:")
print(f"    둘 다 높음: {g_both:.1f}h (n={n_both})")
print(f"    가동시간만 높음: {g_h_only:.1f}h")
print(f"    진동만 높음: {g_v_only:.1f}h")
print(f"    둘 다 낮음: {g_none:.1f}h (n={n_none})")

group_labels = [f'둘 다 높음\n(n={n_both})', '가동시간만\n높음', '진동만\n높음', f'둘 다 낮음\n(n={n_none})']
group_vals   = [g_both, g_h_only, g_v_only, g_none]
group_colors = [DANGER, WARN, WARN, OK]

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
bars = ax.bar(group_labels, group_vals, color=group_colors, edgecolor='white', linewidth=1.5, width=0.55)
ax.set_title('가동시간 + 진동이 동시에 높으면 남은 수명이 얼마나 짧을까?', fontsize=13, fontweight='bold', pad=10)
ax.set_ylabel('평균 잔존수명 (시간)', fontsize=11)
ax.set_facecolor(BG)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, group_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.0f}h', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(group_vals) * 1.25)
save(fig, 'h5_bar_combined_risk.png')

# ════════════════════════════════════════════════════════════════════════════════
# 위험군 도출 — RUL < 100 부품 Machine_ID별 분포
# ════════════════════════════════════════════════════════════════════════════════
print("\n[위험군 도출] RUL < 100 부품")

df_risk = df[df['Remaining_Useful_Life'] < 100].copy()
print(f"  전체 RUL<100 부품 수: {len(df_risk)}")

risk_by_machine = df_risk.groupby('Machine_ID').size().sort_values(ascending=False)
print(f"  장비별 위험 부품 수:\n{risk_by_machine}")

fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
machines = risk_by_machine.index.tolist()
counts   = risk_by_machine.values
bar_colors = [DANGER if c >= 3 else WARN if c >= 1 else OK for c in counts]

bars = ax.bar(machines, counts, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
ax.set_title('장비별 "지금 당장 점검 필요" 부품 수 (잔존수명 100시간 미만)', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('장비 번호 (Machine_ID)', fontsize=11)
ax.set_ylabel('임박 부품 수 (개)', fontsize=11)
ax.set_facecolor(BG)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(counts) + 2)
plt.xticks(rotation=30, ha='right')

legend_patches = [
    mpatches.Patch(color=DANGER, label='3개 이상 — 매우 긴급'),
    mpatches.Patch(color=WARN,   label='1~2개 — 점검 권고'),
]
ax.legend(handles=legend_patches, fontsize=11, loc='upper right')
save(fig, 'risk_machine_bar.png')

# ════════════════════════════════════════════════════════════════════════════════
# 최종 수치 요약 출력 (HTML 작성용)
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("HTML 작성용 수치 요약")
print("="*60)
print(f"총 부품 수: {len(df)}")
print(f"평균 잔존수명: {df['Remaining_Useful_Life'].mean():.0f}h")
print(f"RUL<100 임박 부품: {len(df_risk)}개 ({len(df_risk)/len(df)*100:.1f}%)")
print(f"\n가설1 — 가동시간 구간별 평균 RUL:")
for label, val in zip(q_labels, avg_rul_by_hq.values):
    print(f"  {label.replace(chr(10),' ')}: {val:.0f}h")
print(f"\n가설2 — 진동 구간별 평균 RUL:")
for label, val in zip(v_labels, avg_rul_by_vq.values):
    print(f"  {label.replace(chr(10),' ')}: {val:.0f}h")
print(f"\n가설3 — 부품 타입별 평균 RUL:")
for t, v in avg_rul_by_type.items():
    print(f"  {t}: {v:.0f}h")
print(f"\n가설4 — 온도 구간별 평균 RUL:")
for label, val in zip(t_labels, avg_rul_by_tq.values):
    print(f"  {label}: {val:.0f}h")
print(f"가설4 — 압력 구간별 평균 RUL:")
for label, val in zip(p_labels, avg_rul_by_pq.values):
    print(f"  {label}: {val:.0f}h")
print(f"\n가설5 — 복합 위험 그룹:")
print(f"  둘 다 높음: {g_both:.0f}h")
print(f"  가동시간만 높음: {g_h_only:.0f}h")
print(f"  진동만 높음: {g_v_only:.0f}h")
print(f"  둘 다 낮음: {g_none:.0f}h")
print(f"  가동시간 기준: {hours_thresh:.0f}h 이상")
print(f"  진동 기준: {vib_thresh:.1f} 이상")
print(f"\n위험군 장비 상위 5대:")
for m, c in risk_by_machine.head(5).items():
    print(f"  {m}: {c}개")
print(f"\nGear 평균 RUL: {avg_rul_by_type['Gear']:.0f}h")
print(f"Engine 평균 RUL: {avg_rul_by_type['Engine']:.0f}h")
print(f"Hydraulic Cylinder 평균 RUL: {avg_rul_by_type['Hydraulic Cylinder']:.0f}h")

print("\n완료! 차트 파일 목록:")
for f in sorted(os.listdir(CHART_DIR)):
    print(f"  {f}")
