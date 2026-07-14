"""
可視化: 
1. ベイズ信頼更新の推移(裏切りが起きるたびに信頼度が急落する様子)
2. 「日和見主義者の割合」と「グループ人数」を変えたときの存続率(感度分析)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 日本語フォントが無い環境向けフォールバック

import sys
sys.path.append("notebooks")
from importlib import import_module
mod = import_module("02_bayesian_trust_and_stability")

update_trust = mod.update_trust
monte_carlo_survival_rate = mod.monte_carlo_survival_rate

# -----------------------------------------------------------
# 図1: 信頼度の推移(3パターン: ずっと協力 / たまに裏切り / 頻繁に裏切り)
# -----------------------------------------------------------
def trust_trajectory(history, start=0.9):
    trust = start
    trajectory = [trust]
    for obs in history:
        trust = update_trust(trust, obs)
        trajectory.append(trust)
    return trajectory

np.random.seed(0)
n_periods = 15

# パターンA: ほぼ常に協力(たまにやむを得ず払えない)
history_a = [np.random.random() > 0.05 for _ in range(n_periods)]
# パターンB: 時々裏切る日和見主義者
history_b = [np.random.random() > 0.3 for _ in range(n_periods)]
# パターンC: 頻繁に裏切る
history_c = [np.random.random() > 0.6 for _ in range(n_periods)]

traj_a = trust_trajectory(history_a)
traj_b = trust_trajectory(history_b)
traj_c = trust_trajectory(history_c)

plt.figure(figsize=(8, 5))
plt.plot(traj_a, marker='o', label='Mostly cooperative (honest member)')
plt.plot(traj_b, marker='s', label='Occasional defection (mild opportunist)')
plt.plot(traj_c, marker='^', label='Frequent defection (strong opportunist)')
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Collapse threshold (0.5)')
plt.xlabel('Period')
plt.ylabel('Trust level (Bayesian posterior)')
plt.title('Bayesian Trust Update Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('figures/01_bayesian_trust_trajectory.png', dpi=150)
plt.close()
print("保存: figures/01_bayesian_trust_trajectory.png")

# -----------------------------------------------------------
# 図2: 日和見主義者の割合 x グループ人数 → 存続率の感度分析
# -----------------------------------------------------------
group_sizes = [5, 10, 15, 20, 30]
opportunist_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]

results = np.zeros((len(group_sizes), len(opportunist_fracs)))

for i, n in enumerate(group_sizes):
    for j, frac in enumerate(opportunist_fracs):
        rate, _ = monte_carlo_survival_rate(
            n_members=n, n_periods=12, n_simulations=500,
            defection_prob_opportunist=0.3, frac_opportunist=frac,
        )
        results[i, j] = rate

plt.figure(figsize=(8, 6))
im = plt.imshow(results, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, label='12-period survival rate')
plt.xticks(range(len(opportunist_fracs)), [f"{f:.0%}" for f in opportunist_fracs])
plt.yticks(range(len(group_sizes)), group_sizes)
plt.xlabel('Fraction of opportunistic members')
plt.ylabel('Group size (n)')
plt.title('Tanomoshiko Survival Rate: Group Size vs Opportunist Fraction')

for i in range(len(group_sizes)):
    for j in range(len(opportunist_fracs)):
        plt.text(j, i, f"{results[i,j]*100:.0f}%", ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figures/02_survival_rate_heatmap.png', dpi=150)
plt.close()
print("保存: figures/02_survival_rate_heatmap.png")

print()
print("感度分析の結果(数値):")
print("行=グループ人数, 列=日和見主義者割合")
print(results)
