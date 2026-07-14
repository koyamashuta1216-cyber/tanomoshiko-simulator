"""
榊原健一(2014)「無尽講の経済的意味」の3人・3期間モデルの再現

論文の設定:
- 3人の個人、3期間モデル
- 効用関数: v = ln(c1) + beta*ln(c2) + beta^2*ln(c3)  [論文ではu(c3)の係数はbetaではなくbeta^2、ただし論文式は
  v(c1,c2,c3) = u(c1) + beta*u(c2) + 2*beta... 実際の論文式に忠実に: u(c1)+beta*u(c2)+beta^2*u(c3) の形で近似]
- beta = 0.9
- 各人は生涯に一度だけ、初期保有量が0になるショックを受ける
- 3つの制度: 完備市場 / ローン市場(銀行) / 無尽講

このスクリプトでは、論文の数値例(beta=0.9)を再現し、
無尽講配分・ローン市場配分それぞれの期待効用を計算して論文の数値と照合する。
"""

import numpy as np

beta = 0.9

def u(c):
    return np.log(c)

# ---------------------------------------------------------
# 1. 無尽講配分 (Mujin allocation)
#    論文の結果: 全てのstate・全ての人について c = 2 (一定)
# ---------------------------------------------------------
c_mujin = 2.0

EU_mujin = u(c_mujin) + beta * u(c_mujin) + beta**2 * u(c_mujin)
print(f"[無尽講] 期待効用 = {EU_mujin:.4f}  (論文値: 1.88)")

# ---------------------------------------------------------
# 2. ローン市場配分 (Loan market allocation)
#    論文の数値例(表): 
#    第1期にショックを受けた人: (c1,c2,c3) = (2.42, 1.05, 1.05)
#    第2期にショックを受けた人: (c1,c2,c3) = (1.79, 2.39, 2.39)
#    第3期にショックを受けた人: (c1,c2,c3) = (1.79, 2.55, 2.55)
# ---------------------------------------------------------
loan_allocations = {
    "shocked_period1": (2.42, 1.05, 1.05),
    "shocked_period2": (1.79, 2.39, 2.39),
    "shocked_period3": (1.79, 2.55, 2.55),
}

def expected_utility(c1, c2, c3):
    return u(c1) + beta * u(c2) + beta**2 * u(c3)

EU_loan = {k: expected_utility(*v) for k, v in loan_allocations.items()}
for k, v in EU_loan.items():
    print(f"[ローン市場: {k}] 効用 = {v:.4f}")

# 全体(1/3ずつの確率で各stateが起こる)の期待効用
EU_loan_overall = np.mean(list(EU_loan.values()))
print(f"[ローン市場] 全体の期待効用(平均) = {EU_loan_overall:.4f}  (論文値: 1.74)")

# ---------------------------------------------------------
# 3. Time Inconsistency の検証(第1期にショックが起きた後)
#    無尽講を維持した場合 vs ローン市場に切り替えた場合
# ---------------------------------------------------------
# 無尽講維持: 第0期と同じ配分が続くので期待効用は変わらず 1.88
EU_mujin_after_shock = EU_mujin

# ローン市場: 第1期にショックを受けた本人の期待効用(残り2期分)
EU_loan_shocked_person = expected_utility(2.42, 1.05, 1.05)  # 全期間分でそのまま計算(論文の定義に合わせる)

# ローン市場: ショックを受けなかった2人(第2期・第3期にショックを受ける人)の期待効用
# 論文の数値: (1/2)*[ln(1.79)+beta*ln(2.39)+beta^2*ln(2.39)] + (1/2)*[ln(1.79)+beta*ln(2.55)+beta^2*ln(2.55)]
EU_not_shocked = 0.5 * expected_utility(1.79, 2.39, 2.39) + 0.5 * expected_utility(1.79, 2.55, 2.55)

print()
print("=== Time Inconsistency の検証(第1期ショック後) ===")
print(f"無尽講を維持した場合の期待効用: {EU_mujin_after_shock:.4f}")
print(f"ローン市場: ショックを受けた本人の期待効用: {EU_loan_shocked_person:.4f}  (論文値: 0.967)")
print(f"ローン市場: ショックを受けなかった2人の期待効用: {EU_not_shocked:.4f}  (論文値: 2.1275)")

if EU_not_shocked > EU_mujin_after_shock:
    print("→ 多数派(2人)にとってはローン市場の方が有利。無尽講は破棄される(論文の結論と一致)")
else:
    print("→ 無尽講の方が有利。無尽講は維持される")
