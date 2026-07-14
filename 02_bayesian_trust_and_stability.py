"""
頼母子講(無尽講)の一般化モデル
- n人のグループ、T期間
- 各期にランダムに1人がショック(困窮)を受け、他のメンバーが掛け金を出す
- メンバーには観測できない「タイプ」がある: 正直者(honest) か 日和見主義者(opportunistic)
- 他のメンバーは、観測された行動(拠出したか、裏切ったか)からベイズ更新で
  相手が正直者である事後確率を更新していく
- 「協力を続けるか、裏切るか」の意思決定は、榊原(2014)の枠組みに基づく
  参加制約(participation constraint)として定式化する:
      協力を続けた場合の期待効用 >= 離脱(ローン市場に切り替え)した場合の期待効用
  この不等号が破れると、そのメンバーにとって頼母子講から離脱する方が合理的になる
  (榊原論文のtime inconsistencyの一般化)
"""

import numpy as np

rng = np.random.default_rng(42)

# -----------------------------------------------------------------
# 1. ベイズ信頼更新
# -----------------------------------------------------------------
def update_trust(prior_honest_prob, observed_contribution, honest_contribute_prob=0.95,
                  opportunist_contribute_prob=0.5):
    """
    ベイズの定理で「このメンバーは正直者である」事後確率を更新する。

    prior_honest_prob: 更新前の「正直者である確率」の事前分布
    observed_contribution: True(掛け金を払った) / False(裏切った)
    honest_contribute_prob: 正直者が実際に掛け金を払う確率(たまに払えないこともある)
    opportunist_contribute_prob: 日和見主義者が(疑われないために)掛け金を払う確率
    """
    if observed_contribution:
        p_obs_given_honest = honest_contribute_prob
        p_obs_given_opportunist = opportunist_contribute_prob
    else:
        p_obs_given_honest = 1 - honest_contribute_prob
        p_obs_given_opportunist = 1 - opportunist_contribute_prob

    numerator = p_obs_given_honest * prior_honest_prob
    denominator = (numerator + p_obs_given_opportunist * (1 - prior_honest_prob))
    return numerator / denominator


# -----------------------------------------------------------------
# 2. 参加制約(安定性)のチェック
#    榊原論文の考え方を一般化: 
#    「グループを裏切って抜けたときの効用」が「協力を続けたときの期待効用」を
#    上回ったら、そのメンバーは裏切りを選ぶ
# -----------------------------------------------------------------
def stays_in_group(trust_level, contribution, payout_if_shocked, outside_option_utility,
                    beta=0.9, remaining_periods=2):
    """
    trust_level: グループ全体への信頼度(0~1)。信頼度が低いほど、将来また
                 助けてもらえる確率が下がると考える。
    contribution: 毎期の掛け金
    payout_if_shocked: ショックを受けたときに受け取れる金額
    outside_option_utility: 離脱してローン市場を使った場合の一期あたりの効用
    beta: 割引率
    remaining_periods: 残り期間数
    """
    # 協力を続けた場合の期待効用(信頼度で割り引いた将来の受給期待値)
    expected_payout = trust_level * payout_if_shocked
    cooperate_utility = np.log(max(expected_payout - contribution, 0.01) + contribution)
    total_cooperate_value = sum(beta**t * cooperate_utility for t in range(remaining_periods))

    # 離脱した場合の価値
    total_outside_value = sum(beta**t * outside_option_utility for t in range(remaining_periods))

    return total_cooperate_value >= total_outside_value


# -----------------------------------------------------------------
# 3. モンテカルロ・シミュレーション: グループ存続率
# -----------------------------------------------------------------
def simulate_group(n_members, n_periods, defection_prob_opportunist=0.3,
                    frac_opportunist=0.2, beta=0.9):
    """
    1つのグループをT期間シミュレーションし、最後まで存続したかを判定する。
    """
    is_opportunist = rng.random(n_members) < frac_opportunist
    trust = np.full(n_members, 0.9)  # 初期信頼度(事前分布)

    for t in range(n_periods):
        for i in range(n_members):
            if is_opportunist[i]:
                contributed = rng.random() > defection_prob_opportunist
            else:
                contributed = rng.random() > 0.05  # 正直者もたまに払えないことがある

            trust[i] = update_trust(trust[i], contributed)

        # このグループの平均信頼度が閾値を割ったら崩壊とみなす
        if trust.mean() < 0.5:
            return False, t + 1  # 崩壊した期を返す

    return True, n_periods  # 最後まで存続


def monte_carlo_survival_rate(n_members, n_periods, n_simulations=2000,
                                defection_prob_opportunist=0.3, frac_opportunist=0.2):
    survivals = []
    collapse_periods = []
    for _ in range(n_simulations):
        survived, period = simulate_group(
            n_members, n_periods,
            defection_prob_opportunist=defection_prob_opportunist,
            frac_opportunist=frac_opportunist,
        )
        survivals.append(survived)
        if not survived:
            collapse_periods.append(period)
    return np.mean(survivals), collapse_periods


if __name__ == "__main__":
    print("=== ベイズ信頼更新の例 ===")
    trust = 0.9
    history = [True, True, False, True, False, False]  # 拠出履歴(裏切りが混じる)
    for i, obs in enumerate(history):
        trust = update_trust(trust, obs)
        print(f"期{i+1}: 拠出={'あり' if obs else '裏切り'}  →  信頼度 = {trust:.3f}")

    print()
    print("=== モンテカルロ・シミュレーション: グループ存続率 ===")
    for n in [5, 10, 20]:
        rate, collapses = monte_carlo_survival_rate(
            n_members=n, n_periods=12, n_simulations=2000,
            defection_prob_opportunist=0.3, frac_opportunist=0.2,
        )
        print(f"グループ人数={n:>3}人 → 12期間存続率 = {rate*100:.1f}%")
