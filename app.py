"""
Tanomoshiko Simulator — Streamlit Web App

ローカルで動かす場合:
    pip install streamlit numpy matplotlib
    streamlit run app.py

Streamlit Community Cloudにデプロイする場合:
    このファイルをリポジトリのルートに置き、requirements.txtも一緒にpushすればOK。
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# コアロジック(02_bayesian_trust_and_stability.py と同じもの)
# ---------------------------------------------------------------
def update_trust(prior_honest_prob, observed_contribution, honest_contribute_prob=0.95,
                  opportunist_contribute_prob=0.5):
    if observed_contribution:
        p_obs_given_honest = honest_contribute_prob
        p_obs_given_opportunist = opportunist_contribute_prob
    else:
        p_obs_given_honest = 1 - honest_contribute_prob
        p_obs_given_opportunist = 1 - opportunist_contribute_prob

    numerator = p_obs_given_honest * prior_honest_prob
    denominator = numerator + p_obs_given_opportunist * (1 - prior_honest_prob)
    return numerator / denominator


def simulate_group(n_members, n_periods, defection_prob_opportunist, frac_opportunist):
    is_opportunist = rng.random(n_members) < frac_opportunist
    trust = np.full(n_members, 0.9)
    trust_history = [trust.mean()]

    for t in range(n_periods):
        for i in range(n_members):
            if is_opportunist[i]:
                contributed = rng.random() > defection_prob_opportunist
            else:
                contributed = rng.random() > 0.05
            trust[i] = update_trust(trust[i], contributed)

        trust_history.append(trust.mean())

        if trust.mean() < 0.5:
            return False, t + 1, trust_history

    return True, n_periods, trust_history


def monte_carlo_survival_rate(n_members, n_periods, n_simulations, defection_prob_opportunist, frac_opportunist):
    survivals = []
    for _ in range(n_simulations):
        survived, _, _ = simulate_group(n_members, n_periods, defection_prob_opportunist, frac_opportunist)
        survivals.append(survived)
    return float(np.mean(survivals))


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Tanomoshiko Simulator", layout="centered")

st.title("Tanomoshiko Simulator")
st.markdown(
    "A Bayesian & game-theoretic simulation of Japan's centuries-old mutual-aid "
    "finance system (*tanomoshiko* / *mujin*). Adjust the parameters below and see "
    "whether a trust-based lending group survives or collapses."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    n_members = st.slider("Group size (number of members)", min_value=3, max_value=50, value=10)
    n_periods = st.slider("Number of periods to simulate", min_value=3, max_value=36, value=12)
with col2:
    frac_opportunist = st.slider("Fraction of opportunistic members", 0.0, 1.0, 0.3, step=0.05)
    defection_prob_opportunist = st.slider(
        "How often opportunists defect (skip payment)", 0.0, 1.0, 0.3, step=0.05
    )

n_simulations = st.select_slider(
    "Number of Monte Carlo simulations (higher = more accurate, slower)",
    options=[100, 500, 1000, 2000, 5000],
    value=1000,
)

if st.button("Run simulation", type="primary"):
    with st.spinner("Running Monte Carlo simulation..."):
        rate = monte_carlo_survival_rate(
            n_members, n_periods, n_simulations, defection_prob_opportunist, frac_opportunist
        )
        _, _, example_trajectory = simulate_group(
            n_members, n_periods, defection_prob_opportunist, frac_opportunist
        )

    st.subheader("Result")
    st.metric(
        label=f"Survival rate over {n_periods} periods",
        value=f"{rate * 100:.1f}%",
    )

    if rate > 0.8:
        st.success("This group is likely to remain stable — trust stays high enough to sustain cooperation.")
    elif rate > 0.4:
        st.warning("This group is on the edge — it could survive or collapse depending on how events unfold.")
    else:
        st.error("This group is likely to collapse — too many members lose trust before the simulation ends.")

    st.subheader("Example trust trajectory (one simulated group)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(example_trajectory, marker="o")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Collapse threshold (0.5)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Average group trust level")
    ax.set_title("Average Trust Over Time (one simulated run)")
    ax.legend()
    st.pyplot(fig)

st.divider()
st.caption(
    "Model based on Sakakibara (2014), \"The Economic Meaning of Mujin,\" "
    "extended with Bayesian trust updating and Monte Carlo simulation. "
    "[View source code on GitHub](https://github.com/koyamashuta1216-cyber/tanomoshiko-simulator)"
)
