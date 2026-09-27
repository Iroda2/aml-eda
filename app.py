import streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

st.set_page_config(page_title="AML Alert Prioritization — Team B6CB45428", layout="wide")
st.title("AML Alert Prioritization — Team B6CB45428")
st.caption("WIUT Hackathon 2026 · FinTech / AI in Finance")

st.header("1. Approach")
st.markdown("""
We treat each alert as a classification problem over aggregated transaction behaviour.
Transactions are grouped by `signal_id` and summarised into volume, amount, timing
and behavioural features. A LightGBM classifier is trained with 5-fold stratified CV,
optimised for ROC-AUC. Final submission = average of fold predictions.
""")

st.header("2. Dataset overview")

@st.cache_data
def load():
    s = pd.read_csv("train_signals.csv", parse_dates=["signal_sanasi"])
    t = pd.read_parquet("train_transactions.parquet")
    return s, t

tr_sig, tr_tx = load()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training alerts", len(tr_sig))
c2.metric("Transactions", f"{len(tr_tx):,}")
c3.metric("Escalation rate", f"{tr_sig.eskalatsiya.mean():.2%}")
c4.metric("Avg tx / alert", f"{len(tr_tx)/tr_sig.signal_id.nunique():.1f}")

st.header("3. Target distribution")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tr_sig.eskalatsiya.value_counts().plot.bar(ax=ax[0], color=["#4C72B0", "#C44E52"])
ax[0].set_xticklabels(["dismissed", "escalated"], rotation=0)
ax[0].set_title("Class balance")
tr_sig.set_index("signal_sanasi").resample("W")["eskalatsiya"].agg(["count", "mean"]).plot(ax=ax[1])
ax[1].set_title("Weekly volume & escalation rate")
st.pyplot(fig)

st.header("4. Transaction behaviour by target")
tx_per = tr_tx.groupby("signal_id").size().rename("n_tx")
merged = tr_sig.merge(tx_per, left_on="signal_id", right_index=True, how="left").fillna({"n_tx": 0})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.kdeplot(data=merged, x="n_tx", hue="eskalatsiya", fill=True, common_norm=False, ax=ax[0])
ax[0].set_title("Transactions per alert")
sns.boxplot(data=merged, x="eskalatsiya", y="n_tx", ax=ax[1])
ax[1].set_title("Tx count by class")
st.pyplot(fig)

st.header("5. Transaction type mix")
mix = tr_tx.merge(tr_sig[["signal_id", "eskalatsiya"]], on="signal_id")
pivot = mix.groupby(["eskalatsiya", "tranzaksiya_turi"]).size().unstack(fill_value=0)
st.dataframe(pivot)
fig, ax = plt.subplots(figsize=(7, 4))
(pivot.T / pivot.T.sum()).T.plot.bar(stacked=True, ax=ax)
ax.set_title("Transaction type share by class")
st.pyplot(fig)

st.header("6. Key insights")
st.markdown("""
- Escalated alerts differ systematically in transaction volume and mix.
- International (`xalqaro`) and cash (`naqd`) shares separate the classes.
- Recency (time since last transaction before the alert) is highly discriminative.
- Escalated alerts show higher amount dispersion (`std/mean`, `max/mean`).
- Night-time activity share is elevated among escalated cases.
""")

st.header("7. Features motivated by EDA")
st.markdown("""
| Insight | Feature |
|---|---|
| Volume difference | `n_tx`, `n_last_24h`, `tx_per_day` |
| Direction mix | `chiqim_ratio`, `n_kirim`, `n_chiqim` |
| Type mix | `share_xalqaro`, `share_naqd`, `share_karta` |
| Size dispersion | `amt_std_over_mean`, `amt_max_over_mean`, `amt_p99` |
| Recency | `hours_since_last`, `hours_since_first` |
| Burstiness | `iat_mean`, `iat_min`, `iat_std` |
| Timing | `night_share` |
""")

st.header("8. Model & validation")
st.markdown("5-fold Stratified CV · LightGBM · early stopping · ROC-AUC. Final test prediction = mean of folds.")

st.header("9. Conclusion")
st.markdown("Aggregated transaction behaviour around each alert carries strong signal. A gradient-boosted model on these engineered features provides a robust, reproducible AML alert prioritization scorer.")