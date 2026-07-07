from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import make_synthetic
from src.model import run_rfm_analysis, mine_association_rules, monthly_sales_trend

THEME = {"bg": "#0e1117", "fg": "#ffffff", "grid": "#1a1f2e",
         "cyan": "#22d3ee", "rose": "#f43f5e", "green": "#22c55e", "amber": "#fbbf24"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": THEME["bg"], "axes.facecolor": THEME["bg"],
        "axes.edgecolor": THEME["grid"], "axes.labelcolor": THEME["fg"],
        "text.color": THEME["fg"], "xtick.color": THEME["fg"], "ytick.color": THEME["fg"],
        "grid.color": THEME["grid"], "grid.alpha": 0.3,
    })


st.set_page_config(page_title="RetailPulse | Retail Transaction Analytics", layout="wide", page_icon="\U0001f6d2")

with st.sidebar:
    st.header("⚙ Config")
    n_customers = st.slider("Customers", 100, 2000, 500, 50)
    n_transactions = st.slider("Transactions", 1000, 10000, 3000, 500)
    min_support = st.slider("Min rule support", 0.005, 0.05, 0.02, 0.005)
    min_confidence = st.slider("Min rule confidence", 0.05, 0.5, 0.2, 0.05)
    st.caption("RFM Segmentation · Basket Analysis · Trend Decomposition")

data = make_synthetic(n_customers=n_customers, n_transactions=n_transactions, seed=42)
rfm_result = run_rfm_analysis(data)
rules = mine_association_rules(data["itemsets"], min_support=min_support, min_confidence=min_confidence)
trend = monthly_sales_trend(data["transactions"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{data['n_customers']:,}")
c2.metric("Transactions", f"{data['n_transactions']:,}")
c3.metric("RFM Segments", rfm_result["n_segments"])
c4.metric("Association Rules", len(rules))

t1, t2, t3 = st.tabs(["\U0001f465 RFM Segmentation", "\U0001f6d2 Basket Analysis", "\U0001f4c8 Sales Trend"])

with t1:
    st.subheader("Recency / Frequency / Monetary Segmentation")
    st.latex(r"\text{RFM\_score} = R\_\text{score} + F\_\text{score} + M\_\text{score}")
    st.caption("Each customer is quantile-scored on recency (inverted — recent = high score), "
               "frequency, and monetary value, then summed into a segment tier (Hughes, 1996).")
    st.dataframe(
        rfm_result["summary"].sort_values("RFM_score", ascending=False).round(1),
        use_container_width=True,
    )
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    summary = rfm_result["summary"].sort_values("RFM_score")
    ax.bar(summary["RFM_score"].astype(str), summary["n_customers"], color=THEME["cyan"], alpha=0.8)
    ax.set_xlabel("RFM Score"); ax.set_ylabel("Customers")
    ax.set_title("Customers per RFM Segment", color=THEME["fg"]); ax.grid(True, alpha=0.2, axis="y")
    st.pyplot(fig)
    st.dataframe(rfm_result["rfm"].head(50), use_container_width=True, height=200)

with t2:
    st.subheader("Product Co-Purchase Association Rules")
    st.latex(r"\text{support}(A) = \frac{|\{t : A \subseteq t\}|}{|T|}, \quad "
             r"\text{confidence}(A\to B) = \frac{\text{support}(A \cup B)}{\text{support}(A)}, \quad "
             r"\text{lift}(A\to B) = \frac{\text{confidence}(A\to B)}{\text{support}(B)}")
    st.caption("Rules mined from all 2-itemsets (Agrawal et al., 1993). Lift > 1 means the pair "
               "co-occurs more than chance would predict — the basis for \"customers who bought X also bought Y\".")
    if rules:
        rules_df = pd.DataFrame(rules)
        st.dataframe(rules_df, use_container_width=True, height=280)
        _style()
        fig, ax = plt.subplots(figsize=(8, max(3, len(rules) * 0.4)))
        labels = [f"{r['antecedent']} → {r['consequent']}" for r in rules]
        colors = [THEME["green"] if r["lift"] >= 2 else THEME["amber"] for r in rules]
        ax.barh(labels, [r["lift"] for r in rules], color=colors)
        ax.axvline(1.0, color=THEME["rose"], ls="--", lw=1, label="No association (lift=1)")
        ax.set_xlabel("Lift"); ax.set_title("Rule Lift", color=THEME["fg"])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="x")
        st.pyplot(fig)
    else:
        st.info("No rules clear the support/confidence thresholds — lower them in the sidebar.")

with t3:
    st.subheader("Monthly Sales Trend Decomposition")
    st.latex(r"y_t = \text{trend}_t + \text{seasonal}_t + \text{residual}_t")
    st.caption("Centered moving-average decomposition into trend, seasonal, and residual components.")
    monthly = pd.Series(trend["monthly_sales"])
    _style()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, color=THEME["cyan"], marker="o", lw=2)
    ax.set_title("Monthly Revenue", color=THEME["fg"]); ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

    decomp = trend["decomposition"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, key, color in zip(axes, ["trend", "seasonal", "residual"], [THEME["cyan"], THEME["amber"], THEME["rose"]]):
        ax.plot(decomp[key], color=color, lw=1.5)
        ax.set_ylabel(key.capitalize(), color=THEME["fg"])
        ax.grid(True, alpha=0.2)
    _style()
    st.pyplot(fig)
    st.metric("Trend slope ($/month)", f"{trend['trend_slope']:,.2f}")
