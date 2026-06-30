"""core.py — RFM segmentation and basket analysis metrics for RetailPulse.

Retail analytics metrics, NOT generic classification:
  * **RFM scores** — Recency, Frequency, Monetary value scoring.
  * **Segment sizes** — count of customers in each RFM tier.
  * **Basket support/confidence/lift** — association rule mining metrics.
  * **Trend decomposition** — trend/seasonal/residual decomposition.

References
----------
Hughes (1996), "Strategic Database Marketing." (RFM)
Agrawal et al. (1993), "Mining Association Rules." (basket analysis)
"""
from __future__ import annotations
import numpy as np
from collections import Counter


def compute_rfm(transactions, customer_col="customer_id", date_col="date",
                amount_col="amount", reference_date=None):
    """Compute RFM (Recency, Frequency, Monetary) per customer.

    Returns DataFrame with customer_id, recency, frequency, monetary.
    """
    import pandas as pd
    df = transactions
    if reference_date is None:
        reference_date = df[date_col].max()
    rfm = df.groupby(customer_col).agg(
        recency=(date_col, lambda x: (reference_date - x.max()).days),
        frequency=(customer_col, "count"),
        monetary=(amount_col, "sum"),
    ).reset_index()
    return rfm


def rfm_segment(rfm_df, n_quantiles=5):
    """Segment customers into RFM tiers using quantile scoring."""
    import pandas as pd
    df = rfm_df.copy()
    # Use rank-based scoring to avoid qcut duplicate/bin issues
    df["R_score"] = pd.cut(df["recency"], bins=min(n_quantiles, df["recency"].nunique()),
                           labels=False, include_lowest=True, duplicates="drop")
    df["F_score"] = pd.cut(df["frequency"].rank(method="first"), bins=min(n_quantiles, df["frequency"].nunique()),
                           labels=False, include_lowest=True, duplicates="drop")
    df["M_score"] = pd.cut(df["monetary"], bins=min(n_quantiles, df["monetary"].nunique()),
                           labels=False, include_lowest=True, duplicates="drop")
    # Fill NaN with 0 and convert to int
    for col in ["R_score", "F_score", "M_score"]:
        df[col] = df[col].fillna(0).astype(int) + 1
    # Invert R (lower recency = higher score = better)
    df["R_score"] = n_quantiles + 1 - df["R_score"]
    df["RFM_segment"] = df["R_score"].astype(str) + df["F_score"].astype(str) + df["M_score"].astype(str)
    df["RFM_score"] = df["R_score"] + df["F_score"] + df["M_score"]
    return df


def segment_summary(segmented_df):
    """Summarize segment sizes and average metrics."""
    segments = segmented_df.groupby("RFM_score").agg(
        n_customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).reset_index()
    return segments


def basket_support(itemsets, itemset):
    """Support = fraction of transactions containing the itemset."""
    if not itemsets:
        return 0.0
    count = sum(1 for t in itemsets if set(itemset).issubset(set(t)))
    return count / len(itemsets)


def basket_confidence(itemsets, antecedent, consequent):
    """Confidence = support(A∪C) / support(A)."""
    sup_a = basket_support(itemsets, antecedent)
    if sup_a == 0:
        return 0.0
    sup_ac = basket_support(itemsets, list(set(antecedent) | set(consequent)))
    return sup_ac / sup_a


def basket_lift(itemsets, antecedent, consequent):
    """Lift = confidence / support(C). Lift > 1 = positive correlation."""
    sup_c = basket_support(itemsets, consequent)
    if sup_c == 0:
        return 0.0
    return basket_confidence(itemsets, antecedent, consequent) / sup_c


def trend_decomposition(series, window=12):
    """Simple trend/seasonal/residual decomposition using moving average."""
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < window * 2:
        return {"trend": s.tolist(), "seasonal": [0.0] * n, "residual": [0.0] * n}
    # Trend via centered moving average
    trend = np.convolve(s, np.ones(window) / window, mode="same")
    detrended = s - trend
    # Seasonal = average detrended value per position in cycle
    seasonal = np.zeros(n)
    for i in range(n):
        seasonal[i] = np.mean(detrended[i::window]) if i < window else seasonal[i % window]
    residual = s - trend - seasonal
    return {"trend": trend.tolist(), "seasonal": seasonal.tolist(), "residual": residual.tolist()}