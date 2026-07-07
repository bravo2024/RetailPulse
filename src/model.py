"""model.py — RFM segmentation and basket analysis for RetailPulse.

Two retail analytics approaches (NOT classification):
1. **RFM segmentation** — score customers by Recency/Frequency/Monetary
   and group them into tiers for targeted marketing.
2. **Basket analysis** — mine product co-purchase association rules
   using support/confidence/lift.

This is retail analytics, fundamentally different from the generic
sklearn classification template it replaces.

References
----------
Hughes (1996), "Strategic Database Marketing." (RFM)
Agrawal et al. (1993), "Mining Association Rules between Sets of Items."
"""
from __future__ import annotations
import numpy as np
from itertools import combinations
from typing import Any

from src.core import (
    compute_rfm, rfm_segment, segment_summary,
    basket_support, basket_confidence, basket_lift, trend_decomposition,
)


def run_rfm_analysis(data: dict, n_quantiles: int = 4) -> dict[str, Any]:
    """Run full RFM segmentation pipeline."""
    transactions = data["transactions"]
    rfm = compute_rfm(transactions)
    segmented = rfm_segment(rfm, n_quantiles=n_quantiles)
    summary = segment_summary(segmented)
    return {
        "rfm": segmented, "summary": summary,
        "n_customers": len(segmented),
        "n_segments": len(summary),
    }


def mine_association_rules(itemsets: list[list[str]], min_support: float = 0.02,
                           min_confidence: float = 0.3, max_itemset_size: int = 2) -> list[dict]:
    """Mine frequent itemsets and generate association rules.

    For each frequent itemset of size 2, generate rules A → B with
    support, confidence, and lift metrics.
    """
    n = len(itemsets)
    if n == 0:
        return []

    # Find frequent itemsets of size 2
    itemset_counts = {}
    for transaction in itemsets:
        items = sorted(set(transaction))
        for size in range(2, max_itemset_size + 1):
            for combo in combinations(items, size):
                itemset_counts[combo] = itemset_counts.get(combo, 0) + 1

    frequent = {k: v for k, v in itemset_counts.items() if v / n >= min_support}

    # Generate rules from 2-itemsets
    rules = []
    for itemset, count in frequent.items():
        if len(itemset) != 2:
            continue
        a, b = itemset
        sup = count / n
        conf = basket_confidence(itemsets, [a], [b])
        lift = basket_lift(itemsets, [a], [b])
        if conf >= min_confidence:
            rules.append({
                "antecedent": a, "consequent": b,
                "support": round(sup, 4), "confidence": round(conf, 4),
                "lift": round(lift, 4),
            })

    rules.sort(key=lambda r: r["lift"], reverse=True)
    return rules


def monthly_sales_trend(transactions) -> dict:
    """Compute monthly sales trend and decompose into trend/seasonal/residual."""
    import pandas as pd
    df = transactions.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["amount"].sum().sort_index()
    series = monthly.values
    decomp = trend_decomposition(series, window=min(6, len(series) // 2))
    return {
        "monthly_sales": monthly.to_dict(),
        "decomposition": decomp,
        "trend_slope": float(np.polyfit(np.arange(len(series)), series, 1)[0]) if len(series) > 1 else 0.0,
    }


def fit_and_evaluate(data: dict, seed: int = 42) -> tuple:
    """Run RFM + basket analysis + trend decomposition. Returns (model, metrics)."""
    rfm_result = run_rfm_analysis(data, n_quantiles=4)
    rules = mine_association_rules(data["itemsets"], min_support=0.02, min_confidence=0.2)
    trend = monthly_sales_trend(data["transactions"])

    model = {"rfm": rfm_result, "rules": rules, "trend": trend}
    metrics = {
        "n_customers": rfm_result["n_customers"],
        "n_segments": rfm_result["n_segments"],
        "n_rules": len(rules),
        "top_5_rules": rules[:5],
        "trend_slope": trend["trend_slope"],
        "total_revenue": float(data["transactions"]["amount"].sum()),
    }
    return model, metrics