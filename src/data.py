"""data.py — Synthetic retail transaction data for RetailPulse.

Transaction-level data with customer ID, product, date, and amount.
This is retail transaction data for RFM segmentation and basket analysis,
NOT generic tabular features for classification.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any

PRODUCTS = ["bread", "milk", "eggs", "butter", "cheese", "yogurt", "coffee",
            "tea", "cereal", "pasta", "sauce", "rice", "chicken", "beef",
            "fish", "vegetables", "fruits", "chocolate", "cookies", "juice"]


def make_synthetic(n_customers: int = 500, n_transactions: int = 3000, seed: int = 42) -> dict[str, Any]:
    """Generate synthetic retail transactions with customer purchase patterns.

    Some products are frequently bought together (e.g., bread+butter, pasta+sauce)
    to create discoverable basket association rules.
    """
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-12-31")
    date_range = (end_date - start_date).days

    rows = []
    # Product co-purchase groups
    copurchase_groups = [
        ["bread", "butter", "eggs"],
        ["pasta", "sauce"],
        ["coffee", "milk"],
        ["chocolate", "cookies"],
        ["chicken", "vegetables"],
    ]

    for tx_id in range(n_transactions):
        cust_id = f"C{rng.integers(0, n_customers):04d}"
        date = start_date + pd.Timedelta(days=int(rng.integers(0, date_range)))
        # Pick a co-purchase group 40% of the time, random items otherwise
        if rng.random() < 0.4:
            group = copurchase_groups[rng.integers(0, len(copurchase_groups))]
            n_items = rng.integers(1, len(group) + 1)
            items = [group[rng.integers(0, len(group))] for _ in range(n_items)]
        else:
            n_items = rng.integers(1, 4)
            items = [PRODUCTS[rng.integers(0, len(PRODUCTS))] for _ in range(n_items)]

        for item in items:
            amount = round(float(rng.lognormal(2, 0.5)), 2)
            rows.append({
                "transaction_id": f"T{tx_id:05d}",
                "customer_id": cust_id,
                "date": date,
                "product": item,
                "amount": amount,
            })

    df = pd.DataFrame(rows)
    # Create itemsets (one set per transaction)
    itemsets = df.groupby("transaction_id")["product"].apply(list).tolist()

    return {
        "transactions": df,
        "itemsets": itemsets,
        "n_customers": n_customers,
        "n_transactions": len(itemsets),
        "n_rows": len(df),
        "products": PRODUCTS,
    }