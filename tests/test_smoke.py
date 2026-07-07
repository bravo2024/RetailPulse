"""Smoke tests for RetailPulse — RFM segmentation and basket analysis."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import run_rfm_analysis, mine_association_rules, monthly_sales_trend, fit_and_evaluate
from src.core import basket_support, basket_confidence, basket_lift, trend_decomposition


def test_data():
    d = make_synthetic(n_customers=50, n_transactions=500, seed=42)
    assert d["n_transactions"] > 0
    assert "transactions" in d
    assert "itemsets" in d


def test_rfm_analysis():
    d = make_synthetic(n_customers=50, n_transactions=500, seed=42)
    result = run_rfm_analysis(d, n_quantiles=4)
    assert result["n_customers"] > 0
    assert "summary" in result
    assert "RFM_score" in result["rfm"].columns


def test_basket_metrics():
    """Basket support/confidence/lift compute correctly."""
    itemsets = [["bread", "butter"], ["bread", "butter"], ["bread", "milk"]]
    assert basket_support(itemsets, ["bread"]) > 0.5
    assert basket_confidence(itemsets, ["bread"], ["butter"]) > 0.5
    assert basket_lift(itemsets, ["butter"], ["bread"]) >= 1.0


def test_association_rules():
    d = make_synthetic(n_customers=100, n_transactions=1000, seed=42)
    rules = mine_association_rules(d["itemsets"], min_support=0.02, min_confidence=0.2)
    assert isinstance(rules, list)
    # Should find some rules given the co-purchase patterns
    if rules:
        assert "support" in rules[0]
        assert "confidence" in rules[0]
        assert "lift" in rules[0]


def test_fit_and_evaluate():
    d = make_synthetic(n_customers=100, n_transactions=1000, seed=42)
    model, metrics = fit_and_evaluate(d, seed=42)
    assert "n_customers" in metrics
    assert "n_rules" in metrics
    assert "trend_slope" in metrics


if __name__ == "__main__":
    test_data()
    test_rfm_analysis()
    test_basket_metrics()
    test_association_rules()
    test_fit_and_evaluate()
    print("All RetailPulse smoke tests passed!")
