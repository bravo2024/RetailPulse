"""CLI entrypoint for RetailPulse: RFM segmentation + basket-rule mining + trend decomposition."""
from __future__ import annotations

import argparse

from src.data import make_synthetic
from src.model import fit_and_evaluate
from src.evaluate import save_metrics
from src.persist import save_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--customers", type=int, default=500)
    p.add_argument("--transactions", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = make_synthetic(n_customers=args.customers, n_transactions=args.transactions, seed=args.seed)
    print(f"{data['n_customers']:,} customers, {data['n_transactions']:,} transactions")

    model, metrics = fit_and_evaluate(data, seed=args.seed)
    print(f"RFM segments: {metrics['n_segments']}  |  customers scored: {metrics['n_customers']:,}")
    print(f"Association rules found: {metrics['n_rules']}")
    for r in metrics["top_5_rules"]:
        print(f"  {r['antecedent']} -> {r['consequent']}  lift={r['lift']:.2f}  conf={r['confidence']:.2f}")
    print(f"Monthly sales trend slope: {metrics['trend_slope']:.2f}")
    print(f"Total revenue (synthetic): ${metrics['total_revenue']:,.2f}")

    save_model(model)
    save_metrics(metrics)
    print("Saved model -> models/model.pkl, metrics -> models/metrics.json")


if __name__ == "__main__":
    main()
