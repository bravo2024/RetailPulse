# RetailPulse

Retail transaction analytics: RFM segmentation, basket-rule mining, and sales trend decomposition.

Three classical retail-analytics techniques, each implemented from scratch
against a synthetic transaction log (customers, products, dates, amounts):

- **RFM segmentation** (Hughes, 1996) — quantile-scores each customer on
  Recency, Frequency, and Monetary value, then rolls the three scores into
  a segment tier for targeted marketing.
- **Basket-rule mining** (Agrawal et al., 1993) — computes support,
  confidence, and lift for every 2-item combination to surface
  "customers who bought X also bought Y" rules. The synthetic generator
  seeds five co-purchase groups (bread+butter+eggs, pasta+sauce, …) so the
  miner has real structure to find.
- **Trend decomposition** — a centered moving-average split of monthly
  revenue into trend, seasonal, and residual components.

## Run it

```bash
pip install -r requirements.txt
python train.py        # RFM + basket rules + trend, writes models/metrics.json
pytest -q
streamlit run app.py    # 3 tabs: RFM segments, basket rules, trend decomposition
```

## A note on the rule-mining threshold

The default association-rule confidence threshold (0.3) turns out to be
higher than this generator's actual co-purchase signal (~0.20-0.25
confidence for the seeded pairs), so it silently returns zero rules —
looks broken, isn't. `fit_and_evaluate` uses 0.2 by default; the dashboard
sidebar also exposes both thresholds directly so you can see the
support/confidence/lift trade-off first-hand.

## Layout

```
src/data.py     synthetic transaction generator (customers, products, co-purchase groups)
src/core.py     RFM scoring, basket support/confidence/lift, trend decomposition
src/model.py    RFM pipeline, rule mining, monthly trend, fit_and_evaluate
app.py          Streamlit dashboard
train.py        CLI entrypoint
```

## License

MIT
