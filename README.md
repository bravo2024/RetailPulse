# RetailPulse

> Retail CPG promotion response prediction with ROI and incremental lift analysis.

Trains four classifiers on synthetic store-level data to predict sales uptick from promotions. Dashboard provides data exploration, multi-model comparison, promotion ROI and incremental lift metrics, store-level what-if simulation, and Marketing Mix Model (MMM) style analysis for Fortune 500 retail decision-making.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.692 |
| Gini | 0.383 |
| KS Statistic | 0.292 |
| F1 Score | 0.593 |
| Accuracy | 0.640 |

5-fold CV AUC: 0.684 ± 0.016. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Store dataset overview, sales uptick distribution, retail driver descriptions |
| **Model Lab** | Multi-model comparison, ROC/calibration curves, CV results, MMM-style ROI estimates |
| **Promotion ROI** | Promotion ROI by store segment, incremental lift analysis, gross margin impact |
| **What-If** | Store-level promotion spend simulation, threshold-based decision optimisation |

## Repo Structure

```
RetailPulse/
  src/         data, model, evaluate, persist, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic retail CPG dataset: store size, inventory turnover, promotion spend, foot traffic, competitor distance, basket value, employee count, online presence score, category diversity, seasonal index.

## License

MIT
