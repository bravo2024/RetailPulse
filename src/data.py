from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["store_size_sqft","inventory_turnover","promotion_spend","foot_traffic","competitor_distance","avg_basket_value","employee_count","online_presence_score","product_category_diversity","seasonal_index"]
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = FEATURE_NAMES
TARGET_NAME = "sales_uptick"
def make_synthetic(n=10000,seed=42,mode="classification"):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "store_size_sqft": rng.uniform(500,50000,size=n).round(0),
        "inventory_turnover": rng.beta(5,3,size=n).round(3),
        "promotion_spend": rng.lognormal(mean=9,sigma=1,size=n).round(2),
        "foot_traffic": rng.poisson(lam=500,size=n).clip(50,5000),
        "competitor_distance": rng.exponential(scale=2,size=n).clip(0.1,20).round(1),
        "avg_basket_value": rng.lognormal(mean=4,sigma=0.5,size=n).round(2),
        "employee_count": rng.poisson(lam=15,size=n).clip(2,100),
        "online_presence_score": rng.uniform(0,100,size=n).round(1),
        "product_category_diversity": rng.poisson(lam=8,size=n).clip(1,25),
        "seasonal_index": rng.uniform(0.5,1.5,size=n).round(3),
    })
    size=np.clip(df["store_size_sqft"]/50000,0,1); turnover=df["inventory_turnover"]
    promo=np.log(df["promotion_spend"]+1)/12; traffic=np.clip(df["foot_traffic"]/5000,0,1)
    comp=np.clip(1-df["competitor_distance"]/20,0,1); basket=df["avg_basket_value"]/150
    emp=np.clip(df["employee_count"]/100,0,1); online=df["online_presence_score"]/100
    prod_div=np.clip(df["product_category_diversity"]/25,0,1); seasonal=df["seasonal_index"]
    log_odds = -2.5 + 0.3*size + 0.4*turnover + 0.5*promo + 0.3*traffic - 0.2*comp + 0.3*basket + 0.1*emp + 0.3*online + 0.2*prod_div + 0.3*seasonal + rng.normal(0,0.4,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,60)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(sales_uptick=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean(),"mode":mode}
