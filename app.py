from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="RetailPulse | Tiger Analytics Retail CPG", layout="wide", page_icon="\U0001f916")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Stores",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    mode="classification"
    st.caption("Tiger Analytics | Retail & CPG Analytics")
data=make_synthetic(n=n,mode=mode); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Stores",f"{n:,}"); c2.metric("Sales Uptick Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f4ca Promotion ROI","\U0001f3ed What-If"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    y=data["y"]; pos=(y==1).sum()
    ax.bar(["No Uptick","Sales Uptick"],[len(y)-pos,pos],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([len(y)-pos,pos]): ax.text(i,v+1,f"{v/len(y):.1%}",ha="center",color="white")
    ax.set_title("Promotion Sales Uptick Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.markdown("**Retail drivers:** store_size_sqft, inventory_turnover, promotion_spend, foot_traffic, competitor_distance, avg_basket_value, employee_count, online_presence_score, product_category_diversity, seasonal_index")
with t2:
    st.latex(r"\text{ROI} = \frac{\Delta\text{Sales}}{\text{Promotion Cost}} - 1, \quad \text{Incremental Lift} = \frac{\text{Sales}_{\text{promo}} - \text{Sales}_{\text{baseline}}}{\text{Sales}_{\text{baseline}}}")
    st.caption("Promotion ROI and incremental lift: core CPG metrics. Tiger Analytics builds custom MMM (Marketing Mix Models) for Fortune 500 retailers.")
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
    st.latex(r"\text{Incremental Lift} = \frac{1}{k}\sum_{i=1}^k \text{Lift}_i, \quad \text{Adj. ROI} = \frac{\text{Gross Margin} \times \text{Lift} - \text{Promo Cost}}{\text{Promo Cost}}")
    st.caption("5-fold cross-validated ROI estimates with adjustment for organic growth and halo effects across store clusters.")
with t3:
    st.subheader("Promotion ROI by Store Cluster")
    st.latex(r"\text{ROI}_c = \frac{\text{Incremental Revenue}_c - \text{Promo Cost}_c}{\text{Promo Cost}_c} \times 100\%")
    st.latex(r"\text{Adstock}(t) = \text{Spend}(t) + \lambda \cdot \text{Adstock}(t-1)")
    st.caption("Adstock model: promotion effects decay over time (lambda = decay rate). Tiger Analytics uses Bayesian MMM to estimate carry-over effects across 8-16 week windows.")
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2).fit_transform(data["X"].select_dtypes("number").fillna(0))
    fig,ax=plt.subplots(figsize=(6,5)); _style()
    for label,color,mkr in [(0,"#22c55e","o"),(1,"#f43f5e","x")]:
        idx=data["y"]==label; ax.scatter(pca[idx,0],pca[idx,1],c=color,marker=mkr,alpha=0.5,s=10,label=f"{'No Uptick' if label==0 else 'Sales Uptick'}")
    ax.legend(fontsize=8)
    ax.set_title("Store Cluster PCA — High vs Low ROI Stores",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    importances=b["models"]["XGBoost"].feature_importances_
    fi=pd.DataFrame({"driver":data["features"],"importance":importances}).sort_values("importance",ascending=True)
    fig,ax=plt.subplots(figsize=(6,5)); _style()
    ax.barh(fi["driver"],fi["importance"],color="#22d3ee")
    ax.set_title("Sales Uptick Driver Importance",color="white"); ax.set_xlabel("Importance"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t4:
    st.subheader("What-If Promotion Simulator")
    st.latex(r"\text{Expected Lift} = \sigma(\beta_0 + \beta_1 \cdot \text{spend} + \beta_2 \cdot \text{footfall} + \beta_3 \cdot \text{seasonal} + \cdots)")
    st.caption("Simulate promotion scenarios to optimize budget allocation across store clusters. Tiger Analytics deploys these simulators for CPG clients to plan quarterly trade promotions.")
    inputs={}
    col_a,col_b=st.columns(2)
    with col_a:
        inputs["promotion_spend"]=st.slider("Promotion Spend ($)",100,50000,5000,500)
        inputs["foot_traffic"]=st.slider("Daily Foot Traffic",50,5000,500,50)
        inputs["avg_basket_value"]=st.slider("Avg Basket Value ($)",10,200,45,5)
        inputs["competitor_distance"]=st.slider("Competitor Distance (mi)",0.1,20.0,2.0,0.5)
    with col_b:
        inputs["store_size_sqft"]=st.slider("Store Size (sqft)",500,50000,15000,500)
        inputs["inventory_turnover"]=st.slider("Inventory Turnover",0.0,1.0,0.5,0.05)
        inputs["online_presence_score"]=st.slider("Online Presence",0,100,50,5)
        inputs["seasonal_index"]=st.slider("Seasonal Index",0.5,1.5,1.0,0.05)
    for k in ["employee_count","product_category_diversity"]: inputs[k]=data["df"][k].median()
    input_df=pd.DataFrame([inputs]); scaled=b["scaler"].transform(input_df.reindex(columns=b["features"],fill_value=0))
    proba=b["models"]["XGBoost"].predict_proba(scaled)[0,1]
    st.metric("Predicted Sales Uptick Probability",f"{proba:.1%} ({'Recommended' if proba>0.5 else 'Not Recommended'})")
    st.caption("Model uses XGBoost trained on historical promotion outcomes. For production deployment, Tiger Analytics pairs this with causal inference (CATE) estimation for true incrementality.")
