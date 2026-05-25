
# 🏦 DLPO — Digital Lending Portfolio Optimization
> An enterprise-grade AI platform for intelligent, transparent, and ethically governed digital credit management.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)


---

## 📌 Project Overview

DLPO is a **13-module**, production-ready AI lending intelligence platform built as a capstone project for the IIT Guwahati Executive Programme in Data Science & AI.

It addresses the complete digital lending lifecycle — from origination intelligence to portfolio risk management — with **enterprise-grade explainability**, **fairness monitoring**, and **MLOps pipelines**.

### Key Performance Metrics
| Metric | Value |
|--------|-------|
| Credit Risk ROC-AUC | 0.85 |
| KS Statistic | 0.45 |
| Gini Coefficient | 0.70 |
| Fraud Detection Recall | 88% |
| Fraud False Positive Rate | < 8% |
| Early Warning Lead Time | 7–15 DPD |
| Explainability Coverage | 100% (SHAP) |
| Projected EL Reduction | 18–25% |
| NIM Improvement | +3.7pp |

---

## 🏗️ Architecture

```
DLPO Platform
├── Layer 1: Data Intelligence
│   ├── Module 1  — Synthetic Data Generation
│   ├── Module 2  — Data Engineering & Feature Store
│   └── Module 3  — EDA & Portfolio Intelligence
├── Layer 2: AI/ML Decision Engine
│   ├── Module 4  — Customer Segmentation (10 Personas)
│   ├── Module 5  — Credit Risk ML Engine
│   └── Module 6  — Dynamic Risk-Based Pricing (RAROC)
├── Layer 3: Risk Intelligence
│   ├── Module 7  — Early Warning & Collections
│   ├── Module 9  — Portfolio Optimization (CVaR)
│   └── Module 10 — Fraud Detection & Financial Crime
├── Layer 4: Governance & Explainability
│   ├── Module 8  — XAI & AI Governance (SHAP + LIME + DiCE)
│   └── Module 11 — Executive BI & Analytics
└── Layer 5: Platform & Operations
    ├── Module 12 — End-to-End Streamlit Lending Platform
    └── Module 13 — Enterprise MLOps & Production Operations
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/abhiram/dlpo-lending-platform.git
cd dlpo-lending-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit platform
streamlit run app.py
```

---

## 📁 Project Structure

```
iitg/
├── lending_features/          # Feature store outputs (Module 2)
├── segmentation/              # Borrower personas (Module 4)
├── risk_models/               # Trained ML models (Module 5)
├── pricing_engine/            # Pricing pipeline (Module 6)
├── early_warning/             # EWS models (Module 7)
├── explainable_ai/            # SHAP/LIME outputs (Module 8)
├── portfolio_optimization/    # CVaR models (Module 9)
├── fraud_detection/           # Fraud models (Module 10)
├── executive_bi/              # Dashboard outputs (Module 11)
├── streamlit_app/             # Streamlit platform (Module 12)
├── mlops/                     # MLOps pipelines (Module 13)
└── final_documentation/       # This module (Module 14)
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| ML/AI | XGBoost, LightGBM, CatBoost, scikit-learn |
| XAI | SHAP, LIME, DiCE-ML |
| Optimisation | Optuna, scipy, cvxpy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Platform | Streamlit, SQLite, Python 3.10+ |
| MLOps | MLflow, joblib, custom pipeline registry |
| Monitoring | PSI, CSI, statistical drift detection |

---

## 📊 Module Overview

| Module | Description | Key Output |
|--------|-------------|------------|
| M1 | Synthetic Data Generation | 50,000 synthetic loans |
| M2 | Feature Engineering | 80+ engineered features |
| M3 | EDA & BI | Portfolio intelligence dashboard |
| M4 | Customer Segmentation | 10 borrower personas |
| M5 | Credit Risk ML | PD model (AUC=0.85) |
| M6 | Dynamic Pricing | RAROC-optimised rates |
| M7 | Early Warning | 7-15 DPD alert system |
| M8 | XAI & Governance | SHAP + counterfactual |
| M9 | Portfolio Optimisation | CVaR allocation |
| M10 | Fraud Detection | 88% recall fraud engine |
| M11 | Executive BI | CRO dashboard |
| M12 | Streamlit Platform | Full web application |
| M13 | MLOps | Production pipeline |
| M14 | Documentation | This module |

---

## 🎯 Business Impact

For a ₹1,000 Crore digital lending portfolio:
- **Credit Loss Reduction**: ₹18–25 Crore saved annually
- **Fraud Prevention**: ₹8–12 Crore losses avoided
- **Pricing Optimisation**: +₹15–20 Crore NIM improvement
- **Total Annual Value**: ₹46–65 Crore
- **Platform ROI**: 12–20× on investment

---

## 📜 Regulatory Alignment

- ✅ RBI Digital Lending Guidelines (2022)
- ✅ Adverse Action Notice generation (SHAP)
- ✅ Fair lending — disparate impact analysis
- ✅ Model governance — PSI/CSI monitoring
- ✅ Audit trail — complete decision lineage

---

## 👤 Author

**Abhiram Adabala**  
Data Science & AI  
Indian Institute of Technology , Kharagpur 
📧 abhiram.adabala123@gmail.com 
 


