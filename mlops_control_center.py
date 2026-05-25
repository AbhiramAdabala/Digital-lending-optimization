
"""
╔══════════════════════════════════════════════════════════════════════╗
║  FINLEND AI — MLOPS CONTROL CENTER                                  ║
║  Module 13 — Enterprise MLOps & Model Operations                    ║
║  Run: streamlit run mlops_control_center.py --server.port 8503      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json, time

st.set_page_config(
    page_title="FinLend AI — MLOps Control Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAL = {
    'navy': '#0A1628', 'accent': '#00D4FF', 'success': '#00C896',
    'danger': '#FF4757', 'warning': '#FFA502', 'gold': '#FFB800',
    'purple': '#7B2FBE', 'neutral': '#8892A4', 'bg': '#F0F4F8',
}

st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: "DM Sans", sans-serif; }}
  .main {{ background: {PAL["bg"]}; }}
  .mlops-header {{
    background: linear-gradient(135deg, {PAL["navy"]} 0%, #1A3A6B 100%);
    color: white; padding: 20px 28px; border-radius: 14px;
    margin-bottom: 22px; border: 1px solid rgba(0,212,255,0.2);
  }}
  .kpi-card {{
    background: white; border-radius: 12px; padding: 18px 20px;
    border-top: 3px solid {PAL["accent"]};
    box-shadow: 0 1px 8px rgba(15,23,42,0.07);
  }}
  .model-card {{
    background: white; border-radius: 10px; padding: 16px 18px;
    margin: 6px 0; border-left: 4px solid;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  }}
  .alert-red    {{ background: #FEF2F2; border-left: 4px solid #EF4444 !important; }}
  .alert-yellow {{ background: #FFFBEB; border-left: 4px solid #F59E0B !important; }}
  .alert-green  {{ background: #ECFDF5; border-left: 4px solid #10B981 !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mlops-header">
  <div style="font-size:24px;font-weight:700;color:{PAL["accent"]};
              font-family:monospace;">⚙️ FinLend AI — MLOps Control Center</div>
  <div style="font-size:13px;color:rgba(255,255,255,0.55);margin-top:4px;">
    Enterprise Model Operations · Drift Monitoring · Governance · Retraining
    · {datetime.now().strftime("%B %d, %Y  %H:%M")}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:{PAL["navy"]};padding:16px;border-radius:10px;margin-bottom:12px;">
      <div style="color:{PAL["accent"]};font-size:16px;font-weight:700;">⚙️ MLOps Platform</div>
      <div style="color:{PAL["neutral"]};font-size:11px;">Module 13 — Control Center</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠 Platform Overview",
        "📊 Model Registry",
        "🔄 Drift Monitor",
        "🔁 Retraining",
        "⚖️ Governance",
        "🔌 API Health",
        "🚨 Alerts",
    ])


# ── Data Simulation ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def generate_platform_data():
    np.random.seed(42)
    n = 168
    ts = pd.date_range("2024-12-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp":      ts,
        "requests":       np.random.poisson(450, n),
        "latency_p99":    np.clip(180 + np.random.normal(0, 30, n), 50, 600),
        "error_rate":     np.clip(np.random.exponential(0.3, n), 0, 5),
        "pd_score_mean":  np.clip(0.18 + np.cumsum(np.random.normal(0, 0.001, n)), 0.05, 0.55),
        "fraud_alerts":   np.random.poisson(8, n) + (np.arange(n) > 120) * 5,
        "model_uptime":   np.clip(99.9 - np.random.exponential(0.05, n), 98.0, 100.0),
    })


df_mon = generate_platform_data()

MODELS = {
    "PD_Model":    {"version": "v2.1", "stage": "Production", "auc": 0.812, "psi": 0.089, "last_retrained": "2024-11-28"},
    "Fraud_Model": {"version": "v1.3", "stage": "Production", "auc": 0.921, "psi": 0.142, "last_retrained": "2024-11-22"},
    "EWS_Model":   {"version": "v1.1", "stage": "Staging",    "auc": 0.774, "psi": 0.063, "last_retrained": "2024-12-01"},
}


# ── Page Routing ──────────────────────────────────────────────────────────────
if "Overview" in page:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <div style="font-size:11px;color:{PAL["neutral"]}">Models in Production</div>
          <div style="font-size:28px;font-weight:700;">2</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        avg_p99 = df_mon["latency_p99"].mean()
        color = PAL["danger"] if avg_p99 > 200 else PAL["success"]
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{color};">
          <div style="font-size:11px;color:{PAL["neutral"]}">Avg P99 Latency</div>
          <div style="font-size:28px;font-weight:700;">{avg_p99:.0f}ms</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        uptime = df_mon["model_uptime"].mean()
        color = PAL["success"] if uptime >= 99.9 else PAL["warning"]
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{color};">
          <div style="font-size:11px;color:{PAL["neutral"]}">Model Uptime</div>
          <div style="font-size:28px;font-weight:700;">{uptime:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        max_psi = max(m["psi"] for m in MODELS.values())
        color = PAL["danger"] if max_psi > 0.20 else PAL["warning"] if max_psi > 0.10 else PAL["success"]
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{color};">
          <div style="font-size:11px;color:{PAL["neutral"]}">Max PSI (Drift)</div>
          <div style="font-size:28px;font-weight:700;">{max_psi:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        total_reqs = df_mon["requests"].sum()
        st.markdown(f"""
        <div class="kpi-card">
          <div style="font-size:11px;color:{PAL["neutral"]}">Requests (7d)</div>
          <div style="font-size:28px;font-weight:700;">{total_reqs:,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("API Request Volume (7 days)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_mon["timestamp"], y=df_mon["requests"],
            fill="tozeroy", line=dict(color=PAL["accent"], width=1.5),
            fillcolor="rgba(0,212,255,0.15)",
        ))
        fig.update_layout(height=300, paper_bgcolor="white",
                          yaxis_title="Requests/hr",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("P99 Latency Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_mon["timestamp"], y=df_mon["latency_p99"],
            line=dict(color=PAL["warning"], width=1.5),
        ))
        fig.add_hline(y=200, line_dash="dash", line_color=PAL["danger"],
                       annotation_text="SLO (200ms)")
        fig.update_layout(height=300, paper_bgcolor="white",
                          yaxis_title="Latency (ms)",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


elif "Registry" in page:
    st.subheader("📊 Model Registry — Registered Models")
    for model_name, info in MODELS.items():
        psi_color = PAL["danger"] if info["psi"] > 0.20 else PAL["warning"] if info["psi"] > 0.10 else PAL["success"]
        stage_color = PAL["success"] if info["stage"] == "Production" else PAL["warning"]
        st.markdown(f"""
        <div class="model-card" style="border-color:{stage_color};">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
              <span style="font-weight:700;font-size:15px;">{model_name}</span>
              <span style="background:{stage_color};color:white;padding:3px 10px;
                border-radius:12px;font-size:11px;margin-left:10px;">{info["stage"]}</span>
            </div>
            <span style="font-size:12px;color:{PAL["neutral"]}">Version: {info["version"]}</span>
          </div>
          <div style="margin-top:10px;display:flex;gap:20px;font-size:13px;">
            <span>🎯 AUC-ROC: <b>{info["auc"]:.3f}</b></span>
            <span style="color:{psi_color};">📊 PSI: <b>{info["psi"]:.3f}</b></span>
            <span>📅 Retrained: {info["last_retrained"]}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Governance Actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        model_sel = st.selectbox("Select Model", list(MODELS.keys()))
    with c2:
        action_sel = st.selectbox("Action", ["Promote to Production", "Move to Staging", "Archive", "Rollback"])
    with c3:
        approver_name = st.text_input("Approver Name", "CRO")
    if st.button("✅ Execute Governance Action", type="primary"):
        st.success(f"✅ Action '{action_sel}' applied to {model_sel} by {approver_name} at {datetime.now().strftime('%H:%M:%S')}")


elif "Drift" in page:
    st.subheader("🔄 Feature Drift Monitor")
    np.random.seed(42)
    features = ["credit_score", "pd_score", "financial_stress_index",
                 "missed_payment_ratio", "spending_volatility", "debt_to_income",
                 "income_stability", "loan_amount"]
    psi_vals = np.array([0.18, 0.14, 0.22, 0.09, 0.07, 0.19, 0.05, 0.03])
    ks_vals  = np.array([0.11, 0.09, 0.13, 0.06, 0.04, 0.12, 0.03, 0.02])

    drift_data = pd.DataFrame({"Feature": features, "PSI": psi_vals, "KS": ks_vals})
    drift_data["Status"] = drift_data["PSI"].apply(
        lambda x: "RETRAIN" if x >= 0.25 else "ALERT" if x >= 0.20 else "MONITOR" if x >= 0.10 else "OK"
    )
    drift_data["Color"] = drift_data["Status"].map({
        "RETRAIN": PAL["danger"], "ALERT": PAL["warning"],
        "MONITOR": PAL["gold"], "OK": PAL["success"]
    })

    fig = go.Figure(go.Bar(
        x=drift_data["PSI"], y=drift_data["Feature"],
        orientation="h",
        marker_color=drift_data["Color"].tolist(),
        text=[f"{v:.3f} ({s})" for v, s in zip(drift_data["PSI"], drift_data["Status"])],
        textposition="outside",
    ))
    fig.add_vline(x=0.10, line_dash="dash", line_color=PAL["gold"],    annotation_text="Monitor")
    fig.add_vline(x=0.20, line_dash="dash", line_color=PAL["warning"], annotation_text="Alert")
    fig.add_vline(x=0.25, line_dash="dash", line_color=PAL["danger"],  annotation_text="Retrain")
    fig.update_layout(height=400, paper_bgcolor="white",
                      xaxis_title="PSI Score",
                      margin=dict(l=0, r=80, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    n_alert = (drift_data["Status"].isin(["ALERT", "RETRAIN"])).sum()
    if n_alert > 0:
        st.warning(f"⚠️ {n_alert} feature(s) require attention. Consider triggering retraining.")
    else:
        st.success("✅ All features within acceptable drift thresholds.")


elif "Retraining" in page:
    st.subheader("🔁 Retraining Control Panel")
    c1, c2, c3 = st.columns(3)
    with c1:
        retrain_model = st.selectbox("Model", list(MODELS.keys()))
    with c2:
        trigger_type = st.selectbox("Trigger Reason",
            ["Manual", "PSI Drift", "AUC Degradation", "Fairness Drift", "Scheduled"])
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        launch = st.button("🚀 Launch Retraining", type="primary")

    if launch:
        with st.spinner(f"⚙️ Running retraining pipeline for {retrain_model}..."):
            import time as t_lib
            t_lib.sleep(1.5)
        before_auc = np.random.uniform(0.72, 0.78)
        after_auc  = np.random.uniform(0.79, 0.87)
        st.success(f"""
        ✅ Retraining complete!
        Model: {retrain_model} | Trigger: {trigger_type}
        AUC: {before_auc:.3f} → {after_auc:.3f} (+{(after_auc-before_auc):.3f})
        Status: GOVERNANCE GATE PASSED — Ready for staging review.
        """)

    st.subheader("Recent Retraining Jobs")
    jobs_data = pd.DataFrame([
        {"Job ID": "RTJ-A1B2C3", "Model": "PD_Model",    "Trigger": "PSI Alert",    "Status": "✅ PASSED", "AUC Δ": "+0.031", "Date": "2024-11-28"},
        {"Job ID": "RTJ-D4E5F6", "Model": "Fraud_Model", "Trigger": "Monthly",      "Status": "❌ FAILED", "AUC Δ": "-0.012", "Date": "2024-11-15"},
        {"Job ID": "RTJ-G7H8I9", "Model": "Fraud_Model", "Trigger": "Retry",        "Status": "✅ PASSED", "AUC Δ": "+0.018", "Date": "2024-11-22"},
        {"Job ID": "RTJ-J0K1L2", "Model": "EWS_Model",   "Trigger": "AUC Degrad.", "Status": "✅ PASSED", "AUC Δ": "+0.028", "Date": "2024-12-01"},
    ])
    st.dataframe(jobs_data, use_container_width=True, hide_index=True)


elif "Governance" in page:
    st.subheader("⚖️ AI Governance & Compliance")
    gov_checks = [
        ("Model Explainability",    True,  "SHAP values logged for 100% of decisions"),
        ("Audit Trail Active",      True,  "All predictions persisted to audit_log"),
        ("Fairness Monitoring",     True,  "DI ratio > 0.80 across all groups"),
        ("Model Approval Workflow", True,  "CRO approval required before production"),
        ("PSI Monitoring",          True,  "Drift monitored weekly; alert at PSI>0.20"),
        ("Rollback Capability",     True,  "All model versions retained in registry"),
        ("RBI Compliance",          True,  "Fair Practices Code implemented"),
        ("PMLA Fraud Logging",      True,  "All fraud alerts logged with SLA tracking"),
    ]
    cols = st.columns(2)
    for i, (check, ok, detail) in enumerate(gov_checks):
        with cols[i % 2]:
            if ok:
                st.success(f"✅ **{check}**\n\n{detail}")
            else:
                st.error(f"❌ **{check}**\n\n{detail}")


elif "API" in page:
    st.subheader("🔌 API Health & Performance")
    endpoints = [
        {"Endpoint": "/v1/underwriting/score", "Status": "🟢 Healthy", "Avg Latency": "48ms",  "P99": "182ms", "RPS": "142"},
        {"Endpoint": "/v1/fraud/score",        "Status": "🟢 Healthy", "Avg Latency": "18ms",  "P99": "67ms",  "RPS": "89"},
        {"Endpoint": "/v1/ews/score",           "Status": "🟢 Healthy", "Avg Latency": "52ms",  "P99": "198ms", "RPS": "34"},
        {"Endpoint": "/health",                 "Status": "🟢 Healthy", "Avg Latency": "2ms",   "P99": "8ms",   "RPS": "15"},
        {"Endpoint": "/v1/metrics/predictions", "Status": "🟢 Healthy", "Avg Latency": "12ms",  "P99": "45ms",  "RPS": "5"},
    ]
    st.dataframe(pd.DataFrame(endpoints), use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_mon["timestamp"], y=df_mon["model_uptime"],
        fill="tozeroy", fillcolor="rgba(0,200,150,0.1)",
        line=dict(color=PAL["success"], width=2),
        name="Model Uptime",
    ))
    fig.add_hline(y=99.9, line_dash="dash", line_color=PAL["danger"],
                   annotation_text="SLO (99.9%)")
    fig.update_layout(height=300, paper_bgcolor="white",
                      yaxis_title="Uptime (%)", yaxis_range=[98, 100.2],
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


elif "Alerts" in page:
    st.subheader("🚨 Active Governance & Operational Alerts")
    alerts = [
        {"⏰": "2 min ago",  "Severity": "🔴 CRITICAL", "Source": "Drift Detector",    "Message": "financial_stress_index PSI=0.22 — Alert threshold exceeded",   "Action": "Trigger retraining within 24h"},
        {"⏰": "15 min ago", "Severity": "🟠 HIGH",     "Source": "Governance Engine", "Message": "Fraud_Model PSI=0.142 approaching alert threshold (0.15)",        "Action": "Increase monitoring to hourly"},
        {"⏰": "1 hr ago",   "Severity": "🟠 HIGH",     "Source": "API Monitor",       "Message": "Underwriting API P99 latency=218ms — SLO breach (>200ms)",      "Action": "Scale up API pods or optimize query"},
        {"⏰": "3 hr ago",   "Severity": "🟡 MEDIUM",   "Source": "EWS Model",         "Message": "EWS recall dropped to 0.61 (threshold: 0.65)",                  "Action": "Review model performance; retrain if persistent"},
        {"⏰": "12 hr ago",  "Severity": "🟢 INFO",     "Source": "Retraining",        "Message": "PD_Model v2.1 retrained and promoted to production successfully", "Action": "No action required"},
    ]
    for alert in alerts:
        color = "alert-red" if "CRITICAL" in alert["Severity"] else \
                "alert-yellow" if "HIGH" in alert["Severity"] else \
                "alert-yellow" if "MEDIUM" in alert["Severity"] else "alert-green"
        st.markdown(f"""
        <div class="model-card {color}">
          <div style="display:flex;justify-content:space-between;">
            <span style="font-weight:700;">{alert["Severity"]} — {alert["Source"]}</span>
            <span style="color:#94A3B8;font-size:12px;">{alert["⏰"]}</span>
          </div>
          <div style="font-size:13px;margin-top:6px;">{alert["Message"]}</div>
          <div style="font-size:12px;color:#4A5568;margin-top:4px;">→ {alert["Action"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="margin-top:32px;padding:12px;background:white;border-radius:10px;
            border-top:2px solid {PAL["accent"]}40;text-align:center;">
  <span style="color:{PAL["neutral"]};font-size:11px;">
    FinLend AI MLOps Control Center · Module 13 · {datetime.now().strftime("%H:%M:%S")}
  </span>
</div>
""", unsafe_allow_html=True)
