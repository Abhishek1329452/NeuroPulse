"""NeuroPulse demo — clinical decision support prototype only.
Uses synthetic data; it does not diagnose Alzheimer's disease.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = [
    "age", "moca", "mmse", "ptau217", "ab42_40", "hippocampal_volume",
    "wmh_volume", "comorbidity_count", "cognitive_decline", "brain_age_gap", "apoe4",
]
DISPLAY_NAMES = {
    "age": "Age", "moca": "MoCA score", "mmse": "MMSE score", "ptau217": "p-tau217",
    "ab42_40": "Aβ42/40 ratio", "hippocampal_volume": "Hippocampal volume",
    "wmh_volume": "White-matter hyperintensity", "comorbidity_count": "Comorbidities",
    "cognitive_decline": "12-month cognitive decline", "brain_age_gap": "Brain-age gap", "apoe4": "APOE ε4",
}

st.set_page_config(page_title="NeuroPulse", page_icon="🧠", layout="wide")

@st.cache_data
def make_cohort(n=240, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"NP-{i:04d}" for i in range(1, n + 1)],
        "age": rng.normal(70, 7.5, n).clip(55, 90),
        "moca": rng.normal(24, 3.8, n).clip(10, 30),
        "mmse": rng.normal(26, 2.8, n).clip(15, 30),
        "ptau217": rng.normal(1.6, 0.65, n).clip(0.2, 4.5),
        "ab42_40": rng.normal(0.065, 0.013, n).clip(0.025, 0.11),
        "hippocampal_volume": rng.normal(3.15, 0.42, n).clip(1.8, 4.6),
        "wmh_volume": rng.gamma(2.1, 2.0, n).clip(0, 17),
        "comorbidity_count": rng.poisson(1.4, n).clip(0, 5),
        "cognitive_decline": rng.normal(1.0, 0.65, n).clip(0, 4),
        "brain_age_gap": rng.normal(1.5, 5, n).clip(-12, 18),
        "apoe4": rng.binomial(1, 0.27, n),
    })
    # Synthetic proxy outcome for a demonstrator only — never clinical ground truth.
    score = (
        0.06 * (df.age - 70) - 0.45 * (df.moca - 24) - 0.24 * (df.mmse - 26)
        + 1.10 * (df.ptau217 - 1.6) - 35 * (df.ab42_40 - 0.065)
        - 1.25 * (df.hippocampal_volume - 3.15) + 0.12 * df.wmh_volume
        + 0.28 * df.comorbidity_count + 0.65 * df.cognitive_decline
        + 0.09 * df.brain_age_gap + 0.85 * df.apoe4 + rng.normal(0, 0.65, n)
    )
    probability = 1 / (1 + np.exp(-score))
    df["synthetic_progression_label"] = (probability > np.quantile(probability, 0.56)).astype(int)
    return df

@st.cache_resource
def fit_models(cohort):
    X, y = cohort[FEATURES], cohort["synthetic_progression_label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    xgb = XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", random_state=7,
    ).fit(X_train, y_train)
    forest = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=3, random_state=7).fit(X_train, y_train)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=7)).fit(X_train, y_train)
    return xgb, forest, logistic

def infer(cohort, models):
    xgb, forest, logistic = models
    X = cohort[FEATURES]
    risk = 0.50 * xgb.predict_proba(X)[:, 1] + 0.30 * forest.predict_proba(X)[:, 1] + 0.20 * logistic.predict_proba(X)[:, 1]
    urgency = (cohort.cognitive_decline / 4).clip(0, 1)
    benefit = (1 - (cohort.age - 70).abs() / 25).clip(0, 1)
    output = cohort.copy()
    output["risk_score"] = risk
    output["priority_score"] = (0.65 * risk + 0.23 * urgency + 0.12 * benefit).clip(0, 1)
    output["risk_tier"] = pd.cut(risk, [-0.01, 0.40, 0.70, 1.0], labels=["Low", "Medium", "High"])
    output["next_step"] = output.apply(pathway, axis=1)
    return output.sort_values("priority_score", ascending=False)

def pathway(row):
    if row.risk_score >= 0.85 and row.cognitive_decline >= 1.25:
        return "Specialist review + prioritize PET confirmation"
    if row.risk_score >= 0.65:
        return "MRI assessment + targeted blood biomarkers"
    if row.risk_score >= 0.40:
        return "Detailed cognitive assessment + blood work"
    return "Routine monitoring and repeat screening"

def local_explanation(patient, model):
    # Transparent, deterministic feature contributions for the demo UI.
    baseline = {"age": 70, "moca": 24, "mmse": 26, "ptau217": 1.6, "ab42_40": .065,
                "hippocampal_volume": 3.15, "wmh_volume": 4.2, "comorbidity_count": 1.4,
                "cognitive_decline": 1.0, "brain_age_gap": 1.5, "apoe4": .27}
    direction = {"age": 0.06, "moca": -0.45, "mmse": -0.24, "ptau217": 1.1, "ab42_40": -35,
                 "hippocampal_volume": -1.25, "wmh_volume": .12, "comorbidity_count": .28,
                 "cognitive_decline": .65, "brain_age_gap": .09, "apoe4": .85}
    values = []
    for feature in FEATURES:
        contribution = (float(patient[feature]) - baseline[feature]) * direction[feature]
        values.append({"factor": DISPLAY_NAMES[feature], "impact": contribution})
    return pd.DataFrame(values).sort_values("impact", key=lambda s: s.abs(), ascending=False).head(6)

def risk_color(tier):
    return {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[str(tier)]

def main():
    st.markdown("""<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid='stMetric'] {background: #ffffff; border: 1px solid #e7e9ef; border-radius: 12px; padding: 10px;}
    </style>""", unsafe_allow_html=True)
    st.title("🧠 NeuroPulse")
    st.caption("Adaptive prioritization for early Alzheimer’s diagnostic pathways")
    st.warning("**Clinical decision-support demonstrator only.** It uses synthetic data, does not diagnose disease, and requires clinician review for every decision.")

    cohort = make_cohort()
    models = fit_models(cohort)
    patients = infer(cohort, models)

    with st.sidebar:
        st.header("Cohort filters")
        selected_tiers = st.multiselect("Priority tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        age_range = st.slider("Age", 55, 90, (55, 90))
        patient_search = st.text_input("Patient ID")
        st.divider()
        st.caption("Data source: synthetic ADNI/OASIS-like demonstration cohort. No identifiable health information is included.")

    view = patients[(patients.age.between(*age_range)) & (patients.risk_tier.astype(str).isin(selected_tiers))]
    if patient_search:
        view = view[view.patient_id.str.contains(patient_search.upper())]

    a, b, c, d = st.columns(4)
    a.metric("Patients in view", len(view))
    b.metric("High priority", int((view.risk_tier.astype(str) == "High").sum()))
    c.metric("PET review candidates", int((view.risk_score >= .85).sum()))
    d.metric("Mean risk", f"{view.risk_score.mean():.0%}" if len(view) else "—")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Prioritized diagnostic queue")
        table = view[["patient_id", "risk_tier", "risk_score", "priority_score", "next_step", "age", "moca", "ptau217"]].copy()
        table["risk_tier"] = table["risk_tier"].astype(str).map(lambda x: f"{risk_color(x)} {x}")
        table["risk_score"] = table["risk_score"].map("{:.0%}".format)
        table["priority_score"] = table["priority_score"].map("{:.2f}".format)
        st.dataframe(table, use_container_width=True, hide_index=True, height=480)
        selected_id = st.selectbox("Open patient assessment", view.patient_id.tolist() if len(view) else ["No patients"], label_visibility="collapsed")

    with right:
        st.subheader("Risk distribution")
        distribution = view.assign(tier=view.risk_tier.astype(str)).groupby("tier", as_index=False).size()
        order = ["High", "Medium", "Low"]
        distribution["tier"] = pd.Categorical(distribution.tier, order, ordered=True)
        distribution = distribution.sort_values("tier")
        fig = px.bar(distribution, x="tier", y="size", color="tier", color_discrete_map={"High":"#E55353","Medium":"#F3A712","Low":"#2CA58D"}, labels={"size":"Patients", "tier":"Risk tier"})
        fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=25, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Routing logic**")
        st.caption("Risk + decline rate + expected diagnostic benefit determine the queue. High uncertainty should always be escalated to specialist review.")

    if len(view) and selected_id != "No patients":
        patient = patients.loc[patients.patient_id == selected_id].iloc[0]
        st.divider()
        st.subheader(f"Assessment — {selected_id}")
        x, y = st.columns([1, 1])
        with x:
            st.markdown(f"### {risk_color(patient.risk_tier)} {patient.risk_tier} priority")
            m1, m2 = st.columns(2)
            m1.metric("Risk estimate", f"{patient.risk_score:.0%}")
            m2.metric("Priority score", f"{patient.priority_score:.2f}")
            st.info(f"**Suggested next step:** {patient.next_step}")
            explanation = local_explanation(patient, models[0])
            fig = px.bar(explanation.sort_values("impact"), x="impact", y="factor", orientation="h", color="impact", color_continuous_scale="RdBu_r", title="Contributing factors")
            fig.update_layout(coloraxis_showscale=False, height=310, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with y:
            st.markdown("### What-if exploration")
            st.caption("Educational sensitivity analysis; this is not an intervention recommendation.")
            ptau = st.slider("p-tau217", .2, 4.5, float(patient.ptau217), .1)
            ratio = st.slider("Aβ42/40 ratio", .025, .110, float(patient.ab42_40), .005)
            moca = st.slider("MoCA score", 10, 30, int(round(patient.moca)))
            modified = patient.copy()
            modified["ptau217"], modified["ab42_40"], modified["moca"] = ptau, ratio, moca
            simulation = infer(pd.DataFrame([modified[cohort.columns]]), models).iloc[0]
            q1, q2 = st.columns(2)
            q1.metric("Current risk", f"{patient.risk_score:.0%}")
            q2.metric("Simulated risk", f"{simulation.risk_score:.0%}", delta=f"{simulation.risk_score - patient.risk_score:+.0%}")
            st.caption(f"Simulated routing: **{simulation.next_step}**")

    st.divider()
    st.caption("NeuroPulse v1.0 • Prototype for the Precision Care Challenge 2026 • Synthetic data only • Human oversight required")

if __name__ == "__main__":
    main()
