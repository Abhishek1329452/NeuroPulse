"""NeuroPulse — bilingual clinical decision-support prototype.
Uses synthetic data. Not for diagnosis, treatment recommendations, or emergency assessment.
"""
import time
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

NEUROLOGIST_PHONE = "+918349442116"
FEATURES = ["age", "moca", "mmse", "ptau217", "ab42_40", "hippocampal_volume", "wmh_volume", "comorbidity_count", "cognitive_decline", "brain_age_gap", "apoe4"]
DISPLAY = {
    "age": "Age", "moca": "MoCA score", "mmse": "MMSE score", "ptau217": "p-tau217",
    "ab42_40": "Aβ42/40 ratio", "hippocampal_volume": "Hippocampal volume",
    "wmh_volume": "White-matter hyperintensity", "comorbidity_count": "Comorbidity burden",
    "cognitive_decline": "12-month cognitive decline", "brain_age_gap": "Brain-age gap", "apoe4": "APOE ε4",
}

st.set_page_config(page_title="NeuroPulse | Command Center", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def make_cohort(n=240, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"NP-{i:04d}" for i in range(1, n + 1)],
        "age": rng.normal(70, 7.5, n).clip(55, 90),
        "moca": rng.normal(24, 3.8, n).clip(10, 30),
        "mmse": rng.normal(26, 2.8, n).clip(15, 30),
        "ptau217": rng.normal(1.6, .65, n).clip(.2, 4.5),
        "ab42_40": rng.normal(.065, .013, n).clip(.025, .11),
        "hippocampal_volume": rng.normal(3.15, .42, n).clip(1.8, 4.6),
        "wmh_volume": rng.gamma(2.1, 2, n).clip(0, 17),
        "comorbidity_count": rng.poisson(1.4, n).clip(0, 5),
        "cognitive_decline": rng.normal(1, .65, n).clip(0, 4),
        "brain_age_gap": rng.normal(1.5, 5, n).clip(-12, 18),
        "apoe4": rng.binomial(1, .27, n),
    })
    latent = (
        .06 * (df.age - 70) - .45 * (df.moca - 24) - .24 * (df.mmse - 26)
        + 1.1 * (df.ptau217 - 1.6) - 35 * (df.ab42_40 - .065)
        - 1.25 * (df.hippocampal_volume - 3.15) + .12 * df.wmh_volume
        + .28 * df.comorbidity_count + .65 * df.cognitive_decline
        + .09 * df.brain_age_gap + .85 * df.apoe4 + rng.normal(0, .65, n)
    )
    probability = 1 / (1 + np.exp(-latent))
    df["synthetic_progression_label"] = (probability > np.quantile(probability, .56)).astype(int)
    return df

@st.cache_resource
def fit_models(cohort):
    X_train, _, y_train, _ = train_test_split(
        cohort[FEATURES], cohort["synthetic_progression_label"],
        test_size=.2, random_state=7, stratify=cohort["synthetic_progression_label"],
    )
    xgb = XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=.08, subsample=.9,
        colsample_bytree=.9, eval_metric="logloss", random_state=7,
    ).fit(X_train, y_train)
    forest = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=3, random_state=7).fit(X_train, y_train)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=7)).fit(X_train, y_train)
    return xgb, forest, logistic

def route(row):
    if row.risk_score >= .85 and row.cognitive_decline >= 1.25:
        return "Specialist review + prioritize PET confirmation", "CRITICAL"
    if row.risk_score >= .65:
        return "MRI assessment + targeted blood biomarkers", "HIGH"
    if row.risk_score >= .40:
        return "Detailed cognitive assessment + blood work", "MODERATE"
    return "Routine monitoring and repeat screening", "MONITOR"

def infer(cohort, models):
    xgb, forest, logistic = models
    X = cohort[FEATURES]
    risk = .5 * xgb.predict_proba(X)[:, 1] + .3 * forest.predict_proba(X)[:, 1] + .2 * logistic.predict_proba(X)[:, 1]
    urgency = (cohort.cognitive_decline / 4).clip(0, 1)
    benefit = (1 - (cohort.age - 70).abs() / 25).clip(0, 1)
    out = cohort.copy()
    out["risk_score"] = risk
    out["priority_score"] = (.65 * risk + .23 * urgency + .12 * benefit).clip(0, 1)
    out["risk_tier"] = pd.cut(risk, [-.01, .40, .70, 1.0], labels=["Low", "Medium", "High"])
    routing = out.apply(route, axis=1)
    out["next_step"] = [item[0] for item in routing]
    out["routing_status"] = [item[1] for item in routing]
    return out.sort_values("priority_score", ascending=False)

def explanation(patient):
    baseline = {"age":70, "moca":24, "mmse":26, "ptau217":1.6, "ab42_40":.065, "hippocampal_volume":3.15, "wmh_volume":4.2, "comorbidity_count":1.4, "cognitive_decline":1, "brain_age_gap":1.5, "apoe4":.27}
    direction = {"age":.06, "moca":-.45, "mmse":-.24, "ptau217":1.1, "ab42_40":-35, "hippocampal_volume":-1.25, "wmh_volume":.12, "comorbidity_count":.28, "cognitive_decline":.65, "brain_age_gap":.09, "apoe4":.85}
    return pd.DataFrame([
        {"factor": DISPLAY[feature], "impact": (float(patient[feature]) - baseline[feature]) * direction[feature]}
        for feature in FEATURES
    ]).sort_values("impact", key=lambda col: col.abs(), ascending=False).head(6)

def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#eaf7ff; --muted:#91a8bf; --panel:rgba(9,24,48,.76); --line:rgba(109,224,255,.18); --cyan:#57e7ff; --green:#51e6ab; }
    .stApp { background:radial-gradient(circle at 15% 10%,#12335c 0,transparent 27%),radial-gradient(circle at 85% 25%,#1b2758 0,transparent 29%),radial-gradient(circle at 55% 100%,#062e45 0,#030915 55%,#020611 100%); color:var(--ink); font-family:'Manrope',sans-serif; }
    .stApp::before { content:''; position:fixed; inset:0; pointer-events:none; opacity:.28; background-image:radial-gradient(#bdefff 1px,transparent 1px),radial-gradient(#bdefff 1px,transparent 1px); background-position:0 0,80px 70px; background-size:120px 120px,180px 180px; }
    [data-testid='stSidebar'] { background:linear-gradient(180deg,rgba(6,19,40,.96),rgba(4,10,24,.96)); border-right:1px solid var(--line); } [data-testid='stSidebar'] * { color:var(--ink); }
    .block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:2.5rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em!important; color:#effaff!important; } h1 { font-weight:800!important; }
    .hero { padding:1.7rem 1.8rem; border:1px solid var(--line); border-radius:22px; background:linear-gradient(115deg,rgba(18,45,82,.84),rgba(8,23,49,.75)); box-shadow:0 20px 55px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.06); margin-bottom:1rem; }
    .eyebrow { color:var(--cyan); font-family:'DM Mono',monospace; font-size:.73rem; letter-spacing:.16em; font-weight:500; }.subtitle { color:var(--muted); margin-top:.3rem; font-size:.98rem; }.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 14px var(--green); margin-right:7px; }
    div[data-testid='stMetric'] { background:linear-gradient(145deg,rgba(15,36,67,.85),rgba(7,19,42,.8)); border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; box-shadow:inset 0 1px 0 rgba(255,255,255,.05); } div[data-testid='stMetricLabel'] { color:var(--muted); } div[data-testid='stMetricValue'] { color:#f3fcff; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:17px; padding:1.05rem 1.1rem; box-shadow:0 15px 35px rgba(0,0,0,.18); }.route { border-radius:14px; padding:.75rem .9rem; margin:.55rem 0; background:rgba(12,34,63,.74); border:1px solid rgba(109,224,255,.13); }.route-active { border-color:rgba(87,231,255,.72); box-shadow:0 0 23px rgba(87,231,255,.12); background:linear-gradient(100deg,rgba(15,68,100,.8),rgba(10,34,67,.75)); }.route-title { font-family:'DM Mono',monospace; font-size:.75rem; color:var(--cyan); letter-spacing:.08em; }.route-copy { color:#d5eafa; font-size:.86rem; padding-top:.15rem; }
    .badge { display:inline-block; padding:.25rem .55rem; border-radius:30px; font-family:'DM Mono',monospace; font-size:.67rem; letter-spacing:.08em; border:1px solid; }.badge-critical { color:#ff9bad; border-color:rgba(255,92,124,.5); background:rgba(255,92,124,.12); }.badge-high { color:#ffca73; border-color:rgba(255,191,90,.5); background:rgba(255,191,90,.1); }.badge-moderate { color:#9bd9ff; border-color:rgba(80,140,255,.5); background:rgba(80,140,255,.1); }.badge-monitor { color:#89f4c4; border-color:rgba(81,230,171,.5); background:rgba(81,230,171,.1); }
    .stAlert { border-radius:14px!important; background:rgba(255,191,90,.1)!important; border:1px solid rgba(255,191,90,.28)!important; } [data-testid='stDataFrame'] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }.stButton>button { background:linear-gradient(135deg,#1691bd,#405de6); color:white; border:0; border-radius:10px; font-weight:700; }
    .np-ai-shell { background:linear-gradient(145deg,rgba(228,244,255,.98),rgba(242,248,255,.98)); border:1px solid #d3e1f2; border-radius:22px; padding:1.15rem; box-shadow:0 14px 35px rgba(3,21,47,.15); color:#163665; }.np-ai-head { display:flex; align-items:flex-start; gap:.9rem; }.np-ai-avatar { height:48px; width:48px; min-width:48px; display:flex; align-items:center; justify-content:center; border-radius:50%; background:linear-gradient(145deg,#173c78,#0d2756); box-shadow:0 5px 14px rgba(9,39,86,.25); font-size:1.45rem; }.np-ai-title { color:#173665; font-size:1.13rem; font-weight:800; line-height:1.35; }.np-ai-copy { color:#395987; line-height:1.45; margin-top:.28rem; }.np-ai-privacy { margin:.85rem 0 .35rem; border:1px solid #ffcf59; color:#9a5300; background:#fff9e7; padding:.68rem .8rem; border-radius:11px; font-size:.86rem; }.np-ai-call { display:block; text-align:center; text-decoration:none!important; color:white!important; background:linear-gradient(135deg,#175ccc,#27449a); padding:.65rem; border-radius:10px; font-weight:800; margin:.55rem 0; }.np-ai-caption { color:#5e7799; font-size:.78rem; text-align:center; }
    div[data-testid='stChatInput'] { background:#fff; border:1px solid #cbdced; border-radius:15px; box-shadow:0 7px 20px rgba(10,41,81,.10); } div[data-testid='stChatInput'] textarea { color:#173665!important; }
    </style>""", unsafe_allow_html=True)

def badge(status):
    class_name = {"CRITICAL":"critical", "HIGH":"high", "MODERATE":"moderate", "MONITOR":"monitor"}[status]
    return f"<span class='badge badge-{class_name}'>{status}</span>"

def pathway_panel(status):
    stages = [
        ("01", "COGNITIVE SCREENING", "MoCA / MMSE and baseline clinical data"),
        ("02", "BLOOD BIOMARKERS", "Targeted p-tau217 and Aβ42/40 assessment"),
        ("03", "STRUCTURAL MRI", "AI-assisted neuroimaging evaluation"),
        ("04", "PET PRIORITIZATION", "Specialist-guided advanced confirmation"),
    ]
    active = {"MONITOR":0, "MODERATE":1, "HIGH":2, "CRITICAL":3}[status]
    html = "<div class='panel'><div class='eyebrow'>DIAGNOSTIC FLIGHT PATH</div>"
    for index, (number, title, copy) in enumerate(stages):
        css = "route route-active" if index == active else "route"
        html += f"<div class='{css}'><div class='route-title'>{number} · {title}</div><div class='route-copy'>{copy}</div></div>"
    return html + "</div>"

def assistant_reply(message, language, patient=None):
    text = message.lower().strip()
    hindi = language == "हिन्दी"
    emergency_words = ["emergency", "stroke", "unconscious", "chest pain", "seizure", "suicide", "बेहोश", "दौरा", "सीने में दर्द", "स्ट्रोक"]
    if any(word in text for word in emergency_words):
        return "⚠️ **Urgent safety note:** This app cannot assess an emergency. For sudden or life-threatening symptoms, contact local emergency services immediately. The neurologist line is for non-emergency clinical coordination." if not hindi else "⚠️ **तत्काल सुरक्षा सूचना:** यह ऐप आपातकाल का आकलन नहीं कर सकता। अचानक या जानलेवा लक्षण होने पर तुरंत स्थानीय आपातकालीन सेवाओं से संपर्क करें। न्यूरोलॉजिस्ट लाइन गैर-आपातकालीन क्लिनिकल समन्वय के लिए है।"
    if patient is not None and any(word in text for word in ["risk", "patient", "route", "recommend", "जोखिम", "मरीज", "मार्ग", "सिफारिश"]):
        return f"For **{patient.patient_id}**, NeuroPulse shows an estimated risk of **{patient.risk_score:.0%}** and the suggested next step is **{patient.next_step}**. This is a decision-support summary only; a neurologist must review the full clinical context." if not hindi else f"**{patient.patient_id}** के लिए NeuroPulse **{patient.risk_score:.0%}** अनुमानित जोखिम दिखाता है। सुझाया गया अगला चरण है: **{patient.next_step}**। यह केवल निर्णय-सहायता सारांश है; न्यूरोलॉजिस्ट को पूरा क्लिनिकल संदर्भ देखना चाहिए।"
    if any(word in text for word in ["call", "neurologist", "doctor", "कॉल", "न्यूरोलॉजिस्ट", "डॉक्टर"]):
        return "Use the **Call neurologist** button to open your device dialer. Share the dashboard summary with the clinician; do not use this prototype as a diagnosis." if not hindi else "डिवाइस डायलर खोलने के लिए **न्यूरोलॉजिस्ट को कॉल करें** बटन दबाएं। क्लिनिशियन के साथ डैशबोर्ड सारांश साझा करें; इस प्रोटोटाइप को निदान न मानें।"
    return "I can explain the dashboard, risk tiers, contributing factors, and suggested diagnostic pathway. I cannot diagnose Alzheimer’s disease, recommend treatment, or replace a neurologist." if not hindi else "मैं डैशबोर्ड, जोखिम स्तर, योगदान देने वाले कारक और सुझाए गए डायग्नोस्टिक पाथवे को समझा सकता/सकती हूँ। मैं अल्ज़ाइमर का निदान, उपचार की सलाह या न्यूरोलॉजिस्ट का विकल्प नहीं हूँ।"

def render_ai_assistant(patient=None):
    if "np_chat" not in st.session_state:
        st.session_state.np_chat = []
    st.divider()
    st.markdown("### Care navigator")
    language_col, call_col = st.columns([3, 1])
    with language_col:
        language = st.radio("Language / भाषा", ["English", "हिन्दी"], horizontal=True, key="np_language")
    with call_col:
        st.markdown(f"<a class='np-ai-call' href='tel:{NEUROLOGIST_PHONE}'>☎ Call neurologist</a>", unsafe_allow_html=True)
    st.markdown("""<div class='np-ai-shell'><div class='np-ai-head'><div class='np-ai-avatar'>🤖</div><div><div class='np-ai-title'>Namaste! I’m NeuroPulse Care Navigator.</div><div class='np-ai-copy'>Ask about this patient’s priority tier, contributing factors, or suggested diagnostic pathway. I’ll help explain the dashboard clearly.</div></div></div><div class='np-ai-privacy'>🔒 <b>Privacy:</b> Do not enter names, medical-record numbers, OTPs, passwords, or other identifiable health information. This prototype does not need it.</div></div>""", unsafe_allow_html=True)
    suggestions = ["Explain this patient’s risk", "Why was this patient prioritized?", "What is the next diagnostic step?", "When should I call the neurologist?"] if language == "English" else ["इस मरीज का जोखिम समझाएं", "इस मरीज को प्राथमिकता क्यों दी गई?", "अगला डायग्नोस्टिक चरण क्या है?", "न्यूरोलॉजिस्ट को कब कॉल करें?"]
    st.caption("Try asking / पूछें:")
    columns = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        if columns[index % 2].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.np_pending_message = suggestion
    for entry in st.session_state.np_chat:
        with st.chat_message(entry["role"], avatar="🤖" if entry["role"] == "assistant" else "🧑‍⚕️"):
            st.markdown(entry["content"])
    prompt = st.chat_input("Ask the Care Navigator… / Care Navigator से पूछें…") or st.session_state.pop("np_pending_message", None)
    if prompt:
        st.session_state.np_chat.append({"role":"user", "content":prompt})
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("NeuroPulse is preparing a safe explanation…"):
                time.sleep(.4)
                answer = assistant_reply(prompt, language, patient)
                st.markdown(answer)
        st.session_state.np_chat.append({"role":"assistant", "content":answer})
    st.markdown("<div class='np-ai-caption'>Clinical decision support only · Not a diagnosis or treatment recommendation · Human review required</div>", unsafe_allow_html=True)

def main():
    inject_css()
    cohort = make_cohort()
    models = fit_models(cohort)
    patients = infer(cohort, models)
    st.markdown("""<div class='hero'><div class='eyebrow'><span class='status-dot'></span>NEUROPULSE // CLINICAL INTELLIGENCE COMMAND CENTER</div><h1 style='margin:.35rem 0 0'>Early diagnostic prioritization, made visible.</h1><div class='subtitle'>A transparent workflow for routing synthetic patient cohorts through cognitive screening, biomarker assessment, MRI, and specialist-led PET prioritization.</div></div>""", unsafe_allow_html=True)
    st.warning("Clinical decision-support prototype only. This application uses synthetic data and must not be used to diagnose Alzheimer’s disease or make treatment decisions.")
    with st.sidebar:
        st.markdown("### 🧠 NeuroPulse"); st.caption("PATIENT FLOW CONTROL"); st.divider()
        tiers = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        ages = st.slider("Age range", 55, 90, (55, 90))
        query = st.text_input("Find patient", placeholder="NP-0001")
        st.divider(); st.markdown("**System health**"); st.success("Inference engine online"); st.caption("Synthetic cohort · 3-model ensemble · clinician review required")
    view = patients[patients.age.between(*ages) & patients.risk_tier.astype(str).isin(tiers)]
    if query:
        view = view[view.patient_id.str.contains(query.upper())]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cohort in view", len(view)); m2.metric("High risk", int((view.risk_tier.astype(str) == "High").sum())); m3.metric("PET review queue", int((view.routing_status == "CRITICAL").sum())); m4.metric("Mean risk", f"{view.risk_score.mean():.0%}" if len(view) else "—")
    left, right = st.columns([1.42, .92], gap="large")
    with left:
        st.markdown("### Priority queue"); st.caption("Sorted by composite priority: estimated risk, decline urgency, and diagnostic benefit.")
        queue = view[["patient_id", "routing_status", "risk_tier", "risk_score", "priority_score", "next_step", "age", "moca"]].copy()
        queue.risk_score = queue.risk_score.map("{:.0%}".format); queue.priority_score = queue.priority_score.map("{:.2f}".format)
        queue.columns = ["Patient", "Route status", "Risk tier", "Risk", "Priority", "Suggested next step", "Age", "MoCA"]
        st.dataframe(queue, use_container_width=True, hide_index=True, height=420)
        selected = st.selectbox("Open clinical assessment", view.patient_id.tolist() if len(view) else ["No patient found"], label_visibility="collapsed")
    with right:
        st.markdown("### Population risk orbit")
        if len(view):
            orbit = px.scatter_3d(view, x="moca", y="ptau217", z="hippocampal_volume", color="risk_tier", size="priority_score", hover_name="patient_id", color_discrete_map={"High":"#ff5c7c", "Medium":"#ffbf5a", "Low":"#51e6ab"}, labels={"moca":"MoCA", "ptau217":"p-tau217", "hippocampal_volume":"Hippocampal volume"})
            orbit.update_traces(marker=dict(opacity=.82, line=dict(width=.5, color="#d8f8ff")))
            orbit.update_layout(height=365, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", scene=dict(bgcolor="rgba(0,0,0,0)", xaxis=dict(backgroundcolor="rgba(7,25,52,.35)",gridcolor="rgba(140,226,255,.16)"), yaxis=dict(backgroundcolor="rgba(7,25,52,.35)",gridcolor="rgba(140,226,255,.16)"), zaxis=dict(backgroundcolor="rgba(7,25,52,.35)",gridcolor="rgba(140,226,255,.16)")), legend=dict(bgcolor="rgba(0,0,0,0)"), font=dict(color="#dff5ff"))
            st.plotly_chart(orbit, use_container_width=True)
        st.caption("Each point represents a synthetic patient profile across cognitive, biomarker, and imaging-proxy dimensions.")
    patient = None
    if len(view) and selected != "No patient found":
        patient = patients[patients.patient_id == selected].iloc[0]
        st.divider(); st.markdown(f"### Patient assessment · {patient.patient_id} &nbsp; {badge(patient.routing_status)}", unsafe_allow_html=True)
        a, b, c = st.columns([1.04, 1.04, .92], gap="large")
        with a:
            st.markdown("<div class='panel'>", unsafe_allow_html=True); x, y = st.columns(2); x.metric("Risk estimate", f"{patient.risk_score:.0%}"); y.metric("Priority score", f"{patient.priority_score:.2f}"); st.markdown(f"<div style='color:#91a8bf;font-size:.8rem;margin-top:.7rem'>SUGGESTED NEXT STEP</div><div style='font-weight:700;color:#effaff;margin-top:.25rem'>{patient.next_step}</div></div>", unsafe_allow_html=True)
            factors = explanation(patient).sort_values("impact")
            chart = px.bar(factors, x="impact", y="factor", orientation="h", color="impact", color_continuous_scale=["#4375e8", "#19355e", "#ff7089"], title="Why this patient is prioritized")
            chart.update_layout(height=315, margin=dict(l=0,r=0,t=48,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#dff5ff"), coloraxis_showscale=False, xaxis=dict(gridcolor="rgba(140,226,255,.12)"), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(chart, use_container_width=True)
        with b:
            st.markdown(pathway_panel(patient.routing_status), unsafe_allow_html=True)
        with c:
            st.markdown("<div class='panel'><div class='eyebrow'>WHAT-IF EXPLORATION</div><p style='color:#91a8bf;font-size:.82rem'>Educational sensitivity analysis — not a treatment recommendation.</p></div>", unsafe_allow_html=True)
            p_tau = st.slider("p-tau217", .2, 4.5, float(patient.ptau217), .1); ab_ratio = st.slider("Aβ42/40 ratio", .025, .110, float(patient.ab42_40), .005); moca = st.slider("MoCA score", 10, 30, int(round(patient.moca)))
            modified = patient.copy(); modified["ptau217"], modified["ab42_40"], modified["moca"] = p_tau, ab_ratio, moca
            simulation = infer(pd.DataFrame([modified[cohort.columns]]), models).iloc[0]
            s1, s2 = st.columns(2); s1.metric("Current", f"{patient.risk_score:.0%}"); s2.metric("Simulated", f"{simulation.risk_score:.0%}", delta=f"{simulation.risk_score-patient.risk_score:+.0%}")
            st.caption(f"Simulated route: {simulation.next_step}")
    render_ai_assistant(patient)
    st.markdown("<div style='text-align:center;color:#6f8aa5;padding:2rem 0 .3rem;font-family:DM Mono,monospace;font-size:.72rem;letter-spacing:.08em'>NEUROPULSE V1.3 · SYNTHETIC DATA ONLY · HUMAN OVERSIGHT REQUIRED</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
