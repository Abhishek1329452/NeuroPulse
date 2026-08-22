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

# NeuroPulse prototype: synthetic data only. Not for diagnosis or treatment.
PHONE = "+918349442116"
F = ["age","moca","mmse","ptau217","ab42_40","hippocampal_volume","wmh_volume","comorbidity_count","cognitive_decline","brain_age_gap","apoe4"]
N = {"age":"Age","moca":"MoCA score","mmse":"MMSE score","ptau217":"p-tau217","ab42_40":"Aβ42/40 ratio","hippocampal_volume":"Hippocampal volume","wmh_volume":"White-matter hyperintensity","comorbidity_count":"Comorbidity burden","cognitive_decline":"12-month cognitive decline","brain_age_gap":"Brain-age gap","apoe4":"APOE ε4"}
st.set_page_config(page_title="NeuroPulse", page_icon="🧠", layout="wide")

@st.cache_data
def cohort(n=240):
    r=np.random.default_rng(42)
    d=pd.DataFrame({"patient_id":[f"NP-{i:04d}" for i in range(1,n+1)],"age":r.normal(70,7.5,n).clip(55,90),"moca":r.normal(24,3.8,n).clip(10,30),"mmse":r.normal(26,2.8,n).clip(15,30),"ptau217":r.normal(1.6,.65,n).clip(.2,4.5),"ab42_40":r.normal(.065,.013,n).clip(.025,.11),"hippocampal_volume":r.normal(3.15,.42,n).clip(1.8,4.6),"wmh_volume":r.gamma(2.1,2,n).clip(0,17),"comorbidity_count":r.poisson(1.4,n).clip(0,5),"cognitive_decline":r.normal(1,.65,n).clip(0,4),"brain_age_gap":r.normal(1.5,5,n).clip(-12,18),"apoe4":r.binomial(1,.27,n)})
    z=.06*(d.age-70)-.45*(d.moca-24)-.24*(d.mmse-26)+1.1*(d.ptau217-1.6)-35*(d.ab42_40-.065)-1.25*(d.hippocampal_volume-3.15)+.12*d.wmh_volume+.28*d.comorbidity_count+.65*d.cognitive_decline+.09*d.brain_age_gap+.85*d.apoe4+r.normal(0,.65,n)
    d["target"]=(1/(1+np.exp(-z))>np.quantile(1/(1+np.exp(-z)),.56)).astype(int)
    return d

@st.cache_resource
def models(d):
    x,y=train_test_split(d[F],d.target,test_size=.2,random_state=7,stratify=d.target)[:2]
    # train_test_split above returns X_train, X_test; recover labels using index
    y_train=d.loc[x.index,"target"]
    return (XGBClassifier(n_estimators=80,max_depth=3,learning_rate=.08,eval_metric="logloss",random_state=7).fit(x,y_train),RandomForestClassifier(n_estimators=180,max_depth=7,min_samples_leaf=3,random_state=7).fit(x,y_train),make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(x,y_train))

def score(d,m):
    xgb,rf,lr=m; p=.5*xgb.predict_proba(d[F])[:,1]+.3*rf.predict_proba(d[F])[:,1]+.2*lr.predict_proba(d[F])[:,1]
    o=d.copy();o["risk_score"]=p;o["priority_score"]=(.65*p+.23*(o.cognitive_decline/4).clip(0,1)+.12*(1-(o.age-70).abs()/25).clip(0,1)).clip(0,1);o["risk_tier"]=pd.cut(p,[-.01,.4,.7,1],labels=["Low","Medium","High"])
    def route(r):
        if r.risk_score>=.85 and r.cognitive_decline>=1.25:return "Specialist review + prioritize PET confirmation","CRITICAL"
        if r.risk_score>=.65:return "MRI assessment + targeted blood biomarkers","HIGH"
        if r.risk_score>=.4:return "Detailed cognitive assessment + blood work","MODERATE"
        return "Routine monitoring and repeat screening","MONITOR"
    q=o.apply(route,axis=1);o["next_step"]=[v[0] for v in q];o["status"]=[v[1] for v in q]
    return o.sort_values("priority_score",ascending=False)

def factors(p):
    b={"age":70,"moca":24,"mmse":26,"ptau217":1.6,"ab42_40":.065,"hippocampal_volume":3.15,"wmh_volume":4.2,"comorbidity_count":1.4,"cognitive_decline":1,"brain_age_gap":1.5,"apoe4":.27};w={"age":.06,"moca":-.45,"mmse":-.24,"ptau217":1.1,"ab42_40":-35,"hippocampal_volume":-1.25,"wmh_volume":.12,"comorbidity_count":.28,"cognitive_decline":.65,"brain_age_gap":.09,"apoe4":.85}
    return pd.DataFrame([{"Factor":N[k],"Impact":(p[k]-b[k])*w[k]} for k in F]).sort_values("Impact",key=lambda x:x.abs(),ascending=False).head(6)

def css():
 st.markdown("""<style>
 @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
 .stApp{background:#eef5fb!important;color:#152b4b!important;font-family:Manrope,sans-serif}.stApp::before{display:none}[data-testid='stSidebar']{background:linear-gradient(180deg,#12345e,#0a2545)!important;border-right:1px solid #234a76!important}[data-testid='stSidebar'] *{color:#f6fbff!important}.block-container{max-width:1450px;padding-top:1.4rem}h1,h2,h3{color:#122d52!important}.hero{padding:1.6rem 1.8rem;border-radius:22px;background:linear-gradient(135deg,#123d73,#1e5e97 58%,#207da2);box-shadow:0 14px 30px rgba(26,67,110,.20);margin-bottom:1rem}.hero h1{color:#fff!important;margin:.3rem 0}.eyebrow{color:#9defff;letter-spacing:.12em;font-size:.72rem;font-weight:800}.subtitle{color:#dceeff}.panel,div[data-testid='stMetric']{background:#fff!important;border:1px solid #d5e3f2!important;border-radius:16px;padding:1rem;box-shadow:0 5px 15px rgba(22,58,100,.08)}div[data-testid='stMetricLabel']{color:#53708f!important}div[data-testid='stMetricValue']{color:#15365f!important}.route{background:#f6faff;border:1px solid #d8e8f7;border-radius:13px;padding:.75rem;margin:.55rem 0}.active{background:#e9f7ff;border-color:#4aaee6;box-shadow:0 0 0 2px rgba(74,174,230,.12)}.route b{color:#14639a}.route span{color:#365979;font-size:.86rem}.badge{padding:.2rem .55rem;border-radius:18px;font-size:.68rem;font-weight:800}.CRITICAL{background:#fff0f3;color:#b92748}.HIGH{background:#fff7e9;color:#9a5d00}.MODERATE{background:#edf5ff;color:#17649c}.MONITOR{background:#ecfbf4;color:#187a58}.stAlert{background:#fff8df!important;border-color:#f3cf72!important;color:#654500!important}[data-testid='stDataFrame']{border:1px solid #d5e3f2;border-radius:14px}.ai{background:#f5f9fe;border:1px solid #d2e2f2;border-radius:22px;padding:1.15rem;color:#17365f}.aihead{display:flex;gap:.9rem}.avatar{width:48px;height:48px;border-radius:50%;background:#173c78;display:flex;align-items:center;justify-content:center;font-size:1.4rem}.aititle{font-size:1.15rem;font-weight:800;color:#15375f}.aicopy{color:#3c5f86;margin-top:.25rem}.privacy{margin-top:.8rem;background:#fff9e9;border:1px solid #ffcf59;padding:.65rem;border-radius:10px;color:#8a4c00}.call{display:block;text-align:center;text-decoration:none!important;background:linear-gradient(135deg,#126fc2,#1f4f9e);color:white!important;padding:.65rem;border-radius:10px;font-weight:800}.np-ai-caption{color:#53708f;text-align:center;font-size:.78rem}[data-testid='stChatMessage']{background:#fff!important;border:1px solid #d5e3f2!important;border-radius:14px!important;padding:.65rem .8rem!important;margin:.55rem 0!important;box-shadow:0 3px 10px rgba(22,58,100,.06)}[data-testid='stChatMessage'] *{color:#18395f!important}[data-testid='stChatInput']{background:#fff!important;border:1px solid #b9d1e8!important;border-radius:15px!important}[data-testid='stChatInput'] textarea,[data-testid='stChatInput'] textarea::placeholder{color:#33587e!important;opacity:1!important}.stButton>button{background:#edf5ff;color:#155c97;border:1px solid #bad8ef;border-radius:18px;font-weight:700}
 </style>""",unsafe_allow_html=True)

def answer(msg,lang,p=None):
 t=msg.lower();hi=lang=="हिन्दी"
 if any(x in t for x in ["emergency","stroke","unconscious","chest pain","seizure","बेहोश","दौरा","स्ट्रोक"]):return "⚠️ This app cannot assess emergencies. For sudden or life-threatening symptoms, contact local emergency services immediately." if not hi else "⚠️ यह ऐप आपातकाल का आकलन नहीं कर सकता। अचानक या जानलेवा लक्षणों के लिए तुरंत स्थानीय आपातकालीन सेवाओं से संपर्क करें।"
 if p is not None and any(x in t for x in ["risk","patient","route","recommend","जोखिम","मरीज","मार्ग"]):return f"For **{p.patient_id}**, estimated risk is **{p.risk_score:.0%}**. Suggested next step: **{p.next_step}**. A neurologist must review the full clinical context." if not hi else f"**{p.patient_id}** के लिए अनुमानित जोखिम **{p.risk_score:.0%}** है। सुझाया गया अगला चरण: **{p.next_step}**। न्यूरोलॉजिस्ट को पूरा क्लिनिकल संदर्भ देखना चाहिए।"
 return "I can explain the dashboard, risk tier, factors, and suggested pathway. I cannot diagnose disease or recommend treatment." if not hi else "मैं डैशबोर्ड, जोखिम स्तर, कारक और सुझाए गए पाथवे को समझा सकता/सकती हूँ। मैं निदान या उपचार की सलाह नहीं दे सकता/सकती।"

def chatbot(p):
 if "chat" not in st.session_state:st.session_state.chat=[]
 st.divider();st.markdown("### Care navigator")
 a,b=st.columns([3,1]);lang=a.radio("Language / भाषा",["English","हिन्दी"],horizontal=True);b.markdown(f"<a class='call' href='tel:{PHONE}'>☎ Call neurologist</a>",unsafe_allow_html=True)
 st.markdown("""<div class='ai'><div class='aihead'><div class='avatar'>🤖</div><div><div class='aititle'>Namaste! I’m NeuroPulse Care Navigator.</div><div class='aicopy'>Ask about a patient’s risk, contributing factors, or diagnostic pathway.</div></div></div><div class='privacy'>🔒 <b>Privacy:</b> Do not enter names, medical-record numbers, OTPs, passwords, or identifiable health information.</div></div>""",unsafe_allow_html=True)
 qs=["Explain this patient’s risk","Why was this patient prioritized?","What is the next diagnostic step?","When should I call the neurologist?"] if lang=="English" else ["इस मरीज का जोखिम समझाएं","इस मरीज को प्राथमिकता क्यों दी गई?","अगला डायग्नोस्टिक चरण क्या है?","न्यूरोलॉजिस्ट को कब कॉल करें?"]
 st.caption("Try asking / पूछें:");c=st.columns(2)
 for i,q in enumerate(qs):
  if c[i%2].button(q,key=f"q{i}",use_container_width=True):st.session_state.pending=q
 for z in st.session_state.chat:
  with st.chat_message(z["role"],avatar="🤖" if z["role"]=="assistant" else "🧑‍⚕️"):st.markdown(z["content"])
 prompt=st.chat_input("Ask the Care Navigator… / Care Navigator से पूछें…") or st.session_state.pop("pending",None)
 if prompt:
  st.session_state.chat.append({"role":"user","content":prompt})
  with st.chat_message("user",avatar="🧑‍⚕️"):st.markdown(prompt)
  with st.chat_message("assistant",avatar="🤖"):
   with st.spinner("Preparing a safe explanation…"):
    time.sleep(.35);out=answer(prompt,lang,p);st.markdown(out)
  st.session_state.chat.append({"role":"assistant","content":out})
 st.markdown("<div class='np-ai-caption'>Clinical decision support only · Not a diagnosis or treatment recommendation · Human review required</div>",unsafe_allow_html=True)

def main():
 css();d=score(cohort(),models(cohort()))
 st.markdown("""<div class='hero'><div class='eyebrow'>● NEUROPULSE · CLINICAL INTELLIGENCE COMMAND CENTER</div><h1>Early diagnostic prioritization, made visible.</h1><div class='subtitle'>Transparent routing through cognitive screening, biomarkers, MRI, and specialist-led PET prioritization.</div></div>""",unsafe_allow_html=True)
 st.warning("Clinical decision-support prototype only. This app uses synthetic data and does not diagnose Alzheimer’s disease or make treatment decisions.")
 with st.sidebar:
  st.markdown("### 🧠 NeuroPulse");st.caption("PATIENT FLOW CONTROL");st.divider();tiers=st.multiselect("Risk tier",["High","Medium","Low"],default=["High","Medium","Low"]);ages=st.slider("Age range",55,90,(55,90));q=st.text_input("Find patient",placeholder="NP-0001")
 v=d[d.age.between(*ages)&d.risk_tier.astype(str).isin(tiers)]
 if q:v=v[v.patient_id.str.contains(q.upper())]
 x1,x2,x3,x4=st.columns(4);x1.metric("Cohort in view",len(v));x2.metric("High risk",int((v.risk_tier.astype(str)=="High").sum()));x3.metric("PET review queue",int((v.status=="CRITICAL").sum()));x4.metric("Mean risk",f"{v.risk_score.mean():.0%}" if len(v) else "—")
 l,r=st.columns([1.4,1]);
 with l:
  st.markdown("### Priority queue");tab=v[["patient_id","status","risk_tier","risk_score","priority_score","next_step","age","moca"]].copy();tab.risk_score=tab.risk_score.map("{:.0%}".format);tab.priority_score=tab.priority_score.map("{:.2f}".format);st.dataframe(tab,use_container_width=True,hide_index=True,height=400);sel=st.selectbox("Open clinical assessment",v.patient_id.tolist() if len(v) else ["No patient"])
 with r:
  st.markdown("### Population risk orbit")
  if len(v):
   fig=px.scatter_3d(v,x="moca",y="ptau217",z="hippocampal_volume",color="risk_tier",size="priority_score",hover_name="patient_id",color_discrete_map={"High":"#d63b57","Medium":"#d89213","Low":"#218964"});fig.update_layout(height=390,paper_bgcolor="#ffffff",font=dict(color="#17365f"),scene=dict(bgcolor="#ffffff"));st.plotly_chart(fig,use_container_width=True)
 p=None
 if len(v) and sel!="No patient":
  p=d[d.patient_id==sel].iloc[0];st.divider();st.markdown(f"### Patient assessment · {p.patient_id}")
  a,b,c=st.columns([1,1,1])
  with a:
   st.markdown("<div class='panel'>",unsafe_allow_html=True);u,w=st.columns(2);u.metric("Risk estimate",f"{p.risk_score:.0%}");w.metric("Priority",f"{p.priority_score:.2f}");st.markdown(f"**Suggested next step:**  \\n{p.next_step}</div>",unsafe_allow_html=True);fc=factors(p).sort_values("Impact");fig=px.bar(fc,x="Impact",y="Factor",orientation="h",color="Impact",color_continuous_scale="RdBu_r");fig.update_layout(height=300,paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",font=dict(color="#17365f"),coloraxis_showscale=False);st.plotly_chart(fig,use_container_width=True)
  with b:
   st.markdown("<div class='panel'><div class='eyebrow'>DIAGNOSTIC FLIGHT PATH</div>",unsafe_allow_html=True)
   stages=[("COGNITIVE SCREENING","MoCA / MMSE and baseline data"),("BLOOD BIOMARKERS","Targeted p-tau217 and Aβ42/40"),("STRUCTURAL MRI","AI-assisted imaging evaluation"),("PET PRIORITIZATION","Specialist-guided confirmation")];active={"MONITOR":0,"MODERATE":1,"HIGH":2,"CRITICAL":3}[p.status]
   for i,(h,t) in enumerate(stages):st.markdown(f"<div class='route {'active' if i==active else ''}'><b>{i+1:02d} · {h}</b><br><span>{t}</span></div>",unsafe_allow_html=True)
   st.markdown("</div>",unsafe_allow_html=True)
  with c:
   st.markdown("### What-if exploration");pt=st.slider("p-tau217",.2,4.5,float(p.ptau217),.1);ab=st.slider("Aβ42/40 ratio",.025,.110,float(p.ab42_40),.005);mo=st.slider("MoCA score",10,30,int(round(p.moca)));m=p.copy();m["ptau217"],m["ab42_40"],m["moca"]=pt,ab,mo;s=score(pd.DataFrame([m[cohort().columns]]),models(cohort())).iloc[0];e,f=st.columns(2);e.metric("Current",f"{p.risk_score:.0%}");f.metric("Simulated",f"{s.risk_score:.0%}",delta=f"{s.risk_score-p.risk_score:+.0%}")
 chatbot(p)
if __name__=="__main__":main()
