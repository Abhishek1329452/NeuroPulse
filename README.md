# 🧠 NeuroPulse
## AI-Driven Prioritization for Early Alzheimer’s Diagnostic Pathways

> **Precision Care Challenge 2026 — Prototype**  
> NeuroPulse is an explainable clinical decision-support prototype that prioritizes patients for progressive Alzheimer’s diagnostic evaluation. It is designed to help clinicians allocate limited diagnostic resources such as blood biomarker panels, MRI, and PET scans more consistently.

> **Important safety notice:** NeuroPulse is not a diagnostic, treatment, or autonomous clinical decision system. This prototype uses only a synthetic ADNI/OASIS-like cohort. Every recommendation requires neurologist review.

---

## The problem

Early Alzheimer’s pathways face a resource-allocation bottleneck:

- Large populations receive initial cognitive screening.
- Only a subset can receive advanced and costly diagnostics such as MRI or PET.
- Manual triage can be inconsistent, slow, and difficult to scale.
- Important signals are spread across cognitive, biomarker, imaging, and clinical-risk data.

NeuroPulse turns those signals into a transparent, prioritized diagnostic queue. It helps identify which patients may benefit most from timely follow-up while keeping clinicians in control.

---

## What we implemented

NeuroPulse delivers a deployable, no-Docker Streamlit demonstration with:

- **Multimodal patient profiles** combining cognitive scores, blood-biomarker proxies, structural-imaging proxies, demographics, genetics, and comorbidity burden.
- **Ensemble risk scoring** using XGBoost, Random Forest, and Logistic Regression.
- **Adaptive prioritization** that combines estimated risk, cognitive-decline urgency, and expected diagnostic benefit.
- **Progressive diagnostic routing** from routine monitoring to blood work, MRI, and specialist/PET prioritization.
- **Patient-level explanations** showing the strongest factors influencing prioritization.
- **Interactive what-if exploration** for educational sensitivity analysis.
- **Clinician-focused dashboard** with filters, cohort summary metrics, a ranked queue, and individual patient assessments.

---

## System architecture

```text
                         ┌──────────────────────────────────┐
                         │          Clinician user          │
                         └────────────────┬─────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   NeuroPulse Streamlit Dashboard                       │
│  • Cohort filters        • Priority queue       • Patient assessment   │
│  • Risk explanations     • What-if exploration  • Safety disclaimer    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Clinical Decision-Support Engine                      │
│                                                                        │
│  1. Feature preparation                                                │
│     Cognitive + biomarkers + imaging proxies + comorbidities           │
│                                                                        │
│  2. Ensemble inference                                                 │
│     XGBoost (50%) + Random Forest (30%) + Logistic Regression (20%)   │
│                                                                        │
│  3. Priority engine                                                    │
│     Priority = risk + urgency of decline + expected benefit            │
│                                                                        │
│  4. Explainability                                                     │
│     Patient-specific contributing-factor view                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Suggested next diagnostic step                        │
│  Monitor → Cognitive / blood work → MRI + biomarkers → Specialist/PET │
└────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-end workflow

```text
 ┌────────────────────┐
 │ De-identified data │
 │ / synthetic cohort │
 └─────────┬──────────┘
           │
           ▼
 ┌────────────────────┐
 │ Feature engineering│
 │ • MoCA / MMSE      │
 │ • p-tau217         │
 │ • Aβ42/40          │
 │ • MRI proxies      │
 │ • APOE ε4          │
 │ • Comorbidities    │
 └─────────┬──────────┘
           │
           ▼
 ┌────────────────────┐
 │ Ensemble model     │
 │ calculates risk    │
 └─────────┬──────────┘
           │
           ▼
 ┌────────────────────┐
 │ Adaptive priority  │
 │ score calculation  │
 └─────────┬──────────┘
           │
           ▼
 ┌────────────────────────────────────────────────────────────┐
 │ Risk tier and routing                                       │
 │ • Low: routine monitoring                                  │
 │ • Medium: detailed cognitive assessment + blood work       │
 │ • High: MRI + targeted biomarkers                          │
 │ • Critical: specialist review + PET prioritization         │
 └─────────┬──────────────────────────────────────────────────┘
           │
           ▼
 ┌────────────────────┐
 │ Clinician review   │
 │ and final decision │
 └────────────────────┘
```

---

## Prioritization approach

NeuroPulse does not rely on a single threshold. It produces a resource-aware score:

```text
Priority Score = 0.65 × Estimated Risk
               + 0.23 × Cognitive-Decline Urgency
               + 0.12 × Expected Diagnostic Benefit
```

| Component | Purpose |
|---|---|
| Estimated risk | Ensemble estimate derived from the patient’s multimodal profile. |
| Cognitive-decline urgency | Gives added priority to patients showing faster deterioration. |
| Expected diagnostic benefit | Helps direct limited diagnostic capacity toward patients likely to benefit from earlier investigation. |

### Routing rules used in the prototype

| Risk range / condition | Suggested next step |
|---|---|
| Risk < 40% | Routine monitoring and repeat screening |
| 40%–64% | Detailed cognitive assessment and blood work |
| 65%–84% | MRI assessment and targeted blood biomarkers |
| ≥85% with elevated decline | Specialist review and PET prioritization |

These are demonstrator rules, not clinical guidelines.

---

## Machine-learning design

| Model | Role in NeuroPulse |
|---|---|
| XGBoost | Captures complex, nonlinear patterns in multimodal tabular features. |
| Random Forest | Provides robust complementary predictions across noisy inputs. |
| Logistic Regression | Adds a simpler, more interpretable baseline estimate. |

The final risk estimate is a weighted ensemble:

```text
Risk = 0.50 × XGBoost + 0.30 × Random Forest + 0.20 × Logistic Regression
```

### Input features

- Age
- MoCA and MMSE cognitive screening scores
- p-tau217 proxy
- Aβ42/40 ratio proxy
- Hippocampal-volume proxy
- White-matter-hyperintensity proxy
- Comorbidity count
- Twelve-month cognitive decline
- Brain-age gap proxy
- APOE ε4 status

---

## User interface

The dashboard is organized around how a neurologist would triage a cohort:

1. **Cohort filters** — narrow the queue by age, risk tier, or patient identifier.
2. **Prioritized queue** — see high-priority patients first with their risk, priority score, and next step.
3. **Patient assessment** — inspect a selected patient’s explanation and diagnostic routing.
4. **What-if exploration** — explore how selected input changes affect the model estimate for demonstration and discussion.

### Example demo sequence

1. Filter the dashboard to **High** risk patients.
2. Open the highest-priority patient.
3. Explain the top contributing factors.
4. Show the suggested pathway and the rationale for escalation.
5. Adjust the what-if controls to demonstrate transparent sensitivity analysis.
6. Reinforce that a clinician makes the final decision.

---

## Safety, ethics, and limitations

NeuroPulse is designed around responsible-AI principles:

- **Non-diagnostic use:** The system prioritizes evaluation; it does not diagnose Alzheimer’s disease.
- **Human oversight:** A neurologist must review every suggested action.
- **Synthetic data only:** This demo contains no patient-identifiable information and is not validated for clinical use.
- **Transparent outputs:** The interface presents contributing factors rather than only a black-box score.
- **Uncertainty awareness:** A real deployment should refer uncertain, unusual, or out-of-distribution cases to specialist review.
- **Validation required:** Clinical deployment would require approved datasets, calibration, subgroup fairness testing, prospective validation, security controls, and regulatory assessment.

---

## Technology stack

| Layer | Technology |
|---|---|
| Application and UI | Streamlit |
| Data processing | Pandas, NumPy |
| Machine learning | XGBoost, scikit-learn |
| Visualization | Plotly |
| Deployment | Streamlit Community Cloud or Hugging Face Spaces |

---

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed by Streamlit, typically `http://localhost:8501`.

---

## Deploy without Docker

### Option 1 — Streamlit Community Cloud

1. Create a GitHub repository containing `app.py`, `requirements.txt`, and `README.md`.
2. Open Streamlit Community Cloud and select **Create app**.
3. Select the repository and set the entry point to `app.py`.
4. Deploy and share the generated URL.

### Option 2 — Hugging Face Spaces

1. Create a new Space and choose **Streamlit** as the SDK.
2. Upload `app.py` and `requirements.txt`.
3. Wait for the build to finish.
4. Share the public Space URL.

---

## Project structure

```text
neuropulse/
├── app.py                # Streamlit dashboard and ML inference pipeline
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Future enhancements

- Replace synthetic data with approved, de-identified ADNI/OASIS datasets.
- Add longitudinal patient records and temporal trajectory models.
- Introduce formal uncertainty estimation and referral rules.
- Add fairness and calibration monitoring by demographic subgroup.
- Integrate FHIR-compatible data ingestion for clinical-system pilots.
- Add audit logs, access controls, and model-version governance.

---

## Presentation statement

**NeuroPulse transforms a reactive diagnostic bottleneck into an explainable, structured, and resource-aware prioritization workflow. It helps clinicians focus limited advanced testing capacity on patients most likely to benefit from timely evaluation—while preserving clinical judgment at every stage.**
