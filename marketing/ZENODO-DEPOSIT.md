# Zenodo deposit — ready to fill

Deposit at https://zenodo.org/uploads/new (sign in with your ORCID so the record binds to
it automatically).

**Why Zenodo:** operated by CERN, EU-funded, mints a real DOI with no gatekeeper, indexed
by OpenAIRE and picked up by Google Scholar. It is the fastest route to a formal, citable
record in an authoritative database.

**On permanence — the thing that unblocks depositing now:** Zenodo supports versioning. A
record gets a *concept DOI* that always resolves to the newest version, plus a *version
DOI* for each deposit. Publishing v1 today and v2 when the remaining experiments land is
the intended workflow, not a workaround. You cite the concept DOI and it stays correct
forever. So there is no reason to wait.

---

## 1. Upload type

**Publication → Technical note**

(Not "Software" — deposit the report here and, separately, use Zenodo's GitHub integration
to archive the `BSP` repository as its own Software record. Two DOIs, cross-linked. See §7.)

## 2. Title

```
Fifteen Minutes Ahead: Event-Level Evaluation of a Personalised Glucose Forecasting System
```

## 3. Authors

```
Kaya, Atakan
ORCID: [your ORCID]
Affiliation: [university, or "Independent researcher"]
```

## 4. Description / abstract

```
Continuous glucose monitors report current glucose and a short-term trend arrow, but the
arrow arrives too late to act on: fast carbohydrate takes 15-20 minutes to reach the
blood and corrective insulin longer still. This report documents BSP (Blood Sugar
Predictor), a per-user glucose forecasting system built to move that decision point
fifteen minutes earlier, together with an event-level evaluation of it.

BSP is a serverless pipeline — headless CGM history export, model training on SageMaker
Processing, ONNX export, and scale-to-zero inference on AWS Lambda — producing a
15-minute forecast from twelve five-minute readings with time-of-day encoded as sine and
cosine. Models are trained per subject; an early attempt at cross-subject transfer
degraded performance.

Evaluated on a temporally held-out 10% of a single subject's 90-day record (24,947
readings), the originally deployed configuration reaches 9.3 mg/dL mean absolute error at
the 15-minute horizon, beating persistence (10.7) and linear extrapolation (14.7)
baselines. That aggregate figure proves misleading. Sixty-five percent of test windows
move by 10 mg/dL or less, and error on windows moving more than 20 mg/dL is five times
higher (26.8 vs 5.1 mg/dL). Hypoglycemia detection below 70 mg/dL at the 15-minute
horizon reaches only 8.8% recall — 3 of 34 events. Trained under mean squared error on a
signal that is flat two thirds of the time, the network converges toward a persistence
solution that scores well on average error while remaining blind to the events the system
exists to detect.

Two independent interventions are reported. First, ensemble aggregation: individual
members of the ten-model bag detect up to 16 of the 34 events while their mean detects 3,
because averaging cancels the minority of members that anticipate a fall. Substituting a
low-quantile aggregate for the mean — no retraining, identical weights — raises
hypoglycemia recall to 50.0% at unchanged precision, and out-of-range warning recall from
60.6% to 76.9%. Second, objective design: an asymmetric loss penalising under-prediction
of falls raises recall to 73.8%, and a classifier trained directly on the low-glucose
event reaches 75.7% recall at 66.3% precision.

The aggregation result is the most transferable finding and is not specific to glucose:
where an ensemble is used to detect rare events, mean aggregation systematically discards
the signal from the minority of members that detected them.

Limitations are substantial and stated in full. All data derive from one subject; 34
hypoglycemic test events suffice to demonstrate an eightfold gap but not to tune a
clinical threshold or establish generalisation. There is no clinical validation, ethics
approval, or trial. Precision costs are material — at 43.7% precision more than half of
the asymmetric model's warnings are false. The retrained models are evaluated offline and
are not deployed. BSP is not a medical device and must not inform treatment decisions.
```

## 5. Metadata

| Field | Value |
|---|---|
| **Publication date** | date of deposit |
| **Language** | English |
| **Version** | 1.0.0 |
| **License** | Creative Commons Attribution 4.0 International (CC-BY-4.0) |
| **Keywords** | continuous glucose monitoring; hypoglycemia prediction; time series forecasting; ensemble methods; rare event detection; LSTM; ONNX; serverless machine learning; type 1 diabetes; model evaluation; single-subject study |

**Subjects (optional, improves discovery):**
- Computer Science → Machine Learning
- Health Sciences → Endocrinology and Metabolism

## 6. Related identifiers

| Relation | Identifier |
|---|---|
| `is supplemented by` | https://github.com/Atakan-Kaya35/BSP |
| `is supplemented by` | https://github.com/Atakan-Kaya35/blood-sugar-predictor-web |
| `is supplemented by` | https://github.com/Atakan-Kaya35/blood-sugar-predictor-android |
| `is documented by` | the Medium article URL |
| `is identical to` | the published report URL |

## 7. Second deposit — the software record

Zenodo has a native GitHub integration: **Zenodo → Account → GitHub → toggle on
`Atakan-Kaya35/BSP` → then cut a release on GitHub.** Zenodo archives that release and
mints a Software DOI automatically.

Suggested release tag: `v2.0.0`, titled *Event-level evaluation and aggregation fix*.

Add a `CITATION.cff` at the repository root once you have the DOI, so GitHub renders a
"Cite this repository" button. Template:

```yaml
cff-version: 1.2.0
title: "BSP — Blood Sugar Predictor"
message: "If you use this software, please cite it as below."
type: software
authors:
  - family-names: Kaya
    given-names: Atakan
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
repository-code: "https://github.com/Atakan-Kaya35/BSP"
license: MIT
version: 2.0.0
doi: "10.5281/zenodo.XXXXXXX"
```

## 8. What you can claim once the DOI exists

**CV, under Publications:**
```
Kaya, A. (2026). Fifteen Minutes Ahead: Event-Level Evaluation of a Personalised
Glucose Forecasting System. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

**LinkedIn headline:**
```
Founder, BSP — personalised glucose forecasting | CS undergraduate
```

**LinkedIn "Projects" or "Publications" entry:**
```
BSP (Blood Sugar Predictor) — Founder
2024 – present

Built and deployed a per-user glucose forecasting system as a Type 1 diabetic:
serverless AWS pipeline (Lambda, SageMaker, App Runner, S3), LSTM ensemble exported to
ONNX, React and Kotlin clients. Published an event-level evaluation showing that mean
ensemble aggregation suppressed hypoglycemia detection, and raised 15-minute event recall
from 8.8% to 73.8%. Open source, MIT. DOI: 10.5281/zenodo.XXXXXXX

Not a medical device; research use only.
```

**The one-line version, for anywhere:**
```
I build personalised glucose forecasting systems, starting with my own.
```

## 9. Sequencing

1. Deposit the report as v1.0.0 → get the DOI **today**
2. Update the report's citation block with the real DOI, republish
3. Turn on the GitHub integration, cut `v2.0.0`, get the software DOI
4. Add `CITATION.cff`
5. Update the Medium post to link the DOI
6. Update LinkedIn and CV
7. When the remaining experiments land → deposit v1.1.0; the concept DOI stays valid

## 10. Then arXiv

The aggregation finding is the paper — it is counterintuitive, has a clean mechanism, and
transfers beyond glucose. arXiv `cs.LG` requires endorsement for a first-time submitter;
a Zenodo DOI and a supervisor contact both help. Target venue if you want peer review:
**Machine Learning for Health (ML4H)**, which has a short-paper track suited to exactly
this scope and is explicit about welcoming negative and corrective results.
