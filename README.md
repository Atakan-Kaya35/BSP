# BSP — Cloud-Native Blood Sugar Prediction (Research/Demo)

> **Not a medical device.** Educational & research purposes only.
> **Stack:** AWS Lambda · SageMaker Processing · App Runner · S3 · Docker · Python (Keras/TF/ONNX) · React/Kotlin

<p align="center">
  <img src="./demos/Hero_banner.png" alt="BSP Logo"/>
</p>

---

<p align="center">
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/status-public%20demo-blue"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT%2FApache-lightgrey"></a>
  <a href="#"><img alt="Platform" src="https://img.shields.io/badge/platform-AWS-orange"></a>
  <a href="#"><img alt="Infra" src="https://img.shields.io/badge/infra-Docker%20%7C%20Serverless-green"></a>
</p>

---

## 🧭 Table of Contents

* [Why This Exists](#-why-this-exists)
* [Highlights](#-highlights)
* [Architecture](#-architecture)
* [Repository Map](#-repository-map)
* [Quickstart (Local Demo)](#-quickstart-local-demo)
* [Deploying to AWS](#-deploying-to-aws)
* [Data & Models](#-data--models)
* [API Shape](#-api-shape)
* [Observability](#-observability)
* [Security & Privacy](#-security--privacy)
* [Roadmap & Lessons](#-roadmap--lessons)
* [Contributing](#-contributing)
* [License](#-license)
* [Appendix: Dev Tips](#appendix-dev-tips)

---

## 🎯 Why This Exists

This repository open-sources the **system engineering** behind a complete blood sugar prediction pipeline. The original target was production; after **patent rejection** and **app-store medical constraints**, this is released as a **research/educational demo** so others can learn, reuse, and build.

> ⚠️ **Medical Disclaimer**
> This project is **not a medical device** and must **not** be used for clinical decisions. Educational and research purposes only.

---

## ✨ Highlights

* **Modular Containers**: distinct images for **training** (SageMaker Processing) and **inference** (AWS Lambda).
* **Serverless Inference**: autoscale-to-zero + cold-start tuned.
* **Training Pipeline**: ingest CSV → train (GRU/LSTM) → evaluate → export artifacts.
* **Data Connectors**: Dexcom (via `pydexcom`) & S3 model storage.
* **Frontends (Prototype)**: React (web) & Kotlin/Jetpack Compose (Android) visualizations.
* **Production Mindset**: IaC-ready layout, env isolation, logs/metrics hooks.

<p align="center">
    <img src="./demos/highlights.png" alt="BSP Logo"/>
</p>

---

## 🏗️ Architecture

<p align="center">
    <img src="./demos/BSP_arhitecture.png" alt="BSP Logo"/>
</p>

**Flow**

1. **Ingestion** pulls CGM readings (or uses synthetic CSV) → saved to **S3**.
2. **Training (SageMaker Processing)** runs RNN candidates (e.g., GRU/LSTM), evaluates, exports (`.onnx`, metrics `.json`) → **S3**.
3. **Inference (Lambda)** loads selected model from **S3** → returns **real-time predictions** + indicators.
4. **App** (React/Kotlin) renders last 12 values + next 3 predictions with confidence bands.

---

## 🔗 Related Repositories

- **blood-sugar-predictor-android** — Kotlin / Jetpack Compose Android frontend  
  https://github.com/Atakan-Kaya35/blood-sugar-predictor-android

- **blood-sugar-predictor-web** — React web frontend (Plotly visualization)  
  https://github.com/Atakan-Kaya35/blood-sugar-predictor-web

---

## 🗺️ Repository Map

> Each deployable has its own **folder README** with purpose, runtime/target, entrypoint, env vars, and deployment steps.

| Path                                             | What it is                        | Runtime / Target         |
| ------------------------------------------------ | --------------------------------- | ------------------------ |
| `Inference Container/`                           | Real-time prediction service      | **AWS Lambda** (Python)  |
| `Training Container/`                            | Batch training & evaluation       | **SageMaker Processing** |
| `Histgetter Container for App Runner Sources/`   | Optional ingestion/API            | **AWS App Runner**       |
| `Histgetter Container for Lambda Source (FAIL)/` | Optional ingestion/API            | **AWS Lambda** (Python)  |
| `Models/Modern Model/`                           | Model artifacts from Version 2    | TensorFlow               |
| `Models/Legacy Model/`                           | Model artifacts from Version 1    | TensorFlow               |
| `demos/`                                         | Images for Github                 | Draw.io etc.             |

---

## ⚡ Quickstart (Local Demo)

Runs locally with **synthetic data**—no PHI/PII or external creds.

```bash
# clone
git clone https://github.com/Atakan-Kaya35/BSP.git
cd BSP

# local venv for tooling
python -m venv .venv && source .venv/bin/activate

# build & run inference locally
docker build -t bsp-inference "./Inference Container"
docker run --rm -p 8080:8080 -e MODEL_PATH=/app/models/demo.onnx bsp-inference

# query
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

---

## ☁️ Deploying to AWS

> Exact commands live in each folder’s README (SAM/CDK/Terraform or GitHub Actions). Below is the shape.

### Inference (Lambda)

* **Entry point:** `app.lambda_handler`
* **Inputs:** recent CGM series (or synthetic), S3 model artifact
* **Outputs:** JSON: predictions + trend + bands
* **Env vars:** `ENV`, `TMP_DIR`, `S3_BUCKET`

<p align="center">
    <img src="./demos/Inference_structure.png" alt="BSP Logo"/>
</p>

### Training (SageMaker Processing)

* **Inputs:** S3 CSV(s)
* **Args:** `--epochs`, `--batch_size`, `--early_stop_patience`, `--num_of_models`, `--acceptable_acc_score`, `--username`, `--remaining-tries`, `--num_of_layers`, `--seq_len`
* **Artifacts:** `.onnx` (or `.h5`), metrics `.json`, zipped bundle → **S3**

<p align="center">
    <img src="./demos/Training_structure.png" alt="BSP Logo"/>
</p>

### Optional: Ingestion/API (App Runner)

* **Purpose:** scheduled fetch + normalized endpoints for downstream jobs.
* Can be replaced with a simple Dexcom→S3 step if preferred.

---

## 📦 Data & Models

* **Synthetic demo data** in `examples/`.
* **Model formats:** trained in Keras; exported to **ONNX** in the newest version for portable inference. Was a **h5** export previously.
* Real integrations (Dexcom via `pydexcom`) require env-based credentials (never committed).

---

## 🔌 API Shape

**Request**

```json
{
  "username": "atakanka350@gmail.com",
  "password": "*****",
  "is_first_call": true
}
```

**Response**

```json
{
  "safeness": true,
  "trend": 27,
  "befores": 159159158,
  "befores1": 135143155,
  "befores2": 128123129,
  "afters": 156155153,
  "indicators": 20,
  "confidence": 96.58816483331636,
  "anomalies": 0,
  "model_info": [[0.0, 0.375, 0.0, 0.5859719438877755], [0.16666666666666666, 0.375, 0.0, 0.6092184368737475], [0.16666666666666666, 0.125, 0.0, 0.5867735470941884]]
}
```

---

## 📊 Observability

* **Logs:** structured JSON to stdout (CloudWatch).
* **Metrics/Alarms:** invocation latency, error rate, cold starts.
* **Traceability:** include model name/version in response `meta`.

---

## 🔐 Security & Privacy

* No PHI/PII committed. **Use synthetic data** for demos.
* Secrets via **AWS Secrets Manager / SSM** (never in Git).
* `.gitignore` / `.dockerignore` configured to avoid accidental leakage.
* If secrets ever existed in history, **rotate** and **purge** with `git filter-repo` before publishing.

---

## 🧭 Roadmap & Lessons

This effort encountered **real-world walls** (patent denial, app-store medical policies). The productization path is paused, but the **engineering patterns** are reusable across time-series domains (fitness, sleep, IoT, anomaly detection).

**PR Track (Next):**

* Blog #1: *“Building a Cloud-Native CGM Predictor”*
* Blog #2: *“What Deploying Medical ML Taught Me (as a Student)”*
* 5–7 slide deck + short demo video

<p align="center">
    <img src="./demos/Timeline.png" alt="BSP Logo"/>
</p>

---

## 🤝 Contributing

PRs welcome for:

* Smaller, faster images & cold-start tweaks
* ONNX/TFLite inference variants
* Synthetic data generators
* Docs/diagram improvements

---

## 📜 License

MIT or Apache-2.0 **with the following notice**:

> Provided for research and educational purposes only. **Not intended for medical use.** No warranties.

---

## Appendix: Dev Tips

**Makefile convenience**

```Makefile
.PHONY: infer-build infer-run train-build
infer-build: ; docker build -t bsp-inference containers/inference
infer-run:   ; docker run --rm -p 8080:8080 -e MODEL_PATH=/app/models/demo.onnx bsp-inference
train-build: ; docker build -t bsp-trainer containers/trainer
```

**Pre-publish hygiene**

```bash
git grep -nE 'AWS_|SECRET|TOKEN|PASSWORD|PRIVATE_KEY|BEGIN RSA|BEGIN OPENSSH'
```

**Large artifacts (optional)**

```bash
git lfs install
git lfs track "*.h5" "*.onnx" "*.zip"
git add .gitattributes
```