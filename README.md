# BSP Blood Sugar Model Containers

🚀 This project contains modular Docker-based containers for training and inference of personalized RNN models for blood sugar prediction. It's designed for seamless integration with AWS Lambda and SageMaker for full production deployment.

---

## 🧠 Capabilities

### 🟢 Inference Container (Lambda)

- Fetches blood sugar data from Dexcom via `pydexcom`
- Downloads user-trained model from S3 bucket
- Makes predictions using personalized Keras-based RNN
- Computes health indicators (trend, safety, plateau, etc.)
- Returns results in concise `.json` format
- Scales to 0 (via Lambda), cold start ~40s

### 🔵 Training Container (SageMaker)

- Accepts user-specific CSV training data from S3
- Supports custom model hyperparameters:
  - `--epochs`
  - `--batch_size`
  - `--num_of_models`
  - `--remaining_tries`
  - `--acceptable_acc_score`
- Evaluates model quality using multiple benchmark datasets
- Creates `.h5`, `.json`, and zipped outputs, uploads to S3
- Multi-model support: trains a bag of models, not just one
- Designed for SageMaker Processing Jobs

---

## 📦 S3 Structure

