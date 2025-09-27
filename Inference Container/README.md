# Inference Container

**Purpose:** Runs an inference for the system, predicting the user's blood sugar with a given model.

**Runtime / Deploy Target:** AWS Lambda (Python 3.11)  
**Entry Point:** `app.lambda_handler`  
**Dependecies:** The user's personalised models must be present inside of the zip located at a AWS S3 "folder" with their name. The artifacts must be of the right format, current configuration uses ONNX artifacts thıugh legacy versions were using h5. The user's Dexcom username and password for their current blood sugar data to be scraped from the company's database must be present.  

## Quick Run

```bash
# local test (example)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
