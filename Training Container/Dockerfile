# Base image: TensorFlow 2.19 with GPU
FROM tensorflow/tensorflow:2.19.0-gpu

# Disable bytecode and enable immediate logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Required SageMaker paths
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

# Install Linux build tools (optional)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all necessary code explicitly
COPY app.py .
COPY bsp_util.py .
COPY bsp_cloud_lib.py .
COPY config.py .
COPY pyified_resources.py .
COPY BSP_Extreme_Value_Evaluator_Models.csv .
COPY BSP_Plateau_Evaluator_Models.csv .
COPY BSP_General_Trend_Change_Evaluator_Models.csv .
COPY health_check.py .
# for safety reasons, not applicable for now but maybe later you do not provide the env file
#COPY .env .

# Define training entry point (if used by SageMaker, or manually)
ENTRYPOINT ["python", "app.py"]