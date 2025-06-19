# Start from official SageMaker TensorFlow CPU base image (2.13, Python 3.9)
FROM tensorflow/tensorflow:2.13.0-gpu

# SageMaker expects this
ENV PYTHONUNBUFFERED=TRUE

# Create directories
RUN mkdir -p /opt/ml/code

WORKDIR /opt/ml/code

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY app.py .
COPY bsp_cloud_lib.py .
COPY bsp_util.py .
COPY pyified_resources.py .
COPY BSP_Extreme_Value_Evaluator_Models.csv .
COPY BSP_Plateau_Evaluator_Models.csv .
COPY BSP_General_Trend_Change_Evaluator_Models.csv .

# Define the entrypoint for the container
ENTRYPOINT ["python", "app.py"]
