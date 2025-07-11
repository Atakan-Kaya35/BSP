# AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /var/task

# Copy your Python script and .h5 files into the container,
# The requirement for this backup was no longer there
#COPY 67_acc_model(12,50,24,10).h5 .

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY app.py bsp_cloud_lib.py bsp_util.py pyified_resources.py config.py ./
COPY .env .

# Define Lambda handler entry point
CMD ["app.lambda_handler"]