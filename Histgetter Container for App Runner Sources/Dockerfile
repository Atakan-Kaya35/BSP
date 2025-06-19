# Use the existing Playwright Docker image as the base image
FROM mcr.microsoft.com/playwright/python:v1.47.0-noble

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .

# Install Playwright browser binaries explicitly at runtime
RUN playwright install --with-deps

# Define the command to run the Python script
CMD ["python", "main.py"]