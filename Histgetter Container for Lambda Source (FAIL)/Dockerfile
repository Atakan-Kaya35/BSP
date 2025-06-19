FROM public.ecr.aws/lambda/python:3.11

WORKDIR /app

# Install system dependencies and Chromium
RUN yum install -y \
    alsa-lib \
    atk \
    cups-libs \
    gtk3 \
    ipa-gothic-fonts \
    libXcomposite \
    libXcursor \
    libXdamage \
    libXext \
    libXi \
    libXrandr \
    libXScrnSaver \
    libXtst \
    pango \
    xorg-x11-server-Xvfb \
    nss \
    wget \
    chromium \
    && yum clean all

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser binaries
RUN playwright install chromium

# Copy your Lambda handler
COPY lambda_function.py .

# Set environment variables
ENV HOME=/tmp
ENV TMPDIR=/tmp
ENV PATH="/usr/lib64/chromium-browser:$PATH"

ENTRYPOINT [ "python3", "-m", "awslambdaric" ]
CMD ["lambda_function.lambda_handler"]
