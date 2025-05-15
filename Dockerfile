FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl for kubernetes management
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x ./kubectl \
    && mv ./kubectl /usr/local/bin/kubectl

# Install AWS CLI (for AWS integration)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Install Google Cloud SDK (for GCP integration)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg -o /tmp/cloud.google.gpg \
    && install -o root -g root -m 644 /tmp/cloud.google.gpg /usr/share/keyrings/cloud.google.gpg \
    && rm /tmp/cloud.google.gpg \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt /app/
COPY requirements-dev.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# For development mode, include dev dependencies
ARG DEV_MODE=false
RUN if [ "$DEV_MODE" = "true" ] ; then pip install --no-cache-dir -r requirements-dev.txt ; fi

# Copy project files
COPY . /app/

# Create directory for secrets
RUN mkdir -p /app/secrets
RUN chmod 700 /app/secrets

# Create directory for model cache
RUN mkdir -p /app/weights_cache
RUN chmod 755 /app/weights_cache

# Create non-root user
RUN useradd -m -r -u 1000 hipo
RUN chown -R hipo:hipo /app

# Switch to non-root user
USER hipo

# Expose ports
EXPOSE 5000
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["serve"]