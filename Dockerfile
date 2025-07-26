FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libicu-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p input output logs

# Set environment variables - FIXED
ENV PYTHONPATH=/app:/app/app
ENV PYTHONUNBUFFERED=1

# Language detection environment variables
ENV LANGUAGE_DETECTION_ENABLED=true
ENV LANGUAGE_CONFIDENCE_THRESHOLD=0.4
ENV DEFAULT_LANGUAGE=english
ENV LANGUAGE_DISPLAY_ENABLED=true

# Expose volumes for input/output
VOLUME ["/app/input", "/app/output", "/app/logs"]

# Set the default command - FIXED
ENTRYPOINT ["python", "-m", "app.main"]

# Default command arguments
CMD ["/app/input", "-d", "-o", "/app/output"]

# Health check - FIXED
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.extend(['/app', '/app/app']); from app.utils.pdf_utils import PDFParser; print('OK')" || exit 1

# Metadata
LABEL maintainer="PDF Outline Extractor"
LABEL version="1.1.0"
LABEL description="Extract hierarchical outlines from PDF documents with multilingual support (Japanese, German, Tamil, English)"
LABEL languages="japanese,german,tamil,english"
LABEL features="multilingual,language-detection,configurable-processing"
