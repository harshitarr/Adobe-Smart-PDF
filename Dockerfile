FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
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
LABEL version="1.0.0"
LABEL description="Extract hierarchical outlines from PDF documents"
