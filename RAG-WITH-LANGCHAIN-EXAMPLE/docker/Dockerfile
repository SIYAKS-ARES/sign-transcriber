# =============================================================================
# LGS RAG Soru Uretim Sistemi - Dockerfile
# =============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Development Stage
# =============================================================================
FROM base as development

# Copy requirements first for caching
COPY requirements.txt requirements-dev.txt ./

# Install all dependencies including dev
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p vectorstore data/generated_questions data/logs

# Default command for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production Stage
# =============================================================================
FROM base as production

# Copy requirements
COPY requirements.txt ./

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser

# Copy source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Copy data directory (for default configs.json and dataset_ocr_li.csv files)
# Note: Volume mounts in docker-compose will override these files at runtime
# If data directory doesn't exist in build context, this will fail intentionally
COPY --chown=appuser:appuser data/ ./data/

# Create necessary directories with proper permissions
RUN mkdir -p vectorstore data/generated_questions data/logs && \
    chown -R appuser:appuser vectorstore data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

