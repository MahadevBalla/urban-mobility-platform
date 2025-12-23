# Urban Transit Tool - Docker Container
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for geospatial libraries and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Geospatial libraries
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    # PostgreSQL client
    postgresql-client \
    libpq-dev \
    # Build tools
    gcc \
    g++ \
    make \
    # Utilities
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY app.py test_zone_generation.py ./
COPY README.md ./
# COPY setup_config.yaml ./ # Optional config file

# Create necessary directories
RUN mkdir -p /app/data /app/cache /app/output_cache /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
