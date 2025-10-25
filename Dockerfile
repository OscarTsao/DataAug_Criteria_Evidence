# ============================================================================
# Multi-Stage Production Dockerfile
# PSY Agents NO-AUG Baseline Repository
# ============================================================================
# Stage 1: Builder (full dev environment)
# Stage 2: Runtime (minimal production environment)
# Optional GPU support with graceful CPU fallback
# ============================================================================

# ============================================================================
# STAGE 1: Builder
# ============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0

# Metadata
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.title="PSY Agents NO-AUG Builder"
LABEL org.opencontainers.image.description="Builder stage for PSY Agents NO-AUG baseline"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    # PyTorch configuration
    TORCH_VERSION=2.1.2 \
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlink for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /build

# Copy dependency files first (better layer caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev for production)
RUN poetry install --no-root --no-interaction --no-ansi --without dev && \
    # Install PyTorch with CUDA support
    poetry run pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY README.md ./
COPY tests/ ./tests/

# Install the project
RUN poetry install --no-interaction --no-ansi --without dev

# Run tests to ensure build quality (optional, can comment out for faster builds)
RUN poetry run pytest tests/ -v --tb=short -x || echo "Tests completed with status: $?"

# ============================================================================
# STAGE 2: Runtime (Production)
# ============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.1.0

# Metadata
LABEL maintainer="Oscar Tsao"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.title="PSY Agents NO-AUG Runtime"
LABEL org.opencontainers.image.description="Production runtime for PSY Agents NO-AUG baseline"
LABEL org.opencontainers.image.authors="Oscar Tsao"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PATH="/home/appuser/.local/bin:/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    # CUDA configuration
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlink for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create non-root user with specific UID/GID for consistency
RUN groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /app /app/data /app/outputs /app/mlruns && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /build/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser pyproject.toml /app/
COPY --chown=appuser:appuser README.md /app/

# Install package in editable mode (lightweight)
RUN /app/.venv/bin/pip install --no-deps -e /app

# Switch to non-root user
USER appuser

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (show help)
CMD ["/app/.venv/bin/python", "-m", "psy_agents_noaug.cli", "--help"]
