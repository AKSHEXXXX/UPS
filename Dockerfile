# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure tools are available (git for VCS dependencies, curl for optional uv bootstrap)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=gym

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# HF builder layers may place the cache and target environment on different
# filesystems, where hardlinking is unavailable.
ENV UV_LINK_MODE=copy

# Install dependencies (and the project itself) in one sync pass.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Install the FastAPI runtime into the project virtualenv explicitly.
# Hugging Face/OpenEnv startup should not fall back to binaries from the base image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/env/.venv/bin/python \
    --requirements /app/env/server/requirements.txt

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy source code from build context (lightweight via .dockerignore).
# This avoids re-copying builder's .venv a second time.
COPY . /app/env

# Compliance checks: ensure required runtime and metadata files are present.
RUN test -f /app/env/inference.py && \
    test -f /app/env/openenv.yaml && \
    test -f /app/env/README.md

# Create optional outputs directory used by validators and local tooling.
RUN mkdir -p /app/env/outputs

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports from reorganized folders are always resolvable.
ENV PYTHONPATH="/app/env:/app/env/algorithms:/app/env/evaluation:/app/env/graders:$PYTHONPATH"

# Required LLM inference configuration defaults (override via environment/secrets).
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"
ENV HF_TOKEN=""

# Health check using Python stdlib to avoid runtime curl dependency.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

# Run the FastAPI server with the project virtualenv interpreter so runtime
# imports resolve from /app/.venv instead of any binaries bundled in the base image.
CMD ["sh", "-c", "cd /app/env && /app/.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"]
