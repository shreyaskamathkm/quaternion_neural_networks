# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.10.13
FROM python:${PYTHON_VERSION}-slim-buster as builder

# Prevents python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps python from buffering stdin/stdout.
ENV PYTHONUNBUFFERED=1

# uv configuration
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration files
WORKDIR /app
COPY uv.lock pyproject.toml ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# --- Second Stage ---
FROM python:${PYTHON_VERSION}-slim-buster as production

# Copy virtual env from builder
ENV VENV_PATH=/app/.venv
COPY --from=builder ${VENV_PATH} ${VENV_PATH}

# Add virtual env to path
ENV PATH="${VENV_PATH}/bin:$PATH"

# Copy application code
COPY quaternion_neural_networks /app/quaternion_neural_networks

WORKDIR /app

# Set the default command to run Python
CMD ["python"]

