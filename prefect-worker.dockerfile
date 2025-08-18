FROM python:3.12.11-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install uv package manager
RUN pip install uv

# copy dependency files
COPY pyproject.toml uv.lock ./

# install dependencies
RUN uv sync --frozen

# copy application code
COPY prefect/ ./prefect/
COPY scripts/ ./scripts/

# create data directories
RUN mkdir -p data/raw data/intermediate monitoring/predictions

# set python path
ENV PYTHONPATH=/app

# activate virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# run deployment
CMD ["python", "prefect/deploy.py"]