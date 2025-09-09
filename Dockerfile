# HFT Neurosymbolic AI System - Simplified Dockerfile
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single layer for better caching
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Ensure Python loads our compatibility shim
COPY utils/sitecustomize.py /usr/local/lib/python3.10/site-packages/sitecustomize.py

# Copy requirements files first for better caching
COPY utils/requirements_rdf.txt requirements.txt

# Install Python dependencies in layers for better caching
# Layer 1: Core requirements (changes least often)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Layer 2: Web framework dependencies (changes occasionally)
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    streamlit==1.28.1 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    python-dotenv==1.0.0 \
    httpx==0.25.2 \
    websockets==12.0 \
    aiofiles==23.2.1

# Layer 3: Data science libraries (changes occasionally)
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.17.0 \
    yfinance==0.2.28 \
    ta==0.10.2 \
    ccxt==4.1.77

# Layer 4: Monitoring and development tools (changes rarely)
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    structlog==23.2.0 \
    loguru==0.7.2 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    black==23.11.0 \
    isort==5.12.0 \
    flake8==6.1.0 \
    jsonschema==4.20.0 \
    pyyaml==6.0.1

# Layer 5: Graph database clients (changes rarely)
RUN pip install --no-cache-dir \
    pydgraph==21.3.2 \
    neo4j==5.15.0 \
    redis==5.0.1 \
    rdflib==7.0.0 \
    sparqlwrapper==2.0.0 \
    networkx==3.2.1 \
    pyvis==0.3.2

# Layer 6: AI/ML libraries (CPU versions for ARM64) - most stable
RUN pip install --no-cache-dir \
    torch==2.1.1 \
    torchvision==0.16.1 \
    torchaudio==2.1.1 \
    transformers==4.36.2 \
    datasets==2.15.0 \
    accelerate==0.25.0 \
    protobuf==3.20.3 \
    onnx==1.15.0 \
    onnxruntime==1.16.3 \
    optuna==3.5.0 \
    wandb==0.16.1 \
    mlflow==2.8.1

# Layer 7: Symbolic reasoning libraries (changes rarely)
RUN pip install --no-cache-dir \
    kanren==0.2.3 \
    unification==0.2.2 \
    sympy==1.12 \
    z3-solver==4.12.2.0 \
    pysmt==0.9.5 \
    folium==0.15.1 \
    pydot==1.4.2 \
    graphviz==0.20.1

# Layer 8: HFT-specific libraries (simplified - removed memory-intensive packages)
RUN pip install --no-cache-dir \
    numba==0.58.1 \
    cython==3.0.6 \
    dask==2023.12.0 \
    dask[distributed]==2023.12.0 \
    polars==0.20.1 \
    duckdb==0.9.2 \
    pyarrow==14.0.2

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/config /app/benchmarks

# Copy application code (changes most often - keep at end)
COPY . /app/

# Create non-root user for security
RUN useradd -m -u 1000 hft_user && \
    chown -R hft_user:hft_user /app
USER hft_user

# Expose ports
EXPOSE 8000 8501 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Default command - run FastAPI first, then Streamlit dashboard
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port 8000 & (sleep 5 && streamlit run /app/dashboard.py --server.port 8501 --server.address 0.0.0.0)"]