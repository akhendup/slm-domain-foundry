# CPU / Mac image — no CUDA, no Unsloth.
# For NVIDIA GPU use Dockerfile.gpu + docker-compose.gpu.yml instead.
# docker compose up --build

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure /app is on Python path so "python -m demo.gradio_ui" and "python -m data.*" work
ENV PYTHONPATH=/app

WORKDIR /app

# Minimal system deps — only gcc for packages that can't use pre-built wheels
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from requirements.txt
# --prefer-binary: use pre-built wheels instead of compiling from source
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy project code
COPY README.md .
COPY run_gradio_ui.py .
COPY data/       data/
COPY train/      train/
COPY demo/       demo/
COPY scripts/    scripts/
COPY sample_data/ sample_data/

EXPOSE 7860

# Verify core imports (data prep + UI + torch/transformers)
RUN python -c "import pdfplumber; import gradio as gr; import torch; import transformers; print('Dependencies OK')"

CMD ["bash"]
