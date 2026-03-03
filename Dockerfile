# AI SLM Training - runnable on any platform that supports Docker
# Build: docker build -t ai_slm_training .
# Run data prep: docker run --rm -v "$(pwd)/my_data:/data" ai_slm_training python -m data.prepare_training_data --csv /data/qa.csv --output-dir /data/training_data
# Run web UI: docker run -p 7860:7860 -v "$(pwd)/output_model:/app/model:ro" ai_slm_training python run_gradio_ui.py --model-dir /app/model --host 0.0.0.0
# Shell: docker run -it --rm ai_slm_training bash

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure /app is on Python path so "python -m demo.gradio_ui" and "python -m data.*" work
ENV PYTHONPATH=/app

WORKDIR /app

# System deps for building Python packages
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir unsloth || echo "Unsloth unavailable (no CUDA build tools), skipping"

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
