# Website + live demo on a Hugging Face Space (Docker SDK, CPU).
# Copied to the Space root by deploy/huggingface/deploy.sh.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
# CPU wheels keep the image small; the demo runs the CPU profile anyway.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements-web.txt
COPY --chown=user . .
USER user
ENV YOLO_CONFIG_DIR=/tmp/ultralytics PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
