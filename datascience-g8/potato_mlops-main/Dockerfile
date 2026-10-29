FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       "torch==2.14.1+cpu" \
       "torchvision==0.29.2+cpu" \
    && python -m pip install --no-cache-dir \
       "fastapi>=0.162.0" \
       "pillow>=13.0.0" \
       "python-multipart>=0.0.32" \
       "uvicorn[standard]>=0.54.0"

COPY src/ /app/src/
COPY models/potato_resnet18.pth /app/models/potato_resnet18.pth

EXPOSE 10000

CMD ["sh", "-c", "exec uvicorn potato_mlops.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
