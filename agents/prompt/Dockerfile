FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install fastapi uvicorn transformers torch pydantic accelerate

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]