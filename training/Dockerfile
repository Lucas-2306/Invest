FROM python:3.12-slim

WORKDIR /app

COPY training /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    sqlalchemy \
    psycopg2-binary \
    pydantic-settings \
    pyarrow \
    scikit-learn \
    joblib \
    lightgbm

ENV PYTHONPATH=/app

CMD ["python", "scripts/build_dataset.py"]