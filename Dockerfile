FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.4 && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

COPY src/ ./src/
COPY config/ ./config/

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
