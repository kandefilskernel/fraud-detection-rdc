# Contexte de build : racine du dépôt (docker compose le fait automatiquement)
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

FROM base AS deps
COPY services/scoring-service/requirements.txt .
RUN pip install --prefix=/install -r requirements.txt \
 && pip install --prefix=/install --no-deps xgboost==2.0.3

FROM base AS runtime
RUN useradd --create-home --uid 1000 app
COPY --from=deps /install /usr/local
COPY shared/ shared/
COPY ml/__init__.py ml/__init__.py
COPY ml/features/ ml/features/
COPY ml/models/ ml/models/
COPY ml/serving/ ml/serving/
COPY ml/artifacts/ ml/artifacts/
COPY services/scoring-service/app/ app/
USER app
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "3"]
