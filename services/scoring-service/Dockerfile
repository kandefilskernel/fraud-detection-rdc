# syntax=docker/dockerfile:1
# Contexte de build : racine du dépôt (docker compose le fait automatiquement)
FROM python:3.12-slim AS base
ENV LOG_FORMAT=json PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

FROM base AS deps
COPY services/scoring-service/requirements.txt .
# cache pip persistant + reprises : une coupure réseau ne force pas à tout retélécharger
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 120 --prefix=/install -r requirements.txt \
 && pip install --retries 10 --timeout 120 --prefix=/install --no-deps xgboost==2.0.3

FROM base AS runtime
# un thread par bibliothèque de calcul : le parallélisme vient des workers uvicorn
# (sinon OpenMP/MKL multiplient les threads et se disputent les cœurs)
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN useradd --create-home --uid 1000 app
COPY --from=deps /install /usr/local
COPY shared/ shared/
COPY ml/__init__.py ml/__init__.py
COPY ml/features/ ml/features/
COPY ml/models/ ml/models/
COPY ml/serving/ ml/serving/
COPY ml/nlp/ ml/nlp/
COPY ml/artifacts/ ml/artifacts/
COPY services/scoring-service/app/ app/
USER app
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health').status==200 else 1)"
# métriques agrégées sur tous les workers (répertoire vidé à chaque démarrage)
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 3"]
