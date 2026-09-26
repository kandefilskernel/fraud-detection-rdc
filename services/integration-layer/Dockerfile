FROM python:3.12-slim
ENV LOG_FORMAT=json PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY services/integration-layer/requirements.txt .
RUN pip install -r requirements.txt && useradd --create-home --uid 1000 app
COPY shared/ shared/
COPY services/integration-layer/app/ app/
USER app
EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8002/health').status==200 else 1)"
# métriques agrégées sur tous les workers (répertoire vidé à chaque démarrage)
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && exec uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 2"]
