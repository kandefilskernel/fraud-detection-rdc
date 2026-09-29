FROM python:3.12-slim
ENV LOG_FORMAT=json PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY services/backoffice-api/requirements.txt .
RUN pip install -r requirements.txt && useradd --create-home --uid 1000 app
COPY shared/ shared/
COPY services/backoffice-api/alembic.ini .
COPY services/backoffice-api/alembic/ alembic/
COPY services/backoffice-api/app/ app/
COPY services/backoffice-api/knowledge/ knowledge/
USER app
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8003/health').status==200 else 1)"
# les migrations sont appliquées à chaque démarrage (idempotent)
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 2"]
