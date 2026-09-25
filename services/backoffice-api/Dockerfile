FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY services/backoffice-api/requirements.txt .
RUN pip install -r requirements.txt && useradd --create-home --uid 1000 app
COPY shared/ shared/
COPY services/backoffice-api/alembic.ini .
COPY services/backoffice-api/alembic/ alembic/
COPY services/backoffice-api/app/ app/
USER app
EXPOSE 8003
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8003/health').status==200 else 1)"
# les migrations sont appliquées à chaque démarrage (idempotent)
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 2"]
