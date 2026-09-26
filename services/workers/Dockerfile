# Une image pour tous les workers ; la commande choisit lequel lancer (voir docker-compose.yml)
FROM python:3.12-slim
ENV LOG_FORMAT=json PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY services/workers/requirements.txt .
RUN pip install -r requirements.txt && useradd --create-home --uid 1000 app
COPY shared/ shared/
COPY services/workers/workers/ workers/
USER app
EXPOSE 9100
CMD ["python", "-m", "workers.persister"]
