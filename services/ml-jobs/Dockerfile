# Contexte de build : racine du dépôt
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1
COPY services/ml-jobs/requirements.txt .
RUN pip install --prefix=/install -r requirements.txt \
 && pip install --prefix=/install --no-deps xgboost==2.0.3

FROM python:3.12-slim
# pas de limite de threads ici : le réentraînement doit utiliser tous les cœurs
ENV LOG_FORMAT=json PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
RUN useradd --create-home --uid 1000 app
COPY --from=deps /install /usr/local
# couche séparée : ajouter un petit paquet ne force pas à retélécharger PyTorch
RUN pip install --no-cache-dir httpx==0.27.0
COPY shared/ shared/
COPY ml/ ml/
USER app
EXPOSE 9101
CMD ["python", "-m", "ml.monitoring.drift_detector", "--serve"]
