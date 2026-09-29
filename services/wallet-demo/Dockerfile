# Application « portefeuille client » de démonstration + serveur opérateur simulé.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OPFMT_DIR=/srv/opfmt
WORKDIR /srv
COPY services/wallet-demo/requirements.txt .
RUN pip install -r requirements.txt && useradd --create-home --uid 1000 app
COPY shared/ shared/
# formats propriétaires des opérateurs : mêmes adaptateurs que l'integration-layer
COPY services/integration-layer/app/adapters/ opfmt/adapters/
COPY services/wallet-demo/wallet/ wallet/
USER app
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8010/health').status==200 else 1)"
# un seul processus : les sessions de démonstration sont gardées en mémoire
CMD ["uvicorn", "wallet.main:app", "--host", "0.0.0.0", "--port", "8010"]
