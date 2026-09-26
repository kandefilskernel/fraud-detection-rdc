# Image d'amorçage pour Kubernetes : image du scoring + historique de démonstration.
# (Avec un opérateur réel, l'amorçage lirait son historique, pas ces CSV synthétiques.)
FROM fraud-rdc/scoring-service:dev
COPY ml/data/raw/transactions.csv ml/data/raw/users.csv ml/data/raw/generation_metadata.json ml/data/raw/
