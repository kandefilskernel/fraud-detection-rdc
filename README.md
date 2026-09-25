# Détection de fraude Mobile Money RDC (+ cartes Visa virtuelles)

Mémoire : « Conception et évaluation d'un modèle IA de profilage comportemental pour la détection
de fraude Mobile Money en RDC : extension comparative aux cartes Visa virtuelles ».

Algorithme et architecture cible : voir [docs/ARCHITECTURE_CIBLE.md](docs/ARCHITECTURE_CIBLE.md).

## Lancer le projet en local (Windows + Docker Desktop + VS Code)

Toutes les commandes se tapent dans le terminal PowerShell de VS Code, à la racine du projet.

### 1. Environnement Python (une seule fois)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Si PowerShell refuse d'activer le venv :
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2. Pipeline ML (données → variables → modèles)
```powershell
.\scripts\run_pipeline.ps1                  # complet (~10 min)
.\scripts\run_pipeline.ps1 -SkipGeneration  # réutilise les données existantes (~5 min)
.\scripts\run_pipeline.ps1 -Quick           # essai rapide
python -m pytest ml/tests -q                # tests
```
Résultats : `ml/reports/phase3_model_comparison.csv`, modèles dans `ml/artifacts/`.

### 3. Infrastructure (PostgreSQL/TimescaleDB, Redis, Redpanda)
Démarrer Docker Desktop, puis :
```powershell
copy .env.example .env        # la première fois, puis changer les mots de passe
docker compose up -d
docker compose ps             # tous les services doivent être "healthy"
```
- PostgreSQL : `localhost:5432`
- Redis : `localhost:6379`
- Kafka (Redpanda) : `localhost:19092` depuis Windows, `redpanda:9092` depuis les conteneurs
- Console Kafka : http://localhost:8088

Arrêt : `docker compose down` (ajouter `-v` pour effacer aussi les données).
