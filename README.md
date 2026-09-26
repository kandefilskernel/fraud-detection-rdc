# Détection de fraude Mobile Money RDC (+ cartes Visa virtuelles)

Mémoire : « Conception et évaluation d'un modèle IA de profilage comportemental pour la détection
de fraude Mobile Money en RDC : extension comparative aux cartes Visa virtuelles ».

Algorithme, architecture et limites : [docs/ARCHITECTURE_CIBLE.md](docs/ARCHITECTURE_CIBLE.md).

> **Statut : prototype de recherche.** Toutes les données sont synthétiques. Avant tout usage réel :
> pilote en mode silencieux avec un opérateur, audit de sécurité, validation juridique et
> réglementaire (BCC, protection des données).

## Architecture en bref

```
Opérateurs ──► Nginx /ingest ──► integration-layer ──► scoring-service ──► décision (≈ 20 ms)
 (Vodacom, Airtel,               (adaptateurs)          │ profil Redis + modèle hybride
  Orange, Visa)                                         │ XGBoost + LSTM-Attention + Autoencodeur
                                                        ▼
                                 Kafka (Redpanda) : transactions.scored, fraud.alerts, audit.logs
                                    │ persister → TimescaleDB   │ alerter → dossiers + SMS/e-mail
                                    │ auditor → journal chaîné  │ drift-monitor → Prometheus
                                                        ▼
            Tableau de bord Next.js  ◄── backoffice-api (JWT, rôles, dossiers, rapports, audit)
            Grafana / Prometheus / Loki (supervision)   retrainer (champion / challenger)
```

## Lancer le projet (Windows + Docker Desktop + VS Code)

Commandes à taper dans le terminal PowerShell de VS Code, à la racine du projet.

### 1. Première installation
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # si refusé : Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env               # puis changer les mots de passe
```

### 2. Données et modèle (si `ml/data` est vide)
```powershell
.\scripts\run_pipeline.ps1                  # génération + variables + entraînement (~10 min)
.\scripts\run_pipeline.ps1 -SkipGeneration  # réutilise les données existantes
```

### 3. Démarrer toute la plateforme
Docker Desktop doit être lancé.
```powershell
docker compose up -d --build        # 1re fois : ~15 min (téléchargement de PyTorch)
docker compose ps                   # tous les services « Up » / « healthy »
python scripts\seed_database.py     # comptes de démonstration (analyste, superviseur)
```

| Adresse | Contenu |
|---|---|
| http://localhost | Tableau de bord (comptes dans `scripts/seed_database.py`, admin dans `.env`) |
| http://localhost/ingest/v1/transactions/{vodacom\|airtel\|orange\|visa} | API des opérateurs |
| http://localhost:8001/docs · :8002/docs · :8003/docs | Documentation des API (scoring, intégration, back-office) |
| http://localhost:3001 | Grafana (admin / `GRAFANA_ADMIN_PASSWORD`) |
| http://localhost:9090 | Prometheus (cibles, règles d'alerte) |
| http://localhost:8088 | Console Kafka (Redpanda) |
| http://localhost:8025 | Boîte mail locale (e-mails d'alerte) |

### 4. Démonstration temps réel
```powershell
python scripts\simulate_transactions.py --rate 20                        # trafic des 4 opérateurs
python scripts\simulate_transactions.py --only-fraud-episodes --rate 5   # démo jury
```
Latence réaliste : lancer le simulateur **dans** le réseau Docker (sinon le relais de ports de
Docker Desktop ajoute ~40 ms) :
```powershell
docker run --rm --network fraud-detection-rdc_default -v ${PWD}:/work:ro -w /work --entrypoint sh fraud-rdc/scoring-service:dev -c "pip install -q --user httpx==0.27.0; python scripts/simulate_transactions.py --url http://integration-layer:8002 --rate 40"
```

### 5. Apprentissage continu
```powershell
python scripts\simulate_feedback.py          # verdicts des analystes + plaintes clients
docker compose exec retrainer python -m ml.retraining.retrain --force   # champion vs challenger
```
Le nouveau modèle n'est promu que s'il bat l'ancien sur une période qu'aucun des deux n'a vue ;
le scoring-service le recharge alors à chaud (sans interruption).

### 6. Tests, charge, Kubernetes
```powershell
.\scripts\run_tests.ps1                               # toutes les suites (PostgreSQL requis)
python -m ml.serving.parity_check --n 2000            # parité entraînement / production
python scripts\load\build_payloads.py
docker run --rm --network fraud-detection-rdc_default -v ${PWD}/scripts/load:/load grafana/k6 run /load/k6_ingest.js
kubectl kustomize infra\kubernetes\overlays\kinshasa  # manifests zone Kinshasa (ou katanga)
```

### Arrêter
```powershell
docker compose down        # ajouter -v pour effacer aussi les données
```

## Dépannage
- **La construction échoue en téléchargeant PyTorch** : connexion instable ; relancer
  `docker compose build` — le cache pip conserve ce qui a déjà été téléchargé.
- **Le scoring ne démarre pas (« feature store vide »)** : `python -m ml.serving.seed_feature_store`.
- **Remettre la démo à zéro** : `python -m ml.serving.seed_feature_store` (profils) puis vider les
  tables `scored_transactions`, `cases`, `notifications` (le journal d'audit est inaltérable par conception).
