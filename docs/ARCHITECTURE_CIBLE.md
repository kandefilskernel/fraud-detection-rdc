# Architecture cible et algorithme de détection temps réel

Ce document fixe l'algorithme retenu et les ajustements d'architecture pour que le système
puisse réellement fonctionner en temps réel chez un opérateur Mobile Money en RDC.

## 1. Algorithme : moteur de risque comportemental hybride à décision coût-sensible

Le cœur scientifique du mémoire reste le **modèle hybride déjà entraîné** (XGBoost + LSTM-Attention
+ Autoencodeur, combinés par un méta-apprenant). Il est intégré dans un moteur complet en quatre étages :

```
Transaction ──► [1] Profil comportemental ──► [2] Scoring hybride ──► [3] Décision ──► réponse (< 100 ms)
                    (état Redis, mis à jour      XGBoost (champion)        coût-sensible,
                     à chaque transaction)       LSTM-Attention            3 actions + règles
                                                 Autoencodeur              réglementaires
                                                 → méta-apprenant
                                                 → calibration
                                                            │
                                                            ▼
                                          [4] Événements asynchrones (Kafka/Redpanda) :
                                              alertes, dossiers analystes, audit,
                                              retour des analystes → réentraînement
```

### [1] Profil comportemental en streaming (le « profilage » du sujet)
- `BehavioralFeatureExtractor` (ml/features) est déjà écrit « en ligne » : son état par utilisateur,
  appareil, contrepartie et agent est déplacé dans **Redis** au lieu de la mémoire du processus.
- Même code à l'entraînement (rejeu de l'historique) et en production : pas d'écart entraînement/production.
- Le LSTM a besoin des 10 derniers vecteurs de l'utilisateur : ils sont gardés dans une liste Redis
  (tampon circulaire de 10 éléments) — lecture O(1).

**Variables à ajouter, spécifiques à la RDC :**

| Variable | Pourquoi |
|---|---|
| `hours_since_sim_swap` | Le SIM swap est la 1ʳᵉ fraude des données. En réel, l'opérateur connaît la date du dernier changement de SIM (IMSI). C'est le signal le plus fort utilisé par les opérateurs. |
| `recipient_distinct_senders_24h` | Détection des **comptes mules** (fan-in) : un compte qui reçoit de nombreux expéditeurs inconnus. |
| `recipient_account_age_days`, `recipient_cashout_speed` | Une mule retire vite après réception, souvent sur un compte récent. |
| `agent_fraud_exposure_7d` | Part des cash-out d'un agent liés à des alertes confirmées (fraude d'agent). |

### [2] Scoring hybride
- XGBoost = **champion** (rapide, robuste, explicable par SHAP).
- LSTM-Attention = mémoire de séquence (enchaînements typiques : SIM swap → P2P → cash-out).
- Autoencodeur = détecteur de nouveautés (fraudes jamais vues).
- Méta-apprenant logistique + **calibration isotonique** : la sortie devient une vraie probabilité,
  indispensable pour la décision coût-sensible.
- **Mode dégradé** : si le LSTM ou l'autoencodeur dépasse son budget de temps, le score XGBoost seul
  est utilisé (important avec une infrastructure contrainte).

### [3] Décision coût-sensible (ce qui en fait une vraie solution opérationnelle)
Au lieu d'un seul seuil « fraude / pas fraude », on minimise le coût attendu :

    coût(APPROUVER) = P(fraude) × montant
    coût(VÉRIFIER)  = coût de friction (client qui abandonne, appel, OTP)
    coût(BLOQUER)   = (1 − P(fraude)) × coût d'un faux blocage

Trois actions :
- **APPROUVER** : la transaction passe.
- **VÉRIFIER** (step-up) : confirmation du PIN par USSD, appel du service client ou OTP.
- **BLOQUER** : transaction refusée, dossier ouvert pour un analyste.

Seuils distincts par canal (Mobile Money / Visa virtuelle), et règles réglementaires prioritaires
(ex. SIM swap < 24 h + cash-out élevé → VÉRIFIER au minimum). Les règles sont explicables et
auditables pour la BCC.

### [4] Apprentissage continu
- Chaque décision est journalisée (prédictions + variables + version du modèle).
- Les analystes confirment ou infirment les alertes → étiquettes.
- Dérive surveillée (PSI / Kolmogorov-Smirnov) → réentraînement programmé, nouveau modèle
  promu seulement s'il bat l'ancien (registre MLflow champion/challenger).

## 2. Ajustements d'architecture

| Architecture initiale | Problème | Remplacement |
|---|---|---|
| Transaction → Kafka → feature-engineering-service → Kafka → fraud-detection-service | Chaîne **asynchrone** : impossible de bloquer une transaction avant qu'elle soit exécutée. | **Chemin synchrone** : l'opérateur appelle `POST /v1/score` et reçoit la décision en < 100 ms. Kafka sert uniquement *après* la décision. |
| feature-engineering-service séparé de fraud-detection-service | Saut réseau supplémentaire + risque d'écart entre les variables calculées et celles du modèle. | Fusion en un **scoring-service** unique (variables + modèle + décision). |
| gateway/ en Python (proxy maison) | Réinvente un reverse-proxy, ajoute de la latence. | **Nginx** (TLS, limitation de débit, routage). L'authentification JWT reste dans les services. |
| auth, case-management, reporting = 3 services | Trop de services à maintenir pour un mémoire. | Un **backoffice-api** (modules auth / dossiers / rapports). |
| audit-service sur Elasticsearch | Elasticsearch demande 2 à 4 Go de RAM. | Table d'audit **append-only chaînée par hachage** dans PostgreSQL/TimescaleDB (preuve d'inaltérabilité). |
| Kafka + Zookeeper | Lourd en local, listeners mal configurés. | **Redpanda** (API Kafka, un seul conteneur). |
| Airflow | Lourd (≈ 4 Go) pour un seul DAG. | Tâche de réentraînement programmée (conteneur `retrainer`) + MLflow ; Airflow en option. |
| `backend/` et `services/` en parallèle | Doublon. | Tout sous `services/`. |

### Services retenus

```
fraud-detection-rdc/
├── services/
│   ├── integration-layer/   # adaptateurs Vodacom / Airtel / Orange / Visa → schéma unifié
│   ├── scoring-service/     # POST /v1/score : profil Redis + modèle hybride + décision
│   ├── backoffice-api/      # auth JWT + RBAC, dossiers analystes, rapports, feedback
│   └── workers/             # consommateurs Kafka : alertes (SMS/e-mail), audit, prediction-logger
├── ml/                      # données, variables, modèles, entraînement (existant)
├── frontend/                # tableau de bord Next.js
├── infra/                   # nginx, prometheus, grafana, kubernetes
└── docker-compose.yml
```

## 3. Mesures actuelles (réentraînement du 25/09/2026, jeu de test = dernier mois, jamais vu)

| Mesure | Valeur |
|---|---|
| PR-AUC hybride / XGBoost seul / LSTM seul | 0,991 / 0,986 / 0,986 |
| Faux positifs | ≈ 0,03 % des transactions légitimes |
| Latence modèle hybride, 1 transaction, 1 cœur CPU | p50 = 5,8 ms, p99 = 8,8 ms |
| Latence XGBoost seul (mode dégradé) | p50 = 0,9 ms, p99 = 2,0 ms |
| Rappel par typologie | 100 % sauf SMURFING 99 % et **SOCIAL_ENGINEERING 69 %** |

Le modèle tient donc largement le budget temps réel (< 100 ms avec Redis et le réseau).

**Point faible identifié : l'ingénierie sociale (69 %).** La victime fait elle-même la transaction,
depuis son propre téléphone et à des heures normales : son comportement paraît habituel. Le signal
est du côté du **destinataire** (compte mule récent, qui reçoit de nombreux inconnus puis retire vite).
C'est pourquoi les variables « destinataire » de la section 1 sont prioritaires. C'est aussi un très
bon argument de mémoire : le profilage de l'émetteur seul ne suffit pas, il faut profiler le réseau.

## 4. Rigueur d'évaluation (questions attendues du jury)

Les scores actuels (PR-AUC ≈ 0,99 ; 1,0 sur Visa) sont obtenus sur des **données synthétiques**.
Pour le jury, il faut montrer que le modèle ne fait pas que « reconnaître le générateur » :

1. **Test sur fraude inconnue** : entraîner sans un type de fraude (ex. SMURFING), tester dessus.
2. **Négatifs difficiles** : SIM swaps légitimes, changements d'appareil légitimes, voyages, salaires.
3. **Bruit d'étiquettes** : une partie des fraudes non signalées (réalité terrain).
4. **Métrique métier** : montant de fraude évité, nombre de clients honnêtes dérangés pour 10 000 transactions.
5. **Latence** mesurée (p50 / p99) transaction par transaction.

## 5. Mesures sur la plateforme complète et limites connues

Mesures sur un portable 4 cœurs / 8 Go faisant tourner les 21 conteneurs :

| Mesure | Valeur | Remarque |
|---|---|---|
| Latence bout-en-bout (réseau Docker, 40 tx/s) | p50 ≈ 19 ms, p95 ≈ 79 ms | intégration + scoring |
| Latence depuis Windows | + ~40 ms | relais de ports de Docker Desktop, absent en production |
| Débit soutenu | ≈ 30-40 tx/s | limité par le CPU partagé entre 21 conteneurs ; montée en charge horizontale via HPA |
| Fraudes interceptées en rejeu | 22/22 (1 500 tx), 9/9 (1 500 tx) | 1 client honnête bloqué |

Problèmes rencontrés et corrigés (utiles pour la discussion du mémoire) :
1. **Versions de bibliothèques** : un modèle sérialisé avec d'autres versions ne se recharge pas ;
   versions figées identiques entre entraînement et service.
2. **Parité** : identifiants de portefeuille lus comme flottants, et arbres XGBoost au-delà de
   l'arrêt précoce ; corrigés, parité vérifiée à 1e-7.
3. **Transactions en retard** (hors ordre chronologique) : écarts de temps négatifs ; bornés à 0.
4. **Contention de threads** : OpenMP multipliait les threads par worker ; un thread par worker.
5. **Dérive** : les variables calendaires et cumulatives dérivent par construction ; seules les
   variables comportementales et le score déclenchent l'alarme. Dérive réelle observée : soldes
   (`log_balance_before`, `amount_to_balance`) et activité des agents.
6. **PSI des variables binaires** : les quantiles se confondaient en une classe ; calcul par catégorie.

Limites : données synthétiques, formats d'API opérateurs simulés, pas d'audit de sécurité, pas de
test à l'échelle nationale (milliers de tx/s), ingénierie sociale détectée à 69 % seulement.
