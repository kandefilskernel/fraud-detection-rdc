# Architecture cible et algorithme de détection temps réel

Ce document fixe l'algorithme retenu et les ajustements d'architecture pour que le système
puisse réellement fonctionner en temps réel chez un opérateur Mobile Money en RDC.

## 1. Algorithme : moteur de risque comportemental hybride à décision coût-sensible

Le cœur scientifique du mémoire est un **modèle hybride** : une branche « arbres », un LSTM-Attention
et un autoencodeur, combinés par un méta-apprenant. Depuis le 28/09, la branche arbres est une **forêt
aléatoire** et non plus XGBoost : c'est le modèle qui répond le mieux aux besoins, voir
[SELECTION_MODELE.md](SELECTION_MODELE.md). Il est intégré dans un moteur complet en quatre étages :

```
Transaction ──► [1] Profil comportemental ──► [2] Scoring hybride ──► [3] Décision ──► réponse (< 100 ms)
                    (état Redis, mis à jour      Forêt aléatoire           coût-sensible,
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
| `hours_since_sim_swap` | Le SIM swap est la 1ʳᵉ fraude des données. En réel, l'opérateur connaît la date du dernier changement de SIM (IMSI). C'est le signal le plus fort utilisé par les opérateurs. *(non simulé : l'ajouter au générateur rendrait le résultat circulaire)* |
| `agent_fraud_exposure_7d` | Part des cash-out d'un agent liés à des alertes confirmées (fraude d'agent). |

**Fait le 27/09 — C1, profilage bidirectionnel (famille `recipient_network`, 13 variables).**
La contrepartie (destinataire d'un envoi, expéditeur d'une réception) est profilée à partir des
transactions de TOUS les clients et de tous les opérateurs, en temps réel (Redis `fs:wallet`,
`fs:wfs`, `fs:win`, `fs:wout`) : client KYC ou non, ancienneté, expéditeurs distincts et part de
premiers contacts sur 7 jours, nombre d'opérateurs payeurs, diffusion vers d'autres clients ;
motif « envoi par erreur » (renvoi supérieur au montant reçu de cette personne) ; le titulaire
comme destinataire (réceptions d'inconnus, part de l'argent reçu qui ressort aussitôt).

### [2] Scoring hybride
- **Forêt aléatoire** (200 arbres) = profil instantané. Elle est « aplatie » en tableaux numpy
  (`ml/models/flat_forest.py`) : 0,7 ms au lieu de 14 ms par transaction, avec une explication par
  variable (méthode de Saabas, exacte et additive). Elle remplace XGBoost : meilleure sur la vérité
  terrain (+0,019 de PR-AUC) et sur Visa (+0,069), écarts significatifs.
- LSTM-Attention = mémoire de séquence (enchaînements typiques : SIM swap → P2P → cash-out).
- Autoencodeur = détecteur de nouveautés : inutile sur les fraudes connues, mais +3,3 points en
  moyenne sur des typologies jamais vues à l'entraînement (7 sur 7).
- Méta-apprenant logistique sur les trois scores. Sa probabilité est calibrée sur les fraudes
  *signalées* ; elle sous-estime le risque réel (voir SELECTION_MODELE.md, « Calibration »).
- **Mode dégradé** : si une branche neuronale échoue, sa contribution est neutralisée et la réponse
  l'indique.

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

Friction ciblée contre l'arnaque « envoi par erreur » : la victime utilise son propre téléphone,
depuis son lieu habituel, et le modèle la laisse passer (probabilité faible). Une règle demande une
confirmation PIN quand le client renvoie **plus de 3 fois** ce que ce même numéro vient de lui
envoyer (48 h), variable `refund_ratio_to_cp` du profil réseau (C1). Mesurée sur les données : elle
attrape 36 % de ces arnaques, ne dérange que 0,32 % des envois légitimes, et une alerte sur deux
est une vraie fraude. Comme toutes les règles, elle ne peut que renforcer la décision du modèle.

### [4 bis] Signalements de SMS d'arnaque (NLP)
Les clients transfèrent les SMS suspects au numéro court de leur opérateur. Un classifieur NLP
(n-grammes de caractères et de mots, régression logistique) reconnaît l'arnaque, et les numéros
extraits du message sont marqués. Tout envoi vers un numéro marqué, si le portefeuille a moins de
30 jours, demande une confirmation, jamais un blocage. Avec 1 % de clients qui signalent, près d'un
numéro d'escroc sur deux est marqué avant sa première victime. Détails : `docs/NLP_SIGNALEMENTS_SMS.md`.

### [5] Assistant d'enquête (hors boucle temps réel)
Après la décision, l'analyste peut demander une note d'instruction : recherche des dossiers
passés les plus proches (archive de 4 152 cas, cosinus pondéré par l'importance XGBoost), des
liens avec des fraudes confirmées et des procédures internes (RAG), puis rédaction par Claude
**sans aucun identifiant client**, ou assemblage sans modèle de langage si aucune clé n'est
configurée. Sur la période de test, la typologie proposée par les précédents est juste dans
88 % des cas (24 % pour la réponse constante). Détails et limites : `docs/ASSISTANT_ENQUETE.md`.

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
│   ├── backoffice-api/      # auth JWT + RBAC, dossiers analystes, rapports, feedback,
│   │                        # assistant d'enquête RAG (knowledge/ : archive + procédures)
│   ├── wallet-demo/         # application mobile de démonstration « portefeuille client »
│   └── workers/             # consommateurs Kafka : alertes (SMS/e-mail), audit, prediction-logger
├── ml/                      # données, variables, modèles, entraînement (existant)
├── frontend/                # tableau de bord Next.js
├── infra/                   # nginx, prometheus, grafana, kubernetes
└── docker-compose.yml
```

## 3. Mesures actuelles (réentraînement du 26/09/2026, données « réalistes », test = dernier mois, jamais vu)

| Mesure | Valeur |
|---|---|
| PR-AUC hybride / Random Forest / LSTM seul / XGBoost seul | 0,880 / 0,876 / 0,858 / 0,839 |
| PR-AUC hybride : Mobile Money / Visa virtuelle | 0,901 / 0,780 (Random Forest sur Visa : 0,803) |
| Précision / rappel au seuil choisi sur la validation | 0,90 / 0,81 |
| Faux positifs | ≈ 0,14 % des transactions légitimes |
| Latence modèle hybride, 1 transaction, 1 cœur CPU | p50 = 5,8 ms, p99 = 8,8 ms (mesure du 25/09, architecture inchangée) |
| Latence XGBoost seul (mode dégradé) | p50 = 0,9 ms, p99 = 2,0 ms |
| Rappel par typologie | SIM_SWAP 95 %, CARD_TESTING 91 %, SMURFING 88 %, AGENT_FRAUD 88 %, ACCOUNT_TAKEOVER 85 %, CNP_FRAUD 79 %, **SOCIAL_ENGINEERING 56 %**, **TOPUP_DRAIN 56 %** (n = 18), FRIENDLY_FRAUD 0 % (indétectable par construction) |

Le modèle tient donc largement le budget temps réel (< 100 ms avec Redis et le réseau).

**Constats à discuter dans le mémoire :**
- L'hybride ne dépasse le Random Forest que de 0,004 de PR-AUC : écart probablement non
  significatif (à vérifier par bootstrap, phase 4). Sur Visa, le Random Forest fait mieux.
- La branche séquentielle (LSTM) apporte l'essentiel du gain sur XGBoost seul (+0,02 à +0,04).
- L'autoencodeur n'apporte rien (coefficient du meta-learner ≈ 0,05 ; l'hybride sans lui est
  aussi bon) : ses anomalies sont déjà captées par les variables comportementales.
- Visa est plus difficile que Mobile Money (moins d'historique par carte, fraudes via proxy
  local ou malware sur le téléphone de la victime).

Mesures précédentes (25/09, données « faciles ») : PR-AUC 0,991, 1,000 sur Visa pour TOUS les
modèles. Le générateur rendait la fraude trivialement séparable (appareil jamais partagé chez
les clients légitimes, IP étrangère quasi exclusive aux fraudes) ; corrigé le 26/09 (section 4).

**Évaluation de C1 (27/09)** — comparaison contrôlée (mêmes données, split et graines ;
`ml/reports/comparaison_C1.json`, bootstrap apparié par client ×1000) :

| Mesure (hybride, test) | Sans C1 | Avec C1 | Écart [IC 95 %] |
|---|---|---|---|
| PR-AUC globale | 0,880 | 0,873 | −0,007 [−0,019 ; +0,004] (non significatif) |
| PR-AUC Mobile Money / Visa | 0,901 / 0,780 | 0,890 / 0,790 | non significatifs |
| Ingénierie sociale, avec transfert d'amorçage (n = 46) | 72 % | **96 %** | +24 points |
| Ingénierie sociale, sans amorçage (n = 109) | 50 % | 40 % | −10 points (bruit d'entraînement possible) |
| McNemar sur les décisions | | | 48 erreurs corrigées, 44 introduites, p = 0,76 |

Lecture : C1 capte très bien l'arnaque « envoi par erreur » quand son motif est présent, mais
n'améliore pas le modèle dans son ensemble sur ces données. Les signaux de réseau (fan-in) sont
faibles par construction : 82 % des mules du générateur ne reçoivent que d'une seule victime.
Leur utilité réelle dépend du taux de réutilisation des mules, inconnu → analyse de sensibilité
à faire, et répétition sur plusieurs graines avant toute conclusion.

**Évaluation de C2 (27/09)** — profil unifié portefeuille + carte (production) contre profils
séparés par canal (« silo » : l'émetteur de la carte ne voit que la carte) ; mêmes données, split
et graines ; `ml/reports/comparaison_C2.json`.

| Mesure (hybride, test) | Silo | Unifié | Écart [IC 95 %] |
|---|---|---|---|
| PR-AUC globale | 0,869 | 0,873 | +0,004 [−0,004 ; +0,012] (non significatif) |
| PR-AUC Visa virtuelle | 0,791 | 0,790 | −0,001 [−0,036 ; +0,030] (non significatif) |
| Rappel global | 0,790 | 0,807 | **+0,017 [+0,002 ; +0,033]** |
| Rappel SIM_SWAP (n = 203) | 0,877 | 0,926 | **+0,049 [+0,021 ; +0,084]** |
| McNemar sur les décisions | | | 38 erreurs corrigées, 21 introduites, p = 0,036 |

Lecture : l'hypothèse « le profil du portefeuille aide la carte » n'est PAS confirmée (Visa
inchangée). Le gain observé va dans l'autre sens : l'activité carte enrichit le profil Mobile
Money (SIM swap mieux détecté). Deux limites : (1) les cartes récentes ne sont pas testables —
95 % des cartes du générateur sont actives dès le premier mois, le mois de test ne contient que
5 transactions de cartes neuves et aucune fraude ; (2) en silo, les recharges de carte ne sont
vues que par l'émetteur de la carte, alors qu'en réalité l'opérateur les voit aussi (débit du
portefeuille) : le silo pénalise un peu le Mobile Money. ~18 mesures comparées : un écart
« significatif » isolé peut être dû au hasard ; répéter sur plusieurs graines.

### Données « réalistes v3 » et profil de réputation (28/09)

À partir du 28/09, le générateur reproduit ce que l'opérateur sait réellement (sources dans
[HYPOTHESES_DONNEES.md](HYPOTHESES_DONNEES.md)) :
- **Signalement partiel :** seuls 38,5 % des épisodes de fraude sont signalés ou détectés.
- **Délai de signalement :** 2,6 jours en médiane.
- **Mules réutilisées :** 2 victimes en moyenne, jusqu'à 11.

Les résultats ci-dessus (phase 3, C1, C2) portent sur la version précédente (v2) des données et restent
valables pour elle. Ils sont à refaire sur v3 pour une comparaison homogène.

**Profil de réputation** (famille `reputation`, 5 variables). Les entités déjà impliquées dans des fraudes
**signalées avant** la transaction sont marquées : appareil, portefeuille contrepartie, agent (7 j),
marchand (30 j) et compte (30 j). La réputation est alimentée en production par le topic Kafka
`fraud.confirmed` (verdicts d'analystes, plaintes, retours d'opérateurs). À l'entraînement, les
signalements sont rejoués à leur date de signalement (`replay_history`) : pas de fuite du futur.

| Hybride, test (`ml/reports/comparaison_reputation*.json`) | Sans réputation | Avec | Écart [IC 95 %] |
|---|---|---|---|
| PR-AUC, étiquettes connues de l'opérateur | 0,493 | 0,477 | −0,016 [−0,061 ; +0,028] n.s. |
| **PR-AUC, vérité terrain** | 0,909 | **0,937** | **+0,028 [+0,014 ; +0,043]** |
| PR-AUC vérité, Mobile Money | 0,929 | **0,967** | **+0,038 [+0,027 ; +0,050]** |
| PR-AUC vérité, Visa | 0,844 | 0,743 | **−0,100 [−0,175 ; −0,025]** |
| **Rappel ingénierie sociale (vérité, n = 197)** | 31 % | **53 %** | **+21 points [+14 ; +29]** |
| Rappel test de carte (vérité, n = 64) | 53 % | 11 % | −42 points (significatif) |

Lecture :
- **Ingénierie sociale :** la réputation du **portefeuille destinataire** fait passer sa détection de 31 % à 53 %. Une mule déjà signalée est reconnue par les victimes suivantes (30 % des fraudes Mobile Money vont vers un portefeuille déjà signalé, contre 0 % des opérations légitimes).
- **Visa se dégrade** (surtout le test de carte). La cause n'est pas établie. Piste principale : un modèle et un seuil uniques, dominés par le Mobile Money (6 fois plus de fraudes). À tester : un seuil ou un modèle par canal, et une répétition sur plusieurs graines.
- **Variable inutile :** `rep_user_frauds_30d` n'apporte rien, car le générateur ne simule pas la re-victimisation d'un même compte.
- **Écart entre les deux mesures :** sur les étiquettes connues de l'opérateur, aucune différence n'est visible (0,49), alors que la vérité terrain montre le gain (0,94). Beaucoup d'« erreurs » apparentes sont des fraudes que personne n'a signalées. Un opérateur qui n'évalue son modèle que sur ses étiquettes **sous-estime fortement** son efficacité.

**Point faible identifié : l'ingénierie sociale (56 %).** La victime fait elle-même la transaction,
depuis son propre téléphone et à des heures normales : son comportement paraît habituel. Le signal
est du côté du **destinataire** (compte mule récent, qui reçoit de nombreux inconnus puis retire vite).
C'est pourquoi les variables « destinataire » de la section 1 sont prioritaires. C'est aussi un très
bon argument de mémoire : le profilage de l'émetteur seul ne suffit pas, il faut profiler le réseau.

## 4. Rigueur d'évaluation (questions attendues du jury)

Les scores sont obtenus sur des **données synthétiques**. Pour le jury, il faut montrer que le
modèle ne fait pas que « reconnaître le générateur » :

1. **Test sur fraude inconnue** : entraîner sans un type de fraude (ex. SMURFING), tester dessus.
2. **Négatifs difficiles** *(fait le 26/09)* : téléphones familiaux partagés, changements
   d'appareil et navigateurs, VPN, retraits de tout le solde, rafales, likelemba (tontine),
   micro-achats de jeux, dépôts initiés depuis la ligne de l'agent.
3. **Bruit d'étiquettes** *(fait le 26/09)* : 4 % des épisodes non signalés, contestations
   abusives (« fraude amicale ») ; vérité terrain dans `ml/data/raw/label_noise.csv`.
   Paramètres dans `GeneratorConfig` : les mettre à 0 permet une analyse de sensibilité.
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
