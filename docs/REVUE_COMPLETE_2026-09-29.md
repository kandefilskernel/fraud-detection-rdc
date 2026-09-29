# Revue complète du projet — 29/09/2026

Périmètre : ~15 000 lignes de code (ML, serving, back-office, workers, frontend, infra, docs).
Méthode : lecture du code + reproduction partielle (le générateur, graine 42, redonne exactement
386 828 transactions ; un HistGradientBoosting sans réglage a été entraîné pour contrôle).
Aucun fichier du projet n'a été modifié. Les numéros de ligne se réfèrent à l'état du 29/09.

## 0. Verdict

Le projet est d'un niveau très supérieur à un mémoire de master moyen : split temporel propre,
features causales partagées entraînement/production, distinction étiquettes observées / vérité
terrain (bruit de signalement), bootstrap apparié par client, ablations, LOTO « fraude inconnue ».

Mais trois points peuvent être attaqués par un jury et doivent être corrigés ou assumés :

1. **Le jeu de test a servi à choisir le modèle final** (forêt + LSTM + AE) → les gains annoncés ne
   sont plus une confirmation indépendante.
2. **Les données sont presque parfaitement séparables** avec les vraies étiquettes (PR-AUC vérité
   0,97 MM) ; plusieurs artefacts du générateur trahissent la fraude.
3. **L'autoencodeur est inerte** (coef 0,017) et la règle de sélection déclarée désignerait la
   **forêt seule**, pas l'hybride — ce qui touche au cœur de la thèse (apport du profil séquentiel).

## 1. Priorités (ordre conseillé)

| # | Action | Zone | Effort |
|---|---|---|---|
| P1 | Refaire la sélection de modèle sur la validation seule ; test = confirmation unique (ou 2e test jamais vu) | ML | moyen |
| P2 | Ajouter « forêt seule » en 2e manche + bootstrap forêt+LSTM(+AE) vs forêt seule | ML | faible |
| P3 | Statuer sur l'AE : le retirer, ou l'utiliser comme alerte séparée, ou refaire le LOTO avec le pipeline complet | ML | faible |
| P4 | Seuil (et calibration) par canal pour Visa | ML | faible |
| P5 | Corriger 3 artefacts du générateur (appareil agent, solde, arnaque) puis ré-entraîner | Données | moyen |
| P6 | 5–10 graines, moyenne ± écart-type, correction Holm/BH des comparaisons | ML | calcul |
| P7 | Refuser au démarrage les secrets par défaut (JWT, HMAC, clé interne) hors mode démo | Sécurité | faible |
| P8 | Profils KYC des nouveaux clients (actuellement profil par défaut → âge de compte faux) | Serving | moyen |
| P9 | Workers : DLQ pour messages invalides, commit avant envoi SMS | Workers | moyen |
| P10 | Mettre à jour les chiffres des docs (plusieurs ne correspondent plus aux rapports) | Docs | faible |

## 2. Méthodologie ML (critique pour le mémoire)

**CRITIQUE — Test utilisé pour la sélection.** `ml/training/select_forest.py:5-8`, `select_model.py:235-247`,
`docs/SELECTION_MODELE.md:49-52`. La règle avait retenu « XGB + LSTM » sur la validation ; ce choix a été
rejeté à cause d'un bootstrap *sur le test* (Visa vérité −0,012), puis la 2e manche (forêt) a été conçue
parce qu'elle était « meilleure partout » *sur le test*. → Les +0,019 (global) et +0,069 (Visa) sont biaisés.
Correctif : sélection sur validation (y compris vérité validation si on suppose un échantillon vérifié),
puis un seul passage sur le test, ou nouveau test (période postérieure / autre graine).

**CRITIQUE — La règle déclarée n'est pas appliquée.** `select_forest.py:401-403` ne compare que 3 candidats.
Sur la validation : forêt seule 0,5089 > forêt+LSTM 0,4764 > forêt+LSTM+AE 0,4677. La règle B1/B6 choisirait
la forêt seule ; l'AE fait perdre 0,0087 > seuil d'égalité 0,005. Sur le test, la forêt seule (0,4809) bat
déjà l'ancien hybride XGB (0,4773). → Il faut soit montrer un apport significatif du LSTM, soit l'écrire
honnêtement (« le profil séquentiel n'apporte pas de gain mesurable sur ces données ») — c'est un résultat.

**MAJEUR — Autoencodeur inerte.** `ml/artifacts/metadata.json` : coefficients forêt 2,47 / LSTM 0,69 / AE 0,017.
Le LOTO « +3,3 points » est mesuré avec un méta-apprenant à 2 entrées sans LSTM : ne se transfère pas au modèle
déployé. `ARCHITECTURE_CIBLE.md:154` indique encore « ≈ 0,05 ».

**MAJEUR — Vérité terrain utilisée pour décider** (`select_model.py:215-216`). L'opérateur ne l'a pas.
Présenter C3 comme scénario « échantillon de validation vérifié », jamais avec la vérité du test.

**MAJEUR — Intervalles incomplets.** Une seule graine (`train_model.py:53`) ; plus de 15 tests « significatifs »
sans correction de comparaisons multiples. Les écarts entre modèles (~0,02) sont du même ordre que la variance
d'entraînement probable.

**MAJEUR — Forêt vs XGB non contrôlée.** Forêt entraînée sur train+early_stop (données plus récentes), XGB sur
fit seul (`train_model.py:153-158`). Le gain peut venir des données, pas de l'algorithme.

**MAJEUR — Visa évaluée avec le seuil global.** `train_model.py:78-83`. En vérité, au seuil global : précision 1,0,
FPR 0, rappel 0,316, alors que le rappel à 1 % FPR vaut 0,776. Le seuil est trop haut pour Visa ; l'écart avec
MM est en partie un artefact de seuil. `ARCHITECTURE_CIBLE.md:74` annonce pourtant des seuils par canal.

**MAJEUR — « Dégradation Visa corrigée » surinterprété** (`SELECTION_MODELE.md:21`) : 0,8126 vs 0,8435 sans
réputation → « partiellement compensée », IC large (n = 39 fraudes observées).

**MINEUR — Incohérences de traçabilité.** `schemas.py:15` annonce « TreeSHAP (XGBoost) » alors que c'est Saabas
sur forêt ; restes XGBoost dans `test_models.py:109`, `meta_learner.py:13`, `hybrid_ensemble.py:98`,
`ASSISTANT_ENQUETE.md:21,33` ; `adopt_model.py:236` recopie la config de l'ancien modèle ; latences du LSTM/AE
codées en dur (`select_forest.py:392,410`) et p99 additionnés.

**MINEUR — Réentraînement biaisé en faveur du champion** (`retrain.py:151-156`) : étiquettes issues des alertes
du champion, promotion sur +0,002 sans IC, ~9 fraudes dans la tranche d'évaluation.

Points vérifiés corrects : méta-apprenant ajusté sur la validation (blending), seuil jamais choisi sur le test,
`recall_at_fpr` correct, séquences LSTM causales. Préciser dans le mémoire que « PR-AUC » = average precision.

## 3. Données synthétiques

**CRITIQUE — Problème trop facile avec les vraies étiquettes.** PR-AUC vérité 0,974 (MM) ; un modèle sans réglage
atteint 0,97 MM / 0,92 Visa ; retirer n'importe quelle famille de features ne descend jamais sous 0,91. L'écart
0,50 ↔ 0,97 vient presque entièrement du bruit de signalement. → Faire l'analyse de sensibilité annoncée
(`HYPOTHESES_DONNEES.md` §5) avec des fraudes plus difficiles, ou l'assumer comme limite.

**MAJEUR — Artefact appareil agent.** `generate_synthetic_data.py:706` : 50 % des cash-out frauduleux utilisent
l'appareil de l'agent, jamais en légitime (0/45 168 vs 236/906). Ajouter 10–20 % de retraits légitimes assistés.

**MAJEUR — Tentatives au-delà du solde.** `:974-978` supprime 80 % des légitimes > solde mais garde toutes les
frauduleuses ; l'arnaque a un montant fixe (`:684`) alors que la victime connaît son solde → `amount_to_balance`
devient la variable la plus discriminante (AUC 0,74).

**MAJEUR — Labels du futur dans le train.** `is_fraud = 1` dès que l'épisode sera signalé un jour (`:1021`) ;
6,8 % des positifs du train ne sont signalés qu'après la coupure. Correctif : y = 1 seulement si
`fraud_reported_at` < date de coupure du split concerné (garder la vérité complète pour l'évaluation).

**MAJEUR — Réseau de mules incohérent.** Les comptes `is_mule_account` ne reçoivent jamais d'argent des victimes ;
les P2P ne sont pas en double écriture (pas de P2P_RECEIVE miroir) → `cp_is_customer` = 0 pour 100 % des fraudes,
la contribution « profilage bidirectionnel » (C1) est évaluée sur un graphe incomplet.

**MAJEUR — Fraude amicale (Visa).** 17 % des positifs Visa du train, impossibles à apprendre (`:917`). Publier
les métriques Visa avec et sans.

**MINEUR** — 116 fraudeurs présents en train et en test (ajouter une évaluation GroupKFold par utilisateur) ;
crédits (P2P_RECEIVE) étiquetés fraude ; un seul fuseau pour la RDC (UTC+1 et UTC+2 en réalité) ;
38,5 % d'épisodes signalés dans les docs vs 37,73 % mesurés ; tests anti-fuite faibles
(`test_features.py:212-216` ne vérifie que `notna()`).

### Pourquoi Visa est plus faible (PR-AUC 0,32 vs 0,50)
1. Moins d'étiquettes et plus bruitées : 26 % des fraudes Visa signalées vs 40 % MM ; fraude amicale.
2. Très peu d'exemples : 39 fraudes observées en test → IC ≈ ±0,1.
3. 28 des 76 features sont constantes sur Visa (agent, contrepartie, accès…).
4. Seuil global calé sur Mobile Money (9× plus de fraudes).
5. La réputation dessert Visa : rappel CARD_TESTING 0,53 → 0,11 (vérité).

Pistes : seuil et calibration par canal ; canal en entrée du méta-apprenant ; features carte (micro-achats par
carte/marchand sur 10 min, taux de refus, diversité marchands, pays/BIN) ; pondérer ou neutraliser la réputation
pour Visa ; sélection sur la validation Visa.

## 4. Scoring temps réel et intégration opérateurs

**CRITIQUE — Nouveaux clients mal profilés.** Seul `bulk_load` écrit `fs:profile`/`fs:wallet`
(`ml/serving/feature_store.py:268-314`). Tout client créé après le seed reçoit `DEFAULT_PROFILE` (Kinshasa, KYC 1,
plafond 100 USD) et un âge de compte calculé depuis `period_start` : un compte mule ouvert hier paraît avoir
600 jours. Correctif : flux de synchronisation KYC + date de première apparition.

**MAJEUR** —
- Statut « FAILED » différent entre entraînement (statut réel) et production (seulement si le modèle bloque)
  → `failed_count_24h` inutile contre le brute-force PIN (`scoring main.py:164,172`).
- Horodatage opérateur non borné : une date erronée dans le futur vide les fenêtres 24 h/7 j du client et de l'agent.
- Visa : STAN (6 chiffres) utilisé comme clé d'idempotence 48 h → collisions → 409 sur paiements légitimes.
- Repli quand le scoring est indisponible : perd la règle SIM swap ; délai HTTP (0,8 s) < attente du verrou (1 s) ;
  un 401/422 déclenche silencieusement le repli.
- Rechargement à chaud : `fs:seq` stocke des vecteurs déjà normalisés avec l'ancien scaler.
- Secrets `dev-*` acceptés dans tous les modes ; la clé scoring permet d'écrire dans la réputation
  (`/v1/reputation/report`) sans HMAC.

**MINEUR** — `idem.complete` synchrone hors `try` (double comptage possible) ; Kafka `acks=1` et offsets
auto-commités côté réputation ; numéros de téléphone normalisés différemment selon les chemins
(la règle « bénéficiaire signalé par SMS » ne se déclenche pas pour `0812…`) ; `X-Signature` non ASCII → 500 ;
`TRUSTED_PROXIES=172.16.0.0/12` trop large ; pas de CRL mTLS ; montants < 14,25 CDF arrondis à 0 → 422.

Points forts : même extracteur de features en entraînement et en production + tests de parité ; verrou par
client en Lua ; HMAC bien conçu ; pseudonymisation par HMAC à clé ; repli par défaut fermé (VERIFY).

## 5. Back-office, workers, frontend

**CRITIQUE — Secrets par défaut non refusés.** `backoffice-api/app/config.py:8` (`change-me-in-production`),
`:15` (mot de passe admin), `docker-compose.yml:153` (`dev-internal-key`, port 8003 publié) → jeton admin
forgeable si le déploiement garde la valeur par défaut.

**MAJEUR** —
- Chaîne d'audit : SHA-256 sans clé, tête non ancrée, table possédée par le rôle applicatif → réécriture
  complète possible (`DISABLE TRIGGER`) et troncature indétectable ; topic `audit.logs` sans ACL.
- Événements d'audit perdus si Kafka est indisponible (envoi après commit, sans vérification de livraison) → outbox.
- `alerter.py:93-127` : SMS envoyés avant le commit du lot → SMS en double lors d'un rejeu.
- `common.py:82-91` : un message malformé est réessayé à l'infini et bloque la partition → DLQ.
- `cases.py:111-157` : un analyste peut clore le dossier d'un autre et réécrire la note d'un dossier clos.
- `cases.py:74-100` : un seul analyste peut confirmer une fraude et marquer agent/marchand/appareil dans la
  réputation → validation par un superviseur.
- Pas de limite de débit sur `/auth/login` ni sur l'assistant LLM.

**MINEUR** — consultation de l'assistant en cache non auditée ; proxy Next.js en liste de refus (`/api/docs`,
`/api/metrics` accessibles sans authentification) → liste blanche ; vérification de l'audit sensible au fuseau
PostgreSQL ; paramètres `limit`/`size` négatifs → 500 ; pas d'index sur `details->>'event_id'` ;
JWT en `localStorage` 8 h.

Points forts : aucune injection SQL (ORM / `text()` paramétré) ; rôles relus en base à chaque requête ;
consommateurs idempotents ; RAG sans texte libre envoyé au LLM ; migration Alembic conforme aux modèles.

## 6. Infrastructure, exploitation, documentation

**CRITIQUE** —
- Scripts Windows (`scripts/seed_database.py`, tests back-office, retrain) : `shared/database/session.py:13`
  utilise le mot de passe par défaut `nuru_secure_password_2026` et ne lit pas `.env` → « password
  authentication failed » si `.env` a un autre mot de passe. Charger `.env` (python-dotenv).
- Clone neuf : les CSV sont ignorés par git ; le README présente l'étape « données » comme facultative
  « si `ml/data` est vide », or le dossier n'est jamais vide (JSON) → le seeder échoue et toute la chaîne tombe.
- Kubernetes : l'image `fraud-rdc/seed-data` n'est construite nulle part → le scoring ne démarre jamais.

**MAJEUR** —
- K8s : l'integration-layer ne reçoit pas les secrets HMAC (retombe sur `dev-*`), `INTERNAL_API_KEY` absent,
  la NetworkPolicy bloque integration-layer → backoffice:8003 (retours opérateurs coupés).
- Champion/challenger : promotion automatique sans humain, marge +0,002 sans IC, réessais toutes les 600 s
  tant que la dérive persiste (un challenger finit par gagner par chance) ; biais de sélection de la tranche
  d'évaluation (`retrain.py:86-92`).
- Version Python non imposée : `numpy==1.26.4`/`torch==2.2.1` sans wheels pour Python ≥ 3.13 → utiliser
  `py -3.12 -m venv venv` ; `ml/requirements.txt` non figé (`>=`).

**MINEUR** — MLflow (5000) exposé sans authentification même en prod ; `docker-compose.mtls.yml` monte la clé
privée de l'AC dans nginx ; détecteur de dérive : NaN non gérés par `ks_2samp`, fenêtres de 2 000 tx trop
courtes, le retrainer ignore le `for: 10m` de l'alerte ; pas d'Alertmanager ; `.gitignore`/.dockerignore ne
couvrent pas `ml/artifacts/candidates|archive` ni `mlflow.db` ; nom de projet compose non fixé (le réseau
`fraud-detection-rdc_default` dépend du nom du dossier) ; `ARCHITECTURE_CIBLE.md` parle d'un « registre MLflow »
alors que le registre réel est la table `model_registry`.

Points forts : images non-root, multi-étapes, versions alignées ; healthchecks cohérents ; K8s soigné
(probes, limits, HPA, PDB, deny-all) ; promotion atomique avec archive.

## 7. État du dépôt Git

- 82 fichiers modifiés non commités (~5 000 lignes, surtout frontend) : à committer.
- `.git/index.lock` vide laissé lors de la revue : à supprimer
  (`Remove-Item C:\Repo\fraud-detection-rdc\.git\index.lock`).
- Clés privées `infra/nginx/certs/*.key` et `.env` : non suivis par git (correct).

## 8. Chiffres de référence (test, 57 796 transactions, 382 fraudes observées / 1 014 réelles)

| Modèle | PR-AUC obs. | PR-AUC vérité | MM obs. / vérité | Visa obs. / vérité |
|---|---|---|---|---|
| Production forêt + LSTM + AE | 0,4862 | 0,9555 | 0,5048 / 0,9739 | 0,3247 / 0,8126 |
| Forêt seule | 0,4809 | 0,9435 | 0,4999 / 0,9619 | 0,3191 / 0,8185 |
| Ancien hybride XGB + LSTM + AE | 0,4773 | 0,9365 | 0,5015 / 0,9666 | 0,2615 / 0,7433 |
| Régression logistique | 0,405 | 0,857 | — | 0,099 / — |

Validation : forêt seule 0,5089 · forêt+LSTM 0,4764 · production 0,4677.
