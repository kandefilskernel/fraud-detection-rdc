# Intégration avec les opérateurs (Vodacom, Airtel, Orange, émetteur Visa)

Ce document décrit comment les systèmes des opérateurs communiquent avec la plateforme pour
détecter la fraude en temps réel : contrat d'API, sécurité, idempotence, retours d'information,
signaux télécom, carte Visa côté émetteur et cadre légal. Il sert de guide d'intégration et de
base pour le chapitre « Architecture d'intégration » du mémoire.

> Les formats de messages des opérateurs sont **simulés** (inspirés de leurs API publiques).
> Une intégration réelle exige un partenariat avec chaque opérateur.

## 1. Vue d'ensemble

```
Système de l'opérateur                         Plateforme
────────────────────                         ──────────
AVANT d'exécuter une opération :
POST /ingest/v1/transactions/{opérateur}  ─►  nginx (mTLS, limite de débit)
  X-API-Key, X-Timestamp, X-Signature         └► integration-layer
                                                  sécurité → adaptateur → pseudonymisation
                                                  └► scoring-service (idempotent)
                                                      profil Redis + modèle + règles
◄── {"action": "VERIFY", "verification_method": "PIN_USSD", ...}   (~20 ms)
exécute / fait vérifier / refuse

PLUS TARD (enquête, plainte, contestation) :
POST /ingest/v1/feedback/{opérateur}      ─►  integration-layer ─► back-office (API interne)
                                                  étiquette de la transaction → réentraînement
```

Deux modes d'intégration existent ; la plateforme implémente le premier, le second se branche
sur la même logique :

| Mode | Principe | Usage recommandé |
|---|---|---|
| En ligne (synchrone) | L'opérateur appelle l'API avant d'exécuter | Opérations à risque : retraits, gros transferts, recharges de carte |
| Quasi temps réel (asynchrone) | L'opérateur envoie un flux après exécution (Kafka) | Démarrage : pilote en **mode silencieux** (on note sans bloquer) |

## 2. Contrat d'API

### Envoyer une transaction
`POST /ingest/v1/transactions/{vodacom|airtel|orange|visa}`, corps au format propriétaire de
l'opérateur (voir `services/integration-layer/app/adapters/`).

| En-tête | Contenu |
|---|---|
| `X-API-Key` | Clé d'API propre à l'opérateur |
| `X-Timestamp` | Horodatage Unix (secondes) au moment de l'envoi |
| `X-Signature` | Signature HMAC-SHA256 (ci-dessous), en hexadécimal |

Réponse (extrait) :

```json
{"transaction_id": "TX123", "action": "VERIFY", "risk_level": "ELEVE", "fraud_probability": 0.41,
 "verification_method": "HORS_SIM", "rules_triggered": ["SIM_RECENTE_OPERATION_SORTANTE"],
 "idempotent_replay": false, "iso8583_response_code": "1A", "end_to_end_ms": 21.4}
```

`verification_method` indique COMMENT vérifier quand `action = VERIFY` : `PIN_USSD` (ressaisie du
PIN), `3DS` (carte), `HORS_SIM` (agence avec pièce d'identité ou numéro secondaire : après un
changement de SIM récent, un code envoyé par SMS arriverait chez le fraudeur).
`iso8583_response_code` n'est présent que pour Visa (section 7).

### Signer une requête
Implémentation de référence : [`shared/security/request_signing.py`](../shared/security/request_signing.py).

```
chaîne canonique = X-Timestamp + "\n" + "POST" + "\n" + chemin + "\n" + SHA256_hex(corps)
X-Signature      = HMAC_SHA256(secret_opérateur, chaîne canonique)  en hexadécimal
```

Le **chemin signé est `/v1/...`**, sans le préfixe `/ingest` de la passerelle. Le secret HMAC est
distinct de la clé d'API. Une requête est refusée si son horodatage s'écarte de plus de 5 minutes
de l'horloge du serveur (anti-rejeu) : les serveurs des opérateurs doivent être synchronisés (NTP).

```python
import json, time, hmac, hashlib, httpx
body = json.dumps(message).encode()
path = "/v1/transactions/vodacom"
ts = str(int(time.time()))
canonical = "\n".join([ts, "POST", path, hashlib.sha256(body).hexdigest()]).encode()
sig = hmac.new(SECRET.encode(), canonical, hashlib.sha256).hexdigest()
httpx.post("https://plateforme/ingest" + path, content=body,
           headers={"X-API-Key": KEY, "X-Timestamp": ts, "X-Signature": sig, "Content-Type": "application/json"})
```

### Codes d'erreur

| Code | Cause | Action de l'opérateur |
|---|---|---|
| 401 | Clé d'API invalide, signature absente, invalide ou expirée | Vérifier clé, secret, horloge |
| 403 | IP non autorisée, certificat client absent ou d'un autre opérateur | Vérifier le réseau / le certificat |
| 404 | Opérateur inconnu ou non accepté par cette instance | — |
| 409 | Même `transaction_id` avec un contenu différent, ou envoi simultané en cours | Ne jamais réutiliser un identifiant ; réessayer plus tard si « en cours » |
| 422 | Message mal formé (champ manquant, type inconnu) | Corriger le message |
| 200 + `"risk_level": "INCONNU"` | Scoring indisponible : politique de repli (`FAIL_OPEN`) | Appliquer l'action renvoyée |

## 3. Sécurité en couches

| Couche | Mécanisme | Où |
|---|---|---|
| Réseau | VPN site-à-site (IPsec, WireGuard) ou lien dédié : l'API n'est pas joignable depuis Internet | Infrastructure (hors code) |
| Transport | **mTLS** : chaque opérateur a un certificat client ; nginx le vérifie, son nom (CN) doit correspondre à l'opérateur | `infra/nginx/mtls/`, `docker-compose.mtls.yml` |
| Origine | Liste d'adresses IP autorisées par opérateur (`OPERATOR_IP_ALLOWLIST`) | `app/security.py` |
| Identité | Clé d'API par opérateur (`OPERATOR_API_KEYS`) | `app/security.py` |
| Message | Signature HMAC + horodatage (`OPERATOR_HMAC_SECRETS`, `REQUIRE_SIGNATURE`) | `shared/security/request_signing.py` |

Activer le mTLS (certificats de TEST) :

```powershell
python scripts/security/generate_dev_pki.py
docker compose -f docker-compose.yml -f docker-compose.mtls.yml up -d
python scripts/simulate_transactions.py --url https://localhost/ingest --mtls-dir infra/nginx/certs
```

En production :
- **Certificats.** Une vraie autorité de certification (PKI). Chaque opérateur génère sa clé chez lui et n'envoie que sa demande de signature (CSR). Durées de validité courtes et liste de révocation.
- **Secrets.** Les secrets HMAC et les clés d'API sont stockés dans un coffre (secrets Kubernetes, Vault), jamais dans le code.
- **Proxy de confiance.** `TRUSTED_PROXIES` ne contient que le reverse proxy : les en-têtes `X-Real-IP` et `X-Client-*` ne sont crus que venant de lui.
- **Port direct fermé.** Le port 8002 de l'integration-layer n'est jamais publié.

**Limite connue** : un seul secret HMAC par opérateur. Faire tourner un secret demande un changement coordonné ; accepter deux secrets pendant la transition est une amélioration à prévoir.

## 4. Idempotence

Un opérateur qui n'a pas reçu de réponse (coupure, délai dépassé) **renvoie la même transaction**.
La plateforme garde chaque décision 48 h (Redis, clé `idem:{canal}:{opérateur}:{transaction_id}`) :

- **Même contenu :** la décision d'origine est renvoyée (`"idempotent_replay": true`). Le profil du client n'est **pas** recompté, et rien n'est republié vers la base ou les alertes.
- **Même identifiant, contenu différent :** refus (409). C'est une réutilisation d'identifiant ou un message falsifié.
- **Deux envois simultanés :** le second attend la fin du premier (jusqu'à 1 s), puis reçoit sa décision.
- **Échec du scoring :** la clé est libérée, et l'opérateur peut renvoyer.

Démonstration : `python scripts/simulate_transactions.py --duplicate-rate 0.1` (compteur
`renvoi_deja_score`, métrique Prometheus `scoring_idempotent_replays_total`).

## 5. Retours des opérateurs

`POST /ingest/v1/feedback/{opérateur}` (même sécurité, chemin signé `/v1/feedback/{opérateur}`) :

```json
{"transaction_id": "TX123", "outcome": "CHARGEBACK", "reference": "AIRTEL-2025-0042",
 "comment": "le client ne reconnaît pas l'opération"}
```

| `outcome` | Étiquette | Effet |
|---|---|---|
| `FRAUD_CONFIRMED` (enquête), `CUSTOMER_COMPLAINT` (plainte), `CHARGEBACK` (contestation carte) | 1 | Dossier ouvert ou fermé en FRAUDE_CONFIRMEE |
| `LEGITIMATE` (le client confirme être à l'origine) | 0 | Dossier ouvert fermé en FAUX_POSITIF |

Règles de traitement :
- **Périmètre :** un opérateur ne peut étiqueter que ses propres transactions (403 sinon).
- **Primauté de la fraude :** « légitime » ne remplace jamais une fraude déjà confirmée par un analyste ou une plainte. La plateforme renvoie un conflit (409) et l'arbitrage se fait à la main.
- **Transaction pas encore enregistrée (404) :** renvoyer le retour plus tard.
- **Répétition :** un retour répété ne change rien.
- **Traçabilité :** chaque changement est écrit dans le journal d'audit chaîné (acteur `operateur:{nom}`).

Les étiquettes alimentent **sans modification** le réentraînement champion/challenger
(`ml/retraining/retrain.py`). Les retours sur les fraudes **laissées passer** sont essentiels : sans eux, le modèle
n'apprendrait que de ses propres alertes.

## 6. Signaux télécom : le changement de SIM

Les opérateurs Mobile Money sont aussi opérateurs télécom : ils connaissent la date du dernier
changement de carte SIM (registre IMSI / HLR). Champ optionnel dans chaque format :

| Opérateur | Champ | Format |
|---|---|---|
| Vodacom | `LastSimSwapTime` | `AAAAMMJJhhmmss` |
| Airtel | `subscriber.last_sim_swap` | ISO 8601 |
| Orange | `date_dernier_changement_sim` | `JJ/MM/AAAA HH:MM:SS` |

Règles (période de refroidissement, `ml/serving/decision.py`) :
- **Opération sortante moins de 24 h après un changement de SIM :** au minimum VERIFY, avec la méthode `HORS_SIM`.
- **Même situation, avec vidage du compte ou montant supérieur à 50 % du plafond KYC :** BLOCK.

**Ce signal passe par les règles, pas par le modèle.** Il est absent des données d'entraînement synthétiques : l'y ajouter
puis montrer que le modèle l'utilise serait un résultat circulaire. Son poids réel sera appris
pendant le pilote, sur de vraies données.

## 7. Carte Visa côté émetteur

Pendant une autorisation, Visa transmet la demande (ISO 8583, MTI `0100`) à l'**émetteur** de la
carte ; le processeur de l'émetteur interroge alors la plateforme. Le message transmet aussi :

- `network.risk_score` : le score de risque du réseau (ex. Visa Advanced Authorization, 0-99). La plateforme le **complète** sans le remplacer : ≥ 90 → au minimum VERIFY, ≥ 98 → BLOCK (seuils réglables) ;
- `network.three_ds_authenticated` : l'achat a déjà été authentifié par 3-D Secure.

Réponse traduite en code ISO 8583 (champ 39) : `00` approuvée, `1A` authentification
supplémentaire requise (déclenche 3-D Secure), `59` fraude suspectée. Le délai de réponse
(`SCORING_TIMEOUT_S`, 0,8 s) doit rester dans le budget d'autorisation de l'émetteur ; au-delà,
la politique de repli s'applique.

## 8. Cadre légal et modes de déploiement

Les données de transactions sont des données personnelles et couvertes par le secret
professionnel des établissements de monnaie électronique. Avant tout déploiement réel, une
validation juridique est indispensable. Points à vérifier avec un juriste :
- **La loi applicable :** le Code du numérique de 2023 pour la protection des données personnelles, et les instructions de la BCC sur la monnaie électronique ;
- **La base légale** du traitement ;
- **Une analyse d'impact** (DPIA) ;
- **La durée de conservation** des données ;
- **Le lieu d'hébergement**, qui peut devoir rester en RDC ;
- **Un accord de partage de données** entre opérateurs.

La plateforme prévoit deux architectures (`DEPLOYMENT_MODE`) :

| Mode | Principe | Profilage multi-opérateurs (C1) |
|---|---|---|
| `operator_instance` | Chaque opérateur héberge sa propre instance, chez lui ; seul `OPERATOR_ID` est accepté | Limité à ses propres clients |
| `shared_platform` | Un organisme commun (plateforme d'interopérabilité, par exemple) mutualise les données ; **pseudonymisation et signature obligatoires**, sinon refus de démarrer | Complet : c'est la condition pour voir une mule payée depuis plusieurs réseaux |
| `demo` | Environnement local : tous les opérateurs, identifiants en clair | — |

**Pseudonymisation** (`PSEUDONYMIZE_IDS`, `PSEUDONYMIZATION_KEY`,
[`shared/privacy/pseudonymize.py`](../shared/privacy/pseudonymize.py)) :
- **Ce qui est remplacé :** l'identifiant client, le numéro de téléphone, la carte et l'IMEI deviennent un pseudonyme HMAC, déterministe mais non réversible sans la clé. Les identifiants d'agents et de marchands (des entreprises) restent en clair.
- **Aucun effet sur le modèle :** il ne lit jamais la valeur d'un identifiant, seulement « est-ce le même ? ». Les variables sont identiques, comme le vérifie le test `test_pseudonymization_does_not_change_any_feature`.
- **Ré-identification :** la plateforme mutualisée ne voit aucun numéro en clair. Seul le détenteur de la clé peut ré-identifier un client après une alerte, par exemple pour l'appeler.
- **Cohérence obligatoire :** la même clé doit servir à l'amorçage (`seed_feature_store`) et au trafic temps réel. Changer de clé oblige à ré-amorcer.

## 9. Déploiement progressif recommandé

1. **Pilote silencieux.** Flux asynchrone d'un opérateur, décisions calculées mais jamais appliquées. On mesure la précision et le rappel sur de vraies données.
2. **Mode en ligne sur les opérations à risque**, avec la politique de repli `FAIL_OPEN=true` (continuité de service).
3. **Extension à toutes les opérations**, puis retours des opérateurs branchés pour l'apprentissage continu.
4. **Plateforme mutualisée**, une fois le cadre légal validé.

## 10. Vérifier

```powershell
.\scripts\run_tests.ps1        # 5 suites (dont services/integration-layer/tests/test_security_api.py)
python scripts/simulate_transactions.py --url http://localhost/ingest --duplicate-rate 0.1
```
