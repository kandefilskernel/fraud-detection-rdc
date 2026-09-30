# Données réelles des opérateurs : du fichier reçu au modèle en production

Ce guide sert à deux choses :

1. **Le cahier des charges** à remettre à un opérateur (Vodacom, Airtel, Orange, émetteur Visa) :
   quelles données, dans quel format, avec quelles précautions.
2. **La procédure** pour les intégrer : import, contrôle qualité, apprentissage, pilote silencieux,
   puis mise en production.

Le module est `ml/onboarding`. Toutes les données réelles restent dans `ml/data/workspaces/<nom>/`,
ignoré par git : elles ne doivent jamais être committées ni envoyées ailleurs.

---

## 1. Cahier des charges des données (à remettre à l'opérateur)

### 1.1 Export des transactions (obligatoire)

- **Toutes** les opérations de la période, pas seulement les suspectes : le modèle apprend ce
  qu'est un comportement normal.
- **Période** : 6 mois ou plus recommandés (minimum technique : 30 jours ; en dessous de 90 jours,
  les habitudes mensuelles et les fins de mois sont mal apprises).
- **Volume** : 100 000 transactions ou plus recommandées (minimum : 10 000). Un échantillon de
  clients complets (tout l'historique de chaque client retenu) vaut mieux qu'un échantillon
  aléatoire de transactions.
- **Format** : CSV UTF-8 (ou Parquet, ou Excel), une ligne par transaction.

| Champ | Obligatoire | Pourquoi |
|---|---|---|
| Identifiant de la transaction | oui | rapprochement avec les signalements de fraude |
| Date et heure (avec le fuseau utilisé) | oui | profil horaire, vélocité, découpage temporel |
| Numéro du titulaire (MSISDN) | oui | profil comportemental (pseudonymisé à l'import) |
| Type d'opération (envoi, réception, dépôt, retrait, paiement...) | oui | règles et variables par type |
| Montant et devise (CDF / USD) | oui | écart au montant habituel |
| Statut (réussi / échoué) | recommandé | tentatives échouées répétées |
| Numéro de la contrepartie (transferts) | recommandé | comptes mules, nouveaux bénéficiaires |
| Code agent (dépôts, retraits) | recommandé | agents à risque, retraits après SIM swap |
| Code marchand, catégorie | recommandé | paiements marchands |
| Identifiant d'appareil (IMEI, identifiant d'application) | recommandé | changement d'appareil, SIM swap |
| Canal d'accès (application, USSD, agent) | recommandé | changement de canal |
| Province ou région (cellule, agent) | recommandé | opérations loin du domicile |
| Solde avant l'opération | recommandé | vidage du compte |

Un champ absent n'empêche pas l'import : la famille de variables correspondante est
**neutralisée**, et le rapport qualité le signale.

### 1.2 Fraudes signalées (obligatoire pour entraîner)

Sans exemples de fraudes, le modèle ne peut pas apprendre à les reconnaître.

- **Contenu** : fraudes confirmées, plaintes de clients, contestations, fraudes découvertes par
  les équipes de l'opérateur.
- **Colonnes** : identifiant de la transaction frauduleuse, **date du signalement**, typologie si
  elle est connue (SIM swap, ingénierie sociale, prise de contrôle, mule...).
- **Délai d'extraction** : extraire ce fichier au moins **30 jours après la fin de la période**.
  Sinon les dernières semaines sont sous-étiquetées : des fraudes n'ont pas encore été signalées
  et passent pour légitimes.
- **Date de signalement** : elle compte. Une transaction ne doit « voir » que les fraudes déjà
  connues au moment où elle a lieu. Faute de date, un délai par défaut est appliqué
  (`labels.default_report_delay_days`).

### 1.3 Référentiel clients KYC (recommandé)

Province de résidence, niveau KYC, plafond de transaction, date d'ouverture du compte. Sans ce
référentiel, un profil prudent par défaut est appliqué.

### 1.4 À ne PAS transmettre

Noms, numéros de pièce d'identité, adresses, codes PIN, mots de passe, numéros de carte complets.
S'ils sont présents, l'import les ignore : seules les colonnes déclarées dans le fichier de
correspondance sont lues. Mieux vaut qu'ils ne quittent jamais les systèmes de l'opérateur.

### 1.5 Transfert et cadre légal

Point à faire valider par un juriste congolais : ce guide n'est pas un avis juridique.

- **Convention signée** avec l'opérateur avant tout transfert : finalité (détection de la
  fraude), durée de conservation, sécurité, sous-traitance, suppression en fin de contrat.
- **Conformité** au Code du numérique (protection des données personnelles : déclaration du
  traitement, délégué à la protection des données) et aux instructions de la BCC applicables à
  l'opérateur.
- **Transfert chiffré** : SFTP, ou archive chiffrée dont la clé est transmise par un autre canal.
  Jamais par e-mail ni par clé USB non chiffrée.
- **Hébergement en RDC** recommandé pour la production (datacenter de Kinshasa).
- **Pseudonymisation** : la clé `PSEUDONYMIZATION_KEY` peut rester chez l'opérateur, ou chez un
  tiers de confiance. Sans elle, les pseudonymes ne permettent pas de retrouver un numéro.

---

## 2. Procédure (PowerShell, à la racine du projet)

### 2.1 Une fois : la clé de pseudonymisation

```powershell
python scripts/security/generate_secrets.py   # ajoute PSEUDONYMIZATION_KEY dans .env si absente
```

Dans `.env`, mettre aussi `PSEUDONYMIZE_IDS=true` : le trafic temps réel doit être pseudonymisé
avec la **même clé**, sinon tous les clients apparaissent inconnus. Cette clé n'est jamais changée
par `--rotate`. La changer impose de tout réimporter.

### 2.2 Le fichier de correspondance

Copier `ml/onboarding/mappings/modele_operateur.yaml` en `vodacom.yaml`, puis y indiquer :

- les noms de colonnes de l'export ;
- les libellés de l'opérateur, par exemple « Retrait » → `CASH_OUT` ;
- le format des dates, le fuseau, le séparateur, la virgule décimale et le taux CDF/USD.

Une erreur dans ce fichier est expliquée en clair dès le chargement.

### 2.3 Import, qualité, apprentissage

```powershell
# un import par export (plusieurs opérateurs = même espace de travail)
python -m ml.onboarding.onboard import --workspace pilote_2026 --mapping ml/onboarding/mappings/vodacom.yaml `
       --input D:\exports\vodacom_tx.csv --frauds D:\exports\vodacom_fraudes.csv --kyc D:\exports\vodacom_kyc.csv
python -m ml.onboarding.onboard import --workspace pilote_2026 --mapping ml/onboarding/mappings/airtel.yaml `
       --input D:\exports\airtel_tx.csv --frauds D:\exports\airtel_fraudes.csv

python -m ml.onboarding.onboard build --workspace pilote_2026   # fusion + rapport_qualite.md
python -m ml.onboarding.onboard train --workspace pilote_2026   # --quick pour un premier essai
```

- **Essai rapide sur un gros export** : `--sample-users 20000` à l'import garde 20 000 clients
  avec tout leur historique. On peut aussi utiliser `--since 2026-01-01`.
- **Plusieurs opérateurs** : un même numéro est reconnu d'un opérateur à l'autre, puisque la clé
  de pseudonymisation est la même. Le réseau des contreparties (comptes mules) est alors vu en
  entier.

### 2.4 Lire les rapports

**`ml/data/workspaces/<nom>/rapport_qualite.md`**

- **Erreurs bloquantes**, qui refusent l'entraînement :
  - volume ou période trop courts ;
  - aucune fraude, ou moins de 20 fraudes dans les tranches de validation et de test ;
  - plus de 10 % des lignes écartées ;
  - identifiants non pseudonymisés ;
  - taux de fraude invraisemblable.
- **Avertissements** : variables neutralisées, valeurs non reconnues (à ajouter dans `values`),
  signalements tardifs, faible couverture KYC.

**`reports/comparaison.md`** : le nouveau modèle et le modèle en production, sur la **même période
de test**, la plus récente et jamais vue à l'entraînement.

| Indicateur | Ce qu'il mesure |
|---|---|
| PR-AUC | qualité globale |
| Rappel | part des fraudes détectées |
| Précision | part des alertes justes |
| Alertes par jour | charge des analystes : à valider avec l'opérateur |

---

## 3. Mise en production progressive

```powershell
python -m ml.onboarding.onboard promote --workspace pilote_2026 --yes
docker compose run --rm feature-store-seeder python -m ml.serving.seed_feature_store `
       --raw-dir ml/data/workspaces/pilote_2026/raw --until all
```

- **Promotion** : l'ancien modèle est archivé dans `ml/artifacts/archive/`. Le service de scoring
  recharge le nouveau à chaud.
- **Réamorçage** : il remplace les profils de démonstration par l'historique réel des clients.
  Il est obligatoire, car le scaler (normalisation des variables) a changé.

### 3.1 Pilote silencieux (4 à 8 semaines recommandées)

Dans `.env` : `SHADOW_MODE_OPERATORS=vodacom` (ou `vodacom,airtel`, ou `all`), puis :

```powershell
docker compose up -d integration-layer worker-alerter
```

Pour ces opérateurs :

- **Côté opérateur** : il reçoit **toujours APPROVE**. La vraie décision est renvoyée dans
  `shadow_action`, et elle est enregistrée dans la base et le tableau de bord.
- **Côté analystes** : les dossiers sont ouverts, ils instruisent les alertes comme en production.
- **Côté clients** : **aucun SMS** ne leur est envoyé, et aucun webhook ne part vers l'opérateur.

À mesurer avec l'opérateur :

- fraudes réelles détectées ;
- clients honnêtes qui auraient été bloqués ;
- charge d'alertes ;
- latence.

Les critères de bascule sont à fixer par écrit avant le pilote, par exemple « moins de 1 client
honnête bloqué pour 10 000 transactions ».

### 3.2 Bascule

- **Retirer l'opérateur** de `SHADOW_MODE_OPERATORS` : ses décisions sont désormais appliquées.
- **Commencer prudemment** : blocage seulement au niveau CRITIQUE, et vérification pour les autres
  niveaux.

### 3.3 Apprentissage continu

- **Étiquettes** : les verdicts des analystes, les retours des opérateurs (`/v1/feedback`) et les
  plaintes alimentent le réentraînement champion/challenger (`ml/retraining`).
- **Promotion automatique** : un challenger n'est promu que s'il bat le champion sur la période la
  plus récente.
- **Historique utilisé** : après une promotion depuis un espace de travail, le réentraînement
  repart de l'historique réel (`training_data` dans `metadata.json`).

---

## 4. Cohérence avec le temps réel (important)

Le modèle doit recevoir en temps réel des données préparées **exactement** comme à l'entraînement.

- **Fuseau** : l'heure utilisée est celle de `timezone.target` (Kinshasa par défaut). Les
  adaptateurs temps réel doivent fournir la même heure locale.
- **Pseudonymisation** : même clé à l'import et en temps réel (`PSEUDONYMIZE_IDS=true`).
- **Champs absents de l'historique** : si le solde ou l'appareil manquent dans l'export, ils
  doivent aussi manquer, ou être neutralisés de la même façon, dans le flux temps réel de cet
  opérateur. Conventions : solde inconnu = 1e9 USD, appareil inconnu = « UNK-<client> ».
- **Adaptateurs** : ceux de `services/integration-layer/app/adapters/` suivent des formats
  simulés. Ils sont à adapter au format réel de l'API de chaque opérateur au moment du
  raccordement.

## 5. Tester sans données réelles

```powershell
python scripts/demo_export_operateur.py --operator VODACOM --users 600
$env:PSEUDONYMIZATION_KEY="cle-de-test-uniquement-0123456789"
python -m ml.onboarding.onboard import --workspace demo_vodacom --mapping ml/onboarding/mappings/demo_vodacom.yaml `
       --input data/exports_operateurs/demo_vodacom/transactions.csv `
       --frauds data/exports_operateurs/demo_vodacom/fraudes.csv --kyc data/exports_operateurs/demo_vodacom/kyc.csv
python -m ml.onboarding.onboard build --workspace demo_vodacom
python -m ml.onboarding.onboard train --workspace demo_vodacom --quick
```

Le faux export imite les défauts d'un vrai fichier : dates en UTC, virgule décimale, numéros au
format +243, libellés en français, colonnes personnelles à ignorer. Les tests automatiques sont
dans `ml/tests/test_onboarding.py`.
