# Hypothèses des données synthétiques : ce qui est sourcé, ce qui est supposé

Aucune donnée réelle d'opérateur n'est disponible : le jeu de données est généré par
[`ml/generator/generate_synthetic_data.py`](../ml/generator/generate_synthetic_data.py). Pour que les résultats
aient un sens, les paramètres qui conditionnent la détection doivent **ressembler à la réalité**.
Ce document les recense, avec leur source quand elle existe, ou leur statut d'hypothèse sinon.
Tous sont réglables dans `GeneratorConfig`.

> **Règle suivie** : ne jamais ajouter au générateur un signal fabriqué pour qu'une méthode
> réussisse (résultat circulaire). On règle des conditions réalistes, puis on mesure.

## 1. Ce que l'opérateur sait : signalement des fraudes

Le point le plus important. Un opérateur ne connaît pas toutes les fraudes : seulement celles
que les victimes signalent ou que ses équipes détectent, et seulement après un délai.

| Paramètre | Valeur | Justification | Statut |
|---|---|---|---|
| Part des épisodes signalés ou détectés (`report_prob`) | 38,5 % en moyenne : SIM swap 60 %, prise de contrôle 50 %, fraude carte 50 %, agent 35 %, mules 30 %, arnaque 25 %, test de carte 20 % | En Ouganda, environ 23 % des victimes de fraude signalent, 35,6 % après une campagne de sensibilisation ([IPA](https://poverty-action.org/mobile-financial-services-consumer-protection-and-dispute-resolution-uganda)) ; sous-signalement massif dans les pays moins riches ([enquête 12 pays](https://arxiv.org/pdf/2407.12896)). S'y ajoute la détection par l'opérateur lui-même | Niveau moyen **sourcé** ; répartition par type **supposée** (pertes visibles = plus signalées) |
| Effet du montant (`report_amount_effect`) | ±30 % autour de la base | Une grosse perte est plus souvent signalée | Hypothèse |
| Délai médian de signalement (`report_delay_median_days`) | SIM swap 1 j, arnaque et prise de contrôle 2 j, agent 5 j, carte 10-12 j, mules 21 j (enquête), fraude amicale 25 j | « La plupart des retours sont disponibles après une semaine », parfois des mois ([Le Borgne et al., manuel ULB](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)) | Ordre de grandeur **sourcé** ; valeurs par type supposées |
| Délai maximal (`report_delay_max_days`) | 120 jours | Délai de contestation carte des réseaux ([Chargebacks911](https://chargebacks911.com/chargeback-rules/chargeback-time-limits/)) | Sourcé |
| Fraude « amicale » (`friendly_fraud_rate`) | 0,3 % des achats carte | Contestations abusives par le titulaire | Hypothèse |

Résultat obtenu : 38,5 % des épisodes connus, délai médian 2,6 jours, 90 % en moins de 24 jours.

**Conséquences pour l'évaluation** : on mesure les modèles de deux façons.
- **Étiquettes connues de l'opérateur :** ce qu'il pourrait mesurer lui-même.
- **Vérité terrain,** conservée dans `label_noise.csv` : ce que le modèle détecte réellement, y compris des fraudes que personne n'a signalées. Un modèle qui les attrape est pénalisé à tort par la première mesure.

## 2. Réseaux de fraude : réutilisation puis abandon des entités

| Paramètre | Valeur | Justification | Statut |
|---|---|---|---|
| Parc de comptes mules (`mule_pool_size`) | 400 | Réglé pour obtenir en moyenne 2 victimes par mule, jusqu'à 11 ; des comptes mules sont « utilisés plusieurs fois, pour différents types de fraude » ([FCA](https://www.fca.org.uk/publications/multi-firm-reviews/money-mules-activity-cashing-out-findings)) | Réutilisation **sourcée** (qualitativement) |
| Durée de vie d'une mule (`mule_lifespan_days`) | 14 jours (log-normale) | Gel après signalement ou abandon par le réseau | Hypothèse |
| Mules neuves à usage unique (`fresh_mule_rate`) | 20 % | — | Hypothèse |
| Appareils des fraudeurs : parc et durée de vie | 400 appareils, 30 jours ; 45 % d'appareils neufs | Rotation des téléphones et émulateurs par les réseaux | Hypothèse |
| Agent compromis : durée de la fraude (`agent_compromise_days`) | 21 jours, puis suspension ou départ | — | Hypothèse |

## 3. Comportements légitimes qui ressemblent à de la fraude

Sans eux, un seul signal (« nouvel appareil », « IP étrangère ») suffisait à tout détecter : les
scores dépassaient 0,99, ce qui n'est pas crédible.

| Paramètre | Valeur | Justification | Statut |
|---|---|---|---|
| Téléphones familiaux partagés | 15 % des comptes, 2 à 4 comptes par téléphone | Usage courant en RDC | Hypothèse plausible |
| Téléphone emprunté | 2 % des opérations | Idem | Hypothèse |
| Changement de téléphone sur 6 mois | 15 % des clients | Perte, casse, remplacement | Hypothèse |
| Achat carte depuis un nouveau navigateur / cybercafé | 10 % | — | Hypothèse |
| IP étrangère légitime (VPN, voyage) | 10 % des achats carte | — | Hypothèse |
| Dépôt initié depuis la ligne de l'agent | tous les dépôts | Fonctionnement réel du cash-in | Plausible |
| Likelemba (tontine) : cagnotte reçue puis retirée | 25 % des clients | Pratique courante en RDC | Hypothèse plausible |
| Retrait de tout le solde (transfert de la diaspora) | 6 % des retraits | — | Hypothèse |
| Micro-achats de jeux en série | 15 % des titulaires de carte | Ressemble au test de carte | Hypothèse |

## 4. Fraudes discrètes

| Paramètre | Valeur | Statut |
|---|---|---|
| Prise de contrôle à distance du téléphone de la victime (`fraud_victim_device_rate`) | 35 % | Hypothèse (arnaques à l'application d'assistance à distance, documentées qualitativement) |
| Fraude carte via proxy local, IP congolaise (`fraud_local_proxy_rate`) | 45 % | Hypothèse |
| Préférence des fraudeurs pour la nuit (`fraud_night_bias_scale`) | atténuée de moitié | Hypothèse |

## 4 bis. Archive de l'assistant d'enquête (RAG)

| Paramètre | Valeur | Statut |
|---|---|---|
| Part des transactions alertées puis instruites (`--alert-rate`) | 3 % par canal | Hypothèse, proche du taux d'alerte observé sur la plateforme |
| Fraudes alertées classées à tort sans suite (`--miss-rate`) | 10 % | Hypothèse (client injoignable, complice, instruction trop tardive) |
| Issue des alertes instruites | vérité terrain, sauf les 10 % ci-dessus | Hypothèse : l'appel au client lève le doute, contrairement aux fraudes jamais alertées qui ne sont connues que par plainte |
| Plaintes de complaisance (`FRIENDLY_FRAUD`) | classées sans suite | Hypothèse |

## 5. Ce qui manque et comment le justifier devant le jury

- **Validation :** tous les résultats portent sur des données synthétiques. La seule validation définitive est un **pilote en mode silencieux** chez un opérateur (voir `docs/INTEGRATION_OPERATEURS.md`).
- **Analyse de sensibilité recommandée** pour les paramètres « hypothèse » qui influencent le plus les conclusions :
  - `report_prob` (15 % à 60 %) : le profil de réputation dépend directement du nombre de fraudes connues ;
  - `mule_pool_size` et `mule_lifespan_days` (réutilisation des mules) : c'est la condition d'utilité du profilage réseau (C1) et de la réputation ;
  - `fraud_victim_device_rate` : difficulté des prises de contrôle.

  Procédure : régénérer avec une valeur différente, relancer le pipeline, puis comparer avec `ml/training/compare_runs.py`.
- **Transparence :** les valeurs utilisées sont enregistrées dans `ml/data/raw/generation_metadata.json` (section `config`) et reproductibles (graine fixe).
