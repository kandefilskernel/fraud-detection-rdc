# Choix du modèle de production : selon les besoins, pas selon une préférence

État au 28/09/2026, données « réalistes v3 ». Scripts : `ml/training/select_model.py` (1ʳᵉ manche),
`ml/training/select_forest.py` (2ᵉ manche), `ml/training/adopt_model.py` (mise en production).
Rapports : `ml/reports/selection_modele.json`, `ml/reports/selection_foret.json`.

## Statut méthodologique (mis à jour le 29/09/2026)

Les deux premières manches ci-dessous ont **utilisé le test pour décider** : le choix de la règle
(XGB + LSTM) a été rejeté sur un bootstrap du test, puis la 2ᵉ manche a été conçue parce que la
forêt + LSTM était meilleure sur le test. De plus, la règle B1/B6 n'a pas été réappliquée en 2ᵉ
manche : la forêt seule, devenue éligible grâce à la forêt aplatie, était première sur la
validation (0,509 contre 0,476 et 0,468). Leurs écarts « test » sont donc **exploratoires** : ils
ne constituent pas une confirmation indépendante.

**Décision finale : 3ᵉ manche, protocole pré-enregistré** (`ml/training/select_final.py`) :

- la règle est écrite dans le code avant tout calcul et recopiée dans le rapport ;
- décision sur la validation seule : B3/B4 sur la chaîne complète (prédiction + explication)
  mesurée dans les mêmes conditions pour tous, B2 (Visa) testé sur la validation, B1 puis B6 ;
- critère principal : étiquettes observées ; variante déclarée : vérité terrain de la validation
  (scénario « échantillon vérifié ») ;
- comparaisons déclarées, sur la validation puis sur le test, avec correction de Holm :
  forêt + LSTM − forêt seule (apport du profil séquentiel), forêt + LSTM + AE − forêt + LSTM
  (apport de l'autoencodeur empilé), modèle choisi − modèle en production ;
- le test est lu une seule fois, après la décision ;
- l'autoencodeur sort du méta-apprenant (coefficient 0,017, soit un effet quasi nul) et devient une
  **veille des anomalies** séparée : seuil au quantile 99,5 % des transactions de validation
  non signalées, sans effet sur la décision, signal pour les analystes.

Résultats : `ml/reports/selection_finale.md` (généré par le script). Mise en production :
commande `adopt_model` proposée à la fin du rapport, lancée manuellement après relecture.

## Résultat des manches exploratoires (1 et 2)

**Modèle en production : forêt aléatoire + LSTM-attention + autoencodeur**, combinés par un
méta-apprenant (régression logistique). Il remplace l'hybride XGBoost + LSTM + autoencodeur.

| Test (dernier mois, vérité terrain) | Ancien hybride | **Nouveau modèle** | Écart [IC 95 %] |
|---|---|---|---|
| PR-AUC globale | 0,9365 | **0,9555** | **+0,019 [+0,012 ; +0,026]** |
| PR-AUC Mobile Money | 0,9666 | **0,9739** | **+0,007 [+0,003 ; +0,013]** |
| PR-AUC Visa virtuelle | 0,7433 | **0,8126** | **+0,069 [+0,025 ; +0,113]** |
| Rappel à 1 % de faux positifs | 0,949 | 0,961 | |
| Latence du modèle, 1 transaction, 1 cœur (p99) | 6,9 ms | 12,1 ms | budget 20 ms |

Intervalles : bootstrap apparié par client, 1 000 tirages. Tous les écarts sont significatifs.
La dégradation de Visa causée par le profil de réputation (0,844 → 0,743) est ainsi corrigée.

## Les besoins, fixés avant les mesures

| Besoin | Critère mesurable |
|---|---|
| B1 Détection | PR-AUC sur la validation (étiquettes connues de l'opérateur) ; le test sert seulement à confirmer |
| B2 Deux canaux | Visa ne doit pas se dégrader significativement |
| B3 Temps réel | modèle ≤ 20 ms au 99ᵉ centile, 1 transaction, 1 cœur (budget total 100 ms) |
| B4 Explication | contributions par variable de chaque alerte ≤ 15 ms (audit BCC, analystes) |
| B5 Fraude inconnue | mesurer l'apport de l'autoencodeur sur une typologie jamais vue |
| B6 Simplicité | à 0,005 près sur B1, le modèle le plus simple l'emporte |

## Première manche : 8 candidats

| Candidat | Val. (étiquettes) | Val. (vérité) | Test (vérité) | p99 modèle | Explication p99 |
|---|---|---|---|---|---|
| Hybride actuel (XGB + LSTM + AE) | 0,461 | 0,910 | 0,9365 | 6,9 ms | 7,2 ms |
| XGB + LSTM | 0,472 | 0,908 | 0,9344 | 6,1 ms | 7,2 ms |
| XGB optimisé + LSTM | 0,496 | 0,913 | 0,9367 | 6,1 ms | 17,4 ms |
| XGB optimisé + LSTM + AE | 0,485 | 0,912 | 0,9373 | 6,8 ms | 17,4 ms |
| Forêt aléatoire + LSTM | 0,476 | **0,938** | **0,9550** | 42,3 ms | indisponible |
| Forêt aléatoire seule | 0,509 | 0,935 | 0,9435 | 37,1 ms | indisponible |
| XGB optimisé seul | **0,521** | 0,878 | 0,8947 | 0,9 ms | 17,4 ms |
| LSTM seul | 0,434 | 0,873 | 0,9007 | 5,2 ms | 0 |

« XGB optimisé » : profondeur 8, sans pondération des fraudes, meilleure de 4 configurations sur la validation.

1. La règle a d'abord choisi **XGB + LSTM**, sans autoencodeur. La confirmation sur le test l'a
   rejeté : il dégrade Visa de façon significative (−0,012 [−0,023 ; −0,002]), ce qui viole B2.
2. **Forêt aléatoire + LSTM** est le seul candidat meilleur partout, mais il était écarté : trop
   lent (scikit-learn parcourt 200 arbres un par un en Python) et sans explication par variable.

## Deuxième manche : rendre la forêt compatible avec le temps réel

`ml/models/flat_forest.py` range les 330 016 nœuds des 200 arbres dans des tableaux numpy
(9 Mo). Les 200 arbres descendent ensemble, un niveau par itération. L'explication suit la méthode
de Saabas, additive et exacte : `P(fraude) = biais + Σ contributions`.

| Forêt aplatie | Valeur |
|---|---|
| Écart avec scikit-learn | 7,8 × 10⁻¹⁶ (identique) |
| Prédiction, p50 / p99 | 0,7 ms / 5 ms (au lieu de 14 / 37 ms) |
| Explication, p50 / p99 | 0,9 ms / 3,9 ms |
| Parité entraînement / production (1 500 transactions via Redis) | écart max 6,4 × 10⁻⁸ |

La méthode de Saabas n'est pas « cohérente » au sens de TreeSHAP (Lundberg, 2017). TreeSHAP sur
des arbres de profondeur 49 serait beaucoup trop lent pour le temps réel ; c'est un compromis assumé.

## L'autoencodeur : inutile sur les fraudes connues, utile sur les nouvelles

Pour chaque typologie T, les arbres sont entraînés **sans aucune fraude de type T**. On mesure
ensuite la part des fraudes T détectées sur le test, au seuil qui donne 1 % de fausses alertes
sur la validation.

| Typologie jamais vue | n | Forêt sans T | + autoencodeur | Autoencodeur seul |
|---|---|---|---|---|
| SIM swap | 191 | 92,7 % | 93,7 % | 33,5 % |
| Prise de contrôle | 109 | 77,1 % | 78,0 % | 13,8 % |
| Ingénierie sociale | 197 | 44,7 % | **52,8 %** | 41,6 % |
| Agent compromis | 173 | 15,6 % | 19,1 % | 34,7 % |
| Mules / fractionnement | 192 | 28,1 % | 33,3 % | 33,9 % |
| Test de carte | 64 | 32,8 % | 34,4 % | 62,5 % |
| Carte non présente | 52 | 76,9 % | 82,7 % | 46,2 % |
| Recharge puis dépense | 36 | 86,1 % | 86,1 % | 22,2 % |

- Le gain est positif sur 7 typologies sur 7 hors égalité, +3,3 points en moyenne. Test du signe :
  p ≈ 0,016. Même constat avec XGBoost : +1,8 point, 7 sur 7.
- Sur les fraudes connues, l'apport est minime (+0,0005 de PR-AUC, IC [0,0 ; +0,001]).
- Limite de ce test : le méta-apprenant y est réajusté à 2 entrées (forêt sans T + AE), sans LSTM.
  Dans le modèle déployé, l'AE ne pèse presque rien (coefficient 0,017) : ce gain ne se transfère
  pas. D'où la veille séparée de la 3ᵉ manche plutôt que la branche empilée.
- Limite : le LSTM, lui, a vu toutes les typologies ; ce test porte sur les arbres et l'autoencodeur.
- Piste : seul, l'autoencodeur détecte mieux les fraudes d'agent, les mules et le test de carte
  jamais vus. Un tableau de « veille des anomalies » séparé, sans friction pour le client,
  exploiterait mieux ce signal que le méta-apprenant linéaire.

## Ce que la sélection apprend au mémoire (contribution C3)

- **Choisir un modèle sur les étiquettes de l'opérateur peut conduire au mauvais choix.** Sur la
  validation, XGB optimisé seul est premier avec les étiquettes connues (0,521) et dernier sur la
  vérité (0,878). La forêt + LSTM est moyenne sur les étiquettes (0,476) et première sur la vérité
  (0,938). Le classement sur la vérité est stable de la validation au test ; celui sur les
  étiquettes ne l'est pas.
- **Explication probable.** L'arrêt précoce de XGBoost optimise la PR-AUC sur les fraudes
  *signalées* : il apprend autant « ce qui est signalé » que « ce qui est frauduleux ». La forêt
  aléatoire n'a pas d'arrêt précoce et s'entraîne sur une période plus récente.
- **Recommandation à un opérateur.** Constituer un échantillon de validation vérifié : enquêtes
  sur les alertes et audit d'un échantillon aléatoire. Ne pas se contenter des plaintes.
- **Observé en direct.** Lors d'un rejeu, le simulateur, qui ne connaît que les fraudes signalées,
  comptait 11 « clients honnêtes bloqués ». Les 7 retrouvés dans le jeu de données étaient tous de
  vraies fraudes jamais signalées : prise de contrôle, SIM swap, mules.
- **Calibration.** Les probabilités sont calibrées sur les fraudes *signalées* (moyenne 0,70 % pour
  un taux signalé de 0,66 %), alors que le taux réel est de 1,75 %. La décision coût-sensible sous-
  estime donc le risque d'un facteur d'environ 2,6. Piste : recalibrer sur un échantillon vérifié.

## Plateforme complète (rejeu de 1 500 transactions à 25 tx/s, portable 4 cœurs, 23 conteneurs)

| | Ancien modèle | Nouveau modèle |
|---|---|---|
| Latence dans le service de scoring, p50 / p95 / p99 | 6,4 / 20,5 / 40,9 ms | 9,8 / 35,2 / 57,7 ms |
| Latence de bout en bout, p50 / p95 / p99 | 17 / 47 / 87 ms | 24 / 83 / 115 ms |
| Fraudes du rejeu interceptées | 7 / 7 | 20 / 20 |

Le nouveau modèle est plus lent sous charge, mais reste dans le budget de 100 ms au 95ᵉ centile.
Le 99ᵉ centile dépasse ce budget sur ce portable partagé entre 23 conteneurs ; en production, un
cœur par worker de scoring est prévu.

## Taux d'erreur en conditions d'exploitation (mois de test, vérité terrain)

Décision réelle de la plateforme (coût attendu), **sans compter les règles réglementaires**, qui ne
peuvent qu'ajouter des vérifications. 57 796 transactions, dont 1 014 fraudes (1,75 %).

| Mesure | Valeur |
|---|---|
| Fraudes arrêtées (vérification ou blocage) | **84,0 %** (16,0 % manquées) |
| Montant frauduleux arrêté | **99,5 %** : les fraudes manquées sont surtout de petits montants |
| Clients honnêtes à qui l'on demande une confirmation | **0,66 %** |
| Clients honnêtes bloqués | **0,0 %** |
| Part de vraies fraudes parmi les alertes | 69,4 % |
| Mobile Money : fraudes arrêtées / honnêtes dérangés | 88,2 % / 0,45 % |
| Visa virtuelle : fraudes arrêtées / honnêtes dérangés | 60,5 % / 3,35 % |

Par typologie : mules 100 %, SIM swap 89 %, ingénierie sociale 88 %, recharge puis dépense 86 %,
carte non présente 85 %, agent 81 %, prise de contrôle 78 %, **test de carte 27 %**. Le test de carte
passe par construction : des micro-achats de 0,5 à 2,5 USD coûtent moins que la gêne d'une
vérification. C'est le gros achat qui suit qui est arrêté.

Compromis possible en jouant sur le seuil seul :

| Honnêtes alertés | 0,1 % | 0,5 % | 1 % | 2 % |
|---|---|---|---|---|
| Fraudes détectées | 86,4 % | 93,8 % | 96,1 % | 97,4 % |

Ces taux sont mesurés sur des données synthétiques. Les vrais taux d'erreur ne seront connus
qu'après un pilote en mode silencieux chez un opérateur.
