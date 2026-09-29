# Assistant d'enquête (RAG) pour les analystes

L'assistant aide l'analyste à **instruire une alerte déjà décidée**. Il retrouve les dossiers
passés qui ressemblent à la transaction et les procédures internes applicables, puis rédige une
note d'instruction qui cite ses sources. Il **ne décide jamais** et ne tourne pas dans la boucle
de scoring.

## Pourquoi pas de RAG dans la décision temps réel

| | Décision (scoring) | Enquête (assistant) |
|---|---|---|
| Moment | pendant la transaction | après, à la demande de l'analyste |
| Latence acceptable | ≈ 20 ms | 10 à 60 s |
| Qui décide | modèle hybride + règles, auditables | l'analyste |
| Rôle d'un modèle de langage | aucun : trop lent, non déterministe, difficile à auditer | rédaction d'un avis, sources citées |

## Fonctionnement

```
alerte ─► 1. faits (variables traduites en clair : shared/investigation/signals.py)
          2. cas passés proches (archive, cosinus pondéré par l'importance XGBoost)
          3. verdicts récents d'analystes sur des transactions semblables
          4. liens avec des fraudes confirmées (même appareil, bénéficiaire, agent, marchand)
          5. procédures internes (BM25 + bonus typologie supposée / règle déclenchée)
                │
                ├─► mode « llm » : Claude (claude-opus-5) rédige la note, cite [C#] et [P#]
                └─► mode « extractif » : note assemblée sans modèle de langage
                    (pas de clé d'API, erreur réseau, quota, refus)
```

- **Archive des cas passés** (`ml/rag/build_case_archive.py`) : reconstituée à partir des données
  **antérieures à la période de test** (aucune fuite). 4 152 dossiers :
  - 2 997 alertes instruites (les 3 % de transactions les plus risquées selon la branche XGBoost, sur
    les périodes qu'elle n'a pas apprises) : 1 102 fraudes confirmées, 1 895 classées sans suite ;
  - 1 155 plaintes clients (fraudes signalées, y compris celles que le modèle n'aurait pas alertées).
  - Hypothèse : l'enquête établit la vérité, sauf pour 10 % des fraudes alertées, classées à tort
    (client injoignable, complice, trop tard) : 121 dossiers « classés » sont en réalité des fraudes.
- **Procédures** (`services/backoffice-api/knowledge/procedures/*.md`) : procédures types par
  typologie (SIM swap, ingénierie sociale, prise de contrôle, agent, mules, carte virtuelle, faux
  positifs) et cadre général. À faire valider par la conformité de l'opérateur.

## Protection des données

- Le texte envoyé au modèle de langage **ne contient aucun identifiant** : ni numéro, ni client,
  ni appareil, ni agent, ni marchand, ni identifiant de transaction. Seulement des propriétés
  (« appareil jamais vu pour ce client »), des montants, la province et l'heure. Un test vérifie
  qu'aucun identifiant ne sort (`tests/test_assistant.py`).
- Les cas sont désignés par des alias (C1, C2…) ; la correspondance reste dans le back-office.
- Chaque consultation est inscrite au journal d'audit inaltérable (`ASSISTANT_ENQUETE`) avec les
  sources montrées à l'analyste, le mode et le modèle utilisés.
- Sans clé d'API, rien ne quitte la plateforme (mode extractif).

## Évaluation de la recherche (période de test, jamais vue)

`python -m ml.rag.evaluate_retrieval` → `ml/reports/rag_retrieval_eval.json` (k = 8 voisins).

**Typologie proposée** (typologie majoritaire des fraudes confirmées les plus proches), sur 1 014
fraudes réelles :

| | Exactitude |
|---|---|
| Assistant (précédents) | **88,3 %** |
| Réponse constante « typologie la plus fréquente du canal » | 24,0 % |

Par typologie : blanchiment/mules 99,5 %, ingénierie sociale 96,4 %, recharge-puis-dépense 94,4 %,
SIM swap 93,7 %, carte non présente 92,3 %, agent 91,3 %, prise de contrôle 59,6 %,
test de carte 46,9 % (souvent confondu avec la carte non présente, dont il est proche).

Le modèle de scoring ne donne qu'une probabilité ; **la typologie est l'apport propre de
l'assistant** : elle oriente directement les vérifications (appeler l'opérateur pour un SIM swap,
le superviseur de réseau pour un agent, la conformité pour des mules).

**Tri des alertes** (1 733 alertes, 54,6 % de vraies fraudes) :

| Score utilisé pour classer les alertes | ROC-AUC | PR-AUC |
|---|---|---|
| Part de fraudes parmi les précédents | 0,936 | 0,943 |
| Score du modèle hybride | 0,955 | 0,965 |
| Moyenne des deux rangs | 0,965 | 0,976 |

Les précédents seuls ne font pas mieux que le modèle, ce qui est attendu : ils servent à
**expliquer** (« 5 des 6 cas les plus proches étaient des SIM swap confirmés »), pas à remplacer
le score. La combinaison est légèrement meilleure sur cet échantillon (écart non testé
statistiquement : piste pour une version ultérieure, pas un résultat du mémoire).

## Limites

- Archive et procédures construites sur des données synthétiques et des procédures types.
- La qualité de la note rédigée n'est pas mesurée automatiquement : à évaluer avec des analystes
  (grille : exactitude des faits cités, utilité des vérifications, temps d'instruction gagné).
- Le modèle de langage peut se tromper ; les consignes l'obligent à citer ses sources et
  l'interface permet de vérifier chaque référence [C#] / [P#] en un clic.

## Utilisation

- Tableau de bord → une transaction alertée → **« Préparer la note d'instruction »**.
- API : `POST /transactions/{id}/assistant` (`?mode=extractif` pour forcer le mode sans modèle de
  langage, `?refresh=true` pour régénérer) ; `GET /assistant/status`.
- Activer la rédaction par Claude : renseigner `ANTHROPIC_API_KEY` dans `.env`, puis
  `docker compose up -d backoffice-api`. Réglages : `ASSISTANT_MODEL` (défaut `claude-opus-5`),
  `ASSISTANT_EFFORT` (`low` / `medium` / `high`), `ASSISTANT_USE_FALLBACKS` (repli côté serveur si
  le modèle principal refuse une demande).
- Reconstruire l'archive après un réentraînement : `python -m ml.rag.build_case_archive`, puis
  reconstruire l'image du back-office.
