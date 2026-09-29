# NLP : analyse des SMS d'arnaque signalés par les clients

## Le problème visé

L'arnaque « envoi par erreur » est la plus difficile à détecter : la victime paie elle-même, depuis son
téléphone, à une heure normale. Le profil comportemental du client ne voit rien d'anormal. Le profil
de réputation, lui, ne reconnaît un escroc qu'**après** une fraude confirmée.

Or l'escroc envoie d'abord des dizaines de SMS (« je vous ai envoyé 50 000 FC par erreur, renvoyez
au… »). Certains destinataires ne tombent pas dans le piège et **signalent** le message au numéro court
de leur opérateur. Ce texte permet de marquer le numéro de l'escroc **avant** que ses victimes suivantes
ne paient.

## Fonctionnement

```
Client ──SMS transféré──► Opérateur ──► couche d'intégration ──► scoring-service
                                           │ extrait les numéros       │ classifieur NLP
                                           │ (même déguisés)           │ P(arnaque) ≥ 0,70 ?
                                           │ les pseudonymise          │   oui : fs:scam:{numéro}
                                           │ les masque : <NUMERO>     ▼
                                                        tout envoi vers ce numéro (portefeuille
                                                        de moins de 30 jours) : confirmation PIN
                                                        (règle BENEFICIAIRE_SIGNALE_PAR_SMS)
```

- **Extraction des numéros** (`shared/nlp/phone_numbers.py`) : `+243 81 234 5678`, `081-234-56-78`,
  `(+243) 812345678`, `0 8 1 2 3…`, et même en lettres (« zéro huit un… »). Codes courts, montants et
  dates sont ignorés. Le numéro du client qui signale n'est jamais marqué.
- **Classifieur** (`ml/nlp/scam_sms.py`) :
  - entrée : le texte masqué, précédé du type d'expéditeur (particulier ou opérateur) ;
  - n-grammes de caractères (2 à 5), robustes aux fautes, aux abréviations SMS et au mélange de langues ;
  - n-grammes de mots (1 à 2) ;
  - régression logistique à 5 classes : faux envoi par erreur, faux agent, faux gain, demande de code, légitime.
  - Environ 5 ms par message, sans GPU.
- **Règle** (`ml/serving/decision.py`) : un envoi vers un numéro signalé dans les 30 derniers jours
  demande une confirmation. Deux protections :
  1. **Jamais de blocage** sur ce seul motif : un escroc pourrait signaler le numéro d'un innocent.
  2. **Seulement si le portefeuille bénéficiaire a moins de 30 jours.** 97 % des arnaques visent un
     numéro récent, contre 19 % des envois honnêtes. Une relation établie n'est donc jamais mise en
     cause par un signalement, même malveillant.
- **Messages de l'opérateur** : un message envoyé sous le nom d'un opérateur ne sert jamais à marquer
  un numéro, car les vraies confirmations de dépôt contiennent le numéro de l'expéditeur.

## Données

Aucun jeu public de SMS d'arnaque Mobile Money congolais n'existe. `ml/nlp/sms_corpus.py` génère
8 000 signalements à partir de 71 modèles de messages :
- **4 arnaques** : faux envoi par erreur, faux agent, faux gain, demande de code ;
- **des messages légitimes signalés par doute** : vraies confirmations de l'opérateur, messages
  familiaux, tontines, promotions ;
- **des variations** : fautes, abréviations, accents perdus, numéros déguisés ;
- **les langues** : français majoritaire, lingala et swahili. **Les phrases en langues nationales sont
  des approximations à faire valider par des locuteurs natifs.**

## Résultats

### Classifieur

Validation croisée **groupée par modèle de message** : chaque test porte sur des formulations jamais
vues à l'apprentissage. `ml/reports/nlp_scam_sms.json`.

| | Formulations jamais vues | Découpage aléatoire (optimiste) |
|---|---|---|
| ROC-AUC arnaque / légitime | **0,967** | 1,000 |
| Arnaques repérées (seuil 0,70) | **86,9 %** | 100 % |
| Messages légitimes marqués à tort | **7,2 %** | 0 % |

Le découpage aléatoire donne un score parfait parce que le test contient des phrases déjà vues : c'est
pourquoi seul le protocole groupé est retenu.

| Langue | Arnaques repérées | Légitimes marqués à tort |
|---|---|---|
| Français | 89,3 % | 4,8 % |
| Lingala | 90,4 % | 25,6 % |
| Swahili | 73,1 % | 0,6 % |

La distinction entre les 4 types d'arnaque est moins bonne (F1 macro 0,65) : « envoi par erreur » et
« demande de code » se ressemblent. Pour la décision, seule compte la distinction arnaque / légitime.

### Apport à la détection (mois de test, vérité terrain)

`python -m ml.nlp.evaluate_sms_rule`, résultats dans `ml/reports/nlp_sms_rule_eval.json`.

Simulation :
- chacune des 353 mules d'arnaque envoie 40 à 200 SMS ;
- une part *r* des destinataires les signale ;
- chaque signalement passe par la vraie chaîne : texte jamais vu du classifieur, extraction, classifieur ;
- des messages légitimes contenant les numéros de vrais clients sont aussi signalés (38 % des
  signalements) ;
- la décision est prise par la vraie politique : modèle et toutes les règles.

| Part des destinataires qui signalent | Numéros d'escrocs marqués | … avant la 1re victime | Ingénierie sociale arrêtée | Fraudes en plus | Honnêtes dérangés en plus |
|---|---|---|---|---|---|
| Sans signalements | | | 87,8 % | | |
| 0,5 % | 41 % | 27 % | 88,8 % | +5 | 0 |
| 1 % | 70 % | **48 %** | 88,8 % | +7 | 0 |
| 2 % | 87 % | **65 %** | 89,8 % | +13 | +1 |
| 5 % | 99 % | 88 % | 90,4 % | +14 | +2 |

Lecture :
- **Gain supplémentaire modeste**, parce que le système arrêtait déjà 88 % de ces arnaques (modèle,
  réseau du bénéficiaire, règle « renvoi supérieur au reçu »).
- **Coût quasi nul** : au plus 2 clients honnêtes dérangés en plus sur un mois.
- **Vraie force : l'anticipation.** Avec 1 % de clients qui signalent, près d'un numéro d'escroc sur
  deux est marqué avant sa première victime. Aucun modèle entraîné sur les fraudes passées ne peut le
  faire.
- **Effet de bord utile** : les mules servent aussi aux SIM swap et aux prises de contrôle. Le SIM swap
  arrêté passe de 89,0 % à 92,7 % (2 % de signalements).
- Sans la condition « portefeuille de moins de 30 jours », la même règle dérangeait jusqu'à 300
  clients honnêtes de plus pour 13 fraudes. La condition est indispensable.

## Protection des données et cadre légal

- **Seuls les SMS volontairement transférés** par le client sont utilisés. Un opérateur ne peut pas
  lire les SMS de ses clients.
- La couche d'intégration **masque** les numéros et les **pseudonymise** si la plateforme est
  mutualisée : le service de scoring ne voit jamais un numéro en clair.
- Chaque signalement est compté (métriques Prometheus `scoring_scam_sms_reports_total`).

## Démonstration

Application portefeuille → « Recevoir de l'argent d'un inconnu » → **« Signaler ce SMS comme
arnaque »**. Un autre client qui veut ensuite envoyer de l'argent à ce numéro doit confirmer par PIN,
avec l'avertissement : « D'autres clients ont signalé ce numéro pour des SMS d'arnaque ».

API opérateurs : `POST /ingest/v1/scam-reports/{vodacom|airtel|orange}` (signé HMAC comme les
transactions) avec `report_id`, `received_at`, `text`, `sender` et `reporter`.

## Limites

- Corpus et campagnes **synthétiques**. Le taux de signalement réel est inconnu, d'où les 4 valeurs testées.
- Langues nationales encore faibles : faux positifs en lingala, moins bonne détection en swahili. Il
  faudra de vrais messages annotés par des locuteurs natifs.
- Un escroc peut changer de SIM souvent. La règle reste utile tant qu'il réutilise un numéro plus de
  quelques heures, ce qui est le cas des mules du générateur (14 jours en moyenne).
