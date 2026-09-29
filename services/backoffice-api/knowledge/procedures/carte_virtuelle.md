---
titre: Fraudes sur carte Visa virtuelle
typologies: CARD_TESTING, CNP_FRAUD, TOPUP_DRAIN
regles: SCORE_RESEAU_CARTE_ELEVE, SCORE_RESEAU_CARTE_CRITIQUE
---
## Signes typiques
- Test de carte : rafale de micro-achats (moins de 3 USD) en quelques minutes chez des marchands
  en ligne, parfois suivie d'un gros achat.
- Carte non présente : achats chez des marchands à risque ou étrangers avec des données de carte volées.
- Trans-canal : recharge de la carte depuis le portefeuille mobile puis dépenses immédiates, souvent à l'étranger.
- Le score de risque du réseau Visa complète notre modèle ; un score élevé justifie à lui seul une vérification.

## Vérifications
- Authentification 3-D Secure réussie ou non ; adresse IP et pays du marchand.
- Délai entre la recharge et l'achat ; le client a-t-il rechargé lui-même ?
- Le client reconnaît-il les marchands ? A-t-il saisi sa carte sur un site inconnu récemment ?

## Mesures
- Bloquer la carte et en réémettre une nouvelle ; ne pas débloquer l'ancienne.
- Contestation (rétrofacturation) auprès du réseau pour les achats non reconnus, dans les délais du réseau.
- Si la recharge venait d'une prise de contrôle du portefeuille : appliquer aussi la procédure prise de contrôle.

## Éléments à décharge
- Achat habituel du client (abonnements, marchands déjà utilisés), 3-D Secure validé, IP congolaise
  cohérente avec son historique.
