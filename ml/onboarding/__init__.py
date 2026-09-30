"""Intégration des données RÉELLES d'un ou plusieurs opérateurs (voir docs/DONNEES_REELLES.md).

    import   : export de l'opérateur -> format pivot, pseudonymisé dès la lecture
    build    : fusion des sources + contrôle qualité (rapport, erreurs bloquantes)
    train    : variables, split temporel, entraînement dans l'espace de travail, comparaison
    promote  : mise en production sur commande explicite (ancien modèle archivé)

Les données réelles restent dans ml/workspaces/<nom>/ (ignoré par git).
"""
