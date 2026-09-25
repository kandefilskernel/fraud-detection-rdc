"""
Meta-learner (stacking) : combine les scores des trois branches.

Il est entraîné sur la tranche VALIDATION, que les branches n'ont jamais vue : il apprend
donc la fiabilité réelle de chaque branche hors échantillon (pas de sur-apprentissage
du stacking). Les entrées sont standardisées : les coefficients de la régression
logistique sont directement comparables et mesurent la contribution de chaque branche.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BRANCH_NAMES = ["xgboost", "lstm_attention", "autoencoder"]


def build_meta_learner() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
    ])


def meta_coefficients(meta: Pipeline, names=BRANCH_NAMES) -> dict:
    coefs = meta.named_steps["logreg"].coef_[0]
    return {n: round(float(c), 4) for n, c in zip(names, coefs)}
