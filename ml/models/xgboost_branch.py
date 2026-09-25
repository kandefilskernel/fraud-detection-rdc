"""Branche 1 : XGBoost sur les variables tabulaires (profil comportemental instantané)."""
import numpy as np
import xgboost as xgb


def build_xgboost_branch(scale_pos_weight: float = 1.0, n_estimators: int = 600,
                         seed: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,  # compense le déséquilibre des classes
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )


def train_xgboost_branch(X_fit: np.ndarray, y_fit: np.ndarray, X_es: np.ndarray, y_es: np.ndarray,
                         n_estimators: int = 600, seed: int = 42) -> xgb.XGBClassifier:
    spw = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    model = build_xgboost_branch(scale_pos_weight=spw, n_estimators=n_estimators, seed=seed)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model


def xgb_logit(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Score brut (log-odds) : mieux adapté que la probabilité comme entrée du meta-learner."""
    return model.predict(X, output_margin=True).astype(np.float32)
