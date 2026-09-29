"""
Index des cas passés pour l'assistant d'enquête (partie « R » du RAG).

Chaque cas est une transaction déjà instruite : fraude confirmée (avec sa typologie) ou
alerte classée sans suite (faux positif). Il est représenté par son vecteur de variables
comportementales, standardisé comme à l'entraînement du modèle, puis pondéré par
l'importance des variables (gain XGBoost) : deux cas sont « proches » s'ils se ressemblent
sur ce qui distingue une fraude, pas sur des détails sans intérêt.

La similarité est un cosinus pondéré. Le même code sert à l'évaluation hors ligne
(ml/rag/evaluate_retrieval.py) et au back-office.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

CONFIRMED, CLEARED = "FRAUDE_CONFIRMEE", "FAUX_POSITIF"


class CaseIndex:
    def __init__(self, vectors: np.ndarray, cases: list[dict], feature_names: list[str],
                 mean: np.ndarray, scale: np.ndarray, weights: np.ndarray, clip: float = 10.0):
        self.cases = cases
        self.feature_names = list(feature_names)
        self.mean, self.scale, self.weights, self.clip = mean, scale, weights, clip
        self.channels = np.array([c["channel"] for c in cases])
        self._unit = self._normalize(vectors.astype(np.float32) * weights)

    # ------------------------------------------------------------------ chargement
    @classmethod
    def load(cls, directory: str | Path) -> "CaseIndex":
        d = Path(directory)
        z = np.load(d / "case_archive.npz", allow_pickle=False)
        cases = json.loads((d / "case_archive.json").read_text(encoding="utf-8"))["cases"]
        return cls(z["vectors"], cases, [str(n) for n in z["feature_names"]], z["mean"], z["scale"],
                   z["weights"], float(z["clip"]))

    def save(self, directory: str | Path, vectors: np.ndarray, meta: dict) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "case_archive.npz", vectors=vectors.astype(np.float16),
                            feature_names=np.array(self.feature_names), mean=self.mean,
                            scale=self.scale, weights=self.weights, clip=np.float32(self.clip))
        (d / "case_archive.json").write_text(json.dumps({"meta": meta, "cases": self.cases},
                                                        ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ vecteurs
    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.maximum(norms, 1e-9)

    def scale_features(self, features: dict) -> np.ndarray:
        """Variables brutes -> vecteur standardisé (variable absente = valeur moyenne)."""
        raw = np.array([float(features.get(n, self.mean[i]) or 0.0) for i, n in enumerate(self.feature_names)])
        return np.clip((raw - self.mean) / self.scale, -self.clip, self.clip).astype(np.float32)

    # ------------------------------------------------------------------ recherche
    def search(self, features: dict, channel: str | None = None, k: int = 8,
               exclude_ids: set[str] | None = None, min_similarity: float = 0.0) -> list[dict]:
        """k cas les plus proches, un seul par épisode (une fraude en plusieurs opérations ne
        doit pas occuper toute la liste)."""
        q = self._normalize(self.scale_features(features) * self.weights)
        sims = self._unit @ q
        if channel is not None:
            sims = np.where(self.channels == channel, sims, -np.inf)
        order = np.argsort(-sims)
        out, seen = [], set()
        for i in order:
            if not np.isfinite(sims[i]) or sims[i] < min_similarity or len(out) >= k:
                break
            c = self.cases[i]
            episode = c.get("episode") or c["id"]
            if episode in seen or (exclude_ids and c.get("source_id") in exclude_ids):
                continue
            seen.add(episode)
            out.append({**c, "similarity": round(float(sims[i]), 3)})
        return out


def summarize_neighbors(neighbors: list[dict]) -> dict:
    """Ce que disent les précédents : part de fraudes confirmées, typologies dominantes.
    Pondéré par la similarité (un cas très proche compte plus qu'un cas lointain)."""
    if not neighbors:
        return {"n": 0, "fraud_share": None, "median_similarity": None, "typologies": []}
    w_total = sum(max(n["similarity"], 0.0) for n in neighbors) or 1.0
    w_fraud = sum(max(n["similarity"], 0.0) for n in neighbors if n["outcome"] == CONFIRMED)
    typo = Counter()
    for n in neighbors:
        if n["outcome"] == CONFIRMED and n.get("typology"):
            typo[n["typology"]] += max(n["similarity"], 0.0)
    total_typo = sum(typo.values()) or 1.0
    sims = sorted(n["similarity"] for n in neighbors)
    return {
        "n": len(neighbors),
        "median_similarity": round(float(np.median(sims)), 3),
        "n_confirmed": sum(n["outcome"] == CONFIRMED for n in neighbors),
        "n_cleared": sum(n["outcome"] == CLEARED for n in neighbors),
        "fraud_share": round(w_fraud / w_total, 3),
        "typologies": [{"typology": t, "share": round(v / total_typo, 3)} for t, v in typo.most_common(3)],
    }
