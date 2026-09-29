"""
Classifieur des SMS signalés comme arnaque (NLP).

Entrée : le texte du SMS dont les numéros sont masqués (<NUMERO>), précédé du type
d'expéditeur (<EXP_NUMERO> un particulier, <EXP_OPERATEUR> un nom d'opérateur). Le modèle
apprend la FORME du message, jamais les numéros : ceux-ci sont extraits à part
(shared/nlp/phone_numbers.py) et marqués dans le profil de réputation.

Représentation :
    - n-grammes de caractères (2 à 5, dans les mots) : robustes aux fautes, abréviations
      SMS (« svp », « mrc »), accents perdus et mélange de langues (français, lingala,
      swahili), là où un découpage en mots échoue ;
    - n-grammes de mots (1 à 2) : expressions (« par erreur », « code pin », « frais de dossier »).
Modèle : régression logistique multiclasse, pondérée (classes équilibrées). Rapide (< 1 ms),
explicable (poids des n-grammes), sans GPU.

Évaluation : validation croisée GROUPÉE PAR MODÈLE DE MESSAGE (GroupKFold) : chaque pli teste
des formulations jamais vues à l'apprentissage. Sans cela, le score mesurerait la mémoire.

    python -m ml.nlp.scam_sms          # entraîne, évalue, enregistre ml/artifacts/nlp/
"""
from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.nlp.sms_corpus import CATEGORIES, LABELS, SCAM, generate
from shared.nlp.phone_numbers import extract_numbers, is_phone_number, mask_numbers

ROOT = Path(__file__).absolute().parents[2]
ARTIFACT_DIR = ROOT / "ml" / "artifacts" / "nlp"
REPORT = ROOT / "ml" / "reports" / "nlp_scam_sms.json"
SCAM_THRESHOLD = 0.70        # P(arnaque) au-delà de laquelle les numéros sont marqués


def prepare(text: str, sender: str | None) -> str:
    """Texte présenté au modèle : type d'expéditeur + message masqué, minuscules, sans accents."""
    kind = "<EXP_NUMERO>" if is_phone_number(sender) else "<EXP_OPERATEUR>" if sender else "<EXP_INCONNU>"
    t = unicodedata.normalize("NFKD", mask_numbers(text).lower())
    t = "".join(c for c in t if not unicodedata.combining(c)).replace("<numero>", " NUMERO ")
    return f"{kind} {t}"


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("car", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)),
        ("mots", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 token_pattern=r"(?u)<?\b\w[\w']*\b>?", max_features=30000)),
    ])
    return Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))])


class ScamSmsClassifier:
    """Chargé par le scoring-service : classe un signalement et dit s'il faut marquer ses numéros."""

    def __init__(self, pipeline: Pipeline, threshold: float = SCAM_THRESHOLD, version: str = "?"):
        self.pipe, self.threshold, self.version = pipeline, threshold, version
        self.classes = list(pipeline.named_steps["clf"].classes_)

    @classmethod
    def load(cls, directory: str | Path = ARTIFACT_DIR) -> "ScamSmsClassifier":
        d = Path(directory)
        meta = json.loads((d / "scam_sms.json").read_text(encoding="utf-8"))
        return cls(joblib.load(d / "scam_sms.joblib"), meta["threshold"], meta["trained_at"])

    def classify(self, masked_text: str, sender_is_number: bool | None) -> dict:
        """Le texte arrive déjà masqué (couche d'intégration) : aucun numéro en clair ici."""
        sender = "243810000000" if sender_is_number else ("OPERATEUR" if sender_is_number is False else None)
        proba = self.pipe.predict_proba([prepare(masked_text, sender)])[0]
        by_class = dict(zip(self.classes, proba))
        p_scam = 1.0 - float(by_class.get("LEGITIME", 0.0))
        top = max(SCAM, key=lambda c: by_class.get(c, 0.0))
        category = top if p_scam >= 0.5 else "LEGITIME"
        return {"category": category, "category_label": LABELS[category], "scam_probability": round(p_scam, 4),
                "is_scam": p_scam >= self.threshold, "probabilities": {c: round(float(p), 4) for c, p in by_class.items()}}

    def top_terms(self, category: str, k: int = 12) -> list[str]:
        """N-grammes les plus caractéristiques d'une catégorie (explicabilité du modèle)."""
        names = self.pipe.named_steps["tfidf"].get_feature_names_out()
        coef = self.pipe.named_steps["clf"].coef_[self.classes.index(category)]
        return [names[i].split("__", 1)[1] for i in np.argsort(-coef)[:k]]


def evaluate_and_train(n: int = 8000, seed: int = 7) -> dict:
    t0 = time.perf_counter()
    df = pd.DataFrame(generate(n, seed))
    df["x"] = [prepare(t, s) for t, s in zip(df.text, df.sender)]
    df["scam"] = (df.category != "LEGITIME").astype(int)
    # extraction des numéros : sur les arnaques, le numéro où envoyer l'argent doit être trouvé
    df["n_numbers"] = [len(extract_numbers(t)) for t in df.text]

    oof_proba = np.zeros((len(df), len(CATEGORIES)))
    for tr, te in GroupKFold(n_splits=5).split(df.x, df.category, groups=df.template):
        pipe = build_pipeline().fit(df.x.iloc[tr], df.category.iloc[tr])
        idx = [list(pipe.classes_).index(c) for c in CATEGORIES]
        oof_proba[te] = pipe.predict_proba(df.x.iloc[te])[:, idx]
    pred = np.array(CATEGORIES)[oof_proba.argmax(1)]
    p_scam = 1 - oof_proba[:, CATEGORIES.index("LEGITIME")]
    flagged = p_scam >= SCAM_THRESHOLD

    report = {
        "n_messages": int(len(df)), "n_templates": int(df.template.nunique()),
        "protocole": "validation croisée groupée par modèle de message (5 plis) : formulations jamais vues",
        "macro_f1": round(float(f1_score(df.category, pred, average="macro")), 4),
        "par_categorie": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score")}
                          for k, v in classification_report(df.category, pred, output_dict=True).items() if k in CATEGORIES},
        "matrice_confusion": {"ordre": CATEGORIES, "valeurs": confusion_matrix(df.category, pred, labels=CATEGORIES).tolist()},
        "arnaque_vs_legitime": {
            "roc_auc": round(float(roc_auc_score(df.scam, p_scam)), 4),
            "seuil": SCAM_THRESHOLD,
            "arnaques_marquees": round(float(flagged[df.scam == 1].mean()), 4),
            "legitimes_marques_a_tort": round(float(flagged[df.scam == 0].mean()), 4),
        },
        "par_langue": {lang: {"n": int((df.lang == lang).sum()),
                              "arnaques_marquees": round(float(flagged[(df.lang == lang) & (df.scam == 1)].mean()), 3)
                              if ((df.lang == lang) & (df.scam == 1)).any() else None,
                              "legitimes_marques_a_tort": round(float(flagged[(df.lang == lang) & (df.scam == 0)].mean()), 3)
                              if ((df.lang == lang) & (df.scam == 0)).any() else None}
                       for lang in ("fr", "ln", "sw")},
        "extraction_numeros": {"arnaques_avec_numero_trouve": round(float((df.n_numbers[df.scam == 1] > 0).mean()), 4)},
    }

    # protocole optimiste, pour comparaison : découpage aléatoire (des formulations déjà vues en test)
    from sklearn.model_selection import StratifiedKFold
    rnd = np.zeros(len(df))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(df.x, df.category):
        pipe = build_pipeline().fit(df.x.iloc[tr], df.category.iloc[tr])
        rnd[te] = 1 - pipe.predict_proba(df.x.iloc[te])[:, list(pipe.classes_).index("LEGITIME")]
    report["comparaison_decoupage_aleatoire"] = {
        "roc_auc": round(float(roc_auc_score(df.scam, rnd)), 4),
        "arnaques_marquees": round(float((rnd >= SCAM_THRESHOLD)[df.scam == 1].mean()), 4),
        "legitimes_marques_a_tort": round(float((rnd >= SCAM_THRESHOLD)[df.scam == 0].mean()), 4),
        "note": "optimiste : le test contient des formulations vues à l'apprentissage",
    }

    # modèle final : tous les messages
    final = build_pipeline().fit(df.x, df.category)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, ARTIFACT_DIR / "scam_sms.joblib", compress=3)
    trained_at = pd.Timestamp.now().isoformat(timespec="seconds")
    (ARTIFACT_DIR / "scam_sms.json").write_text(json.dumps(
        {"trained_at": trained_at, "threshold": SCAM_THRESHOLD, "categories": CATEGORIES, "evaluation": report},
        ensure_ascii=False, indent=1), encoding="utf-8")
    clf = ScamSmsClassifier(final, SCAM_THRESHOLD, trained_at)
    report["termes_caracteristiques"] = {c: clf.top_terms(c, 10) for c in CATEGORIES}
    sample = df.x.iloc[:200].tolist()
    t = time.perf_counter()
    for x in sample:
        final.predict_proba([x])
    report["latence_ms_par_message"] = round((time.perf_counter() - t) / len(sample) * 1000, 3)
    report["duree_s"] = round(time.perf_counter() - t0, 1)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = evaluate_and_train()
    print(json.dumps({k: v for k, v in r.items() if k != "termes_caracteristiques"}, ensure_ascii=False, indent=1))
    for c, terms in r["termes_caracteristiques"].items():
        print(f"{c:<14} {' | '.join(terms)}")
