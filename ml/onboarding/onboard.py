"""Données réelles d'un ou plusieurs opérateurs : de l'export au modèle, en 4 commandes.

    python -m ml.onboarding.onboard import  --workspace pilote_vodacom --mapping ml/onboarding/mappings/vodacom.yaml \\
                                            --input exports/vodacom_2026.csv --frauds exports/fraudes.csv --kyc exports/kyc.csv
    python -m ml.onboarding.onboard build   --workspace pilote_vodacom      # fusion + rapport qualité
    python -m ml.onboarding.onboard train   --workspace pilote_vodacom      # entraînement ISOLÉ + comparaison
    python -m ml.onboarding.onboard promote --workspace pilote_vodacom --yes  # mise en production

Plusieurs opérateurs : lancer « import » une fois par export (même --workspace), puis « build ».
Tout reste dans ml/data/workspaces/<nom>/ (ignoré par git) ; la production (ml/artifacts)
n'est modifiée que par « promote ».
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

WORKSPACES = Path("ml/data/workspaces")
PRODUCTION = Path(os.getenv("ARTIFACTS_DIR", "ml/artifacts"))


def log(msg: str) -> None:
    print(msg, flush=True)


def workspace_dir(name: str) -> Path:
    if not name or any(c in name for c in "/\\:") or name.startswith("."):
        raise SystemExit(f"nom d'espace de travail invalide : {name!r} (lettres, chiffres, _ et -)")
    return WORKSPACES / name


def _pseudonymizer(allow_clear: bool):
    from shared.env_file import load_env
    from shared.privacy.pseudonymize import Pseudonymizer
    load_env()
    key = os.getenv("PSEUDONYMIZATION_KEY", "")
    if key:
        return Pseudonymizer(key)
    if allow_clear:
        log("ATTENTION : identifiants gardés EN CLAIR (--allow-clear-ids) : réservé aux données de test")
        return None
    raise SystemExit("PSEUDONYMIZATION_KEY absente (.env) : obligatoire pour importer des données réelles.\n"
                     "Générer : python scripts/security/generate_secrets.py (puis PSEUDONYMIZE_IDS=true), "
                     "ou --allow-clear-ids pour des données de TEST uniquement.")


# ---------------------------------------------------------------------- import / build
def cmd_import(a) -> None:
    from ml.onboarding.importer import import_operator
    from ml.onboarding.mapping import load_mapping
    ws = workspace_dir(a.workspace)
    m = load_mapping(a.mapping)
    t0 = time.perf_counter()
    info = import_operator(m, a.input, ws, frauds_path=a.frauds, kyc_path=a.kyc,
                           pseudo=_pseudonymizer(a.allow_clear_ids), since=a.since, sample_users=a.sample_users)
    log(f"[OK] {m.operator} ({m.name}) : {info['rows_kept']:,} transactions gardées sur {info['rows_read']:,} lues, "
        f"{info['labels']['frauds']:,} fraudes, en {time.perf_counter() - t0:.0f} s")
    for reason, k in info["dropped"].items():
        if k:
            log(f"    écartées : {reason} ({k:,})")
    for what, k in info["defaults"].items():
        log(f"    par défaut : {what} ({k:,})")
    log(f"    -> {ws / 'sources' / m.name}")


def cmd_build(a) -> bool:
    from ml.onboarding.importer import build_workspace
    from ml.onboarding.quality import assess, write_report
    ws = workspace_dir(a.workspace)
    tx, users, logs = build_workspace(ws)
    result = assess(tx, users, logs)
    report = write_report(result, ws)
    log(f"[{'OK' if result['ready'] else 'BLOQUÉ'}] {len(tx):,} transactions, {len(users):,} clients -> {ws / 'raw'}")
    for b in result["blocking"]:
        log(f"  ERREUR  {b}")
    for w in result["warnings"]:
        log(f"  attention  {w}")
    log(f"    rapport : {report}")
    return result["ready"]


# ---------------------------------------------------------------------- entraînement
def _champion_branches() -> tuple:
    meta = PRODUCTION / "metadata.json"
    if meta.exists():
        order = json.loads(meta.read_text(encoding="utf-8")).get("branch_order")
        if order:
            return tuple(order)
    return ("random_forest", "lstm_attention", "autoencoder")


def build_drift_reference(ws: Path, seed: int = 42, n: int = 10_000) -> None:
    import numpy as np
    import pandas as pd
    from ml.features.feature_engineering import FEATURE_NAMES
    from ml.models.hybrid_ensemble import HybridEnsemble
    proc, art = ws / "processed", ws / "artifacts"
    rng = np.random.default_rng(seed)
    split = np.load(proc / "split.npy")
    train_rows, val_rows = np.flatnonzero(split <= 1), np.flatnonzero(split == 2)
    feats = pd.read_csv(proc / "features.csv", usecols=FEATURE_NAMES)[FEATURE_NAMES].to_numpy(np.float32)
    model = HybridEnsemble.load(art)
    rows = np.sort(rng.choice(val_rows, size=min(n, len(val_rows)), replace=False))
    np.savez_compressed(art / "drift_reference.npz",
                        features=feats[rng.choice(train_rows, size=min(n, len(train_rows)), replace=False)],
                        scores=model.predict_proba(np.load(proc / "X_all.npy"), np.load(proc / "seq_idx.npy"),
                                                   rows).astype(np.float32),
                        feature_names=np.array(FEATURE_NAMES))


def compare_with_champion(ws: Path) -> dict:
    """Même période de test (la plus récente, jamais vue) pour le candidat et le modèle en production."""
    import joblib
    import numpy as np
    import pandas as pd
    from ml.features.feature_engineering import FEATURE_NAMES
    from ml.models.hybrid_ensemble import HybridEnsemble
    from ml.training.evaluation import evaluate, evaluate_by_group
    proc = ws / "processed"
    split, seq, y = np.load(proc / "split.npy"), np.load(proc / "seq_idx.npy"), np.load(proc / "y_all.npy")
    meta = pd.read_csv(proc / "meta.csv", usecols=["timestamp", "channel"])
    test = np.flatnonzero(split == 3)
    days = max((pd.to_datetime(meta["timestamp"].iloc[test]).max()
                - pd.to_datetime(meta["timestamp"].iloc[test]).min()).total_seconds() / 86400, 1e-9)
    out = {"n_test": int(len(test)), "frauds_test": int(y[test].sum()), "test_days": round(days, 1)}

    def block(model, X):
        p = model.predict_proba(X, seq, test)
        m = evaluate(y[test], p, model.threshold)
        m["alerts_per_day"] = round(float((p >= model.threshold).sum()) / days, 1)
        m["by_channel"] = evaluate_by_group(y[test], p, model.threshold, meta["channel"].to_numpy()[test])
        return m

    cand = HybridEnsemble.load(ws / "artifacts")
    out["candidate"] = block(cand, np.load(proc / "X_all.npy"))
    if (PRODUCTION / "metadata.json").exists():
        try:
            prep = json.loads((PRODUCTION / "preprocessing.json").read_text(encoding="utf-8"))
            raw = pd.read_csv(proc / "features.csv", usecols=FEATURE_NAMES)[FEATURE_NAMES].to_numpy(np.float32)
            Xc = np.clip(joblib.load(PRODUCTION / "scaler.pkl").transform(raw), -prep["clip"], prep["clip"])
            champ = HybridEnsemble.load(PRODUCTION)
            out["champion"] = block(champ, Xc.astype(np.float32))
            out["champion_version"] = json.loads((PRODUCTION / "metadata.json").read_text(encoding="utf-8")).get(
                "trained_at")
        except Exception as e:  # noqa: BLE001 — modèle en production incompatible : comparaison impossible
            out["champion_error"] = str(e)
    c, ch = out["candidate"], out.get("champion")
    if "pr_auc" not in c:
        out["recommendation"] = "NE PAS PROMOUVOIR : une seule classe dans le test"
    elif ch and "pr_auc" in ch and ch["pr_auc"] >= c["pr_auc"]:
        out["recommendation"] = "NE PAS PROMOUVOIR : le modèle en production fait au moins aussi bien"
    else:
        out["recommendation"] = ("PROMOUVOIR EN MODE SILENCIEUX d'abord (SHADOW_MODE_OPERATORS), puis activer "
                                 "le blocage après validation par l'opérateur")
    return out


def _fmt(m: dict | None, k: str) -> str:
    if not m or k not in m:
        return "—"
    v = m[k]
    return f"{v:.1%}" if k in ("precision", "recall", "false_positive_rate", "recall_at_1pct_fpr") else f"{v}"


def write_comparison(ws: Path, cmp: dict) -> Path:
    c, ch = cmp["candidate"], cmp.get("champion")
    rows = [("PR-AUC (qualité globale)", "pr_auc"), ("Fraudes détectées (rappel)", "recall"),
            ("Alertes justes (précision)", "precision"), ("Détection à 1 % de faux positifs", "recall_at_1pct_fpr"),
            ("Faux positifs (part des légitimes)", "false_positive_rate"), ("Alertes par jour", "alerts_per_day")]
    lines = [f"# Comparaison — {ws.name}", "",
             f"Période de test : {cmp['test_days']} jours les plus récents, {cmp['n_test']:,} transactions, "
             f"{cmp['frauds_test']:,} fraudes (jamais vues à l'entraînement).", "",
             f"| Indicateur | Nouveau modèle (données réelles) | Modèle en production"
             f"{' (' + cmp['champion_version'] + ')' if cmp.get('champion_version') else ''} |", "|---|---|---|"]
    lines += [f"| {label} | {_fmt(c, k)} | {_fmt(ch, k)} |" for label, k in rows]
    if cmp.get("champion_error"):
        lines += ["", f"Modèle en production non évaluable sur ces données : {cmp['champion_error']}"]
    lines += ["", f"**Recommandation : {cmp['recommendation']}**", "",
              "Mise en production : `python -m ml.onboarding.onboard promote --workspace "
              f"{ws.name} --yes` (l'ancien modèle est archivé)."]
    path = ws / "reports" / "comparaison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ws / "reports" / "comparaison.json").write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def cmd_train(a) -> None:
    from ml.preprocessing.train_test_split_normalize import run_preprocessing
    from ml.training.train_model import TrainConfig, train_pipeline
    ws = workspace_dir(a.workspace)
    q = ws / "rapport_qualite.json"
    if not q.exists():
        raise SystemExit("lancer d'abord « build » (rapport qualité absent)")
    if not json.loads(q.read_text(encoding="utf-8"))["ready"] and not a.force:
        raise SystemExit(f"données NON prêtes (voir {ws / 'rapport_qualite.md'}) ; --force pour passer outre")
    t0 = time.perf_counter()
    log("[1/4] Variables comportementales (rejeu chronologique)...")
    run_preprocessing(rebuild_features=True, processed_dir=ws / "processed", artifacts_dir=ws / "artifacts",
                      raw_dir=ws / "raw")
    log("[2/4] Entraînement du modèle hybride (espace de travail, pas la production)...")
    cfg = TrainConfig(processed_dir=str(ws / "processed"), artifacts_dir=str(ws / "artifacts"),
                      reports_dir=str(ws / "reports"), use_mlflow=False, skip_baselines=a.quick,
                      production_branches=_champion_branches(), seed=a.seed)
    if a.quick:
        cfg.xgb_estimators, cfg.lstm_epochs, cfg.ae_epochs = 150, 2, 5
    train_pipeline(cfg)
    meta_path = ws / "artifacts" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({"workspace": ws.name, "training_data": (ws / "processed").as_posix(),
                 "raw_data": (ws / "raw").as_posix()})
    meta_path.write_text(json.dumps(meta, indent=4, ensure_ascii=False, default=str), encoding="utf-8")
    log("[3/4] Référence de dérive...")
    build_drift_reference(ws, seed=a.seed)
    log("[4/4] Comparaison avec le modèle en production...")
    report = write_comparison(ws, compare_with_champion(ws))
    log(f"[OK] en {(time.perf_counter() - t0) / 60:.1f} min. Rapport : {report}")
    log(report.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- mise en production
def cmd_promote(a) -> None:
    from ml.retraining.retrain import promote
    ws = workspace_dir(a.workspace)
    cand = ws / "artifacts"
    cmp_path = ws / "reports" / "comparaison.json"
    if not (cand / "metadata.json").exists() or not cmp_path.exists():
        raise SystemExit("aucun modèle entraîné dans cet espace (lancer « train »)")
    cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
    log(f"Recommandation du rapport : {cmp['recommendation']}")
    if cmp["recommendation"].startswith("NE PAS") and not a.force:
        raise SystemExit("promotion refusée par la comparaison ; --force pour passer outre (déconseillé)")
    if not a.yes:
        raise SystemExit("confirmer avec --yes (le modèle en production sera remplacé, l'ancien archivé)")
    promote(cand)
    raw = (ws / "raw").as_posix()
    log("[OK] Modèle promu. Le service de scoring le recharge à chaud (moins de 30 s).\n"
        "Étapes suivantes OBLIGATOIRES :\n"
        f"  1. Amorcer les profils clients avec l'historique réel :\n"
        f"     docker compose run --rm feature-store-seeder python -m ml.serving.seed_feature_store "
        f"--raw-dir {raw} --until all\n"
        "  2. Pilote silencieux : SHADOW_MODE_OPERATORS=<opérateurs> dans .env, puis\n"
        "     docker compose up -d integration-layer worker-alerter\n"
        "  3. Le réentraînement continu utilise désormais cet historique (metadata.json : training_data).")


def main(argv=None) -> None:
    sys.path.insert(0, str(Path.cwd()))
    p = argparse.ArgumentParser(description="Intégration des données réelles des opérateurs")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("import", help="lire un export opérateur (pseudonymisé dès la lecture)")
    s.add_argument("--workspace", required=True)
    s.add_argument("--mapping", required=True, help="fichier de correspondance YAML de l'opérateur")
    s.add_argument("--input", required=True, help="export des transactions (CSV, Excel ou Parquet)")
    s.add_argument("--frauds", help="fichier des fraudes signalées / confirmées")
    s.add_argument("--kyc", help="référentiel clients (KYC)")
    s.add_argument("--since", help="ignorer les transactions antérieures (AAAA-MM-JJ)")
    s.add_argument("--sample-users", type=int, help="garder N clients tirés au hasard (essai rapide)")
    s.add_argument("--allow-clear-ids", action="store_true", help="sans pseudonymisation : données de TEST seulement")
    s = sub.add_parser("build", help="fusionner les sources et produire le rapport qualité")
    s.add_argument("--workspace", required=True)
    s = sub.add_parser("train", help="entraîner dans l'espace de travail et comparer à la production")
    s.add_argument("--workspace", required=True)
    s.add_argument("--quick", action="store_true", help="entraînement rapide (moins d'époques, sans baselines)")
    s.add_argument("--force", action="store_true", help="ignorer les erreurs bloquantes du rapport qualité")
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("promote", help="mettre le modèle de l'espace de travail en production")
    s.add_argument("--workspace", required=True)
    s.add_argument("--yes", action="store_true")
    s.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    {"import": cmd_import, "build": cmd_build, "train": cmd_train, "promote": cmd_promote}[a.cmd](a)


if __name__ == "__main__":
    main()
