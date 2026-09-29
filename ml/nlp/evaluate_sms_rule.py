"""
Ce que les signalements de SMS d'arnaque apportent à la détection, sur le mois de test.

Simulation (hypothèses documentées dans docs/NLP_SIGNALEMENTS_SMS.md) :
  1. Chaque mule utilisée dans une arnaque par ingénierie sociale mène une CAMPAGNE de SMS :
     K messages envoyés à des inconnus pendant sa période d'activité (3 jours avant sa
     première victime jusqu'à 1 jour après la dernière).
  2. Chaque destinataire SIGNALE le message avec une probabilité r (inconnue en réalité :
     plusieurs valeurs testées), avec un délai de quelques minutes à quelques heures.
  3. Chaque signalement passe par la VRAIE chaîne : texte généré (modèle de message jamais
     vu à l'apprentissage du classifieur) -> masquage et extraction des numéros -> classifieur.
     Le numéro n'est marqué que si P(arnaque) >= seuil et l'expéditeur n'est pas un opérateur.
  4. Des messages LÉGITIMES sont aussi signalés par doute (38 % des signalements, comme dans
     le corpus) ; ils contiennent de vrais numéros de clients honnêtes. Une erreur du
     classifieur marque alors un honnête : c'est le coût mesuré de la méthode.
  5. Règle BENEFICIAIRE_SIGNALE_PAR_SMS : tout envoi vers un numéro signalé dans les 30
     derniers jours demande une confirmation. On compare la décision de la plateforme
     (modèle + règles existantes) avec et sans cette règle.

    python -m ml.nlp.evaluate_sms_rule
"""
from __future__ import annotations

import json
import random
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.feature_engineering import canon_id
from ml.nlp.scam_sms import ScamSmsClassifier, build_pipeline, prepare
from ml.nlp.sms_corpus import TEMPLATES, generate, render
from ml.serving.decision import DecisionPolicy
from shared.nlp.phone_numbers import extract_numbers, is_phone_number, mask_numbers

ROOT = Path(__file__).absolute().parents[2]
REPORT = ROOT / "ml" / "reports" / "nlp_sms_rule_eval.json"
DAY = 86400.0
WINDOW_S = 30 * DAY
RULE_FEATURES = ["is_near_full_drain", "amount_to_kyc_limit", "is_new_device", "is_new_province", "refund_ratio_to_cp",
                 "cp_log_age_days"]
# modèles de messages réservés à la simulation : le classifieur utilisé ici ne les a jamais vus
HELD_OUT = {"ee2", "ee6", "ee10", "ee13", "ee14", "dc4", "dc7", "fa9",
            "lg1", "lg6", "lg9", "lg13", "lg20", "lg21", "lg8", "lg22", "lg24"}
SCAM_MIX = {"ENVOI_ERREUR": 0.7, "DEMANDE_CODE": 0.15, "FAUX_AGENT": 0.15}


def to_s(ts: pd.Series) -> np.ndarray:
    return ((ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()


def display(num: str, rng: random.Random) -> str:
    """Numéro canonique 243XXXXXXXXX écrit comme dans un vrai SMS."""
    d = num[3:]
    return rng.choice([f"+243 {d[:2]} {d[2:5]} {d[5:]}", f"0{d}", f"0{d[:2]}-{d[2:5]}-{d[5:7]}-{d[7:]}", f"(+243) {d}", num])


def load() -> tuple[pd.DataFrame, pd.Timestamp]:
    prep = json.loads((ROOT / "ml" / "artifacts" / "preprocessing.json").read_text(encoding="utf-8"))
    test_from = pd.Timestamp(prep["split_bounds"]["test_from"])
    tx = pd.read_csv(ROOT / "ml" / "data" / "raw" / "transactions.csv", low_memory=False,
                     usecols=["transaction_id", "timestamp", "channel", "tx_type", "amount_usd", "counterparty_id"],
                     dtype={"counterparty_id": str}, parse_dates=["timestamp"])
    tx["counterparty_id"] = tx["counterparty_id"].map(canon_id)
    truth = pd.read_csv(ROOT / "ml" / "data" / "processed" / "meta.csv", usecols=["transaction_id", "is_fraud_true", "fraud_type_true"])
    return tx.merge(truth, on="transaction_id"), test_from


def train_simulation_classifier(seed: int = 7) -> ScamSmsClassifier:
    df = pd.DataFrame(generate(8000, seed))
    df = df[~df.template.isin(HELD_OUT)]
    pipe = build_pipeline().fit([prepare(t, s) for t, s in zip(df.text, df.sender)], df.category)
    return ScamSmsClassifier(pipe)


def simulate_reports(tx: pd.DataFrame, clf: ScamSmsClassifier, r: float, seed: int = 11,
                     campaign=(40, 200), legit_share: float = 0.38) -> tuple[dict[str, list[float]], dict]:
    """Horodatages des signalements qui marquent chaque numéro, et statistiques de la simulation."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    se = tx[(tx.fraud_type_true == "SOCIAL_ENGINEERING") & (tx.tx_type == "P2P_SEND") & tx.counterparty_id.notna()]
    ts_all = to_s(se.timestamp)
    spans = pd.DataFrame({"cp": se.counterparty_id.to_numpy(), "t": ts_all}).groupby("cp").t.agg(["min", "max"])
    scam_tpl = {c: [t for t in TEMPLATES[c] if t[0] in HELD_OUT] or TEMPLATES[c] for c in SCAM_MIX}
    legit_tpl = [t for t in TEMPLATES["LEGITIME"] if t[0] in HELD_OUT and "{num}" in t[3]]

    flags: dict[str, list[float]] = defaultdict(list)
    stats = {"campagnes": int(len(spans)), "signalements_arnaque": 0, "arnaques_retenues": 0,
             "signalements_legitimes": 0, "legitimes_marques_a_tort": 0, "numeros_extraits_ok": 0}
    for mule, (t0, t1) in spans.iterrows():
        k = rng.randint(*campaign)
        sends = np_rng.uniform(t0 - 3 * DAY, t1 + DAY, size=k)
        reported = sends[np_rng.random(k) < r]
        for s in reported:
            cat = rng.choices(list(SCAM_MIX), weights=SCAM_MIX.values())[0]
            _, _, _, tpl = rng.choice(scam_tpl[cat])
            text, _ = render(tpl, rng, number=display(mule, rng))
            sender = display(mule, rng) if rng.random() < 0.5 else f"0{rng.randrange(8 * 10**8, 10**9)}"
            res = clf.classify(mask_numbers(text), is_phone_number(sender))
            stats["signalements_arnaque"] += 1
            nums = extract_numbers(text)
            stats["numeros_extraits_ok"] += mule in nums
            if res["scam_probability"] >= clf.threshold and mule in nums:
                stats["arnaques_retenues"] += 1
                flags[mule].append(s + float(np_rng.lognormal(np.log(45 * 60), 1.0)))

    # avance sur les victimes : le numéro est-il marqué avant que la première victime ne paie ?
    first_flag = {m: min(v) for m, v in flags.items() if v}
    stats["mules_marquees"] = int(sum(m in first_flag for m in spans.index))
    stats["mules_marquees_avant_premiere_victime"] = int(sum(m in first_flag and first_flag[m] < t0
                                                             for m, (t0, _) in spans.iterrows()))

    # signalements de messages légitimes contenant le numéro d'un client honnête
    n_legit = int(round(stats["signalements_arnaque"] * legit_share / (1 - legit_share)))
    honest = tx[(tx.is_fraud_true == 0) & (tx.tx_type == "P2P_SEND") & tx.counterparty_id.notna()]
    picks = honest.sample(n_legit, replace=True, random_state=seed)
    for cp_id, t in zip(picks.counterparty_id, to_s(picks.timestamp)):
        _, _, sender_kind, tpl = rng.choice(legit_tpl)
        text, _ = render(tpl, rng, number=display(cp_id, rng))
        sender = "VODACOM" if sender_kind == "operateur" else f"0{rng.randrange(8 * 10**8, 10**9)}"
        stats["signalements_legitimes"] += 1
        if not is_phone_number(sender):
            continue                       # message de l'opérateur : jamais utilisé pour marquer un numéro
        res = clf.classify(mask_numbers(text), True)
        if res["scam_probability"] >= clf.threshold and cp_id in extract_numbers(text):
            stats["legitimes_marques_a_tort"] += 1
            flags[cp_id].append(t + float(np_rng.uniform(-10 * DAY, 10 * DAY)))
    for v in flags.values():
        v.sort()
    return flags, stats


def main(rates=(0.005, 0.01, 0.02, 0.05)) -> dict:
    tx, test_from = load()
    test = tx[tx.timestamp >= test_from].copy()
    preds = pd.read_csv(ROOT / "ml" / "reports" / "test_predictions.csv", usecols=["transaction_id", "score_production"])
    feats = pd.read_csv(ROOT / "ml" / "data" / "processed" / "features.csv", usecols=["transaction_id", *RULE_FEATURES],
                        low_memory=False)
    test = test.merge(preds, on="transaction_id").merge(feats, on="transaction_id")
    policy = DecisionPolicy(threshold=0.2542)
    rows = [({k: row[k] for k in RULE_FEATURES}, p, a, c, t) for p, a, c, t, (_, row) in
            zip(test.score_production, test.amount_usd, test.channel, test.tx_type, test[RULE_FEATURES].iterrows())]

    def decide(reports: np.ndarray) -> np.ndarray:
        """Vraie politique de décision (modèle + toutes les règles), avec le nombre de signalements."""
        return np.array([policy.decide(p, a, c, f, {"tx_type": t, "cp_scam_reports_30d": int(n)})["action"] != "APPROVE"
                         for (f, p, a, c, t), n in zip(rows, reports)])

    base_alert = decide(np.zeros(len(test), int))
    fraud = test.is_fraud_true.to_numpy() == 1
    se = (test.fraud_type_true == "SOCIAL_ENGINEERING").to_numpy()
    p2p = (test.tx_type == "P2P_SEND").to_numpy()
    t_s = to_s(test.timestamp)
    clf = train_simulation_classifier()

    results = {"transactions_test": int(len(test)), "fraudes": int(fraud.sum()), "ingenierie_sociale": int(se.sum()),
               "reference_sans_signalements": {
                   "fraudes_arretees": round(float(base_alert[fraud].mean()), 4),
                   "ingenierie_sociale_arretee": round(float(base_alert[se].mean()), 4),
                   "honnetes_deranges": round(float(base_alert[~fraud].mean()), 5)},
               "par_taux_de_signalement": {}}
    for r in rates:
        flags, stats = simulate_reports(tx, clf, r)
        reports = np.zeros(len(test), int)
        for i in np.flatnonzero(p2p):
            times = flags.get(test.counterparty_id.iat[i])
            if times:
                j = bisect_left(times, t_s[i])            # signalements strictement antérieurs
                reports[i] = int(j > 0 and t_s[i] - times[j - 1] <= WINDOW_S)
        alert = decide(reports)
        new = alert & ~base_alert
        by_type = {t: round(float(alert[(test.fraud_type_true == t).to_numpy()].mean()), 4)
                   for t in ("SOCIAL_ENGINEERING", "SIM_SWAP", "ACCOUNT_TAKEOVER")}
        results["par_taux_de_signalement"][f"{r:.3f}"] = {
            "simulation": stats,
            "fraudes_arretees": round(float(alert[fraud].mean()), 4),
            "ingenierie_sociale_arretee": round(float(alert[se].mean()), 4),
            "par_typologie": by_type,
            "honnetes_deranges": round(float(alert[~fraud].mean()), 5),
            "alertes_ajoutees_par_la_regle": int(new.sum()),
            "dont_vraies_fraudes": int((new & fraud).sum()),
            "honnetes_ajoutes": int((new & ~fraud).sum()),
        }
        x = results["par_taux_de_signalement"][f"{r:.3f}"]
        print(f"r={r:.3f} : ingénierie sociale {results['reference_sans_signalements']['ingenierie_sociale_arretee']:.1%} -> "
              f"{x['ingenierie_sociale_arretee']:.1%} | toutes fraudes -> {x['fraudes_arretees']:.1%} | "
              f"+{x['dont_vraies_fraudes']} fraudes, +{x['honnetes_ajoutes']} honnêtes dérangés", flush=True)
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = main()
    print(json.dumps(out["reference_sans_signalements"], ensure_ascii=False))
