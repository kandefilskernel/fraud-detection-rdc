"""
Traduction des variables du modèle en FAITS lisibles par un analyste.

Utilisée des deux côtés de l'assistant d'enquête :
    - hors ligne, pour décrire les cas de l'archive historique (ml/rag/build_case_archive.py) ;
    - en ligne, pour décrire la transaction à instruire (backoffice-api).

Le même texte des deux côtés permet de comparer « ce qu'on voit ici » avec « ce qu'on
avait vu dans les cas passés ». Les variables arrivent brutes (avant standardisation),
certaines en log1p : on les remet à l'échelle humaine avant de les décrire.

Aucun identifiant (numéro, appareil, agent) n'apparaît dans les faits : seulement des
propriétés (« appareil jamais vu pour ce client »), ce qui permet de les transmettre à un
modèle de langage sans divulguer de données personnelles.
"""
from __future__ import annotations

import math

TYPOLOGY_LABELS = {
    "SIM_SWAP": "Échange frauduleux de carte SIM",
    "ACCOUNT_TAKEOVER": "Prise de contrôle du compte (PIN volé, accès à distance)",
    "SOCIAL_ENGINEERING": "Arnaque par ingénierie sociale (« envoi par erreur », faux agent)",
    "AGENT_FRAUD": "Retrait non autorisé par un agent compromis",
    "SMURFING": "Compte mule / fractionnement (blanchiment)",
    "CARD_TESTING": "Test de carte (rafale de micro-achats)",
    "CNP_FRAUD": "Fraude carte non présente (données de carte volées)",
    "TOPUP_DRAIN": "Recharge de la carte puis dépenses immédiates (trans-canal)",
}

TX_TYPE_LABELS = {
    "P2P_SEND": "envoi à un particulier", "P2P_RECEIVE": "réception d'un particulier",
    "CASH_IN": "dépôt chez un agent", "CASH_OUT": "retrait chez un agent",
    "MERCHANT_PAYMENT": "paiement marchand", "AIRTIME": "achat de crédit téléphonique",
    "BILL_PAYMENT": "paiement de facture", "CARD_TOPUP": "recharge de la carte virtuelle",
    "CARD_PURCHASE": "achat par carte virtuelle",
}


# libellés des variables les plus souvent citées par SHAP (les autres gardent leur nom technique)
FEATURE_LABELS = {
    "log_amount_usd": "montant", "amount_zscore_user": "montant inhabituel pour le client",
    "amount_ratio_user_mean": "montant / moyenne du client", "amount_to_balance": "part du solde débitée",
    "is_near_full_drain": "vidage du compte", "amount_to_kyc_limit": "montant / plafond KYC",
    "log_balance_before": "solde avant opération", "tx_count_1h": "opérations sur 1 h",
    "tx_count_24h": "opérations sur 24 h", "debit_count_1h": "débits sur 1 h",
    "log_debit_sum_24h": "total débité sur 24 h", "log_secs_since_last_tx": "délai depuis la dernière opération",
    "failed_count_24h": "échecs sur 24 h", "is_new_device": "nouvel appareil",
    "log_device_prior_uses": "usages antérieurs de l'appareil", "n_devices_user": "appareils du client",
    "device_n_users": "autres clients sur l'appareil", "access_ussd": "accès USSD", "access_agent": "accès agent",
    "access_app": "accès application", "is_foreign_ip": "IP étrangère", "is_new_province": "nouvelle province",
    "is_away_from_home": "hors province d'origine", "is_new_counterparty": "nouveau bénéficiaire",
    "counterparty_n_users": "clients liés au bénéficiaire", "is_new_agent": "nouvel agent",
    "agent_n_users_24h": "clients de l'agent sur 24 h", "is_high_risk_merchant": "marchand à risque",
    "is_foreign_merchant": "marchand étranger", "is_night": "nuit", "night_unusual": "nuit inhabituelle",
    "hour_deviation_user": "heure inhabituelle", "log_history_days": "ancienneté de l'historique",
    "log_user_tx_count": "nombre d'opérations du client", "log_mins_since_topup": "délai depuis la recharge carte",
    "purchase_to_topup_ratio": "achat / recharge carte", "kyc_level": "niveau KYC",
    "log_account_age_days": "ancienneté du compte", "cp_log_age_days": "ancienneté du bénéficiaire",
    "cp_in_senders_7d": "expéditeurs du bénéficiaire (7 j)", "cp_in_new_ratio_7d": "part de premiers contacts du bénéficiaire",
    "cp_in_operators_7d": "opérateurs des expéditeurs du bénéficiaire", "refund_ratio_to_cp": "renvoi / montant reçu de ce numéro",
    "log_mins_since_received_from_cp": "délai depuis la réception de ce numéro", "in_new_senders_24h": "nouveaux expéditeurs (24 h)",
    "log_inflow_24h": "montant reçu sur 24 h", "passthrough_ratio": "part du montant reçue juste avant",
    "rep_device_frauds": "appareil lié à des fraudes", "rep_cp_frauds": "bénéficiaire lié à des fraudes",
    "rep_agent_frauds_7d": "agent lié à des fraudes (7 j)", "rep_merchant_frauds_30d": "marchand lié à des fraudes (30 j)",
    "rep_user_frauds_30d": "client déjà victime (30 j)",
}


def _exp(v: float) -> float:
    """Inverse de log1p."""
    return math.expm1(max(float(v), 0.0))


LOG1P_COUNTS = {"counterparty_n_users", "agent_n_users_24h", "cp_in_senders_7d", "cp_out_receivers_7d",
                "in_senders_24h", "in_new_senders_24h", "rep_device_frauds", "rep_cp_frauds", "rep_agent_frauds_7d",
                "rep_merchant_frauds_30d", "rep_user_frauds_30d", "log_device_prior_uses", "log_user_tx_count"}


def human_value(name: str, v: float) -> str:
    """Valeur d'une variable à l'échelle humaine (les variables log1p sont reconverties)."""
    v = float(v)
    if name in LOG1P_COUNTS:
        return f"{_exp(v):.0f}"
    if name == "refund_ratio_to_cp" or name == "amount_ratio_user_mean":
        r = _exp(v)
        return f"{r:.1f} fois" if r >= 1 else f"{r:.2f} fois"
    if name.startswith("log_secs"):
        s = _exp(v)
        return f"{s / 60:.0f} min" if s < 86400 else f"{s / 86400:.1f} j"
    if name.startswith("log_mins"):
        m = _exp(v)
        return f"{m:.0f} min" if m < 1440 else f"{m / 1440:.1f} j"
    if name.endswith("_days"):
        return f"{_exp(v):.0f} j"
    if name.startswith("log_"):                       # montants, soldes, revenus
        return f"{_exp(v):.2f} USD"
    if name in ("amount_to_balance", "amount_to_kyc_limit", "passthrough_ratio", "cp_in_new_ratio_7d"):
        return f"{v:.0%}"
    if v in (0.0, 1.0) and (name.startswith(("is_", "access_", "type_", "has_")) or name == "night_unusual"):
        return "oui" if v else "non"
    return f"{v:.2f}"


def describe(f: dict, tx_type: str | None = None, hours_since_sim_swap: float | None = None,
             max_facts: int = 10) -> list[str]:
    """Faits saillants (à charge d'abord), dans l'ordre d'importance pour l'enquête."""
    g = lambda k: float(f.get(k, 0.0) or 0.0)  # noqa: E731
    facts: list[tuple[int, str]] = []     # (priorité, fait) ; priorité faible = plus important

    # --- réputation : entités déjà impliquées dans des fraudes confirmées
    for key, what in (("rep_device_frauds", "cet appareil"), ("rep_cp_frauds", "ce bénéficiaire"),
                      ("rep_agent_frauds_7d", "cet agent (7 derniers jours)"),
                      ("rep_merchant_frauds_30d", "ce marchand (30 derniers jours)"),
                      ("rep_user_frauds_30d", "ce client (30 derniers jours)")):
        n = round(_exp(g(key)))
        if n >= 1:
            facts.append((0, f"{what.capitalize()} : déjà lié à {n} fraude(s) confirmée(s)"))

    if hours_since_sim_swap is not None and hours_since_sim_swap < 72:
        facts.append((0, f"Carte SIM changée il y a {hours_since_sim_swap:.0f} h (signal opérateur)"))

    # --- appareil et localisation
    if g("is_new_device"):
        facts.append((1, "Appareil jamais utilisé par ce client"))
    if g("device_n_users") >= 2:
        facts.append((2, f"Appareil partagé avec {int(g('device_n_users'))} autres clients"))
    if g("is_foreign_ip"):
        facts.append((2, "Connexion depuis une adresse IP étrangère"))
    if g("is_new_province"):
        facts.append((2, "Province jamais vue pour ce client"))

    # --- montant et solde
    if g("is_near_full_drain"):
        facts.append((1, f"Vidage du compte ({min(g('amount_to_balance'), 1.0):.0%} du solde)"))
    elif g("amount_to_balance") >= 0.5:
        facts.append((3, f"Opération sur {g('amount_to_balance'):.0%} du solde"))
    if g("amount_zscore_user") >= 2.5:
        facts.append((2, f"Montant très inhabituel pour ce client (écart de {g('amount_zscore_user'):.1f} σ)"))
    if g("amount_to_kyc_limit") >= 0.8:
        facts.append((2, f"Montant à {g('amount_to_kyc_limit'):.0%} du plafond KYC par opération"))

    # --- vitesse
    if g("debit_count_1h") >= 3:
        facts.append((2, f"{int(g('debit_count_1h'))} débits dans l'heure précédente"))
    elif g("tx_count_1h") >= 4:
        facts.append((3, f"{int(g('tx_count_1h'))} opérations dans l'heure précédente"))
    if g("failed_count_24h") >= 2:
        facts.append((2, f"{int(g('failed_count_24h'))} échecs (PIN, solde) sur 24 h"))

    # --- réseau du bénéficiaire (C1)
    refund = _exp(g("refund_ratio_to_cp"))
    if refund > 3:
        facts.append((0, f"Renvoie {refund:.0f} fois le montant que ce même numéro vient d'envoyer"))
    if g("is_new_counterparty") and tx_type in ("P2P_SEND", None):
        age = _exp(g("cp_log_age_days"))
        if age < 7:
            facts.append((1, f"Bénéficiaire nouveau pour le client, compte de {age:.0f} jour(s)"))
        else:
            facts.append((4, "Bénéficiaire nouveau pour le client"))
    senders = round(_exp(g("cp_in_senders_7d")))
    if senders >= 3:
        new_ratio = g("cp_in_new_ratio_7d")
        facts.append((1, f"Le bénéficiaire a reçu de {senders} expéditeurs différents en 7 j "
                         f"({new_ratio:.0%} de premiers contacts)"))
    if g("cp_in_operators_7d") >= 2:
        facts.append((3, f"Le bénéficiaire reçoit depuis {int(g('cp_in_operators_7d'))} opérateurs"))
    in_new = round(_exp(g("in_new_senders_24h")))
    if in_new >= 3:
        facts.append((1, f"Le client a reçu de {in_new} nouveaux expéditeurs en 24 h"))
    if g("passthrough_ratio") >= 0.8 and in_new >= 2:
        facts.append((1, f"{g('passthrough_ratio'):.0%} du montant vient d'être reçu (fonds de passage)"))

    # --- agent, marchand, carte
    if g("is_new_agent") and tx_type == "CASH_OUT":
        facts.append((3, "Agent jamais utilisé par ce client"))
    if g("is_high_risk_merchant"):
        facts.append((2, "Marchand d'une catégorie à risque"))
    if g("is_foreign_merchant"):
        facts.append((3, "Marchand étranger"))
    if tx_type == "CARD_PURCHASE" and g("log_mins_since_topup"):
        mins = _exp(g("log_mins_since_topup"))
        if mins < 60:
            facts.append((1, f"Achat {mins:.0f} min après une recharge de la carte"))

    # --- heure
    if g("night_unusual") >= 0.5:
        facts.append((2, "Opération de nuit, inhabituelle pour ce client"))
    elif g("hour_deviation_user") >= 0.5:
        facts.append((4, "Heure inhabituelle pour ce client"))

    # --- ancienneté (souvent à décharge)
    hist_days = _exp(g("log_history_days"))
    if 0 < hist_days < 7:
        facts.append((3, f"Historique du client très court ({hist_days:.0f} jour(s))"))

    facts.sort(key=lambda x: x[0])
    return [t for _, t in facts[:max_facts]]


def mitigating(f: dict, tx_type: str | None = None) -> list[str]:
    """Éléments À DÉCHARGE : ce qui ressemble à un comportement habituel du client."""
    g = lambda k: float(f.get(k, 0.0) or 0.0)  # noqa: E731
    out = []
    if not g("is_new_device") and _exp(g("log_device_prior_uses")) >= 20:
        out.append(f"Appareil habituel ({_exp(g('log_device_prior_uses')):.0f} utilisations)")
    # « connu » ne vaut rien si la seule relation est l'envoi qui vient d'arriver (arnaque « erreur »)
    if not g("is_new_counterparty") and tx_type in ("P2P_SEND", "P2P_RECEIVE") and _exp(g("refund_ratio_to_cp")) <= 1.5:
        out.append("Bénéficiaire déjà connu du client")
    if abs(g("amount_zscore_user")) < 1.0 and g("log_user_tx_count") > 0:
        out.append("Montant dans les habitudes du client")
    if not g("is_new_province") and not g("is_foreign_ip"):
        out.append("Lieu habituel")
    if _exp(g("log_history_days")) >= 90:
        out.append(f"Client suivi depuis {_exp(g('log_history_days')):.0f} jours")
    if g("is_month_end") and tx_type in ("P2P_SEND", "BILL_PAYMENT", "CASH_OUT"):
        out.append("Fin/début de mois (salaires, loyers, tontines)")
    return out
