"""
Décision coût-sensible : APPROVE / VERIFY / BLOCK.

Un seuil unique « fraude / pas fraude » ignore le montant : bloquer 2 USD de crédit
téléphonique et laisser passer 900 USD de cash-out n'ont pas le même coût. On choisit
l'action qui minimise la perte attendue (p = probabilité de fraude, m = montant USD) :

    APPROVE : p × m                                   (fraude non stoppée)
    VERIFY  : friction + p × m × (1 − taux_capture)   (confirmation PIN USSD / OTP / appel)
    BLOCK   : (1 − p) × (coût_faux_blocage + part_m × m)  (client honnête bloqué)

Paramètres par canal (Mobile Money / Visa virtuelle), réglables sans réentraîner.
Des règles réglementaires explicites passent ensuite AVANT le coût (auditables BCC).

Le niveau de risque affiché aux analystes (FAIBLE / MOYEN / ÉLEVÉ / CRITIQUE) dépend
uniquement de p, par rapport au seuil F1 choisi sur la validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ACTIONS = ("APPROVE", "VERIFY", "BLOCK")
# refund_ratio_to_cp = log(1 + montant renvoyé / montant reçu de ce numéro) ; seuil = renvoi > 3 × reçu
REFUND_RATIO_LOG_THRESHOLD = math.log1p(3.0)
# cp_log_age_days = log(1 + âge du portefeuille bénéficiaire en jours) ; signalements SMS : moins de 30 jours
SMS_REPORT_MAX_CP_AGE = math.log1p(30.0)
ACTION_RANK = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class ChannelCosts:
    friction_usd: float          # coût d'une vérification (temps client, abandon, SMS)
    step_up_catch_rate: float    # part des fraudes stoppées par la vérification
    false_block_usd: float       # coût fixe d'un client honnête bloqué (appel, churn)
    false_block_amount_share: float  # + part du montant (transaction perdue)


@dataclass
class DecisionPolicy:
    threshold: float                                   # seuil F1 du modèle (validation)
    costs: dict = field(default_factory=lambda: {
        "MOBILE_MONEY": ChannelCosts(0.30, 0.85, 4.0, 0.05),
        # 3-D Secure : vérification moins coûteuse et plus efficace sur carte
        "VISA_VIRTUAL": ChannelCosts(0.15, 0.90, 6.0, 0.05),
    })
    block_floor: float = 0.97   # au-delà : blocage quel que soit le montant
    approve_ceiling: float = 0.01  # en deçà : approbation directe (pas de friction inutile)
    sim_swap_window_h: float = 24.0   # refroidissement après changement de SIM
    network_verify_score: int = 90    # score réseau carte (0-99) déclenchant une vérification
    network_block_score: int = 98     # ... et un blocage

    def risk_level(self, p: float) -> str:
        t = self.threshold
        if p >= max(0.9, t):
            return "CRITIQUE"
        if p >= t:
            return "ELEVE"
        if p >= t / 5:
            return "MOYEN"
        return "FAIBLE"

    def expected_costs(self, p: float, amount_usd: float, channel: str) -> dict:
        c = self.costs.get(channel, self.costs["MOBILE_MONEY"])
        return {
            "APPROVE": p * amount_usd,
            "VERIFY": c.friction_usd + p * amount_usd * (1 - c.step_up_catch_rate),
            "BLOCK": (1 - p) * (c.false_block_usd + c.false_block_amount_share * amount_usd),
        }

    def decide(self, p: float, amount_usd: float, channel: str, feats: dict, tx: dict) -> dict:
        costs = self.expected_costs(p, amount_usd, channel)
        if p >= self.block_floor:
            action, reason = "BLOCK", "probabilité de fraude très élevée"
        elif p < self.approve_ceiling:
            action, reason = "APPROVE", "risque négligeable"
        else:
            action = min(costs, key=costs.get)
            reason = "coût attendu minimal"

        rules = regulatory_rules(feats, tx, self.sim_swap_window_h, self.network_verify_score,
                                 self.network_block_score)
        for rule in rules:
            if ACTION_RANK[rule["min_action"]] > ACTION_RANK[action]:
                action, reason = rule["min_action"], f"règle : {rule['id']}"
        return {
            "action": action,
            "risk_level": self.risk_level(p),
            "reason": reason,
            "expected_costs_usd": {k: round(v, 4) for k, v in costs.items()},
            "rules_triggered": [r["id"] for r in rules],
            "verification_method": verification_method(action, channel, tx, self.sim_swap_window_h),
        }


def verification_method(action: str, channel: str, tx: dict, sim_swap_window_h: float = 24.0) -> str | None:
    """Comment vérifier le client quand l'action est VERIFY.
    Après un changement de SIM récent, un code envoyé par SMS ou un rappel sur le numéro
    arrive chez le fraudeur : la vérification doit passer par un autre canal."""
    if action != "VERIFY":
        return None
    hours = tx.get("hours_since_sim_swap")
    if hours is not None and hours < sim_swap_window_h:
        return "HORS_SIM"          # agence avec pièce d'identité, numéro secondaire enregistré
    if channel == "VISA_VIRTUAL":
        return "3DS"               # authentification 3-D Secure (code 1A côté émetteur)
    return "PIN_USSD"              # ressaisie du PIN / confirmation USSD


def regulatory_rules(feats: dict, tx: dict, sim_swap_window_h: float = 24.0,
                     network_verify_score: int = 90, network_block_score: int = 98) -> list[dict]:
    """Règles métier explicites, prioritaires sur le modèle (conformité et bon sens terrain).
    Elles ne peuvent que RENFORCER la décision, jamais l'assouplir."""
    rules = []
    debit = tx["tx_type"] not in ("P2P_RECEIVE", "CASH_IN")

    # Signal télécom : période de refroidissement après un changement de SIM (pratique
    # courante des banques et opérateurs contre le SIM swap)
    hours = tx.get("hours_since_sim_swap")
    if debit and hours is not None and hours < sim_swap_window_h:
        rules.append({"id": "SIM_RECENTE_OPERATION_SORTANTE", "min_action": "VERIFY"})
        if feats["is_near_full_drain"] or feats["amount_to_kyc_limit"] > 0.5:
            rules.append({"id": "SIM_RECENTE_VIDAGE_OU_MONTANT_ELEVE", "min_action": "BLOCK"})

    # Friction ciblée contre l'arnaque « envoi par erreur » : renvoyer plus de 3 fois ce que ce
    # même numéro vient d'envoyer (48 h). Mesuré sur les données : attrape 36 % des arnaques,
    # 0,32 % des envois légitimes demandent une confirmation, une alerte sur deux est une fraude.
    if tx["tx_type"] == "P2P_SEND" and feats.get("refund_ratio_to_cp", 0.0) > REFUND_RATIO_LOG_THRESHOLD:
        rules.append({"id": "RENVOI_TRES_SUPERIEUR_AU_RECU", "min_action": "VERIFY"})

    # Signalements SMS (NLP, ml/nlp) : le bénéficiaire a été signalé comme escroc par au moins un
    # client dans les 30 derniers jours ET son portefeuille a moins de 30 jours (97 % des arnaques
    # visent un numéro récent, contre 19 % des envois honnêtes). Une relation établie n'est donc
    # jamais mise en cause par un seul signalement, même malveillant. Jamais de blocage sur ce
    # motif : un escroc pourrait signaler le numéro d'un innocent pour le faire bloquer.
    if (tx["tx_type"] == "P2P_SEND" and (tx.get("cp_scam_reports_30d") or 0) >= 1
            and feats.get("cp_log_age_days", 0.0) < SMS_REPORT_MAX_CP_AGE):
        rules.append({"id": "BENEFICIAIRE_SIGNALE_PAR_SMS", "min_action": "VERIFY"})

    # Carte Visa : le réseau a déjà son propre score ; il complète le nôtre sans le remplacer
    score = tx.get("network_risk_score")
    if score is not None and score >= network_block_score:
        rules.append({"id": "SCORE_RESEAU_CARTE_CRITIQUE", "min_action": "BLOCK"})
    elif score is not None and score >= network_verify_score:
        rules.append({"id": "SCORE_RESEAU_CARTE_ELEVE", "min_action": "VERIFY"})

    if debit and feats["is_new_device"] and feats["is_near_full_drain"]:
        rules.append({"id": "NOUVEL_APPAREIL_VIDAGE_COMPTE", "min_action": "VERIFY"})
    if debit and feats["amount_to_kyc_limit"] > 1.0:
        rules.append({"id": "DEPASSEMENT_PLAFOND_KYC", "min_action": "VERIFY"})
    if tx["tx_type"] == "CASH_OUT" and feats["is_new_device"] and feats["is_new_province"]:
        rules.append({"id": "CASH_OUT_APPAREIL_ET_PROVINCE_INHABITUELS", "min_action": "VERIFY"})
    return rules
