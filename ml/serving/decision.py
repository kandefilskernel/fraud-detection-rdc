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

from dataclasses import dataclass, field

ACTIONS = ("APPROVE", "VERIFY", "BLOCK")
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

        rules = regulatory_rules(feats, tx)
        for rule in rules:
            if ACTION_RANK[rule["min_action"]] > ACTION_RANK[action]:
                action, reason = rule["min_action"], f"règle : {rule['id']}"
        return {
            "action": action,
            "risk_level": self.risk_level(p),
            "reason": reason,
            "expected_costs_usd": {k: round(v, 4) for k, v in costs.items()},
            "rules_triggered": [r["id"] for r in rules],
        }


def regulatory_rules(feats: dict, tx: dict) -> list[dict]:
    """Règles métier explicites, prioritaires sur le modèle (conformité et bon sens terrain).
    Elles ne peuvent que RENFORCER la décision, jamais l'assouplir."""
    rules = []
    debit = tx["tx_type"] not in ("P2P_RECEIVE", "CASH_IN")
    if debit and feats["is_new_device"] and feats["is_near_full_drain"]:
        rules.append({"id": "NOUVEL_APPAREIL_VIDAGE_COMPTE", "min_action": "VERIFY"})
    if debit and feats["amount_to_kyc_limit"] > 1.0:
        rules.append({"id": "DEPASSEMENT_PLAFOND_KYC", "min_action": "VERIFY"})
    if tx["tx_type"] == "CASH_OUT" and feats["is_new_device"] and feats["is_new_province"]:
        rules.append({"id": "CASH_OUT_APPAREIL_ET_PROVINCE_INHABITUELS", "min_action": "VERIFY"})
    return rules
