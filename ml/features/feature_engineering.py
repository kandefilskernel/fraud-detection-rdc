"""
Phase 2 : feature engineering comportemental.

Principe : chaque transaction est comparée au PROFIL PASSÉ de son titulaire
(montants habituels, appareils, lieux, horaires, destinataires). Les variables sont
calculées transaction par transaction, dans l'ordre chronologique, à partir de l'état
accumulé jusque-là : aucune information future n'est utilisée (pas de fuite temporelle).

La classe `BehavioralFeatureExtractor` est volontairement « en ligne » (une transaction
à la fois) : le même code sert à l'entraînement (rejeu de l'historique) et, en phase 6,
au feature-engineering-service temps réel, ce qui évite l'écart entraînement/production.

Familles de variables (voir FEATURE_GROUPS) :
    montant, vélocité, appareil, localisation, contrepartie/marchand, temporel,
    trans-canal (Mobile Money -> carte), profil KYC, type de transaction.

Usage :
    python -m ml.features.feature_engineering          # écrit ml/data/processed/features.csv
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("ml/data/raw")
PROCESSED_DIR = Path("ml/data/processed")

DEBIT_TYPES = {"P2P_SEND", "CASH_OUT", "MERCHANT_PAYMENT", "AIRTIME", "BILL_PAYMENT",
               "CARD_TOPUP", "CARD_PURCHASE"}
TX_TYPES = ["P2P_SEND", "P2P_RECEIVE", "CASH_IN", "CASH_OUT", "MERCHANT_PAYMENT",
            "AIRTIME", "BILL_PAYMENT", "CARD_TOPUP", "CARD_PURCHASE"]
# Connaissance métier (catégories marchandes réputées à risque), pas un label
HIGH_RISK_MERCHANT_CATEGORIES = {"ELECTRONICS", "GIFT_CARDS", "CRYPTO_EXCHANGE", "GAMING"}
HOME_COUNTRY = "CD"

HOUR, DAY, WEEK = 3600.0, 86400.0, 7 * 86400.0
NO_HISTORY_GAP_S = 30 * DAY  # valeur par défaut « jamais vu » pour les délais
INFLOW_WINDOW_S = 2 * DAY    # réceptions gardées dans le profil du titulaire
WALLET_WINDOW_S = WEEK       # fenêtre du profil d'un portefeuille contrepartie
WALLET_MAX_EVENTS = 300      # borne mémoire par portefeuille (identique en production, Redis)
P2P_TYPES = ("P2P_SEND", "P2P_RECEIVE")
ID_COLUMNS = ("user_id", "device_id", "counterparty_id", "agent_id", "merchant_id")


def canon_id(v) -> str | None:
    """Identifiant en texte canonique (même règle que shared.schemas.canonical_id) : pandas lit
    les numéros de portefeuille tantôt en texte, tantôt en flottant (243917538864.0) selon les
    blocs du CSV ; sans cette normalisation, un même portefeuille compterait pour deux."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)

FEATURE_GROUPS = {
    "amount": ["log_amount_usd", "amount_zscore_user", "amount_ratio_user_mean",
               "amount_to_balance", "is_near_full_drain", "amount_to_kyc_limit", "log_balance_before"],
    "velocity": ["tx_count_1h", "tx_count_24h", "tx_count_7d", "debit_count_1h",
                 "log_debit_sum_24h", "log_secs_since_last_tx", "failed_count_24h"],
    "device": ["is_new_device", "log_device_prior_uses", "n_devices_user", "device_n_users",
               "is_smartphone", "access_app", "access_ussd", "access_agent"],
    "location": ["is_foreign_ip", "is_new_province", "is_away_from_home"],
    "counterparty": ["is_new_counterparty", "counterparty_n_users", "is_new_agent",
                     "agent_n_users_24h", "is_new_merchant", "is_high_risk_merchant",
                     "is_foreign_merchant"],
    "temporal": ["hour_sin", "hour_cos", "is_night", "user_night_ratio", "night_unusual",
                 "hour_deviation_user", "is_weekend", "is_month_end", "log_history_days",
                 "log_user_tx_count"],
    "cross_channel": ["is_card", "log_mins_since_topup", "purchase_to_topup_ratio"],
    "kyc_profile": ["kyc_level", "log_account_age_days", "log_monthly_income", "has_visa_virtual"],
    "tx_type": [f"type_{t}" for t in TX_TYPES],
    # C1 — profilage bidirectionnel : la contrepartie (destinataire d'un envoi, expéditeur
    # d'une réception) et le titulaire EN TANT QUE destinataire, sur tous les opérateurs.
    "recipient_network": ["cp_is_customer", "cp_log_age_days", "cp_in_senders_7d", "cp_in_new_ratio_7d",
                          "cp_in_operators_7d", "cp_out_receivers_7d", "cp_log_in_amount_7d",
                          "refund_ratio_to_cp", "log_mins_since_received_from_cp",
                          "in_senders_24h", "in_new_senders_24h", "log_inflow_24h", "passthrough_ratio"],
    # Profil de réputation : entités (appareil, portefeuille, agent, marchand, compte) déjà
    # impliquées dans des fraudes SIGNALÉES avant la transaction (délai de signalement respecté)
    "reputation": ["rep_device_frauds", "rep_cp_frauds", "rep_agent_frauds_7d", "rep_merchant_frauds_30d",
                   "rep_user_frauds_30d"],
}
REPUTATION_KINDS = ("dev", "cp", "agent", "merchant", "user")
MONTH = 30 * DAY
FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]
META_COLUMNS = ["transaction_id", "timestamp", "user_id", "channel", "tx_type", "amount_usd",
                "is_fraud", "fraud_type"]
# Champs d'une transaction lus par l'extracteur (rejeu hors ligne, amorçage, production)
EXTRACTOR_COLUMNS = ["user_id", "operator", "tx_type", "channel", "amount_usd", "balance_before_usd",
                     "status", "device_id", "device_type", "access_channel", "ip_country",
                     "location_province", "counterparty_id", "agent_id", "merchant_id",
                     "merchant_category", "merchant_country"]


@dataclass
class _UserState:
    n: int = 0
    log_amt_sum: float = 0.0
    log_amt_sq: float = 0.0
    amt_sum: float = 0.0
    first_ts: float | None = None
    last_ts: float | None = None
    recent: deque = field(default_factory=deque)       # (ts, amount_usd, is_debit) sur 7 jours
    failures: deque = field(default_factory=deque)     # ts des échecs sur 24 h
    devices: dict = field(default_factory=dict)        # device_id -> nb d'utilisations
    provinces: set = field(default_factory=set)
    counterparties: set = field(default_factory=set)
    agents: set = field(default_factory=set)
    merchants: set = field(default_factory=set)
    night_count: int = 0
    hour_sin_sum: float = 0.0
    hour_cos_sum: float = 0.0
    last_topup_ts: float | None = None
    last_topup_amt: float = 0.0
    inflows: deque = field(default_factory=deque)      # (ts, portefeuille, montant, 1er contact) sur 48 h


def _wallet_events() -> deque:
    return deque(maxlen=WALLET_MAX_EVENTS)


@dataclass
class _WalletState:
    """Profil d'un portefeuille vu comme contrepartie (client ou portefeuille d'un autre
    réseau), alimenté par les transactions de TOUS les titulaires et de tous les opérateurs."""
    first_seen: float | None = None
    inflows: deque = field(default_factory=_wallet_events)   # (ts, expéditeur, montant, 1er contact, opérateur)
    outflows: deque = field(default_factory=_wallet_events)  # (ts, bénéficiaire, montant)


class BehavioralFeatureExtractor:
    """Calcule les variables d'une transaction à partir de l'historique vu jusqu'ici,
    puis met l'état à jour. Appeler `process` dans l'ordre chronologique."""

    def __init__(self, users: pd.DataFrame, period_start: pd.Timestamp):
        self.period_start_s = period_start.timestamp()
        users = users.assign(user_id=users["user_id"].map(canon_id))
        self.profiles = users.set_index("user_id")[
            ["province", "kyc_level", "kyc_tx_limit_usd", "account_age_days",
             "monthly_income_usd", "has_visa_virtual"]].to_dict("index")
        self.users: dict[str, _UserState] = {}
        self.device_users: dict[str, set] = {}          # appareil -> titulaires vus
        # portefeuille externe -> titulaires en relation avec lui ; un portefeuille qui
        # reçoit de (ou alimente) nombreux comptes est typique d'un réseau de mules
        self.counterparty_users: dict[str, set] = {}
        self.agent_recent: dict[str, deque] = {}        # agent -> (ts, user) sur 24 h
        # C1 : portefeuille -> titulaire (clients KYC connus) et profil de chaque portefeuille
        self.wallet_owner: dict[str, str] = (
            {canon_id(w): canon_id(u) for w, u in zip(users["wallet_id"], users["user_id"])}
            if "wallet_id" in users.columns else {})
        self.wallets: dict[str, _WalletState] = {}
        # réputation : type d'entité -> identifiant -> {transaction frauduleuse signalée : son horodatage}
        self.reputation: dict[str, dict[str, dict[str, float]]] = {k: {} for k in REPUTATION_KINDS}

    def process(self, tx: dict) -> dict:
        feats = self.extract(tx)
        self.update(tx)
        return feats

    @staticmethod
    def report_entities(event: dict) -> list[tuple[str, str]]:
        """Entités marquées par une fraude signalée : l'appareil utilisé, le portefeuille
        contrepartie (P2P), l'agent, le marchand et le compte concerné (victime compromise ou
        compte mule : dans les deux cas un compte « chaud »)."""
        cp = event.get("counterparty_id") if event.get("tx_type") in P2P_TYPES else None
        pairs = [("dev", event.get("device_id")), ("cp", cp), ("agent", event.get("agent_id")),
                 ("merchant", event.get("merchant_id")), ("user", event.get("user_id"))]
        return [(k, v) for k, v in pairs if isinstance(v, str) and v]

    def report(self, event: dict) -> None:
        """Intègre une fraude SIGNALÉE (plainte, enquête, contestation) au profil de réputation.
        event : transaction_id, ts (horodatage de la transaction frauduleuse) et ses entités.
        Un même signalement reçu deux fois ne compte qu'une fois."""
        for kind, key in self.report_entities(event):
            self.reputation[kind].setdefault(key, {})[event["transaction_id"]] = float(event["ts"])

    # ------------------------------------------------------------------ extraction
    def extract(self, tx: dict) -> dict:
        ts, uid = tx["ts"], tx["user_id"]
        st = self.users.get(uid) or _UserState()
        prof = self.profiles[uid]
        amt = float(tx["amount_usd"])
        log_amt = math.log1p(amt)
        tx_type = tx["tx_type"]
        is_debit = tx_type in DEBIT_TYPES

        f: dict[str, float] = {}

        # --- montant : écart au profil de dépense habituel
        if st.n >= 3:
            mean = st.log_amt_sum / st.n
            std = math.sqrt(max(st.log_amt_sq / st.n - mean ** 2, 0.0)) + 0.25
            f["amount_zscore_user"] = float(np.clip((log_amt - mean) / std, -10, 10))
            f["amount_ratio_user_mean"] = math.log1p(amt / (st.amt_sum / st.n + 1e-6))
        else:
            f["amount_zscore_user"] = 0.0
            f["amount_ratio_user_mean"] = math.log(2.0)
        bal = float(tx["balance_before_usd"])
        f["log_amount_usd"] = log_amt
        f["amount_to_balance"] = min(amt / (bal + 1.0), 5.0) if is_debit else 0.0
        f["is_near_full_drain"] = float(is_debit and bal > 1.0 and amt >= 0.85 * bal)
        f["amount_to_kyc_limit"] = amt / prof["kyc_tx_limit_usd"]
        f["log_balance_before"] = math.log1p(max(bal, 0.0))

        # --- vélocité
        c1h = c24h = d1h = 0
        dsum24 = 0.0
        for t, a, deb in st.recent:
            age = ts - t
            if age <= DAY:
                c24h += 1
                if deb:
                    dsum24 += a
                if age <= HOUR:
                    c1h += 1
                    d1h += deb
        f["tx_count_1h"] = float(c1h)
        f["tx_count_24h"] = float(c24h)
        f["tx_count_7d"] = float(len(st.recent))
        f["debit_count_1h"] = float(d1h)
        f["log_debit_sum_24h"] = math.log1p(dsum24)
        gap = ts - st.last_ts if st.last_ts is not None else NO_HISTORY_GAP_S
        f["log_secs_since_last_tx"] = math.log1p(max(gap, 0.0))
        f["failed_count_24h"] = float(sum(1 for t in st.failures if ts - t <= DAY))

        # --- appareil
        dev = tx["device_id"]
        prior_uses = st.devices.get(dev, 0)
        f["is_new_device"] = float(st.n > 0 and prior_uses == 0)
        f["log_device_prior_uses"] = math.log1p(prior_uses)
        f["n_devices_user"] = float(len(st.devices))
        others = self.device_users.get(dev, set()) - {uid}
        f["device_n_users"] = float(min(len(others), 20))
        f["is_smartphone"] = float(tx["device_type"] == "smartphone")
        access = tx["access_channel"]
        f["access_app"] = float(access == "APP")
        f["access_ussd"] = float(access == "USSD")
        f["access_agent"] = float(access == "AGENT")

        # --- localisation
        ip = tx["ip_country"]
        prov = tx["location_province"]
        f["is_foreign_ip"] = float(isinstance(ip, str) and ip != HOME_COUNTRY)
        f["is_new_province"] = float(st.n > 0 and prov not in st.provinces)
        f["is_away_from_home"] = float(prov != prof["province"])

        # --- contrepartie, agent, marchand
        cp = tx["counterparty_id"] if tx_type in ("P2P_SEND", "P2P_RECEIVE") else None
        f["is_new_counterparty"] = float(cp is not None and cp not in st.counterparties)
        linked = self.counterparty_users.get(cp, set()) if cp is not None else set()
        f["counterparty_n_users"] = math.log1p(len(linked - {uid}))
        agent = tx["agent_id"] if isinstance(tx["agent_id"], str) else None
        f["is_new_agent"] = float(agent is not None and agent not in st.agents)
        if agent is not None:
            f["agent_n_users_24h"] = math.log1p(len({u for t, u in self.agent_recent.get(agent, ())
                                                     if ts - t <= DAY and u != uid}))
        else:
            f["agent_n_users_24h"] = 0.0
        merch = tx["merchant_id"] if isinstance(tx["merchant_id"], str) else None
        f["is_new_merchant"] = float(merch is not None and merch not in st.merchants)
        f["is_high_risk_merchant"] = float(tx["merchant_category"] in HIGH_RISK_MERCHANT_CATEGORIES)
        mc = tx["merchant_country"]
        f["is_foreign_merchant"] = float(isinstance(mc, str) and mc != HOME_COUNTRY)

        # --- temporel : l'heure est-elle inhabituelle POUR CET utilisateur ?
        hour = tx["hour"]
        ang = 2 * math.pi * hour / 24
        f["hour_sin"], f["hour_cos"] = math.sin(ang), math.cos(ang)
        is_night = hour <= 5
        f["is_night"] = float(is_night)
        night_ratio = (st.night_count + 0.05) / (st.n + 1)  # lissage : prior faible
        f["user_night_ratio"] = night_ratio
        f["night_unusual"] = float(is_night) * (1 - night_ratio)
        if st.n >= 3:
            mx, my = st.hour_cos_sum / st.n, st.hour_sin_sum / st.n
            resultant = math.hypot(mx, my)
            if resultant > 0.1:
                diff = abs(math.atan2(my, mx) - ang) % (2 * math.pi)
                f["hour_deviation_user"] = min(diff, 2 * math.pi - diff) / math.pi * resultant
            else:
                f["hour_deviation_user"] = 0.0
        else:
            f["hour_deviation_user"] = 0.0
        f["is_weekend"] = float(tx["weekday"] >= 5)
        f["is_month_end"] = float(tx["day"] >= 25 or tx["day"] <= 3)
        f["log_history_days"] = math.log1p(max(ts - st.first_ts, 0.0) / DAY) if st.first_ts is not None else 0.0
        f["log_user_tx_count"] = math.log1p(st.n)

        # --- trans-canal : recharge de carte suivie d'un achat
        f["is_card"] = float(tx["channel"] == "VISA_VIRTUAL")
        if tx_type == "CARD_PURCHASE" and st.last_topup_ts is not None:
            # max(.., 0) : en production une transaction peut arriver en retard (reprise,
            # horloges d'opérateurs décalées) ; à l'entraînement (ordre chronologique) c'est neutre
            f["log_mins_since_topup"] = math.log1p(max(ts - st.last_topup_ts, 0.0) / 60)
            f["purchase_to_topup_ratio"] = min(amt / (st.last_topup_amt + 1.0), 5.0)
        else:
            f["log_mins_since_topup"] = math.log1p(NO_HISTORY_GAP_S / 60)
            f["purchase_to_topup_ratio"] = 0.0

        # --- profil KYC (données connues de l'opérateur)
        f["kyc_level"] = float(prof["kyc_level"])
        f["log_account_age_days"] = math.log1p(max(prof["account_age_days"] + (ts - self.period_start_s) / DAY, 0.0))
        f["log_monthly_income"] = math.log1p(prof["monthly_income_usd"])
        f["has_visa_virtual"] = float(prof["has_visa_virtual"])

        for t in TX_TYPES:
            f[f"type_{t}"] = float(tx_type == t)

        f.update(self._recipient_network_features(st, uid, ts, cp, amt, tx_type, is_debit))
        f.update(self._reputation_features(uid, ts, dev, cp, agent, merch))
        return f

    # ------------------------------------------------------------------ réputation
    def _reputation_features(self, uid, ts, dev, cp, agent, merch) -> dict:
        rep = self.reputation

        def count(kind, key, window=None) -> float:
            times = rep[kind].get(key) if key is not None else None
            if not times:
                return 0.0
            n = len(times) if window is None else sum(1 for t in times.values() if ts - t <= window)
            return math.log1p(n)

        return {
            "rep_device_frauds": count("dev", dev),
            "rep_cp_frauds": count("cp", cp),
            "rep_agent_frauds_7d": count("agent", agent, WEEK),
            "rep_merchant_frauds_30d": count("merchant", merch, MONTH),
            "rep_user_frauds_30d": count("user", uid, MONTH),
        }

    # ------------------------------------------------------------------ C1 : réseau
    def _recipient_network_features(self, st: _UserState, uid: str, ts: float, cp: str | None,
                                    amt: float, tx_type: str, is_debit: bool) -> dict:
        f: dict[str, float] = {}

        # (a) la contrepartie : son ancienneté et ce que les AUTRES clients font avec elle
        ws = self.wallets.get(cp) if cp is not None else None
        owner = self.wallet_owner.get(cp) if cp is not None else None
        if owner is not None and owner in self.profiles:
            age = self.profiles[owner]["account_age_days"] + (ts - self.period_start_s) / DAY
        elif ws is not None and ws.first_seen is not None:
            age = (ts - ws.first_seen) / DAY   # portefeuille d'un autre réseau : vu depuis...
        else:
            age = 0.0                          # jamais vu
        senders, operators, n_in, n_new, amt_in = set(), set(), 0, 0, 0.0
        receivers = set()
        if ws is not None:
            for t, u, a, first, op in ws.inflows:
                if ts - t <= WALLET_WINDOW_S and u != uid:
                    n_in += 1
                    n_new += first
                    amt_in += a
                    senders.add(u)
                    if op:
                        operators.add(op)
            receivers = {u for t, u, a in ws.outflows if ts - t <= WALLET_WINDOW_S and u != uid}
        f["cp_is_customer"] = float(owner is not None)
        f["cp_log_age_days"] = math.log1p(max(age, 0.0))
        f["cp_in_senders_7d"] = math.log1p(len(senders))
        f["cp_in_new_ratio_7d"] = (n_new + 1) / (n_in + 2)   # lissé : 0,5 sans historique
        f["cp_in_operators_7d"] = float(len(operators))
        f["cp_out_receivers_7d"] = math.log1p(len(receivers))
        f["cp_log_in_amount_7d"] = math.log1p(amt_in)

        # (b) arnaque « envoi par erreur » : je renvoie PLUS que ce que cette personne
        # vient de m'envoyer
        last_from_cp = None
        if tx_type == "P2P_SEND" and cp is not None:
            for t, c, a, _ in st.inflows:
                if c == cp and ts - t <= INFLOW_WINDOW_S:
                    last_from_cp = (t, a)
        if last_from_cp is not None:
            f["refund_ratio_to_cp"] = math.log1p(min(amt / (last_from_cp[1] + 0.01), 100.0))
            f["log_mins_since_received_from_cp"] = math.log1p(max(ts - last_from_cp[0], 0.0) / 60)
        else:
            f["refund_ratio_to_cp"] = 0.0
            f["log_mins_since_received_from_cp"] = math.log1p(NO_HISTORY_GAP_S / 60)

        # (c) le titulaire EN TANT QUE destinataire : argent reçu d'inconnus puis ressorti
        senders24, new24, inflow24 = set(), set(), 0.0
        for t, c, a, first in st.inflows:
            if ts - t <= DAY:
                senders24.add(c)
                inflow24 += a
                if first:
                    new24.add(c)
        f["in_senders_24h"] = math.log1p(len(senders24))
        f["in_new_senders_24h"] = math.log1p(len(new24))
        f["log_inflow_24h"] = math.log1p(inflow24)
        f["passthrough_ratio"] = min(amt, inflow24) / amt if is_debit and amt > 0 else 0.0
        return f

    # ------------------------------------------------------------------ mise à jour
    def update(self, tx: dict) -> None:
        """Intègre la transaction au profil. Le statut (succès/échec) n'est connu
        qu'après l'opération : il n'est utilisé que pour les transactions suivantes."""
        ts, uid = tx["ts"], tx["user_id"]
        st = self.users.setdefault(uid, _UserState())
        amt = float(tx["amount_usd"])
        log_amt = math.log1p(amt)
        tx_type = tx["tx_type"]

        st.n += 1
        st.log_amt_sum += log_amt
        st.log_amt_sq += log_amt ** 2
        st.amt_sum += amt
        if st.first_ts is None:
            st.first_ts = ts
        st.last_ts = ts
        st.recent.append((ts, amt, tx_type in DEBIT_TYPES))
        while st.recent and ts - st.recent[0][0] > WEEK:
            st.recent.popleft()
        if tx["status"] != "SUCCESS":
            st.failures.append(ts)
        while st.failures and ts - st.failures[0] > DAY:
            st.failures.popleft()

        dev = tx["device_id"]
        st.devices[dev] = st.devices.get(dev, 0) + 1
        self.device_users.setdefault(dev, set()).add(uid)
        st.provinces.add(tx["location_province"])
        if tx_type in P2P_TYPES and tx["counterparty_id"] is not None:
            cp = tx["counterparty_id"]
            first_contact = cp not in st.counterparties   # évalué AVANT l'ajout
            st.counterparties.add(cp)
            self.counterparty_users.setdefault(cp, set()).add(uid)
            # C1 : profil du portefeuille contrepartie et réceptions du titulaire
            ws = self.wallets.setdefault(cp, _WalletState())
            if ws.first_seen is None:
                ws.first_seen = ts
            if tx_type == "P2P_SEND":
                ws.inflows.append((ts, uid, amt, first_contact, tx.get("operator")))
            else:
                ws.outflows.append((ts, uid, amt))
                st.inflows.append((ts, cp, amt, first_contact))
            for q in (ws.inflows, ws.outflows):
                while q and ts - q[0][0] > WALLET_WINDOW_S:
                    q.popleft()
        while st.inflows and ts - st.inflows[0][0] > INFLOW_WINDOW_S:
            st.inflows.popleft()
        if isinstance(tx["agent_id"], str):
            st.agents.add(tx["agent_id"])
            q = self.agent_recent.setdefault(tx["agent_id"], deque())
            q.append((ts, uid))
            while q and ts - q[0][0] > DAY:
                q.popleft()
        if isinstance(tx["merchant_id"], str):
            st.merchants.add(tx["merchant_id"])

        hour = tx["hour"]
        ang = 2 * math.pi * hour / 24
        st.hour_sin_sum += math.sin(ang)
        st.hour_cos_sum += math.cos(ang)
        st.night_count += hour <= 5
        if tx_type == "CARD_TOPUP" and tx["status"] == "SUCCESS":
            st.last_topup_ts, st.last_topup_amt = ts, amt


def _period_start(timestamps: pd.Series, raw_dir: Path) -> pd.Timestamp:
    """Début de période (pour l'ancienneté des comptes) : lu dans les métadonnées du générateur."""
    meta = raw_dir / "generation_metadata.json"
    if meta.exists():
        return pd.Timestamp(json.loads(meta.read_text(encoding="utf-8"))["config"]["start_date"])
    return pd.to_datetime(timestamps).min().normalize()


PROFILE_SCOPES = ("unified", "silo")


def reported_seconds(tx: pd.DataFrame) -> np.ndarray | None:
    """Date de signalement de chaque transaction (secondes, même convention que ts), NaN si
    la fraude n'est pas (encore) signalée ; None si le jeu de données n'a pas cette colonne."""
    if "fraud_reported_at" not in tx.columns:
        return None
    return ((pd.to_datetime(tx["fraud_reported_at"]) - pd.Timestamp("1970-01-01"))
            / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def replay_history(records: list[dict], extractor_for, tx_ids, reported_s: np.ndarray | None,
                   until_s: float | None = None) -> list[dict]:
    """Rejoue l'historique dans l'ordre du temps : transactions ET signalements de fraude.

    Chaque signalement est intégré à sa DATE DE SIGNALEMENT : une transaction ne voit que les
    fraudes déjà connues de l'opérateur au moment où elle a lieu (pas de fuite du futur).
    Utilisé à l'identique pour l'entraînement (build_feature_table) et l'amorçage de Redis.
    until_s : intègre aussi les signalements reçus après la dernière transaction et avant until_s."""
    reports = [] if reported_s is None else sorted((s, i) for i, s in enumerate(reported_s) if s == s)
    rows, j = [], 0

    def apply(i):
        extractor_for(records[i]).report({**records[i], "transaction_id": str(tx_ids[i])})

    for r in records:
        # strictement avant : une fraude ne peut pas « voir » son propre signalement
        while j < len(reports) and reports[j][0] < r["ts"]:
            apply(reports[j][1])
            j += 1
        rows.append(extractor_for(r).process(r))
    while until_s is not None and j < len(reports) and reports[j][0] < until_s:
        apply(reports[j][1])
        j += 1
    return rows


def build_feature_table(tx: pd.DataFrame, users: pd.DataFrame,
                        period_start: pd.Timestamp | None = None,
                        profile_scope: str = "unified") -> pd.DataFrame:
    """Rejoue l'historique chronologiquement et renvoie META_COLUMNS + FEATURE_NAMES.

    profile_scope (expérience C2) :
        "unified" : un seul profil par titulaire, alimenté par le portefeuille ET la carte
                    (ce que fait la plateforme en production) ;
        "silo"    : un extracteur indépendant par canal, comme deux institutions qui ne
                    partagent pas leurs données (l'émetteur de la carte ne voit que la carte).
    """
    if profile_scope not in PROFILE_SCOPES:
        raise ValueError(f"profile_scope doit valoir {PROFILE_SCOPES}")
    tx = tx.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    if period_start is None:
        period_start = tx["timestamp"].min().normalize()

    work = tx[[c for c in EXTRACTOR_COLUMNS if c in tx.columns]].copy()
    work = work.astype(object).where(work.notna(), None)
    for col in ID_COLUMNS:
        work[col] = work[col].map(canon_id)
    if "operator" not in work.columns:
        work["operator"] = None
    work["ts"] = (tx["timestamp"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    work["hour"] = tx["timestamp"].dt.hour
    work["weekday"] = tx["timestamp"].dt.weekday
    work["day"] = tx["timestamp"].dt.day

    if profile_scope == "unified":
        single = BehavioralFeatureExtractor(users, period_start)
        extractor_for = lambda r: single  # noqa: E731
    else:
        per_channel = {ch: BehavioralFeatureExtractor(users, period_start) for ch in work["channel"].unique()}
        extractor_for = lambda r: per_channel[r["channel"]]  # noqa: E731

    rows = replay_history(work.to_dict("records"), extractor_for, tx["transaction_id"].to_numpy(),
                          reported_seconds(tx))
    feats = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(np.float32)
    meta = tx[[c for c in META_COLUMNS if c in tx.columns]].reset_index(drop=True)
    return pd.concat([meta, feats], axis=1)


def extract_behavioral_features(tx_path: str | Path = RAW_DIR / "transactions.csv",
                                users_path: str | Path = RAW_DIR / "users.csv") -> tuple[pd.DataFrame, pd.Series]:
    """Compatibilité avec l'ancienne API : renvoie (X, y)."""
    table = load_or_build_features(tx_path, users_path)
    return table[FEATURE_NAMES], table["is_fraud"]


def load_or_build_features(tx_path: str | Path = RAW_DIR / "transactions.csv",
                           users_path: str | Path = RAW_DIR / "users.csv",
                           cache_path: str | Path | None = None,
                           profile_scope: str = "unified") -> pd.DataFrame:
    tx_path, users_path = Path(tx_path), Path(users_path)
    if not tx_path.exists() or not users_path.exists():
        raise FileNotFoundError(f"Fichiers introuvables : {tx_path} / {users_path}. "
                                "Lancez d'abord ml/generator/generate_synthetic_data.py")
    tx = pd.read_csv(tx_path, low_memory=False)
    users = pd.read_csv(users_path)
    table = build_feature_table(tx, users, _period_start(tx["timestamp"], tx_path.parent), profile_scope)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path, index=False)
    return table


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    out = PROCESSED_DIR / "features.csv"
    table = load_or_build_features(cache_path=out)
    print(f"[OK] {len(table):,} transactions, {len(FEATURE_NAMES)} variables "
          f"en {time.perf_counter() - t0:.0f} s -> {out}")
