"""
Générateur de données synthétiques : Mobile Money (RDC) et cartes Visa virtuelles.

Produit un jeu de données transactionnel réaliste pour l'entraînement et l'évaluation
du modèle de profilage comportemental :

    - utilisateurs avec profils socio-économiques (province, opérateur, segment, KYC)
    - comportement normal : rythmes horaires individuels, cycles hebdomadaires,
      pics de fin de mois, frais scolaires de septembre, abonnements récurrents
    - cas « difficiles » légitimes : changements de téléphone, téléphones familiaux partagés,
      voyages et VPN, usagers nocturnes, nouveaux destinataires, rafales d'opérations,
      retraits de tout le solde, montants inhabituels mais légitimes
    - 8 typologies de fraude injectées par épisodes (5 Mobile Money, 3 Visa virtuelle,
      dont une trans-canal Mobile Money -> carte), avec des variantes discrètes (prise de
      contrôle à distance du téléphone de la victime, proxy local, montants partiels)
    - soldes cohérents : les fraudes « vident » le compte réel de la victime et échouent
      si le solde est insuffisant
    - étiquettes bruitées comme chez un opérateur : fraudes non signalées, contestations
      abusives (« fraude amicale ») ; vérité terrain dans label_noise.csv

Tous les paramètres de réalisme sont dans GeneratorConfig (les mettre à 0 rend les données
nettement plus faciles : utile pour une analyse de sensibilité dans le mémoire).

Hypothèses de simulation (à citer dans le mémoire) :
    - un solde unique par portefeuille, exprimé en équivalent USD (en réalité CDF et USD
      sont des sous-comptes séparés) ; les montants sont affichés dans la devise préférée
    - plafonds KYC par transaction : paramètres de simulation, PAS les plafonds officiels BCC
    - cartes Visa virtuelles prépayées, alimentées depuis le portefeuille Mobile Money

Usage :
    python ml/generator/generate_synthetic_data.py --n-users 3000 --n-days 180 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("synthetic_generator")

# --------------------------------------------------------------------------- Référentiels

# province : (ville principale, part des utilisateurs Mobile Money)
PROVINCES = {
    "Kinshasa": ("Kinshasa", 0.40),
    "Haut-Katanga": ("Lubumbashi", 0.15),
    "Nord-Kivu": ("Goma", 0.10),
    "Sud-Kivu": ("Bukavu", 0.07),
    "Kongo-Central": ("Matadi", 0.07),
    "Kasai-Oriental": ("Mbuji-Mayi", 0.06),
    "Lualaba": ("Kolwezi", 0.06),
    "Tshopo": ("Kisangani", 0.05),
    "Ituri": ("Bunia", 0.04),
}
OPERATORS = {"VODACOM": 0.45, "AIRTEL": 0.32, "ORANGE": 0.23}

# segment : (poids, montant typique USD, tx/jour, P(smartphone), P(carte Visa si éligible))
SEGMENTS = {
    "etudiant": (0.15, 6.0, 0.35, 0.70, 0.30),
    "commercant": (0.20, 35.0, 0.90, 0.60, 0.40),
    "salarie": (0.20, 25.0, 0.50, 0.75, 0.55),
    "informel": (0.25, 9.0, 0.45, 0.35, 0.15),
    "agriculteur": (0.10, 7.0, 0.20, 0.15, 0.05),
    "diaspora_beneficiaire": (0.10, 45.0, 0.30, 0.65, 0.45),
}

KYC_LEVEL_PROBS = {1: 0.45, 2: 0.40, 3: 0.15}
KYC_TX_LIMIT_USD = {1: 150.0, 2: 1500.0, 3: 5000.0}  # paramètres de simulation

MM_TX_TYPES = ["P2P_SEND", "P2P_RECEIVE", "CASH_IN", "CASH_OUT",
               "MERCHANT_PAYMENT", "AIRTIME", "BILL_PAYMENT"]
MM_TYPE_PROBS = {
    "default": [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05],
    "commercant": [0.20, 0.25, 0.20, 0.15, 0.08, 0.07, 0.05],
    "diaspora_beneficiaire": [0.15, 0.35, 0.05, 0.25, 0.08, 0.08, 0.04],
    "agriculteur": [0.20, 0.25, 0.10, 0.25, 0.05, 0.12, 0.03],
}
# Multiplicateurs calibrés pour que entrées et sorties s'équilibrent en moyenne
AMOUNT_MULTIPLIER = {"P2P_SEND": 1.0, "P2P_RECEIVE": 1.5, "CASH_IN": 2.5, "CASH_OUT": 1.5,
                     "MERCHANT_PAYMENT": 0.8, "AIRTIME": 0.15, "BILL_PAYMENT": 1.0}
CREDIT_TYPES = {"P2P_RECEIVE", "CASH_IN"}
DEBIT_TX_TYPES = {"P2P_SEND", "CASH_OUT", "MERCHANT_PAYMENT", "AIRTIME", "BILL_PAYMENT"}

MM_MERCHANT_CATEGORIES = ["SUPERMARCHE", "PHARMACIE", "RESTAURANT", "CARBURANT",
                          "BOUTIQUE", "TELECOM", "ECOLE", "UTILITES"]
BILL_CATEGORIES = ["ECOLE", "UTILITES"]

# catégorie : (poids achats normaux, montant typique USD, sigma log-normal, niveau de risque)
CARD_MERCHANT_CATEGORIES = {
    "ECOMMERCE_INTL": (0.30, 25.0, 0.8, "medium"),
    "LOCAL_ONLINE": (0.20, 15.0, 0.7, "low"),
    "STREAMING": (0.10, 9.0, 0.3, "low"),
    "SOFTWARE": (0.08, 15.0, 0.5, "low"),
    "GAMING": (0.08, 10.0, 0.8, "high"),
    "EDUCATION": (0.07, 40.0, 0.6, "low"),
    "ELECTRONICS": (0.06, 120.0, 0.6, "high"),
    "TRAVEL": (0.05, 180.0, 0.6, "medium"),
    "GIFT_CARDS": (0.03, 30.0, 0.5, "high"),
    "CRYPTO_EXCHANGE": (0.03, 60.0, 0.8, "high"),
}
SUBSCRIPTION_CATEGORIES = ["STREAMING", "SOFTWARE"]
CARD_TESTING_CATEGORIES = ["ECOMMERCE_INTL", "SOFTWARE", "GAMING", "STREAMING"]
HIGH_RISK_CATEGORIES = ["ELECTRONICS", "GIFT_CARDS", "CRYPTO_EXCHANGE", "GAMING", "TRAVEL"]
FOREIGN_COUNTRIES = ["US", "FR", "BE", "CN", "ZA", "AE", "GB", "NG", "KE", "IN", "TR", "DE"]

# Profil horaire de base (activité relative par heure 0..23)
BASE_HOUR_WEIGHTS = np.array([0.3, 0.2, 0.15, 0.15, 0.2, 0.5, 1.5, 3.0, 4.5, 5.5, 6.0, 6.0,
                              5.5, 5.5, 5.5, 5.5, 5.5, 5.0, 4.5, 4.0, 3.0, 2.0, 1.2, 0.6])
NIGHT_HOURS = [0, 1, 2, 3, 4, 5]
WEEKDAY_FACTOR = np.array([1.0, 0.95, 0.95, 1.0, 1.15, 1.2, 0.75])  # lundi..dimanche

MM_FRAUD_MIX = {"SIM_SWAP": 0.30, "SOCIAL_ENGINEERING": 0.25, "ACCOUNT_TAKEOVER": 0.15,
                "AGENT_FRAUD": 0.12, "SMURFING": 0.18}
CARD_FRAUD_MIX = {"CARD_TESTING": 0.35, "CNP_FRAUD": 0.40, "TOPUP_DRAIN": 0.25}
# Fraude « amicale » : le titulaire conteste un achat qu'il a bien fait. Comportement
# légitime, étiquette frauduleuse : c'est du bruit que AUCUN modèle comportemental ne peut capter.
RELABELED_FRAUD_TYPES = ["FRIENDLY_FRAUD"]
BURST_TX_TYPES = ["P2P_SEND", "P2P_SEND", "AIRTIME", "MERCHANT_PAYMENT"]

MM, CARD = "MOBILE_MONEY", "VISA_VIRTUAL"
LEGIT_KEPT_AFTER_BALANCES = {MM: 0.92, CARD: 0.73}


@dataclass
class GeneratorConfig:
    n_users: int = 3000
    n_days: int = 180
    start_date: str = "2025-06-01"
    seed: int = 42
    usd_cdf_rate: float = 2850.0
    mm_fraud_rate: float = 0.012      # part cible de transactions frauduleuses (Mobile Money)
    card_fraud_rate: float = 0.025    # part cible de transactions frauduleuses (Visa virtuelle)
    mule_account_rate: float = 0.01   # part des utilisateurs servant de comptes « mules »
    visa_adoption_multiplier: float = 1.0  # > 1 pour enrichir le canal Visa (étude comparative)

    # --- Réalisme (1/2) : comportements LÉGITIMES qui ressemblent à de la fraude.
    # Sans eux, un signal comme « nouvel appareil » ou « IP étrangère » suffirait à tout détecter.
    legit_device_change_rate: float = 0.15   # téléphone perdu, cassé ou remplacé sur la période
    shared_device_rate: float = 0.15         # comptes sur un téléphone familial partagé (2 à 4 comptes)
    borrowed_device_rate: float = 0.02       # opération depuis le téléphone d'un proche
    card_new_browser_rate: float = 0.10      # achat carte depuis un nouveau navigateur / cybercafé
    legit_vpn_rate: float = 0.10             # achat carte avec IP étrangère (VPN, itinérance, voyage)
    legit_full_cashout_rate: float = 0.06    # retrait de tout le solde (transfert de la diaspora, salaire)
    legit_burst_rate: float = 0.08           # plusieurs opérations en quelques minutes
    legit_new_recipient_rate: float = 0.20   # envoi à un destinataire jamais utilisé
    legit_micro_burst_users: float = 0.15    # titulaires de carte achetant des crédits de jeu en série
    legit_tontine_rate: float = 0.25         # membres d'un likelemba (tontine) recevant la cagnotte

    # --- Réalisme (2/2) : fraudes DISCRÈTES et bruit d'étiquetage
    fraud_victim_device_rate: float = 0.35   # prise de contrôle à distance / malware : appareil de la victime
    fraud_fresh_device_rate: float = 0.45    # appareil à usage unique (sinon parc partagé du réseau)
    fraud_local_proxy_rate: float = 0.45     # fraude carte via proxy local (IP congolaise)
    fraud_night_bias_scale: float = 0.5      # atténue la préférence des fraudeurs pour la nuit
    friendly_fraud_rate: float = 0.003       # achats légitimes contestés par le titulaire : étiquetés fraude

    # --- Ce que l'opérateur SAIT (≠ vérité terrain). Sources et discussion : docs/HYPOTHESES_DONNEES.md
    # Probabilité qu'un épisode soit un jour signalé (plainte) ou détecté (enquête). Référence :
    # ~23 % des victimes signalent en Ouganda, 36 % après sensibilisation (IPA) ; l'opérateur
    # détecte aussi une partie par lui-même -> environ 40 % en moyenne ici.
    report_prob: dict = field(default_factory=lambda: {
        "SIM_SWAP": 0.60, "ACCOUNT_TAKEOVER": 0.50, "SOCIAL_ENGINEERING": 0.25, "AGENT_FRAUD": 0.35,
        "SMURFING": 0.30, "CARD_TESTING": 0.20, "CNP_FRAUD": 0.50, "TOPUP_DRAIN": 0.50})
    report_amount_effect: float = 0.6        # grosses pertes signalées plus souvent (±30 % autour de la base)
    # Délai médian entre la fraude et son signalement, en jours (log-normal). Référence : la plupart
    # des retours arrivent en ~1 semaine (Le Borgne et al.) ; contestations carte jusqu'à 120 jours.
    report_delay_median_days: dict = field(default_factory=lambda: {
        "SIM_SWAP": 1.0, "ACCOUNT_TAKEOVER": 2.0, "SOCIAL_ENGINEERING": 2.0, "AGENT_FRAUD": 5.0,
        "SMURFING": 21.0, "CARD_TESTING": 10.0, "CNP_FRAUD": 12.0, "TOPUP_DRAIN": 3.0, "FRIENDLY_FRAUD": 25.0})
    report_delay_sigma: float = 0.8
    report_delay_max_days: float = 120.0

    # --- Cycle de vie des entités frauduleuses (réutilisées puis abandonnées ou gelées). La FCA
    # constate des comptes mules « utilisés plusieurs fois » ; durées = hypothèses paramétrables.
    mule_pool_size: int = 400
    mule_lifespan_days: float = 14.0         # compte mule actif ~2 semaines avant gel / abandon
    fresh_mule_rate: float = 0.2             # part d'épisodes utilisant une mule neuve (usage unique)
    fraud_device_pool_size: int = 400
    fraud_device_lifespan_days: float = 30.0
    agent_compromise_days: float = 21.0      # durée pendant laquelle un agent compromis fraude

    n_agents: int = 800
    n_mm_merchants: int = 600
    n_card_merchants: int = 300
    output_dir: str = "ml/data/raw"


class SyntheticDataGenerator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start = pd.Timestamp(cfg.start_date)
        self.horizon_s = cfg.n_days * 86400
        self.warmup_days = min(14, cfg.n_days // 3)  # historique minimal avant toute fraude
        self._episode_counter = 0

        dates = pd.date_range(self.start, periods=cfg.n_days, freq="D")
        self.day_weekday = dates.weekday.to_numpy()
        self.day_dom = dates.day.to_numpy()
        self.day_month = dates.month.to_numpy()

    # ------------------------------------------------------------------ identifiants
    def _ids(self, prefix: str, n: int, digits: int) -> list[str]:
        nums = self.rng.choice(10 ** digits, size=n, replace=False)
        return [f"{prefix}{x:0{digits}d}" for x in nums]

    def _device_ids(self, n: int) -> list[str]:
        return [f"DEV{x:012x}" for x in self.rng.integers(0, 16 ** 12, size=n)]

    def _pick(self, seq):
        return seq[self.rng.integers(len(seq))]

    def _lifecycles(self, n: int, median_days: float, sigma: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """Début et fin d'activité (secondes) d'entités réutilisées puis abandonnées : durées
        log-normales ; débuts étalés pour qu'à tout instant une partie du parc soit active."""
        life = median_days * 86400 * self.rng.lognormal(0, sigma, n)
        start = self.rng.uniform(-median_days * 86400, self.horizon_s, n)
        return start, start + life

    @staticmethod
    def _active(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        return np.flatnonzero((start <= t) & (end > t))

    # ------------------------------------------------------------------ référentiels
    def generate_agents(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        provs = list(PROVINCES)
        weights = np.array([PROVINCES[p][1] for p in provs])
        agents = pd.DataFrame({
            "agent_id": self._ids("A", cfg.n_agents, 6),
            "province": rng.choice(provs, size=cfg.n_agents, p=weights / weights.sum()),
            "operator": rng.choice(list(OPERATORS), size=cfg.n_agents, p=list(OPERATORS.values())),
            "device_id": self._device_ids(cfg.n_agents),
            # Vérité terrain cachée : NE PAS utiliser comme variable
            "is_compromised": (rng.random(cfg.n_agents) < 0.03).astype(int),
        })
        if agents["is_compromised"].sum() == 0:
            agents.loc[0, "is_compromised"] = 1
        # Un agent compromis ne fraude que pendant une fenêtre (avant suspension / départ)
        start, until = self._lifecycles(cfg.n_agents, cfg.agent_compromise_days)
        comp = agents["is_compromised"].to_numpy() == 1
        agents["compromised_from_s"] = np.where(comp, start, np.nan)
        agents["compromised_until_s"] = np.where(comp, until, np.nan)
        return agents

    def generate_merchants(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        provs = list(PROVINCES)
        pw = np.array([PROVINCES[p][1] for p in provs])
        mm = pd.DataFrame({
            "merchant_id": self._ids("M", cfg.n_mm_merchants, 7),
            "channel": MM,
            "category": rng.choice(MM_MERCHANT_CATEGORIES, size=cfg.n_mm_merchants),
            "country": "CD",
            "province": rng.choice(provs, size=cfg.n_mm_merchants, p=pw / pw.sum()),
            "risk_level": "low",
            "typical_amount_usd": np.nan,
        })
        cats = list(CARD_MERCHANT_CATEGORIES)
        cw = np.array([CARD_MERCHANT_CATEGORIES[c][0] for c in cats])
        card_cats = rng.choice(cats, size=cfg.n_card_merchants, p=cw / cw.sum())
        # garantir au moins 2 marchands par catégorie
        card_cats[: 2 * len(cats)] = np.repeat(cats, 2)
        typical = np.array([CARD_MERCHANT_CATEGORIES[c][1] for c in card_cats])
        typical = typical * rng.lognormal(0, 0.3, size=cfg.n_card_merchants)
        is_sub = np.isin(card_cats, SUBSCRIPTION_CATEGORIES)
        typical[is_sub] = np.floor(typical[is_sub]) + 0.99  # prix d'abonnement fixes
        card = pd.DataFrame({
            "merchant_id": self._ids("V", cfg.n_card_merchants, 7),
            "channel": CARD,
            "category": card_cats,
            "country": [("CD" if c == "LOCAL_ONLINE" else self._pick(FOREIGN_COUNTRIES)) for c in card_cats],
            "province": None,
            "risk_level": [CARD_MERCHANT_CATEGORIES[c][3] for c in card_cats],
            "typical_amount_usd": typical.round(2),
        })
        return pd.concat([mm, card], ignore_index=True)

    def generate_users(self, agents: pd.DataFrame) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        n = cfg.n_users
        provs = list(PROVINCES)
        pw = np.array([PROVINCES[p][1] for p in provs])
        segs = list(SEGMENTS)
        sw = np.array([SEGMENTS[s][0] for s in segs])

        province = rng.choice(provs, size=n, p=pw / pw.sum())
        segment = rng.choice(segs, size=n, p=sw / sw.sum())
        kyc = rng.choice(list(KYC_LEVEL_PROBS), size=n, p=list(KYC_LEVEL_PROBS.values()))
        kyc = np.where((segment == "commercant") & (kyc == 1) & (rng.random(n) < 0.5), 2, kyc)

        p_smart = np.array([SEGMENTS[s][3] for s in segment])
        device_type = np.where(rng.random(n) < p_smart, "smartphone", "feature_phone").astype(object)
        primary_device = np.array(self._device_ids(n), dtype=object)

        # Téléphones familiaux : 2 à 4 comptes de la même province sur un même appareil.
        # Un appareil partagé n'est donc PAS en soi un signal de fraude.
        shared = np.where(rng.random(n) < cfg.shared_device_rate)[0]
        for p in provs:
            members = rng.permutation(shared[province[shared] == p])
            i = 0
            while i < len(members) - 1:
                grp = members[i:i + min(int(rng.integers(2, 5)), len(members) - i)]
                primary_device[grp] = primary_device[grp[0]]
                device_type[grp] = device_type[grp[0]]
                i += len(grp)

        p_visa =np.clip(np.array([SEGMENTS[s][4] for s in segment]) * cfg.visa_adoption_multiplier, 0, 1)
        eligible = (device_type == "smartphone") & (kyc >= 2)
        has_visa = eligible & (rng.random(n) < p_visa)

        typical = np.array([SEGMENTS[s][1] for s in segment]) * rng.lognormal(0, 0.4, n)
        tx_rate = np.array([SEGMENTS[s][2] for s in segment]) * rng.lognormal(0, 0.5, n)

        # Changement de téléphone légitime (cas difficile pour le modèle)
        changes = rng.random(n) < cfg.legit_device_change_rate
        change_ts = np.where(changes, rng.uniform(0.2, 0.9, n) * self.horizon_s, np.inf)

        # Tous les portefeuilles (clients, contacts externes, mules) partagent le même espace
        # de numéros : un préfixe ne doit jamais trahir un compte frauduleux.
        wallets_all = self._ids("243", n + 20000 + 300 + cfg.mule_pool_size + 4000, 9)
        cuts = np.cumsum([n, 20000, 300, cfg.mule_pool_size])
        user_wallets, self.legit_pool, self.hub_pool, self.mule_pool, self.fresh_mules = (
            wallets_all[:cuts[0]], wallets_all[cuts[0]:cuts[1]], wallets_all[cuts[1]:cuts[2]],
            wallets_all[cuts[2]:cuts[3]], wallets_all[cuts[3]:])
        self.mule_start, self.mule_end = self._lifecycles(cfg.mule_pool_size, cfg.mule_lifespan_days)

        users = pd.DataFrame({
            "user_id": [f"U{i:06d}" for i in range(n)],
            "wallet_id": user_wallets,
            "operator": rng.choice(list(OPERATORS), size=n, p=list(OPERATORS.values())),
            "province": province,
            "city": [PROVINCES[p][0] for p in province],
            "segment": segment,
            "kyc_level": kyc,
            "kyc_tx_limit_usd": [KYC_TX_LIMIT_USD[k] for k in kyc],
            "device_type": device_type,
            "primary_device_id": primary_device,
            "secondary_device_id": self._device_ids(n),
            "device_change_ts": change_ts,
            "account_age_days": rng.integers(30, 3000, n),
            "monthly_income_usd": (typical * rng.uniform(8, 20, n)).round(0),
            "preferred_currency": np.where(rng.random(n) < 0.6, "CDF", "USD"),
            "has_visa_virtual": has_visa,
            "card_id": [f"CARD{x:010d}" if v else None
                        for x, v in zip(rng.choice(10 ** 10, n, replace=False), has_visa)],
            "typical_amount_usd": typical.round(2),
            "tx_rate_per_day": tx_rate.round(3),
            "hour_shift": rng.choice([-1, 0, 0, 1, 2], size=n),
            "night_owl": rng.random(n) < 0.06,
            # Vérité terrain cachée : NE PAS utiliser comme variable
            "is_mule_account": rng.random(n) < cfg.mule_account_rate,
        })
        if not users["is_mule_account"].any():
            users.loc[0, "is_mule_account"] = True

        # Agents habituels (même province) et contacts réguliers. Les portefeuilles « populaires »
        # (vendeurs du marché, bailleurs, tontines, églises) reçoivent de nombreux clients,
        # comme le compte collecteur d'un réseau de mules.
        agents_by_prov = agents.groupby("province")["agent_id"].apply(list).to_dict()
        all_agents = agents["agent_id"].tolist()
        home_agents, contacts = [], []
        for p in province:
            pool = agents_by_prov.get(p, all_agents)
            home_agents.append(list(rng.choice(pool, size=min(len(pool), rng.integers(1, 4)), replace=False)))
            k = rng.integers(3, 11)
            c = [self._pick(user_wallets) if rng.random() < 0.5 else self._pick(self.legit_pool) for _ in range(k)]
            c += [self._pick(self.hub_pool) for _ in range(rng.integers(0, 3))]
            contacts.append(c)
        users["home_agents"] = home_agents
        users["contacts"] = contacts
        self.devices_by_prov = users.groupby("province")["primary_device_id"].apply(list).to_dict()
        return users

    # ------------------------------------------------------------------ outils communs
    def _hour_probs(self, u) -> np.ndarray:
        w = np.roll(BASE_HOUR_WEIGHTS, u.hour_shift).copy()
        if u.night_owl:
            w[[22, 23, 0, 1, 2, 3]] += 2.0
        return w / w.sum()

    def _sample_seconds(self, day: int, hour_probs: np.ndarray) -> float:
        hour = self.rng.choice(24, p=hour_probs)
        return day * 86400 + hour * 3600 + self.rng.uniform(0, 3600)

    def _fraud_start(self, night_bias: float) -> float:
        day = self.rng.integers(self.warmup_days, self.cfg.n_days)
        if self.rng.random() < night_bias * self.cfg.fraud_night_bias_scale:
            hour = self._pick(NIGHT_HOURS)
        else:
            hour = self.rng.choice(24, p=BASE_HOUR_WEIGHTS / BASE_HOUR_WEIGHTS.sum())
        return day * 86400 + hour * 3600 + self.rng.uniform(0, 3600)

    def _device_at(self, u, ts: float) -> str:
        return u.secondary_device_id if ts >= u.device_change_ts else u.primary_device_id

    def _fraud_device(self, ctx, t: float) -> str:
        """Appareil du fraudeur : neuf (usage unique) ou issu du parc du réseau ACTIF à l'instant t."""
        active = self._active(ctx["fraud_device_start"], ctx["fraud_device_end"], t)
        if self.rng.random() < self.cfg.fraud_fresh_device_rate or len(active) == 0:
            return self._device_ids(1)[0]
        return ctx["fraud_devices"][self._pick(active)]

    def _fraud_mule(self, t: float) -> str:
        """Portefeuille de réception : mule du réseau active à l'instant t (réutilisée jusqu'à son
        gel ou abandon), ou mule neuve à usage unique."""
        active = self._active(self.mule_start, self.mule_end, t)
        if self.fresh_mules and (self.rng.random() < self.cfg.fresh_mule_rate or len(active) == 0):
            return self.fresh_mules.pop()
        return self.mule_pool[self._pick(active)] if len(active) else self._pick(self.mule_pool)

    def _compromised_agent_at(self, ctx, t: float):
        a = ctx["compromised"]
        active = self._active(a["compromised_from_s"].to_numpy(), a["compromised_until_s"].to_numpy(), t)
        return a.iloc[self._pick(active)] if len(active) else None

    def _uses_victim_device(self) -> bool:
        return self.rng.random() < self.cfg.fraud_victim_device_rate

    def _row(self, u, ts: float, channel: str, tx_type: str) -> dict:
        access = "USSD" if u.device_type == "feature_phone" else ("APP" if self.rng.random() < 0.7 else "USSD")
        if channel == CARD:
            access = "APP"
        return {
            "ts": ts, "uidx": u.Index, "user_id": u.user_id, "wallet_id": u.wallet_id,
            "card_id": u.card_id if channel == CARD else None,
            "channel": channel, "operator": u.operator, "tx_type": tx_type, "access_channel": access,
            "amount_usd": np.nan, "drain_frac": np.nan,
            "counterparty_id": None, "merchant_id": None, "merchant_category": None,
            "merchant_country": None, "agent_id": None,
            "device_id": self._device_at(u, ts), "device_type": u.device_type,
            "ip_country": "CD" if access == "APP" else None,
            "location_province": u.province,
            "is_fraud": 0, "fraud_type": None, "fraud_episode_id": None,
        }

    def _mark_fraud(self, rows: list[dict], fraud_type: str) -> list[dict]:
        self._episode_counter += 1
        ep = f"EP{self._episode_counter:05d}"
        for r in rows:
            r.update(is_fraud=1, fraud_type=fraud_type, fraud_episode_id=ep)
        return rows

    def _clip_amount(self, amount: float, u) -> float:
        return float(np.clip(amount, 0.2, u.kyc_tx_limit_usd * 0.95))

    # ------------------------------------------------------------------ comportement normal
    def generate_normal_mobile_money(self, users, agents, merchants) -> list[dict]:
        rng = self.rng
        rows: list[dict] = []
        mm_merch = merchants[merchants.channel == MM]
        merch_by_prov_cat = {k: g["merchant_id"].tolist()
                             for k, g in mm_merch.groupby(["province", "category"])}
        merch_by_cat = {k: g["merchant_id"].tolist() for k, g in mm_merch.groupby("category")}
        agents_by_prov = agents.groupby("province")["agent_id"].apply(list).to_dict()
        agent_device = dict(zip(agents["agent_id"], agents["device_id"]))
        month_end = (self.day_dom >= 25) | (self.day_dom <= 3)
        day_factor = WEEKDAY_FACTOR[self.day_weekday] * np.where(month_end, 1.25, 1.0)
        pay_cats = [c for c in MM_MERCHANT_CATEGORIES if c not in BILL_CATEGORIES]

        def merchant_for(u, cat):
            pool = merch_by_prov_cat.get((u.province, cat)) or merch_by_cat[cat]
            return self._pick(pool)

        cfg = self.cfg

        def make_row(u, ts, tx_type) -> dict:
            r = self._row(u, ts, MM, tx_type)
            amount = u.typical_amount_usd * AMOUNT_MULTIPLIER[tx_type] * rng.lognormal(0, 0.7)
            if rng.random() < 0.04:  # voyage légitime
                r["location_province"] = self._pick(list(PROVINCES))
            if rng.random() < cfg.borrowed_device_rate:  # téléphone d'un proche
                r["device_id"] = self._pick(self.devices_by_prov[u.province])
            if r["access_channel"] == "APP" and rng.random() < 0.01:  # VPN
                r["ip_country"] = self._pick(FOREIGN_COUNTRIES)
            if tx_type in ("P2P_SEND", "P2P_RECEIVE"):
                if rng.random() >= cfg.legit_new_recipient_rate:
                    r["counterparty_id"] = self._pick(u.contacts)
                else:
                    r["counterparty_id"] = self._pick(self.hub_pool if rng.random() < 0.3 else self.legit_pool)
                if tx_type == "P2P_SEND" and rng.random() < cfg.legit_full_cashout_rate / 2:
                    r["drain_frac"] = rng.uniform(0.5, 0.9)  # l'essentiel du salaire à la famille
            elif tx_type in ("CASH_IN", "CASH_OUT"):
                r["access_channel"], r["ip_country"] = "AGENT", None
                r["agent_id"] = (self._pick(u.home_agents) if rng.random() < 0.8
                                 else self._pick(agents_by_prov.get(r["location_province"], u.home_agents)))
                if tx_type == "CASH_IN":  # le dépôt est initié par l'agent, depuis sa propre ligne
                    r["device_id"] = agent_device[r["agent_id"]]
                if tx_type == "CASH_OUT" and rng.random() < cfg.legit_full_cashout_rate:
                    r["drain_frac"] = rng.uniform(0.9, 1.0)  # retrait de tout le transfert reçu
            elif tx_type == "MERCHANT_PAYMENT":
                cat = self._pick(pay_cats)
                r["merchant_id"], r["merchant_category"], r["merchant_country"] = merchant_for(u, cat), cat, "CD"
            elif tx_type == "BILL_PAYMENT":
                cat = self._pick(BILL_CATEGORIES)
                r["merchant_id"], r["merchant_category"], r["merchant_country"] = merchant_for(u, cat), cat, "CD"
            elif tx_type == "AIRTIME":
                r["counterparty_id"] = u.operator
            if np.isnan(r["drain_frac"]):
                r["amount_usd"] = self._clip_amount(amount, u)
            return r

        for u in users.itertuples():
            hp = self._hour_probs(u)
            probs = MM_TYPE_PROBS.get(u.segment, MM_TYPE_PROBS["default"])
            burst_rate = cfg.legit_burst_rate * (2 if u.segment == "commercant" else 1)
            counts = rng.poisson(u.tx_rate_per_day * day_factor)
            days = np.repeat(np.arange(self.cfg.n_days), counts)
            # tirages vectorisés (heure, type) : rng.choice avec probabilités est lent appel par appel
            stamps = days * 86400 + rng.choice(24, size=len(days), p=hp) * 3600 + rng.uniform(0, 3600, len(days))
            kinds = rng.choice(len(MM_TX_TYPES), size=len(days), p=probs)
            for ts, kind in zip(stamps, kinds):
                tx_type = MM_TX_TYPES[kind]
                rows.append(make_row(u, ts, tx_type))
                # Rafale légitime : plusieurs opérations enchaînées en quelques minutes
                if tx_type in DEBIT_TX_TYPES and rng.random() < burst_rate:
                    for _ in range(rng.integers(1, 4)):
                        ts += rng.uniform(2, 40) * 60
                        rows.append(make_row(u, ts, self._pick(BURST_TX_TYPES)))

            # Likelemba (tontine) : le bénéficiaire du tour reçoit les cotisations de plusieurs
            # membres puis retire la cagnotte. Même forme qu'un compte mule, mais légitime.
            if rng.random() < cfg.legit_tontine_rate:
                ts = self._sample_seconds(rng.integers(self.cfg.n_days), hp)
                contribution = u.typical_amount_usd * rng.uniform(1.0, 4.0)
                for _ in range(rng.integers(5, 16)):
                    r = make_row(u, ts, "P2P_RECEIVE")
                    r.update(counterparty_id=self._pick(self.legit_pool), drain_frac=np.nan,
                             amount_usd=self._clip_amount(contribution, u))
                    rows.append(r)
                    ts += rng.uniform(0.2, 6) * 3600
                r = make_row(u, ts + rng.uniform(10, 180) * 60, "CASH_OUT")
                r.update(drain_frac=rng.uniform(0.85, 1.0), amount_usd=np.nan)
                rows.append(r)

            # Frais scolaires de septembre : montants élevés mais légitimes
            sept_days = np.where((self.day_month == 9) & (self.day_dom <= 20))[0]
            if len(sept_days) and rng.random() < 0.5:
                for _ in range(rng.integers(1, 3)):
                    r = self._row(u, self._sample_seconds(self._pick(sept_days), hp), MM, "BILL_PAYMENT")
                    r["merchant_id"], r["merchant_category"], r["merchant_country"] = merchant_for(u, "ECOLE"), "ECOLE", "CD"
                    r["amount_usd"] = self._clip_amount(u.typical_amount_usd * 4 * rng.lognormal(0, 0.4), u)
                    rows.append(r)
        return rows

    def generate_normal_card(self, users, merchants) -> list[dict]:
        rng = self.rng
        rows: list[dict] = []
        card_merch = merchants[merchants.channel == CARD]
        by_cat = {k: g for k, g in card_merch.groupby("category")}
        cats = list(CARD_MERCHANT_CATEGORIES)
        cw = np.array([CARD_MERCHANT_CATEGORIES[c][0] for c in cats])
        cw = cw / cw.sum()
        months = self.cfg.n_days / 30.0

        for u in users[users.has_visa_virtual].itertuples():
            hp = self._hour_probs(u)
            activity = rng.lognormal(0, 0.4)

            for _ in range(rng.poisson(1.0 * months * activity)):  # recharges « de précaution »
                r = self._row(u, self._sample_seconds(rng.integers(self.cfg.n_days), hp), CARD, "CARD_TOPUP")
                r["amount_usd"] = self._clip_amount(u.typical_amount_usd * 1.5 * rng.lognormal(0, 0.5), u)
                if rng.random() < self.cfg.legit_vpn_rate / 2:
                    r["ip_country"] = self._pick(FOREIGN_COUNTRIES)
                rows.append(r)

            # Abonnements récurrents (même marchand, même montant, même jour du mois)
            if rng.random() < 0.45:
                for _ in range(rng.integers(1, 3)):
                    m = by_cat[self._pick(SUBSCRIPTION_CATEGORIES)].sample(1, random_state=rng).iloc[0]
                    dom = rng.integers(1, 29)
                    for day in np.where(self.day_dom == dom)[0]:
                        r = self._row(u, self._sample_seconds(day, hp), CARD, "CARD_PURCHASE")
                        r.update(merchant_id=m.merchant_id, merchant_category=m.category,
                                 merchant_country=m.country, amount_usd=float(m.typical_amount_usd))
                        rows.append(r)

            for _ in range(rng.poisson(3.0 * months * activity)):
                cat = cats[rng.choice(len(cats), p=cw)]
                m = by_cat[cat].sample(1, random_state=rng).iloc[0]
                sigma = CARD_MERCHANT_CATEGORIES[cat][2]
                r = self._row(u, self._sample_seconds(rng.integers(self.cfg.n_days), hp), CARD, "CARD_PURCHASE")
                r.update(merchant_id=m.merchant_id, merchant_category=cat, merchant_country=m.country,
                         amount_usd=self._clip_amount(m.typical_amount_usd * rng.lognormal(0, sigma), u))
                if rng.random() < self.cfg.legit_vpn_rate:  # VPN, itinérance, voyage
                    r["ip_country"] = self._pick(FOREIGN_COUNTRIES)
                if rng.random() < self.cfg.card_new_browser_rate:  # nouveau navigateur / cybercafé
                    r["device_id"] = self._device_ids(1)[0]
                rows.append(r)
                if rng.random() < 0.6:  # carte prépayée : recharge juste avant l'achat
                    tp = self._row(u, max(0.0, r["ts"] - rng.uniform(1, 30) * 60), CARD, "CARD_TOPUP")
                    tp["amount_usd"] = self._clip_amount(r["amount_usd"] * rng.uniform(1.0, 1.5), u)
                    rows.append(tp)

            # Joueurs : séries de micro-achats (crédits de jeu) qui ressemblent à un test de carte
            if rng.random() < self.cfg.legit_micro_burst_users:
                gaming = pd.concat([by_cat["GAMING"], by_cat["STREAMING"]])
                for _ in range(rng.poisson(0.7 * months)):
                    t = self._sample_seconds(rng.integers(self.cfg.n_days), hp)
                    m = gaming.sample(1, random_state=rng).iloc[0]
                    for _ in range(rng.integers(2, 7)):
                        r = self._row(u, t, CARD, "CARD_PURCHASE")
                        r.update(merchant_id=m.merchant_id, merchant_category=m.category,
                                 merchant_country=m.country, amount_usd=round(rng.uniform(0.5, 5.0), 2))
                        rows.append(r)
                        t += rng.uniform(30, 600)
        return rows

    # ------------------------------------------------------------------ fraudes Mobile Money
    def _fraud_sim_swap(self, u, ctx) -> list[dict]:
        """Échange de carte SIM : la SIM passe dans un autre téléphone (nouvel appareil
        obligatoire), puis le compte est vidé en quelques transferts."""
        rng = self.rng
        t = self._fraud_start(night_bias=0.6)
        dev = self._fraud_device(ctx, t)
        dtype = "smartphone" if rng.random() < 0.5 else "feature_phone"
        prov = u.province if rng.random() < 0.7 else self._pick(list(PROVINCES))
        k = rng.integers(2, 6)
        rows = []
        for i in range(k):
            last = i == k - 1
            tx_type = "CASH_OUT" if last and rng.random() < 0.6 else "P2P_SEND"
            r = self._row(u, t, MM, tx_type)
            r.update(device_id=dev, device_type=dtype, location_province=prov,
                     access_channel="USSD" if dtype == "feature_phone" else "APP",
                     drain_frac=rng.uniform(0.6, 1.0) if last else rng.uniform(0.3, 0.9))
            r["ip_country"] = "CD" if r["access_channel"] == "APP" else None
            if tx_type == "CASH_OUT":
                bad = self._compromised_agent_at(ctx, t) if rng.random() < 0.4 else None
                r.update(access_channel="AGENT", ip_country=None,
                         agent_id=bad.agent_id if bad is not None
                         else self._pick(ctx["agents_by_prov"].get(prov, ctx["all_agents"])))
            else:
                r["counterparty_id"] = self._fraud_mule(t)
            rows.append(r)
            t += rng.uniform(3, 40) * 60
        return self._mark_fraud(rows, "SIM_SWAP")

    def _fraud_account_takeover(self, u, ctx) -> list[dict]:
        """Vol d'identifiants (PIN / phishing) depuis un autre appareil, ou prise de contrôle
        à distance du téléphone de la victime (application d'assistance installée sous un
        faux prétexte) : même appareil, même lieu. Montants partiels pour passer inaperçu."""
        rng = self.rng
        t = self._fraud_start(night_bias=0.25)
        remote = self._uses_victim_device()
        dev = self._device_at(u, t) if remote else self._fraud_device(ctx, t)
        prov = u.province if remote or rng.random() < 0.7 else self._pick(list(PROVINCES))
        rows = []
        for _ in range(rng.integers(2, 5)):
            tx_type = "P2P_SEND" if rng.random() < 0.7 else "MERCHANT_PAYMENT"
            r = self._row(u, t, MM, tx_type)
            r.update(device_id=dev, device_type="smartphone", access_channel="APP", ip_country="CD",
                     location_province=prov, drain_frac=rng.uniform(0.15, 0.5))
            if tx_type == "P2P_SEND":
                r["counterparty_id"] = self._fraud_mule(t)
            else:
                m = ctx["mm_merchants"].sample(1, random_state=rng).iloc[0]
                r.update(merchant_id=m.merchant_id, merchant_category=m.category, merchant_country="CD")
            rows.append(r)
            t += rng.uniform(10, 90) * 60
        return self._mark_fraud(rows, "ACCOUNT_TAKEOVER")

    def _fraud_social_engineering(self, u, ctx) -> list[dict]:
        """Arnaque « envoi par erreur » / faux agent : la victime paie elle-même
        (même appareil, même lieu) -> fraude la plus difficile à détecter."""
        rng = self.rng
        t = self._fraud_start(night_bias=0.0)
        mule = self._fraud_mule(t)
        rows = []
        if rng.random() < 0.4:  # transfert d'amorçage pour crédibiliser l'arnaque
            r = self._row(u, t, MM, "P2P_RECEIVE")
            r.update(counterparty_id=mule, amount_usd=round(rng.uniform(1, 5), 2))
            rows.append(r)
            t += rng.uniform(5, 30) * 60
        for _ in range(rng.integers(1, 3)):
            r = self._row(u, t, MM, "P2P_SEND")
            r.update(counterparty_id=mule,
                     amount_usd=self._clip_amount(u.typical_amount_usd * rng.uniform(2, 6), u))
            rows.append(r)
            t += rng.uniform(5, 45) * 60
        return self._mark_fraud(rows, "SOCIAL_ENGINEERING")

    def _fraud_agent(self, u, ctx) -> list[dict]:
        """Agent compromis effectuant des retraits non autorisés sur le compte du client :
        depuis sa propre ligne avec le PIN du client, ou directement sur le téléphone que le
        client lui a confié pour l'aider (pratique courante auprès des clients peu lettrés)."""
        rng = self.rng
        comp = ctx["compromised"]
        lo_bound, hi_bound = self.warmup_days * 86400, self.horizon_s - 3600
        for _ in range(20):
            agent = comp.iloc[rng.integers(len(comp))]
            lo, hi = max(agent.compromised_from_s, lo_bound), min(agent.compromised_until_s, hi_bound)
            if hi > lo:
                break
        else:
            lo, hi = lo_bound, hi_bound
        day_t = rng.uniform(lo, hi)
        hour = rng.choice(24, p=BASE_HOUR_WEIGHTS / BASE_HOUR_WEIGHTS.sum())
        t = float(np.clip(np.floor(day_t / 86400) * 86400 + hour * 3600 + rng.uniform(0, 3600), lo, hi))
        dev = self._device_at(u, t) if rng.random() < 0.5 else agent.device_id
        rows = []
        for _ in range(rng.integers(1, 3)):
            r = self._row(u, t, MM, "CASH_OUT")
            r.update(access_channel="AGENT", ip_country=None, agent_id=agent.agent_id,
                     device_id=dev, location_province=agent.province,
                     drain_frac=rng.uniform(0.2, 0.7))
            rows.append(r)
            t += rng.uniform(1, 10) * 60
        return self._mark_fraud(rows, "AGENT_FRAUD")

    def _fraud_smurfing(self, u, ctx) -> list[dict]:
        """Compte mule : nombreuses réceptions juste sous le plafond KYC, puis retraits."""
        rng = self.rng
        t = self._fraud_start(night_bias=0.0)
        rows = []
        for _ in range(rng.integers(5, 13)):
            r = self._row(u, t, MM, "P2P_RECEIVE")
            r.update(counterparty_id=self._fraud_mule(t),
                     amount_usd=round(u.kyc_tx_limit_usd * rng.uniform(0.4, 0.95), 2))
            rows.append(r)
            t += rng.uniform(0.5, 8) * 3600
        for _ in range(rng.integers(1, 4)):
            t += rng.uniform(10, 120) * 60
            r = self._row(u, t, MM, "CASH_OUT")
            r.update(access_channel="AGENT", ip_country=None, drain_frac=rng.uniform(0.85, 1.0),
                     agent_id=self._pick(ctx["agents_by_prov"].get(u.province, ctx["all_agents"])))
            rows.append(r)
        return self._mark_fraud(rows, "SMURFING")

    # ------------------------------------------------------------------ fraudes Visa virtuelle
    def _card_fraud_row(self, u, t, ctx, categories, dev, foreign_ip) -> dict:
        m = ctx["card_merchants"][ctx["card_merchants"].category.isin(categories)].sample(1, random_state=self.rng).iloc[0]
        r = self._row(u, t, CARD, "CARD_PURCHASE")
        r.update(merchant_id=m.merchant_id, merchant_category=m.category, merchant_country=m.country,
                 device_id=dev, device_type="smartphone", ip_country=foreign_ip)
        return r

    def _card_fraud_access(self, u, t, ctx) -> tuple[str, str]:
        """(appareil, pays IP) du fraudeur : malware sur le téléphone de la victime (appareil
        connu, IP congolaise) ou appareil du fraudeur, derrière un proxy local ou non."""
        if self.rng.random() < self.cfg.fraud_victim_device_rate * 0.6:
            return self._device_at(u, t), "CD"
        ip = "CD" if self.rng.random() < self.cfg.fraud_local_proxy_rate else self._pick(FOREIGN_COUNTRIES)
        return self._fraud_device(ctx, t), ip

    def _fraud_card_testing(self, u, ctx) -> list[dict]:
        """Test de carte : rafale de micro-achats, puis éventuellement un gros achat."""
        rng = self.rng
        t = self._fraud_start(night_bias=0.5)
        dev, ip = self._card_fraud_access(u, t, ctx)
        rows = []
        for _ in range(rng.integers(4, 13)):
            r = self._card_fraud_row(u, t, ctx, CARD_TESTING_CATEGORIES, dev, ip)
            r["amount_usd"] = round(rng.uniform(0.5, 2.5), 2)
            rows.append(r)
            t += rng.uniform(20, 180)
        if rng.random() < 0.5:
            r = self._card_fraud_row(u, t + rng.uniform(5, 60) * 60, ctx, HIGH_RISK_CATEGORIES, dev, ip)
            r["drain_frac"] = rng.uniform(0.7, 1.0)
            rows.append(r)
        return self._mark_fraud(rows, "CARD_TESTING")

    def _fraud_cnp(self, u, ctx) -> list[dict]:
        """Fraude carte non présente : achats avec les données de carte volées."""
        rng = self.rng
        t = self._fraud_start(night_bias=0.5)
        dev, ip = self._card_fraud_access(u, t, ctx)
        rows = []
        for _ in range(rng.integers(1, 4)):
            r = self._card_fraud_row(u, t, ctx, HIGH_RISK_CATEGORIES, dev, ip)
            r["drain_frac"] = rng.uniform(0.3, 1.0)
            rows.append(r)
            t += rng.uniform(2, 60) * 60
        return self._mark_fraud(rows, "CNP_FRAUD")

    def _fraud_topup_drain(self, u, ctx) -> list[dict]:
        """Fraude trans-canal : prise de contrôle du portefeuille, recharge de la carte,
        puis dépenses immédiates à l'étranger."""
        rng = self.rng
        t = self._fraud_start(night_bias=0.35)
        dev, ip = self._card_fraud_access(u, t, ctx)
        r = self._row(u, t, CARD, "CARD_TOPUP")
        r.update(device_id=dev, device_type="smartphone", ip_country=ip, drain_frac=rng.uniform(0.6, 0.95))
        rows = [r]
        for _ in range(rng.integers(1, 4)):
            t += rng.uniform(5, 30) * 60
            r = self._card_fraud_row(u, t, ctx, HIGH_RISK_CATEGORIES, dev, ip)
            r["drain_frac"] = rng.uniform(0.6, 1.0)
            rows.append(r)
        return self._mark_fraud(rows, "TOPUP_DRAIN")

    def inject_fraud(self, users, agents, merchants, n_normal_mm: int, n_normal_card: int) -> list[dict]:
        rng = self.rng
        ctx = {
            "agents": agents,
            "all_agents": agents["agent_id"].tolist(),
            "agents_by_prov": agents.groupby("province")["agent_id"].apply(list).to_dict(),
            "compromised": agents[agents.is_compromised == 1].reset_index(drop=True),
            "mm_merchants": merchants[merchants.channel == MM],
            "card_merchants": merchants[merchants.channel == CARD],
            # Réseaux de fraudeurs : un parc d'appareils réutilisés d'un épisode à l'autre
            "fraud_devices": self._device_ids(self.cfg.fraud_device_pool_size),
        }
        ctx["fraud_device_start"], ctx["fraud_device_end"] = self._lifecycles(
            self.cfg.fraud_device_pool_size, self.cfg.fraud_device_lifespan_days)
        victims_mm = users[~users.is_mule_account]
        victims_card = users[users.has_visa_virtual & ~users.is_mule_account]
        mules = users[users.is_mule_account]

        handlers = {
            "SIM_SWAP": (self._fraud_sim_swap, victims_mm),
            "SOCIAL_ENGINEERING": (self._fraud_social_engineering, victims_mm),
            "ACCOUNT_TAKEOVER": (self._fraud_account_takeover, victims_mm[victims_mm.device_type == "smartphone"]),
            "AGENT_FRAUD": (self._fraud_agent, victims_mm),
            "SMURFING": (self._fraud_smurfing, mules),
            "CARD_TESTING": (self._fraud_card_testing, victims_card),
            "CNP_FRAUD": (self._fraud_cnp, victims_card),
            "TOPUP_DRAIN": (self._fraud_topup_drain, victims_card),
        }
        budgets = {}
        # Une partie des opérations légitimes générées sera retirée par le rejeu des soldes (un
        # client ne tente pas un achat supérieur à son solde) : le budget de fraude porte sur ce
        # qui RESTE, sinon le taux de fraude réel dépasse la cible (mesuré : 92 % MM, 73 % carte).
        mm_target = self.cfg.mm_fraud_rate / (1 - self.cfg.mm_fraud_rate) * n_normal_mm * LEGIT_KEPT_AFTER_BALANCES[MM]
        card_target = (self.cfg.card_fraud_rate / (1 - self.cfg.card_fraud_rate) * n_normal_card
                       * LEGIT_KEPT_AFTER_BALANCES[CARD])
        budgets.update({k: w * mm_target for k, w in MM_FRAUD_MIX.items()})
        budgets.update({k: w * card_target for k, w in CARD_FRAUD_MIX.items()})

        rows: list[dict] = []
        for fraud_type, budget in budgets.items():
            fn, pool = handlers[fraud_type]
            if pool.empty:
                logger.warning("Aucune victime possible pour %s, typologie ignorée", fraud_type)
                continue
            produced = 0
            while produced < budget:
                u = next(pool.sample(1, random_state=rng).itertuples())
                ep_rows = fn(u, ctx)
                rows.extend(ep_rows)
                produced += len(ep_rows)
        return rows

    # ------------------------------------------------------------------ finalisation
    def _round_amounts(self, df: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        """Les montants saisis par une personne (transferts Mobile Money, recharges de carte),
        y compris ceux dictés par un escroc, sont arrondis : 500 FC en CDF, 1 USD / 0,5 USD
        en USD. Restent non arrondis : les prix fixés par les marchands (achats carte) et
        les montants de vidage calculés sur le solde (encore inconnus ici, NaN)."""
        rate = self.cfg.usd_cdf_rate
        currency = np.where(df["channel"] == CARD, "USD",
                            users["preferred_currency"].to_numpy()[df["uidx"].to_numpy()])
        df["currency"] = currency
        amt = df["amount_usd"].to_numpy(dtype=float, copy=True)
        explicit = ~np.isnan(amt) & (df["tx_type"].to_numpy() != "CARD_PURCHASE")
        cdf = explicit & (currency == "CDF")
        step = np.where(df["tx_type"].to_numpy() == "AIRTIME", 100, 500)
        amt[cdf] = np.maximum(step[cdf], np.round(amt[cdf] * rate / step[cdf]) * step[cdf]) / rate
        usd = explicit & (currency == "USD")
        amt[usd] = np.where(amt[usd] >= 5, np.round(amt[usd]), np.maximum(0.5, np.round(amt[usd] * 2) / 2))
        df["amount_usd"] = amt
        return df

    def _floor_human(self, amount_usd: float, currency: str) -> float:
        """Arrondi vers le bas à 500 FC ou 1 USD (0,5 USD sous 5 USD), au minimum un pas."""
        if currency == "CDF":
            rate = self.cfg.usd_cdf_rate
            return max(1, math.floor(amount_usd * rate / 500)) * 500 / rate
        if amount_usd >= 5:
            return float(math.floor(amount_usd))
        return max(0.5, math.floor(amount_usd * 2) / 2)

    def _report_delay_s(self, fraud_type: str, n: int = 1) -> np.ndarray:
        cfg = self.cfg
        days = cfg.report_delay_median_days[fraud_type] * self.rng.lognormal(0, cfg.report_delay_sigma, n)
        return np.minimum(days, cfg.report_delay_max_days) * 86400

    def _apply_label_noise(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Étiquettes telles qu'un opérateur les connaîtrait réellement :
        - chaque épisode de fraude n'est signalé (plainte) ou détecté (enquête) qu'avec une
          probabilité qui dépend du type et du montant perdu, et seulement après un délai
          -> colonne fraud_reported_at ; les épisodes jamais signalés sont étiquetés légitimes ;
        - quelques achats carte légitimes sont contestés par le titulaire -> étiquetés fraude.
        La vérité terrain de ces lignes est conservée à part (label_noise.csv)."""
        rng, cfg = self.rng, self.cfg
        df["fraud_reported_s"] = np.nan
        fraud = df[df.is_fraud == 1]
        lost = fraud["amount_usd"].where(fraud["tx_type"].isin(DEBIT_TX_TYPES | {"CARD_TOPUP", "CARD_PURCHASE"})
                                         & (fraud["status"] == "SUCCESS"), 0.0)
        episodes = (fraud.assign(lost=lost).groupby("fraud_episode_id")
                    .agg(fraud_type=("fraud_type", "first"), lost=("lost", "sum"), last_ts=("ts", "max")))
        base = episodes["fraud_type"].map(cfg.report_prob).fillna(0.4).to_numpy()
        eff = cfg.report_amount_effect
        p = np.clip(base * (1 - eff / 2 + eff * np.minimum(episodes["lost"].to_numpy() / 150, 1)), 0.02, 0.98)
        reported = rng.random(len(episodes)) < p
        delays = np.array([self._report_delay_s(t)[0] for t in episodes["fraud_type"]])
        reported_at = pd.Series(np.where(reported, episodes["last_ts"].to_numpy() + delays, np.nan),
                                index=episodes.index)
        df["fraud_reported_s"] = df["fraud_episode_id"].map(reported_at)
        unreported = episodes.index[~reported]
        m_unrep = df["fraud_episode_id"].isin(unreported).to_numpy()
        self._report_stats = {
            "episodes": int(len(episodes)), "reported_share": round(float(reported.mean()), 4),
            "reported_share_by_type": {t: round(float(reported[(episodes["fraud_type"] == t).to_numpy()].mean()), 3)
                                       for t in sorted(episodes["fraud_type"].unique())},
            "median_delay_days_by_type": {t: round(float(np.median(delays[(episodes["fraud_type"] == t).to_numpy()
                                                                          & reported]) / 86400), 2)
                                          for t in sorted(episodes["fraud_type"].unique())
                                          if ((episodes["fraud_type"] == t).to_numpy() & reported).any()},
        }
        m_friendly = ((df["is_fraud"] == 0) & (df["tx_type"] == "CARD_PURCHASE")
                      & (df["status"] == "SUCCESS")).to_numpy() & (rng.random(len(df)) < self.cfg.friendly_fraud_rate)

        noise = pd.concat([
            pd.DataFrame({"row": np.where(m_unrep)[0], "observed_label": 0, "true_label": 1,
                          "true_fraud_type": df.loc[m_unrep, "fraud_type"].to_numpy(),
                          "noise_type": "UNREPORTED_FRAUD"}),
            pd.DataFrame({"row": np.where(m_friendly)[0], "observed_label": 1, "true_label": 0,
                          "true_fraud_type": None, "noise_type": "FRIENDLY_FRAUD"}),
        ], ignore_index=True)

        df.loc[m_unrep, ["is_fraud", "fraud_type", "fraud_episode_id"]] = [0, None, None]
        df.loc[m_unrep, "fraud_reported_s"] = np.nan
        for i in np.where(m_friendly)[0]:
            self._episode_counter += 1
            df.loc[i, ["is_fraud", "fraud_type", "fraud_episode_id"]] = [1, "FRIENDLY_FRAUD",
                                                                         f"EP{self._episode_counter:05d}"]
            df.loc[i, "fraud_reported_s"] = df.loc[i, "ts"] + self._report_delay_s("FRIENDLY_FRAUD")[0]
        return df, noise

    def _apply_balances(self, df: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        """Rejoue les transactions dans l'ordre chronologique pour tenir les soldes.

        Un utilisateur légitime connaît généralement son solde : 80 % des opérations
        légitimes qui dépasseraient le solde ne sont jamais tentées (ligne supprimée),
        les 20 % restantes échouent. Les tentatives frauduleuses échouées sont conservées."""
        rng = self.rng
        typical = users["typical_amount_usd"].to_numpy()
        limits = users["kyc_tx_limit_usd"].to_numpy()
        wallet = typical * 3 * rng.lognormal(0, 0.5, len(users))
        card = np.where(users["has_visa_virtual"].to_numpy(), typical * rng.lognormal(0, 0.5, len(users)), 0.0)

        n = len(df)
        uidx = df["uidx"].to_numpy()
        currency = df["currency"].to_numpy()
        tx_type = df["tx_type"].to_numpy()
        amount = df["amount_usd"].to_numpy(dtype=float).copy()
        drain = df["drain_frac"].to_numpy(dtype=float)
        is_fraud = df["is_fraud"].to_numpy()
        before, after = np.zeros(n), np.zeros(n)
        status = np.empty(n, dtype=object)
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            u, t = uidx[i], tx_type[i]
            if t in CREDIT_TYPES:
                before[i] = wallet[u]
                wallet[u] += amount[i]
                after[i], status[i] = wallet[u], "SUCCESS"
                continue
            bal = card[u] if t == "CARD_PURCHASE" else wallet[u]
            if not np.isnan(drain[i]):
                amt = min(drain[i] * bal, limits[u])
                # montant saisi par une personne (client ou fraudeur) : arrondi vers le bas
                amount[i] = (max(0.5, round(amt, 2)) if t == "CARD_PURCHASE"
                             else self._floor_human(amt, currency[i]))
            before[i] = bal
            if amount[i] > bal + 1e-9:
                if is_fraud[i] == 0 and rng.random() < 0.8:
                    keep[i] = False
                    continue
                after[i], status[i] = bal, "FAILED_INSUFFICIENT_FUNDS"
                continue
            if t == "CARD_PURCHASE":
                card[u] -= amount[i]
                after[i] = card[u]
            else:
                wallet[u] -= amount[i]
                after[i] = wallet[u]
                if t == "CARD_TOPUP":
                    card[u] += amount[i]
            status[i] = "SUCCESS"

        df["amount_usd"] = amount.round(2)
        df["balance_before_usd"] = before.round(2)
        df["balance_after_usd"] = after.round(2)
        df["status"] = status
        return df[keep].reset_index(drop=True)

    def run(self) -> dict[str, pd.DataFrame]:
        t0 = time.perf_counter()
        agents = self.generate_agents()
        merchants = self.generate_merchants()
        users = self.generate_users(agents)
        logger.info("Référentiels : %d utilisateurs, %d agents, %d marchands",
                    len(users), len(agents), len(merchants))

        normal_mm = self.generate_normal_mobile_money(users, agents, merchants)
        normal_card = self.generate_normal_card(users, merchants)
        logger.info("Transactions normales : %d Mobile Money, %d Visa", len(normal_mm), len(normal_card))

        fraud = self.inject_fraud(users, agents, merchants, len(normal_mm), len(normal_card))
        logger.info("Transactions frauduleuses injectées : %d", len(fraud))

        df = pd.DataFrame(normal_mm + normal_card + fraud)
        df = df[df["ts"] < self.horizon_s].sort_values("ts", kind="mergesort").reset_index(drop=True)
        df = self._round_amounts(df, users)
        df = self._apply_balances(df, users)
        df, noise = self._apply_label_noise(df)

        rate = self.cfg.usd_cdf_rate
        df["amount"] = np.where(df["currency"] == "CDF", (df["amount_usd"] * rate).round(0), df["amount_usd"])
        df["timestamp"] = self.start + pd.to_timedelta(df["ts"], unit="s")
        df["timestamp"] = df["timestamp"].dt.floor("s")
        # date à laquelle l'opérateur APPREND la fraude (peut dépasser la fin de la période)
        df["fraud_reported_at"] = (self.start + pd.to_timedelta(df["fraud_reported_s"], unit="s")).dt.floor("s")
        df.insert(0, "transaction_id", [f"TX{x:016x}" for x in self.rng.integers(0, 2 ** 63, len(df))])
        df = df[TRANSACTION_COLUMNS]
        noise.insert(0, "transaction_id", df["transaction_id"].to_numpy()[noise.pop("row").to_numpy()])
        logger.info("Bruit d'étiquetage : %d fraudes non signalées, %d contestations abusives",
                    (noise.noise_type == "UNREPORTED_FRAUD").sum(), (noise.noise_type == "FRIENDLY_FRAUD").sum())

        users_out = users.drop(columns=["device_change_ts", "home_agents", "contacts",
                                        "secondary_device_id", "tx_rate_per_day", "hour_shift", "night_owl"])
        logger.info("Génération terminée en %.1f s", time.perf_counter() - t0)
        return {"transactions": df, "users": users_out, "agents": agents, "merchants": merchants,
                "label_noise": noise}

    def save(self, data: dict[str, pd.DataFrame]) -> Path:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in data.items():
            frame.to_csv(out / f"{name}.csv", index=False)
        meta = {
            "config": asdict(self.cfg),
            "summary": summarize(data["transactions"]),
            "label_noise": data["label_noise"]["noise_type"].value_counts().to_dict(),
            "reporting": getattr(self, "_report_stats", {}),
            "data_dictionary": DATA_DICTIONARY,
            "hidden_labels": HIDDEN_LABEL_COLUMNS,
        }
        (out / "generation_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                                      encoding="utf-8")
        return out


TRANSACTION_COLUMNS = [
    "transaction_id", "timestamp", "user_id", "wallet_id", "card_id", "channel", "operator",
    "tx_type", "access_channel", "amount", "currency", "amount_usd", "balance_before_usd",
    "balance_after_usd", "status", "counterparty_id", "merchant_id", "merchant_category",
    "merchant_country", "agent_id", "device_id", "device_type", "ip_country",
    "location_province", "is_fraud", "fraud_type", "fraud_episode_id", "fraud_reported_at",
]

# Colonnes à exclure des variables d'entrée (fuite de label garantie sinon)
HIDDEN_LABEL_COLUMNS = {
    "transactions.csv": ["is_fraud (cible)", "fraud_type", "fraud_episode_id", "fraud_reported_at"],
    "users.csv": ["is_mule_account"],
    "agents.csv": ["is_compromised", "compromised_from_s", "compromised_until_s"],
    "label_noise.csv": ["fichier entier : vérité terrain des lignes à étiquette bruitée"],
}

DATA_DICTIONARY = {
    "transaction_id": "Identifiant unique de la transaction",
    "timestamp": "Date et heure (heure locale RDC)",
    "user_id": "Titulaire du compte (portefeuille et carte)",
    "wallet_id": "Numéro de portefeuille Mobile Money (format MSISDN fictif)",
    "card_id": "Identifiant de la carte Visa virtuelle (canal VISA_VIRTUAL uniquement)",
    "channel": "MOBILE_MONEY ou VISA_VIRTUAL",
    "operator": "Opérateur du portefeuille : VODACOM, AIRTEL, ORANGE",
    "tx_type": "P2P_SEND, P2P_RECEIVE, CASH_IN, CASH_OUT, MERCHANT_PAYMENT, AIRTIME, "
               "BILL_PAYMENT, CARD_TOPUP (portefeuille -> carte), CARD_PURCHASE",
    "access_channel": "USSD, APP, AGENT",
    "amount": "Montant dans la devise de la transaction",
    "currency": "CDF ou USD",
    "amount_usd": "Montant converti en USD",
    "balance_before_usd": "Solde du compte débité/crédité avant la transaction (USD)",
    "balance_after_usd": "Solde après la transaction (inchangé si échec)",
    "status": "SUCCESS ou FAILED_INSUFFICIENT_FUNDS",
    "counterparty_id": "Portefeuille contrepartie (P2P) ou opérateur (AIRTIME)",
    "merchant_id": "Marchand (paiements et achats carte)",
    "merchant_category": "Catégorie du marchand",
    "merchant_country": "Pays du marchand (ISO-2)",
    "agent_id": "Agent Mobile Money (CASH_IN / CASH_OUT)",
    "device_id": "Empreinte de l'appareil utilisé",
    "device_type": "smartphone ou feature_phone",
    "ip_country": "Pays de l'adresse IP (accès APP uniquement)",
    "location_province": "Province d'où la transaction est initiée",
    "is_fraud": "CIBLE : étiquette connue de l'opérateur, 1 si la transaction fait partie d'un épisode "
                "de fraude signalé (tentatives échouées incluses). Bruitée comme dans la réalité : "
                "vérité terrain dans label_noise.csv",
    "fraud_type": "Typologie de fraude (analyse uniquement, jamais en entrée du modèle) ; "
                  "FRIENDLY_FRAUD = achat légitime contesté par le titulaire",
    "fraud_episode_id": "Regroupe les transactions d'un même épisode (analyse uniquement)",
    "fraud_reported_at": "Date à laquelle l'opérateur apprend la fraude (plainte, enquête, contestation). "
                         "Ne sert qu'à rejouer les signalements dans l'ordre du temps (profil de réputation) ; "
                         "jamais en entrée du modèle",
}


def summarize(tx: pd.DataFrame) -> dict:
    by_channel = tx.groupby("channel").agg(n=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"))
    return {
        "n_transactions": int(len(tx)),
        "period": [str(tx["timestamp"].min()), str(tx["timestamp"].max())],
        "fraud_rate_global": round(float(tx["is_fraud"].mean()), 5),
        "by_channel": {c: {"n": int(r.n), "fraud_rate": round(float(r.fraud_rate), 5)}
                       for c, r in by_channel.iterrows()},
        "fraud_transactions_by_type": tx.loc[tx.is_fraud == 1, "fraud_type"].value_counts().to_dict(),
        "fraud_episodes_by_type": tx.loc[tx.is_fraud == 1].groupby("fraud_type")["fraud_episode_id"]
                                    .nunique().to_dict(),
        "failure_rate_legit": round(float((tx.loc[tx.is_fraud == 0, "status"] != "SUCCESS").mean()), 5),
        "failure_rate_fraud": round(float((tx.loc[tx.is_fraud == 1, "status"] != "SUCCESS").mean()), 5),
        "tx_type_distribution": tx["tx_type"].value_counts().to_dict(),
    }


def parse_args() -> GeneratorConfig:
    d = GeneratorConfig()
    p = argparse.ArgumentParser(description="Génère le jeu de données synthétique Mobile Money RDC + Visa virtuelle")
    p.add_argument("--n-users", type=int, default=d.n_users)
    p.add_argument("--n-days", type=int, default=d.n_days)
    p.add_argument("--start-date", default=d.start_date)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--mm-fraud-rate", type=float, default=d.mm_fraud_rate)
    p.add_argument("--card-fraud-rate", type=float, default=d.card_fraud_rate)
    p.add_argument("--visa-adoption-multiplier", type=float, default=d.visa_adoption_multiplier)
    p.add_argument("--output-dir", default=d.output_dir)
    a = p.parse_args()
    return GeneratorConfig(n_users=a.n_users, n_days=a.n_days, start_date=a.start_date, seed=a.seed,
                           mm_fraud_rate=a.mm_fraud_rate, card_fraud_rate=a.card_fraud_rate,
                           visa_adoption_multiplier=a.visa_adoption_multiplier,
                           output_dir=a.output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    gen = SyntheticDataGenerator(parse_args())
    data = gen.run()
    out = gen.save(data)
    print(json.dumps(summarize(data["transactions"]), indent=2, ensure_ascii=False))
    logger.info("Fichiers écrits dans %s", out.resolve())


if __name__ == "__main__":
    main()
